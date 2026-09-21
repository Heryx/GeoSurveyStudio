# -*- coding: utf-8 -*-
"""Pannello agganciabile con i radargrammi processati, nel canvas di QGIS.

Serve a continuare a lavorare con le time-slice sul canvas anche a Profile Viewer
chiuso: si sceglie variante, profilo e canale, si legge la profondita' e la si
confronta con la finestra della time-slice selezionata nel pannello Livelli.

- e' un ``QDockWidget``: si aggancia in basso o ai lati, oppure si stacca e si
  sposta liberamente;
- legge i radargrammi processati direttamente dai .npz della variante
  (``radargram_store``), senza il Profile Viewer;
- clic sulla mappa (strumento "Punta sulla mappa") -> apre profilo, canale e
  traccia piu' vicini; movimento sul radargramma -> punto giallo sulla mappa;
- selezionando una time-slice nel pannello Livelli mostra la sua finestra di
  profondita' sul radargramma (da ``sliceset.json``).
"""

from __future__ import annotations

import os

from qgis.PyQt.QtCore import QObject, QPoint, QRect, QSettings, QSize, Qt, QTimer, pyqtSignal
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsPointXY,
    QgsProject,
    QgsRasterLayer,
    QgsWkbTypes,
)
from qgis.gui import QgsMapToolEmitPoint, QgsRubberBand

import numpy as np

from .radargram_store import (
    VariantStore,
    discover_variants,
    slice_info_for_path,
    variant_dir_from_slice_dir,
)

try:
    from .gpr_viewer_common import (
        FigureCanvasQTAgg,
        Figure,
        HAS_MPL,
        _BlitManager,
        _populate_cmap_combo,
        normalize_display,
    )
    from .gpr_viewer_radargram import RadargramViewMixin
    from matplotlib.patches import Polygon
except Exception:                                    # matplotlib/scipy mancanti
    HAS_MPL = False

_SETTINGS = "GeoSurveyStudio/radargram_dock/"
_MAX_DISPLAY_COLS = 6000        # oltre, il radargramma si mostra decimato (media a blocchi)
_CURSOR_MAP_DELAY_MS = 50


def _fmt(v, nd=2, unit=""):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "n/d"
    return f"{v:.{nd}f}{unit}" if np.isfinite(v) else "n/d"


class _FlowLayout(QLayout):
    """Dispone i widget in riga e va a capo quando manca spazio.

    Il pannello si puo' agganciare in basso (largo) o ai lati (stretto): con righe
    fisse la larghezza minima dei controlli lo impedirebbe.
    """

    def __init__(self, parent=None, hspacing: int = 8, vspacing: int = 3):
        super().__init__(parent)
        self._items: list = []
        self._h, self._v = int(hspacing), int(vspacing)
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, i: int):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i: int):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientations(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._arrange(QRect(0, 0, int(width), 0), apply=False)

    def setGeometry(self, rect) -> None:
        super().setGeometry(rect)
        self._arrange(rect, apply=True)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for it in self._items:
            size = size.expandedTo(it.minimumSize())
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _arrange(self, rect, apply: bool) -> int:
        m = self.contentsMargins()
        area = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, line_h = area.x(), area.y(), 0
        for it in self._items:
            hint = it.sizeHint()
            if x > area.x() and x + hint.width() > area.right() + 1:
                x, y, line_h = area.x(), y + line_h + self._v, 0
            if apply:
                it.setGeometry(QRect(QPoint(x, y), hint))
            x += hint.width() + self._h
            line_h = max(line_h, hint.height())
        return y + line_h - rect.y() + m.bottom()


def _group(*widgets) -> QWidget:
    """Controlli che devono restare vicini quando il pannello va a capo."""
    box = QWidget()
    lay = QHBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(3)
    for w in widgets:
        lay.addWidget(w)
    return box


# ---------------------------------------------------------------------------
# Ponte verso il canvas: tracce, marcatore, strumento "Punta sulla mappa"
# ---------------------------------------------------------------------------

class _MapBridge(QObject):
    """Tutto cio' che tocca la mappa QGIS, in un posto solo."""

    picked = pyqtSignal(float, float, float)   # est, nord (CRS del profilo), tolleranza m
    tool_released = pyqtSignal()               # l'utente ha scelto un altro strumento

    _TRACK_MAX_POINTS = 300

    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self._iface = iface
        self._canvas = iface.mapCanvas() if iface is not None else None
        self._epsg = 0
        self._rb_line = None
        self._rb_pt = None
        self._tool = None
        self._tr_cache: dict = {}
        self._visible = True

    # -- CRS ---------------------------------------------------------------

    def set_epsg(self, epsg: int) -> None:
        self._epsg = int(epsg or 0)

    def _profile_crs(self):
        if self._epsg > 0:
            crs = QgsCoordinateReferenceSystem.fromEpsgId(self._epsg)
            if crs.isValid():
                return crs
        return QgsProject.instance().crs()      # EPSG assente: come fa il viewer

    def _transform(self, to_canvas: bool):
        if self._canvas is None:
            return None
        prof, dst = self._profile_crs(), self._canvas.mapSettings().destinationCrs()
        if not prof.isValid() or not dst.isValid() or prof == dst:
            return None
        key = (prof.authid(), dst.authid(), to_canvas)
        tr = self._tr_cache.get(key)
        if tr is None:
            tr = (QgsCoordinateTransform(prof, dst, QgsProject.instance()) if to_canvas
                  else QgsCoordinateTransform(dst, prof, QgsProject.instance()))
            self._tr_cache[key] = tr
        return tr

    def to_canvas(self, e: float, n: float) -> QgsPointXY:
        pt = QgsPointXY(float(e), float(n))
        tr = self._transform(True)
        try:
            return tr.transform(pt) if tr is not None else pt
        except Exception:
            return pt

    def from_canvas(self, pt: QgsPointXY) -> QgsPointXY:
        tr = self._transform(False)
        try:
            return tr.transform(pt) if tr is not None else QgsPointXY(pt)
        except Exception:
            return QgsPointXY(pt)

    # -- rubber band ---------------------------------------------------------

    def set_visible(self, on: bool) -> None:
        self._visible = bool(on)
        for rb in (self._rb_line, self._rb_pt):
            if rb is not None:
                rb.setVisible(self._visible)

    def _band(self, kind: str):
        if self._canvas is None:
            return None
        if kind == "line":
            if self._rb_line is None:
                self._rb_line = QgsRubberBand(self._canvas, QgsWkbTypes.LineGeometry)
                self._rb_line.setColor(QColor(255, 120, 0))
                self._rb_line.setWidth(3)
            self._rb_line.setVisible(self._visible)
            return self._rb_line
        if self._rb_pt is None:
            self._rb_pt = QgsRubberBand(self._canvas, QgsWkbTypes.PointGeometry)
            self._rb_pt.setColor(QColor(255, 220, 0))
            self._rb_pt.setIconSize(12)
            self._rb_pt.setWidth(3)
        self._rb_pt.setVisible(self._visible)
        return self._rb_pt

    def show_track(self, east, north, pan_if_hidden: bool = False) -> None:
        rb = self._band("line")
        if rb is None:
            return
        rb.reset(QgsWkbTypes.LineGeometry)
        e, n = np.asarray(east, dtype=np.float64), np.asarray(north, dtype=np.float64)
        ok = np.isfinite(e) & np.isfinite(n)
        e, n = e[ok], n[ok]
        if e.size < 2:
            return
        step = max(1, e.size // self._TRACK_MAX_POINTS)
        idx = np.unique(np.r_[np.arange(0, e.size, step), e.size - 1])
        for k, i in enumerate(idx):
            rb.addPoint(self.to_canvas(e[i], n[i]), k == len(idx) - 1)
        if pan_if_hidden:
            try:
                box = rb.asGeometry().boundingBox()
                if not self._canvas.extent().intersects(box):
                    self._canvas.setCenter(box.center())      # solo pan: la scala resta quella dell'utente
                    self._canvas.refresh()
            except Exception:
                pass

    def show_point(self, pos) -> None:
        rb = self._band("point")
        if rb is None:
            return
        rb.reset(QgsWkbTypes.PointGeometry)
        if pos is not None:
            rb.addPoint(self.to_canvas(pos[0], pos[1]), True)

    def clear(self) -> None:
        if self._rb_line is not None:
            self._rb_line.reset(QgsWkbTypes.LineGeometry)
        if self._rb_pt is not None:
            self._rb_pt.reset(QgsWkbTypes.PointGeometry)

    def dispose(self) -> None:
        self.set_pick_active(False)
        for rb in (self._rb_line, self._rb_pt):
            if rb is None:
                continue
            try:
                rb.reset()
                if self._canvas is not None and self._canvas.scene() is not None:
                    self._canvas.scene().removeItem(rb)
            except Exception:
                pass
        self._rb_line = self._rb_pt = None

    # -- strumento "Punta sulla mappa" -----------------------------------------

    def set_pick_active(self, on: bool) -> None:
        if self._canvas is None:
            return
        if on:
            if self._tool is None:
                self._tool = QgsMapToolEmitPoint(self._canvas)
                self._tool.setCursor(Qt.CrossCursor)
                self._tool.canvasClicked.connect(self._on_canvas_clicked)
                self._canvas.mapToolSet.connect(self._on_map_tool_set)
            self._canvas.setMapTool(self._tool)
        elif self._tool is not None:
            try:
                if self._canvas.mapTool() is self._tool:
                    self._canvas.unsetMapTool(self._tool)
            except Exception:
                pass

    def _on_map_tool_set(self, new_tool, _old) -> None:
        if self._tool is not None and new_tool is not self._tool:
            self.tool_released.emit()

    def _on_canvas_clicked(self, pt, button) -> None:
        if button != Qt.LeftButton:
            return
        here = self.from_canvas(QgsPointXY(pt))
        # 10 px sullo schermo, espressi nel CRS del profilo (metri): il clic e' tollerante a ogni scala
        px = QgsPointXY(pt.x() + 10.0 * self._canvas.mapUnitsPerPixel(), pt.y())
        tol = max(2.0, float(here.distance(self.from_canvas(px))))
        self.picked.emit(float(here.x()), float(here.y()), tol)


# ---------------------------------------------------------------------------
# Pannello
# ---------------------------------------------------------------------------

class RadargramDock(QDockWidget):
    """Radargrammi processati, agganciabili al canvas di QGIS."""

    # Tolleranza minima del clic sulla mappa (m): sotto non c'e' nessun profilo "sotto" il punto.
    MIN_PICK_TOL_M = 2.0

    def __init__(self, iface, workspace_provider=None, geometry_provider=None, parent=None):
        super().__init__("Radargrammi processati", parent or (iface.mainWindow() if iface else None))
        self.setObjectName("GeoSurveyRadargramDock")
        self.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
            | QDockWidget.DockWidgetClosable
        )
        self.iface = iface
        self._workspace_provider = workspace_provider
        self._geometry_provider = geometry_provider
        self._store: VariantStore | None = None
        self._pi = 0
        self._ci = 0
        self._cur = None                 # ChannelData mostrato
        self._base = None                # radargramma normalizzato (prima del gain)
        self._decim = 1
        self._cursor_x = None
        self._cursor_z = None
        self._ref_z = None               # profondita' di riferimento (spin / clic / time-slice)
        self._band = None                # (z_from, z_to) della time-slice selezionata
        self._updating = False
        self._fig = self._ax = self._canvas = self._blit = None
        self._im = None
        self._vline = self._hline = self._ref_line = self._band_patch = None
        self._pan = None
        self._map = _MapBridge(iface, self) if (iface is not None and HAS_MPL) else None
        self._layer_signal = None

        self._map_timer = QTimer(self)
        self._map_timer.setSingleShot(True)
        self._map_timer.setInterval(_CURSOR_MAP_DELAY_MS)
        self._map_timer.timeout.connect(self._flush_map_point)

        self._build_ui()
        if self._map is not None:
            self._map.picked.connect(self._on_map_picked)
            self._map.tool_released.connect(self._on_pick_released)
        self._connect_layer_signal()
        self.refresh_variants()
        if iface is not None:
            self._on_current_layer_changed(iface.activeLayer())

    # ------------------------------------------------------------------
    # Interfaccia
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        host = QWidget(self)
        root = QVBoxLayout(host)
        root.setContentsMargins(4, 4, 4, 2)
        root.setSpacing(3)

        if not HAS_MPL:
            root.addWidget(QLabel("matplotlib non trovato: impossibile mostrare i radargrammi.\n"
                                  "Installa con: pip install matplotlib scipy"))
            self.setWidget(host)
            return

        def _combo(min_chars):
            cb = QComboBox()
            cb.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            cb.setMinimumContentsLength(min_chars)
            return cb

        def _tool(text, tip):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.setAutoRaise(True)
            return b

        ctrl = QWidget()
        flow = _FlowLayout(ctrl)
        root.addWidget(ctrl)

        self._cb_variant = _combo(14)
        self._cb_variant.setToolTip("Radargrammi processati salvati nel progetto (GPR_Processing/processed).")
        self._cb_variant.currentIndexChanged.connect(self._on_variant_changed)
        self._btn_refresh = _tool("\u27f3", "Rileggi le varianti e le coordinate (dopo un nuovo posizionamento o salvataggio).")
        self._btn_refresh.clicked.connect(self.refresh)
        self._btn_browse = _tool("\u2026", "Scegli a mano una cartella-variante (.npz).")
        self._btn_browse.clicked.connect(self._browse_variant)
        flow.addWidget(_group(QLabel("Variante"), self._cb_variant, self._btn_refresh, self._btn_browse))

        self._btn_prev_p = _tool("\u25c0", "Profilo precedente (PgSu)")
        self._btn_prev_p.clicked.connect(lambda: self.select_profile(self._pi - 1))
        self._cb_profile = _combo(20)
        self._cb_profile.setMaxVisibleItems(30)
        self._cb_profile.currentIndexChanged.connect(self._on_profile_combo)
        self._btn_next_p = _tool("\u25b6", "Profilo successivo (PgGiu)")
        self._btn_next_p.clicked.connect(lambda: self.select_profile(self._pi + 1))
        flow.addWidget(_group(QLabel("Profilo"), self._btn_prev_p, self._cb_profile, self._btn_next_p))

        self._btn_prev_c = _tool("\u25c0", "Canale precedente (freccia sinistra)")
        self._btn_prev_c.clicked.connect(lambda: self.select_profile(self._pi, self._ci - 1))
        self._cb_channel = _combo(3)
        self._cb_channel.currentIndexChanged.connect(self._on_channel_combo)
        self._btn_next_c = _tool("\u25b6", "Canale successivo (freccia destra)")
        self._btn_next_c.clicked.connect(lambda: self.select_profile(self._pi, self._ci + 1))
        flow.addWidget(_group(QLabel("Canale"), self._btn_prev_c, self._cb_channel, self._btn_next_c))

        self._spin_depth = QDoubleSpinBox()
        self._spin_depth.setDecimals(3)
        self._spin_depth.setSingleStep(0.05)
        self._spin_depth.setRange(0.0, 50.0)
        self._spin_depth.setSuffix(" m")
        self._spin_depth.setToolTip(
            "Profondit\u00e0 di riferimento (riga verde): si imposta qui, con un clic sul\n"
            "radargramma, con le frecce su/gi\u00f9, o seguendo la time-slice selezionata.")
        self._spin_depth.valueChanged.connect(self._on_depth_spin)
        flow.addWidget(_group(QLabel("Profondit\u00e0"), self._spin_depth))

        self._spin_gain = QDoubleSpinBox()
        self._spin_gain.setRange(0.2, 30.0)
        self._spin_gain.setSingleStep(0.5)
        try:
            self._spin_gain.setValue(float(QSettings().value(_SETTINGS + "gain", 1.0)))
        except (TypeError, ValueError):
            self._spin_gain.setValue(1.0)
        self._spin_gain.setToolTip("Moltiplicatore di contrasto (solo visualizzazione).")
        self._spin_gain.valueChanged.connect(self._on_look_changed)
        self._cb_cmap = _combo(8)
        _populate_cmap_combo(self._cb_cmap, current=str(QSettings().value(_SETTINGS + "cmap", "") or "") or None)
        self._cb_cmap.currentIndexChanged.connect(self._on_look_changed)
        flow.addWidget(_group(QLabel("Gain"), self._spin_gain, self._cb_cmap))

        self._chk_follow = QCheckBox("Segui time-slice")
        self._chk_follow.setChecked(True)
        self._chk_follow.setToolTip(
            "Selezionando una time-slice nel pannello Livelli, il radargramma mostra la sua\n"
            "finestra di profondit\u00e0 (fascia verde) e passa alla variante da cui \u00e8 stata generata.")
        self._chk_follow.toggled.connect(self._on_follow_toggled)
        flow.addWidget(self._chk_follow)
        self._chk_map = QCheckBox("Sulla mappa")
        self._chk_map.setChecked(True)
        self._chk_map.setToolTip("Disegna sulla mappa la traccia del profilo e il punto sotto il cursore.")
        self._chk_map.toggled.connect(self._on_map_toggled)
        flow.addWidget(self._chk_map)
        self._btn_pick = QPushButton("Punta sulla mappa")
        self._btn_pick.setCheckable(True)
        self._btn_pick.setToolTip(
            "Attiva lo strumento: un clic sulla mappa apre il profilo, il canale e la traccia\n"
            "pi\u00f9 vicini al punto. Si disattiva scegliendo un altro strumento di QGIS.")
        self._btn_pick.toggled.connect(self._on_pick_toggled)
        self._btn_zoom = _tool("\u2922", "Vista intera (doppio clic sul radargramma)")
        self._btn_zoom.clicked.connect(self._reset_zoom)
        flow.addWidget(_group(self._btn_pick, self._btn_zoom))

        self._fig = Figure(figsize=(8, 2.6))
        self._ax = self._fig.add_axes([0.06, 0.17, 0.92, 0.80])
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setMinimumHeight(120)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._canvas.setFocusPolicy(Qt.StrongFocus)
        self._blit = _BlitManager(self._canvas)
        self._canvas.mpl_connect("resize_event", self._layout_axes)
        self._canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self._canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self._canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self._canvas.mpl_connect("scroll_event", self._on_scroll)
        self._canvas.mpl_connect("axes_leave_event", self._on_axes_leave)
        self._canvas.mpl_connect("key_press_event", self._on_key)
        root.addWidget(self._canvas, 1)

        self._lbl_status = QLabel("")
        self._lbl_status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self._lbl_status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self._lbl_status)
        self.setWidget(host)
        self._show_message("Nessuna variante processata: salva una variante dal Profile Viewer "
                           "oppure scegli una cartella con …")

    # ------------------------------------------------------------------
    # Varianti
    # ------------------------------------------------------------------

    def _workspace_root(self) -> str:
        try:
            return str(self._workspace_provider() or "") if self._workspace_provider else ""
        except Exception:
            return ""

    def refresh_variants(self, select: str | None = None, force: bool = False) -> None:
        """Rilegge l'elenco delle varianti. `select` = nome o cartella da selezionare;
        `force` ricarica anche la variante gia' aperta."""
        if not HAS_MPL:
            return
        keep_dir = select or (self._store.path if self._store is not None else "")
        refs = discover_variants(self._workspace_root())
        self._updating = True
        try:
            self._cb_variant.clear()
            for r in refs:
                self._cb_variant.addItem(f"{r.name}  ({r.n_files})", r.path)
            # una cartella scelta a mano fuori dal progetto resta nell'elenco
            if keep_dir and os.path.isdir(keep_dir) and all(
                    os.path.normcase(os.path.normpath(keep_dir)) != os.path.normcase(r.path) for r in refs):
                self._cb_variant.addItem(os.path.basename(os.path.normpath(keep_dir)), os.path.normpath(keep_dir))
            want = -1
            for i in range(self._cb_variant.count()):
                d = str(self._cb_variant.itemData(i))
                if keep_dir and (os.path.normcase(d) == os.path.normcase(os.path.normpath(keep_dir))
                                 or os.path.basename(d) == keep_dir):
                    want = i
                    break
            if want < 0 and self._cb_variant.count():
                names = [os.path.basename(str(self._cb_variant.itemData(i))) for i in range(self._cb_variant.count())]
                want = names.index("Lavoro") if "Lavoro" in names else 0
            self._cb_variant.setCurrentIndex(want)
        finally:
            self._updating = False
        if self._cb_variant.count() == 0:
            self._set_store(None)
        else:
            self._open_variant_dir(str(self._cb_variant.currentData()), force=force)

    def refresh(self) -> None:
        """Rilegge varianti e coordinate (dopo un nuovo posizionamento o un nuovo salvataggio)."""
        if self._store is not None:
            self._store.invalidate_geometry()
        self.refresh_variants(select=self._store.path if self._store is not None else None, force=True)

    def open_variant(self, name_or_dir: str, profile_stem: str | None = None, channel: int | None = None) -> bool:
        """Apre la variante indicata (nome o cartella), opzionalmente su un profilo/canale."""
        if not HAS_MPL:
            return False
        target = str(name_or_dir or "").strip()
        if not target:
            return False
        if not any(os.path.basename(str(self._cb_variant.itemData(i))) == target
                   or os.path.normcase(str(self._cb_variant.itemData(i))) == os.path.normcase(os.path.normpath(target))
                   for i in range(self._cb_variant.count())):
            self.refresh_variants(select=target)
        for i in range(self._cb_variant.count()):
            d = str(self._cb_variant.itemData(i))
            if os.path.basename(d) == target or os.path.normcase(d) == os.path.normcase(os.path.normpath(target)):
                if self._cb_variant.currentIndex() != i:
                    self._cb_variant.setCurrentIndex(i)      # -> _on_variant_changed
                break
        else:
            return False
        if self._store is None:
            return False
        pi = self._store.profile_index_for_stem(profile_stem) if profile_stem else -1
        if pi >= 0 or channel is not None:
            self.select_profile(pi if pi >= 0 else self._pi, channel)
        return True

    def _browse_variant(self) -> None:
        start = self._store.path if self._store is not None else (self._workspace_root() or os.path.expanduser("~"))
        d = QFileDialog.getExistingDirectory(self, "Cartella con i radargrammi processati (.npz)", start)
        if d:
            self.refresh_variants(select=d)

    def _on_variant_changed(self, _i: int) -> None:
        if self._updating:
            return
        d = self._cb_variant.currentData()
        if d:
            self._open_variant_dir(str(d))

    def _open_variant_dir(self, path: str, force: bool = False) -> None:
        same = (self._store is not None
                and os.path.normcase(self._store.path) == os.path.normcase(os.path.normpath(path)))
        if same and not force and len(self._store) and self._cur is not None \
                and not self._store.is_stale():
            return                      # gia' aperta e invariata su disco
        prev_stem = None
        if self._store is not None and 0 <= self._pi < len(self._store):
            prev_stem = self._store.profiles[self._pi].stem
        store = VariantStore(path, geometry_provider=self._geometry_provider)
        self._set_store(store, keep_stem=prev_stem)

    def _set_store(self, store, keep_stem: str | None = None) -> None:
        self._store = store
        self._cur = None
        self._updating = True
        try:
            self._cb_profile.clear()
            self._cb_channel.clear()
            if store is not None and len(store):
                n = len(store)
                for i, p in enumerate(store.profiles):
                    self._cb_profile.addItem(f"{i + 1}/{n}  {p.stem}")
        finally:
            self._updating = False
        if store is None or not len(store):
            self._draw_empty("Nessuna variante processata: salva una variante dal Profile Viewer "
                             "oppure scegli una cartella con …")
            if self._map is not None:
                self._map.clear()
            return
        pi = store.profile_index_for_stem(keep_stem) if keep_stem else -1
        self._pi = pi if pi >= 0 else 0
        self._ci = min(self._ci, store.profiles[self._pi].n_channels - 1)
        self._load_current(reset_view=True)

    # ------------------------------------------------------------------
    # Selezione profilo / canale
    # ------------------------------------------------------------------

    def select_profile(self, pi: int, ci: int | None = None) -> None:
        """Unico punto di cambio profilo/canale (pulsanti, combo, tasti, mappa)."""
        if self._store is None or not len(self._store):
            return
        pi = int(np.clip(int(pi), 0, len(self._store) - 1))
        n_ch = self._store.profiles[pi].n_channels
        ci = int(np.clip(int(self._ci if ci is None else ci), 0, n_ch - 1))
        if pi == self._pi and ci == self._ci and self._cur is not None:
            return
        same_profile = pi == self._pi
        self._pi, self._ci = pi, ci
        self._load_current(reset_view=not same_profile, keep_y=True)

    def _on_profile_combo(self, i: int) -> None:
        if not self._updating and i >= 0:
            self.select_profile(int(i))

    def _on_channel_combo(self, i: int) -> None:
        if not self._updating and i >= 0:
            self.select_profile(self._pi, int(i))

    def _sync_combos(self) -> None:
        info = self._store.profiles[self._pi]
        self._updating = True
        try:
            self._cb_profile.setCurrentIndex(self._pi)
            if self._cb_channel.count() != info.n_channels:
                self._cb_channel.clear()
                self._cb_channel.addItems([str(i) for i in range(info.n_channels)])
            self._cb_channel.setCurrentIndex(self._ci)
            for b, ok in ((self._btn_prev_p, self._pi > 0),
                          (self._btn_next_p, self._pi < len(self._store) - 1),
                          (self._btn_prev_c, self._ci > 0),
                          (self._btn_next_c, self._ci < info.n_channels - 1)):
                b.setEnabled(ok)
        finally:
            self._updating = False

    def _load_current(self, reset_view: bool = True, keep_y: bool = False) -> None:
        try:
            ch = self._store.channel(self._pi, self._ci)
        except Exception as exc:
            self._cur = None
            self._draw_empty(f"Impossibile leggere il canale: {exc}")
            return
        y_keep = self._ax.get_ylim() if (keep_y and self._im is not None) else None
        self._cur = ch
        self._prepare_display(ch)
        self._spin_depth.blockSignals(True)
        self._spin_depth.setRange(0.0, max(0.05, ch.depth_max_m))
        if self._ref_z is not None:
            self._spin_depth.setValue(float(np.clip(self._ref_z, 0.0, ch.depth_max_m)))
        self._spin_depth.blockSignals(False)
        self._sync_combos()
        self._redraw(reset_view=reset_view, y_keep=y_keep)
        if self._map is not None:
            self._map.set_epsg(self._store.profiles[self._pi].epsg)
            self._map.show_track(ch.east, ch.north, pan_if_hidden=True)
            self._map.show_point(None)
        self._update_status()

    # ------------------------------------------------------------------
    # Disegno
    # ------------------------------------------------------------------

    def _prepare_display(self, ch) -> None:
        data = ch.data
        self._decim = 1
        if data.shape[1] > _MAX_DISPLAY_COLS:
            self._decim = int(np.ceil(data.shape[1] / float(_MAX_DISPLAY_COLS)))
        # stessa normalizzazione del Profile Viewer
        base = normalize_display(data).astype(np.float32, copy=False)
        dist = ch.dist
        self._base = RadargramViewMixin._resample_disp_to_uniform_grid(base, dist)
        if self._decim > 1:
            n = (self._base.shape[1] // self._decim) * self._decim
            self._base = self._base[:, :n].reshape(self._base.shape[0], n // self._decim,
                                                   self._decim).mean(axis=2, dtype=np.float32)

    def _layout_axes(self, _event=None) -> None:
        """Margini fissi in punti: con il pannello stretto le etichette degli assi non si tagliano."""
        fig = self._fig
        w, h = float(fig.bbox.width), float(fig.bbox.height)
        if w < 60 or h < 60:
            return
        k = fig.dpi / 72.0
        left, right, top, bottom = 38 * k, 8 * k, 5 * k, 27 * k
        self._ax.set_position([left / w, bottom / h,
                               max(0.1, 1.0 - (left + right) / w), max(0.1, 1.0 - (top + bottom) / h)])

    def _draw_empty(self, text: str) -> None:
        if not HAS_MPL or self._ax is None:
            return
        self._im = None
        self._base = None
        self._ax.clear()
        self._ax.set_axis_off()
        self._ax.text(0.5, 0.5, text, ha="center", va="center", transform=self._ax.transAxes, fontsize=9,
                      color="gray")
        self._vline = self._hline = self._ref_line = self._band_patch = None
        self._blit.set_artists([])
        self._canvas.draw_idle()
        self._show_message(text)

    def _show_message(self, text: str) -> None:
        if getattr(self, "_lbl_status", None) is not None:
            self._lbl_status.setText(text)
            self._lbl_status.setToolTip(text)

    def _redraw(self, reset_view: bool = True, y_keep=None) -> None:
        ch = self._cur
        if ch is None or self._base is None:
            return
        ax = self._ax
        x_view = None if reset_view else ax.get_xlim()
        ax.clear()
        ax.set_axis_on()
        disp = np.clip(self._base * float(self._spin_gain.value()), -1.0, 1.0)
        length = max(ch.length_m, 1e-6)
        depth_max = ch.depth_max_m
        self._im = ax.imshow(
            disp, aspect="auto", cmap=self._cb_cmap.currentText() or "RdBu_r", vmin=-1.0, vmax=1.0,
            extent=[0.0, length, depth_max, 0.0], interpolation="bilinear",
            interpolation_stage="data", resample=True,
        )
        ax.set_xlim(*(x_view if x_view is not None else (0.0, length)))
        if y_keep is not None:
            lo, hi = sorted(float(v) for v in y_keep)
            y_keep = (min(hi, depth_max), max(lo, 0.0)) if lo < depth_max else None
        ax.set_ylim(*(y_keep if y_keep is not None else (depth_max, 0.0)))
        ax.set_xlabel("Distanza (m)", fontsize=8, labelpad=1)
        ax.set_ylabel("Profondità (m)", fontsize=8, labelpad=1)
        ax.tick_params(labelsize=8, pad=1)
        self._layout_axes()
        # artisti animati: sono ridisegnati a ogni movimento del mouse senza rifare l'immagine
        self._band_patch = Polygon(
            [[0, 0], [0, 0], [1, 0], [1, 0]], closed=True, transform=ax.get_yaxis_transform(),
            facecolor="#00c853", edgecolor="none", alpha=0.25, visible=False)
        ax.add_artist(self._band_patch)
        self._ref_line = ax.axhline(0.0, color="#00e676", lw=1.2, visible=False)
        self._vline = ax.axvline(0.0, color="yellow", lw=1, visible=False)
        self._hline = ax.axhline(0.0, color="cyan", lw=1, ls="--", visible=False)
        self._blit.set_artists([self._band_patch, self._ref_line, self._vline, self._hline])
        self._apply_reference()
        self._blit.invalidate()
        self._canvas.draw_idle()

    def _apply_reference(self) -> None:
        """Riporta profondita' di riferimento e fascia della time-slice sugli artisti."""
        if self._ref_line is None:
            return
        if self._ref_z is not None:
            self._ref_line.set_ydata([self._ref_z, self._ref_z])
            self._ref_line.set_visible(True)
        else:
            self._ref_line.set_visible(False)
        if self._band is not None and self._band_patch is not None:
            z0, z1 = self._band
            self._band_patch.set_xy([[0, z0], [0, z1], [1, z1], [1, z0]])
            self._band_patch.set_visible(True)
        elif self._band_patch is not None:
            self._band_patch.set_visible(False)

    def _reset_zoom(self) -> None:
        if self._cur is not None:
            self._redraw(reset_view=True)

    def _on_look_changed(self, *_a) -> None:
        if self._cur is None:
            return
        try:
            s = QSettings()
            s.setValue(_SETTINGS + "gain", float(self._spin_gain.value()))
            s.setValue(_SETTINGS + "cmap", self._cb_cmap.currentText())
        except Exception:
            pass
        self._redraw(reset_view=False, y_keep=self._ax.get_ylim() if self._im is not None else None)

    # ------------------------------------------------------------------
    # Profondita' di riferimento e time-slice
    # ------------------------------------------------------------------

    def set_reference_depth(self, z: float | None, band=None, from_spin: bool = False) -> None:
        """Profondita' di riferimento (riga verde) ed eventuale fascia (z_da, z_a)."""
        self._ref_z = None if z is None else float(z)
        self._band = None if band is None else (float(band[0]), float(band[1]))
        if self._ref_z is not None and not from_spin:
            self._spin_depth.blockSignals(True)
            self._spin_depth.setValue(float(np.clip(self._ref_z, self._spin_depth.minimum(), self._spin_depth.maximum())))
            self._spin_depth.blockSignals(False)
        self._apply_reference()
        if self._blit is not None:
            self._blit.update()
        self._update_status()

    def _on_depth_spin(self, v: float) -> None:
        self._band = None       # a mano: la fascia della time-slice non vale piu'
        self.set_reference_depth(float(v), from_spin=True)

    def _on_follow_toggled(self, on: bool) -> None:
        if on:
            self._on_current_layer_changed(self.iface.activeLayer() if self.iface else None)
        else:
            self._band = None
            self._apply_reference()
            self._blit.update()

    def _connect_layer_signal(self) -> None:
        if self.iface is None:
            return
        try:
            self.iface.currentLayerChanged.connect(self._on_current_layer_changed)
            self._layer_signal = self.iface.currentLayerChanged
        except Exception:
            self._layer_signal = None

    def _on_current_layer_changed(self, layer) -> None:
        if not HAS_MPL or not self._chk_follow.isChecked() or layer is None:
            return
        if not isinstance(layer, QgsRasterLayer):
            return
        info = slice_info_for_path(layer.source())
        if info is None:
            return
        variant_dir = variant_dir_from_slice_dir(info["dir"], info.get("variant", ""))
        if variant_dir and (self._store is None
                            or os.path.normcase(self._store.path) != os.path.normcase(os.path.normpath(variant_dir))):
            self.open_variant(variant_dir)
        self.set_reference_depth(info["z_center"], band=(info["z_from"], info["z_to"]))

    # ------------------------------------------------------------------
    # Mouse e tastiera
    # ------------------------------------------------------------------

    def _on_mouse_move(self, event) -> None:
        if self._pan is not None and event.inaxes is self._ax and event.xdata is not None:
            self._do_pan(event)
            return
        if event.inaxes is not self._ax or self._cur is None or event.xdata is None:
            return
        self._cursor_x, self._cursor_z = float(event.xdata), float(event.ydata)
        self._vline.set_xdata([self._cursor_x, self._cursor_x])
        self._vline.set_visible(True)
        self._hline.set_ydata([self._cursor_z, self._cursor_z])
        self._hline.set_visible(True)
        self._update_status()
        self._blit.update()
        self._map_timer.start()

    def _on_axes_leave(self, _event) -> None:
        if self._vline is not None:
            self._vline.set_visible(False)
            self._hline.set_visible(False)
            self._blit.update()

    def _on_mouse_press(self, event) -> None:
        if event.inaxes is not self._ax or self._cur is None or event.xdata is None:
            return
        self._canvas.setFocus()
        if event.button == 1:
            if getattr(event, "dblclick", False):
                self._reset_zoom()
            else:
                # clic: la profondita' cliccata diventa il riferimento
                self._band = None
                self.set_reference_depth(float(np.clip(event.ydata, 0.0, self._cur.depth_max_m)))
        elif event.button in (2, 3):
            self._pan = (event.x, event.y, self._ax.get_xlim(), self._ax.get_ylim())

    def _on_mouse_release(self, _event) -> None:
        self._pan = None

    def _do_pan(self, event) -> None:
        x0, y0, xl, yl = self._pan
        bb = self._ax.bbox
        dx = (event.x - x0) / max(bb.width, 1.0) * (xl[1] - xl[0])
        dy = (event.y - y0) / max(bb.height, 1.0) * (yl[1] - yl[0])
        self._set_view(xl[0] - dx, xl[1] - dx, yl[0] - dy, yl[1] - dy)

    def _set_view(self, x0, x1, y0, y1) -> None:
        ch = self._cur
        if ch is None:
            return

        def _clamp(a, b, lo, hi):
            span = min(abs(b - a), hi - lo)
            rev = a > b
            lo_v = min(max(min(a, b), lo), hi - span)
            return (lo_v + span, lo_v) if rev else (lo_v, lo_v + span)

        self._ax.set_xlim(*_clamp(x0, x1, 0.0, max(ch.length_m, 1e-6)))
        self._ax.set_ylim(*_clamp(y0, y1, 0.0, ch.depth_max_m))
        self._blit.invalidate()
        self._canvas.draw_idle()

    def _on_scroll(self, event) -> None:
        if event.inaxes is not self._ax or self._cur is None or event.xdata is None:
            return
        scale = 1.0 / 1.2 if (getattr(event, "step", 0) > 0 or event.button == "up") else 1.2
        x0, x1 = self._ax.get_xlim()
        y0, y1 = self._ax.get_ylim()
        x, y = float(event.xdata), float(event.ydata)
        self._set_view(x - (x - x0) * scale, x + (x1 - x) * scale,
                       y - (y - y0) * scale, y + (y1 - y) * scale)

    def _on_key(self, event) -> None:
        key = str(getattr(event, "key", "") or "")
        step = float(self._spin_depth.singleStep())
        if key == "pageup":
            self.select_profile(self._pi - 1)
        elif key == "pagedown":
            self.select_profile(self._pi + 1)
        elif key == "left":
            self.select_profile(self._pi, self._ci - 1)
        elif key == "right":
            self.select_profile(self._pi, self._ci + 1)
        elif key in ("up", "down") and self._cur is not None:
            z = float(self._ref_z if self._ref_z is not None else (self._cursor_z or 0.0))
            self._band = None
            self.set_reference_depth(float(np.clip(z + (step if key == "down" else -step), 0.0, self._cur.depth_max_m)))

    # ------------------------------------------------------------------
    # Lettura del cursore
    # ------------------------------------------------------------------

    def _cursor_readout(self):
        ch = self._cur
        if ch is None or self._cursor_x is None or self._cursor_z is None:
            return None
        tr = ch.trace_at_distance(self._cursor_x)
        row = ch.row_at_depth(self._cursor_z)
        info = self._store.profiles[self._pi]
        return {
            "trace": tr, "row": row,
            "amp": float(ch.data[row, tr]),
            "time_ns": (row + 0.5) * info.dt_ns,
            "pos": ch.position(tr),
        }

    def _update_status(self) -> None:
        if self._cur is None or self._store is None:
            return
        info = self._store.profiles[self._pi]
        head = (f"Profilo {self._pi + 1}/{len(self._store)}  Ch {self._ci}  "
                f"({self._store.name})")
        ref = ""
        if self._ref_z is not None:
            ref = f"  |  riferimento {self._ref_z:.3f} m"
            if self._band is not None:
                ref += f" (time-slice {self._band[0]:.2f}–{self._band[1]:.2f} m)"
        r = self._cursor_readout()
        cur = ""
        if r is not None:
            pos = r["pos"]
            cur = (f"  |  x {self._cursor_x:.2f} m  z {self._cursor_z:.3f} m  ({r['time_ns']:.1f} ns)  "
                   f"traccia {r['trace']}  amp {_fmt(r['amp'], 4)}")
            if pos is not None:
                cur += f"  E {pos[0]:.2f}  N {pos[1]:.2f}"
        geom = "" if self._cur.geometry_source == "viewer" else "  |  coordinate: salvate nel .npz"
        dec = f"  |  vista decimata x{self._decim}" if self._decim > 1 else ""
        self._show_message(head + ref + cur + geom + dec)

    # ------------------------------------------------------------------
    # Mappa
    # ------------------------------------------------------------------

    def _flush_map_point(self) -> None:
        if self._map is None or self._cur is None or not self._chk_map.isChecked():
            return
        r = self._cursor_readout()
        self._map.show_point(r["pos"] if r is not None else None)

    def _on_map_toggled(self, on: bool) -> None:
        if self._map is None:
            return
        self._map.set_visible(on)
        if on and self._cur is not None:
            self._map.set_epsg(self._store.profiles[self._pi].epsg)
            self._map.show_track(self._cur.east, self._cur.north)

    def _on_pick_toggled(self, on: bool) -> None:
        if self._map is not None:
            self._map.set_pick_active(bool(on))

    def _on_pick_released(self) -> None:
        self._btn_pick.blockSignals(True)
        self._btn_pick.setChecked(False)
        self._btn_pick.blockSignals(False)

    def _on_map_picked(self, east: float, north: float, tol_m: float) -> None:
        self.go_to_world_point(east, north, tol_m=max(self.MIN_PICK_TOL_M, tol_m))

    def go_to_world_point(self, east: float, north: float, tol_m: float | None = None) -> bool:
        """Apre profilo, canale e traccia piu' vicini a (east, north) (CRS dei profili)."""
        if self._store is None or not len(self._store):
            self._show_message("Nessuna variante aperta.")
            return False
        tol = self.MIN_PICK_TOL_M if tol_m is None else float(tol_m)
        hit = self._store.nearest(east, north, tol)
        if hit is None:
            self._show_message(f"Nessun profilo entro {tol:.1f} m dal punto cliccato.")
            return False
        dist, pi, ci, tr = hit
        self.select_profile(pi, ci)
        ch = self._cur
        if ch is None:
            return False
        self._cursor_x = float(ch.dist[min(tr, ch.dist.size - 1)])
        self._cursor_z = float(self._ref_z if self._ref_z is not None else 0.0)
        self._vline.set_xdata([self._cursor_x, self._cursor_x])
        self._vline.set_visible(True)
        self._hline.set_ydata([self._cursor_z, self._cursor_z])
        self._hline.set_visible(self._ref_z is not None)
        # se la traccia e' fuori dalla vista zoomata, la vista si sposta
        lo, hi = sorted(self._ax.get_xlim())
        if not (lo <= self._cursor_x <= hi):
            span = hi - lo
            self._set_view(self._cursor_x - span / 2, self._cursor_x + span / 2, *self._ax.get_ylim())
        self._blit.update()
        self._update_status()
        self._show_message(self._lbl_status.text() + f"  |  (a {dist:.2f} m dal punto)")
        self._flush_map_point()
        return True

    # ------------------------------------------------------------------
    # Ciclo di vita
    # ------------------------------------------------------------------

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._store is not None and self._store.is_stale():
            # la variante e' stata riscritta (es. il viewer salva "Lavoro" alla chiusura)
            self.refresh_variants(select=self._store.path)
        if self._map is not None and self._chk_map_checked():
            self._map.set_visible(True)
            if self._cur is not None:
                self._map.show_track(self._cur.east, self._cur.north)

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        if self._map is not None:
            self._map.set_visible(False)        # dock chiuso: niente tracce orfane sulla mappa
            self._btn_pick.setChecked(False)

    def _chk_map_checked(self) -> bool:
        chk = getattr(self, "_chk_map", None)
        return bool(chk is None or chk.isChecked())

    def cleanup(self) -> None:
        """Da chiamare allo scaricamento del plugin: stacca segnali e rimuove le tracce dalla mappa."""
        try:
            if self._layer_signal is not None:
                self._layer_signal.disconnect(self._on_current_layer_changed)
        except Exception:
            pass
        self._layer_signal = None
        try:
            self._map_timer.stop()
        except Exception:
            pass
        if self._map is not None:
            self._map.dispose()
