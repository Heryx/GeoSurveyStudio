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
  traccia piu' vicini, con la profondita' della time-slice visibile in quel
  momento sul canvas; movimento sul radargramma -> punto giallo sulla mappa;
- selezionando una time-slice nel pannello Livelli mostra la sua finestra di
  profondita' sul radargramma (da ``sliceset.json``);
- doppio clic sul radargramma -> mostra sul canvas la time-slice la cui fascia
  di profondita' contiene il punto cliccato (nasconde le altre dello stesso
  gruppo dell'albero dei livelli);
- qualunque cambio di visibilita' nell'albero dei livelli (checkbox a mano, o lo
  slider di navigazione del dialogo principale) aggiorna dal vivo la riga verde,
  se "Segui time-slice" e' attivo;
- la fascia di profondita' di un raster si legge prima da ``sliceset.json``
  (accanto al file) e, se assente, dal catalogo di progetto
  (``project_catalog.json``): non serve alcuna relazione di cartelle tra le
  time-slice e la variante processata aperta nel pannello.
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
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QMenu,
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
    processed_dir_for,
    slice_info_for_path,
    variant_dir_from_slice_dir,
)
from .gpr_hyperbola_math import (
    DEFAULT_VELOCITY_M_NS,
    hyperbola_curve_depth,
    m_ns_to_velocity_m_s,
    rdp_to_velocity_m_s,
    velocity_from_two_points,
    velocity_m_s_to_m_ns,
    velocity_to_rdp,
)
from .gpr_velocity_store import write_velocity_override
from .gpr_material_presets import material_preset
from .gpr_training_export import export_variant_for_training

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
_HYPER_VEL_STEP_MNS = 0.002     # passo per tacca di rotellina nello strumento "Calcola con iperbole"


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
            if it.isEmpty():
                continue
            size = size.expandedTo(it.minimumSize())
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _arrange(self, rect, apply: bool) -> int:
        m = self.contentsMargins()
        area = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, line_h = area.x(), area.y(), 0
        for it in self._items:
            if it.isEmpty():                 # widget nascosto: non occupa spazio
                continue
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

    def __init__(self, iface, workspace_provider=None, geometry_provider=None,
                 feature_sink=None, feature_query=None, parent=None, zones_provider=None):
        super().__init__("Radargrammi processati", parent or (iface.mainWindow() if iface else None))
        self.setObjectName("GeoSurveyRadargramDock")
        self.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
            | QDockWidget.DockWidgetClosable
        )
        self.iface = iface
        self._workspace_provider = workspace_provider
        # Aree indagate del progetto: callable -> [{"name", "base", "active"}, ...]. Senza,
        # il pannello ha un'unica area implicita: la cartella di `workspace_provider`.
        self._zones_provider = zones_provider
        self._zones: list = []
        self._active_zone_seen: str | None = None
        self._viewer_zone = ""             # area attiva nel Profile Viewer (anche se senza varianti)
        self._geometry_provider = geometry_provider
        self._feature_sink = feature_sink
        self._feature_query = feature_query
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
        self._tree_signal = None

        # Disegno anomalie/interfacce direttamente sul radargramma.
        self._draw_mode: str | None = None          # None | "point" | "line" | "polygon"
        self._draw_vertices: list = []               # [{"x_m","depth_m","trace","east","north","amp"}, ...]
        self._draw_mode_buttons: dict = {}
        self._draw_preview_line = None
        self._draw_preview_fill = None   # riempimento del poligono durante il disegno attivo

        # Geometrie gia' salvate su questo profilo/canale, ridisegnate come overlay
        # persistente (non solo durante il disegno attivo) - vedi _refresh_persisted_features.
        self._persisted_artists: list = []

        # "Misura velocita'": senza apice, popup informativo col valore preimpostato
        # del profilo (sola lettura). Un clic fissa l'apice di un'iperbole/anomalia:
        # da li' in poi il popup e la curva d'anteprima si aggiornano dal vivo
        # muovendo il puntatore lungo un'ala (velocity_from_two_points), per
        # confrontare v/RDP coi valori tipici di materiale - round 4.
        self._vel_mode_active = False
        self._vel_popup = None                        # matplotlib annotation
        self._vel_apex = None                          # (x_m, z_m) sul radargramma

        # "Calcola con iperbole": click fissa l'apice, la rotellina del mouse
        # regola la velocita'/larghezza della curva (niente piu' ricalcolo dal
        # movimento del cursore, round 3), il tasto destro blocca/sblocca il fit
        # prima di salvarlo come preimpostato.
        self._hyper_mode_active = False
        self._hyper_apex = None                        # (x_m, z_m) sul radargramma
        self._hyper_v_mns = None                        # velocita' candidata corrente (m/ns)
        self._hyper_locked = False                      # tasto destro: blocca la regolazione a rotellina
        self._vel_curve_line = None
        self._vel_apex_marker = None

        self._map_timer = QTimer(self)
        self._map_timer.setSingleShot(True)
        self._map_timer.setInterval(_CURSOR_MAP_DELAY_MS)
        self._map_timer.timeout.connect(self._flush_map_point)

        self._build_ui()
        if self._map is not None:
            self._map.picked.connect(self._on_map_picked)
            self._map.tool_released.connect(self._on_pick_released)
        self._connect_layer_signal()
        self._connect_tree_visibility_signal()
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

        def _sep():
            f = QFrame()
            f.setFrameShape(QFrame.VLine)
            f.setFrameShadow(QFrame.Sunken)
            return f

        # Riga 1: cosa si guarda (area, variante, profilo, canale) e come lo si vede.
        view_bar = QWidget()
        flow = _FlowLayout(view_bar)
        root.addWidget(view_bar)

        self._lbl_zone = QLabel("Area")
        self._cb_zone = _combo(10)
        self._cb_zone.setToolTip("Area indagata del progetto: ognuna ha le sue varianti e i suoi profili.")
        self._cb_zone.currentIndexChanged.connect(self._on_zone_changed)
        self._grp_zone = _group(self._lbl_zone, self._cb_zone)
        flow.addWidget(self._grp_zone)

        self._cb_variant = _combo(14)
        self._cb_variant.setToolTip("Radargrammi processati salvati nel progetto (Processing/processed).")
        self._cb_variant.currentIndexChanged.connect(self._on_variant_changed)
        self._btn_refresh = _tool("⟳", "Rileggi le varianti e le coordinate (dopo un nuovo posizionamento o salvataggio).")
        self._btn_refresh.clicked.connect(self.refresh)
        self._btn_more = _tool("⋯", "Altre azioni sulla variante")
        self._btn_more.setPopupMode(QToolButton.InstantPopup)
        more_menu = QMenu(self._btn_more)
        self._act_browse = more_menu.addAction("Scegli a mano una cartella-variante (.npz)…")
        self._act_browse.triggered.connect(self._browse_variant)
        self._act_export_training = more_menu.addAction("Esporta per training…")
        self._act_export_training.setToolTip(
            "Scrive su disco, in una cartella a scelta, tutti i profili/canali della\n"
            "variante aperta in un formato semplice per una pipeline di Deep Learning\n"
            "esterna: un radargramma.npz per canale, la geometria del posizionamento\n"
            "attuale (geometry.json) e i parametri di processing (meta.json).\n"
            "Non modifica nulla nel progetto QGIS.")
        self._act_export_training.triggered.connect(self._export_training)
        self._btn_more.setMenu(more_menu)
        flow.addWidget(_group(QLabel("Variante"), self._cb_variant, self._btn_refresh, self._btn_more))

        self._btn_prev_p = _tool("◀", "Profilo precedente (PgSu)")
        self._btn_prev_p.clicked.connect(lambda: self.select_profile(self._pi - 1))
        self._cb_profile = _combo(20)
        self._cb_profile.setMaxVisibleItems(30)
        self._cb_profile.currentIndexChanged.connect(self._on_profile_combo)
        self._btn_next_p = _tool("▶", "Profilo successivo (PgGiu)")
        self._btn_next_p.clicked.connect(lambda: self.select_profile(self._pi + 1))
        flow.addWidget(_group(QLabel("Profilo"), self._btn_prev_p, self._cb_profile, self._btn_next_p))

        self._btn_prev_c = _tool("◀", "Canale precedente (freccia sinistra)")
        self._btn_prev_c.clicked.connect(lambda: self.select_profile(self._pi, self._ci - 1))
        self._cb_channel = _combo(3)
        self._cb_channel.currentIndexChanged.connect(self._on_channel_combo)
        self._btn_next_c = _tool("▶", "Canale successivo (freccia destra)")
        self._btn_next_c.clicked.connect(lambda: self.select_profile(self._pi, self._ci + 1))
        flow.addWidget(_group(QLabel("Canale"), self._btn_prev_c, self._cb_channel, self._btn_next_c))

        self._spin_depth = QDoubleSpinBox()
        self._spin_depth.setDecimals(3)
        self._spin_depth.setSingleStep(0.05)
        self._spin_depth.setRange(0.0, 50.0)
        self._spin_depth.setSuffix(" m")
        self._spin_depth.setToolTip(
            "Profondità di riferimento (riga verde): si imposta qui, con un clic sul\n"
            "radargramma, con le frecce su/giù, o seguendo la time-slice selezionata.")
        self._spin_depth.valueChanged.connect(self._on_depth_spin)
        flow.addWidget(_group(QLabel("Profondità"), self._spin_depth))

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
        self._btn_zoom = _tool("⤢", "Vista intera")
        self._btn_zoom.clicked.connect(self._reset_zoom)
        flow.addWidget(_group(QLabel("Gain"), self._spin_gain, self._cb_cmap, self._btn_zoom))

        # Riga 2: strumenti. Ogni strumento attivo mostra accanto i suoi comandi, gli altri restano nascosti.
        tool_bar = QWidget()
        tools = _FlowLayout(tool_bar, hspacing=6)
        root.addWidget(tool_bar)

        self._btn_pick = QPushButton("Punta sulla mappa")
        self._btn_pick.setCheckable(True)
        self._btn_pick.setToolTip(
            "Attiva lo strumento: un clic sulla mappa apre il profilo, il canale e la traccia\n"
            "più vicini al punto. Si disattiva scegliendo un altro strumento di QGIS.")
        self._btn_pick.toggled.connect(self._on_pick_toggled)
        tools.addWidget(self._btn_pick)
        tools.addWidget(_sep())

        def _draw_btn(text, mode, tip):
            b = QPushButton(text)
            b.setCheckable(True)
            b.setToolTip(tip)
            b.toggled.connect(lambda on, m=mode: self._on_draw_mode_toggled(m, on))
            b.toggled.connect(self._update_tool_context)
            self._draw_mode_buttons[mode] = b
            return b

        self._btn_draw_point = _draw_btn("Punto", "point",
            "Disegna un punto sul radargramma: un clic lo posiziona e lo salva.\n"
            "Profondita' e coordinate reali vengono dal punto cliccato.")
        self._btn_draw_line = _draw_btn("Linea", "line",
            "Disegna una linea (es. un'interfaccia): un clic per vertice, poi\n"
            "'Termina' (o Invio). 'Annulla vertice' (Backspace) toglie l'ultimo.")
        self._btn_draw_polygon = _draw_btn("Poligono", "polygon",
            "Disegna un poligono (es. il contorno di un'anomalia): un clic per\n"
            "vertice (almeno 3), poi 'Termina' (o Invio) per chiudere il contorno.")
        self._btn_draw_undo = _tool("⌫", "Rimuovi l'ultimo vertice (Backspace)")
        self._btn_draw_undo.clicked.connect(self._undo_draw_vertex)
        self._btn_draw_finish = _tool("✓", "Termina e salva la geometria (Invio)")
        self._btn_draw_finish.clicked.connect(self._finalize_drawn_feature)
        self._btn_draw_cancel = _tool("✕", "Annulla il disegno in corso (Esc)")
        self._btn_draw_cancel.clicked.connect(self._cancel_draw)
        tools.addWidget(_group(
            QLabel("Disegna"), self._btn_draw_point, self._btn_draw_line, self._btn_draw_polygon))
        self._grp_draw_ctx = _group(self._btn_draw_undo, self._btn_draw_finish, self._btn_draw_cancel)
        tools.addWidget(self._grp_draw_ctx)
        tools.addWidget(_sep())

        self._spin_vel_mns = QDoubleSpinBox()
        self._spin_vel_mns.setDecimals(4)
        self._spin_vel_mns.setRange(0.01, 0.30)
        self._spin_vel_mns.setSingleStep(0.005)
        self._spin_vel_mns.setSuffix(" m/ns")
        self._spin_vel_mns.setToolTip(
            "Velocità preimpostata per questo profilo (di base 0,1 m/ns se non\n"
            "ancora calcolata). Modificandola si salva subito e si aggiorna la\n"
            "profondità mostrata sul radargramma.")
        self._spin_vel_mns.valueChanged.connect(self._on_vel_spin_changed)
        self._btn_vel = QPushButton("Misura")
        self._btn_vel.setCheckable(True)
        self._btn_vel.setToolTip(
            "Senza apice, mostra vicino al puntatore la velocità preimpostata di\n"
            "questo profilo (il campo qui a sinistra). Un clic su un'iperbole/anomalia\n"
            "fissa l'apice: muovendo poi il puntatore lungo un'ala il valore si legge\n"
            "dal vivo. Esc toglie l'apice e torna alla lettura preimpostata.")
        self._btn_vel.toggled.connect(self._on_vel_toggled)
        self._btn_vel.toggled.connect(self._update_tool_context)
        self._btn_hyper = QPushButton("Iperbole")
        self._btn_hyper.setCheckable(True)
        self._btn_hyper.setToolTip(
            "Calcola la velocità da un'iperbole di diffrazione. Un clic sull'apice lo\n"
            "fissa. Con l'apice fissato: la rotellina del mouse cambia la velocità/larghezza\n"
            "della curva, il tasto destro blocca/sblocca il fit corrente (bloccato, la\n"
            "rotellina non lo modifica più). 'Salva' registra il valore mostrato.")
        self._btn_hyper.toggled.connect(self._on_hyper_toggled)
        self._btn_hyper.toggled.connect(self._update_tool_context)
        self._btn_hyper_save = QPushButton("Salva")
        self._btn_hyper_save.setEnabled(False)
        self._btn_hyper_save.setToolTip("Salva il valore calcolato con l'iperbole come velocità preimpostata del profilo.")
        self._btn_hyper_save.clicked.connect(self._save_hyper_fit_as_preset)
        self._btn_vel_clear = _tool("✕", "Pulisci l'apice dell'iperbole")
        self._btn_vel_clear.clicked.connect(self._clear_hyper_apex)
        self._lbl_vel = QLabel("v=n/d")
        self._lbl_vel.setToolTip("Velocità (m/ns) e permittività dielettrica (εr).")
        tools.addWidget(_group(QLabel("Velocità"), self._spin_vel_mns, self._btn_vel, self._btn_hyper))
        self._grp_hyper_ctx = _group(self._btn_hyper_save, self._btn_vel_clear, self._lbl_vel)
        tools.addWidget(self._grp_hyper_ctx)

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
        self._chk_follow = QCheckBox("Segui time-slice")
        self._chk_follow.setChecked(True)
        self._chk_follow.setToolTip(
            "Selezionando una time-slice nel pannello Livelli, il radargramma mostra la sua\n"
            "finestra di profondit\u00e0 (fascia verde) e passa alla variante da cui \u00e8 stata generata.")
        self._chk_follow.toggled.connect(self._on_follow_toggled)
        self._chk_map = QCheckBox("Sulla mappa")
        self._chk_map.setChecked(True)
        self._chk_map.setToolTip("Disegna sulla mappa la traccia del profilo e il punto sotto il cursore.")
        self._chk_map.toggled.connect(self._on_map_toggled)
        status_row = QHBoxLayout()
        status_row.setSpacing(10)
        status_row.addWidget(self._lbl_status, 1)
        status_row.addWidget(self._chk_follow)
        status_row.addWidget(self._chk_map)
        root.addLayout(status_row)
        self.setWidget(host)
        self._update_tool_context()
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

    # -- aree ---------------------------------------------------------------------------

    def _zone_list(self) -> list:
        """Aree del progetto come [{"name", "base", "active"}]. Senza un fornitore di aree
        (progetto senza project.json, uso da solo) c'e' un'unica area senza nome."""
        zones = []
        if self._zones_provider is not None:
            try:
                for z in (self._zones_provider() or []):
                    zones.append({"name": str(z.get("name") or ""), "base": str(z.get("base") or ""),
                                  "active": bool(z.get("active"))})
            except Exception:
                zones = []
        if not zones:
            self._viewer_zone = ""
            return [{"name": "", "base": self._workspace_root(), "active": True}]
        self._viewer_zone = next((z["name"] for z in zones if z["active"]), zones[0]["name"])
        # Solo le aree con radargrammi processati (se nessuna ne ha, resta l'elenco intero: il
        # pannello dira' che non c'e' nulla da mostrare).
        with_data = [z for z in zones if discover_variants(z["base"])]
        return with_data or zones

    def _zone_named(self, name: str):
        return next((z for z in self._zones if z["name"] == name), None)

    def _current_zone(self):
        name = self._cb_zone.currentData()
        return self._zone_named(str(name)) if name is not None else (self._zones[0] if self._zones else None)

    def _zone_of_variant(self, target: str):
        """Nome dell'area che contiene la variante `target` (cartella o solo nome), o None."""
        target = str(target or "").strip()
        if not target:
            return None
        norm = os.path.normcase(os.path.normpath(target))
        if os.path.isabs(target) or os.sep in target or "/" in target:
            parent = os.path.normcase(os.path.dirname(norm))
            for z in self._zones:
                proc = processed_dir_for(z["base"])
                if proc and os.path.normcase(os.path.normpath(proc)) == parent:
                    return z["name"]
            return None
        cur = self._current_zone()
        ordered = ([cur] if cur else []) + [z for z in self._zones if z is not cur]
        for z in ordered:
            if any(r.name == target for r in discover_variants(z["base"])):
                return z["name"]
        return None

    def _make_geometry_provider(self, zone_name: str):
        """Il Profile Viewer ha in memoria SOLO i profili dell'area attiva: per le altre aree
        (o dopo un cambio d'area) la geometria resta quella salvata nei .npz."""
        if self._geometry_provider is None:
            return None

        def provider(stem):
            z = self._zone_named(zone_name)
            if z is not None and not z["active"]:
                return None
            return self._geometry_provider(stem)
        return provider

    def _fill_zone_combo(self, want: str | None) -> None:
        self._cb_zone.clear()
        for z in self._zones:
            self._cb_zone.addItem(z["name"] or "-", z["name"])
        idx = next((i for i, z in enumerate(self._zones) if z["name"] == want), -1)
        if idx < 0:
            idx = next((i for i, z in enumerate(self._zones) if z["active"]), 0)
        self._cb_zone.setCurrentIndex(idx)
        self._grp_zone.setVisible(len(self._zones) > 1)

    def _fill_variant_combo(self, keep_dir: str = "") -> None:
        """Varianti dell'area scelta; `keep_dir` (se presente) resta selezionata."""
        zone = self._current_zone()
        refs = discover_variants(zone["base"]) if zone else []
        self._cb_variant.clear()
        for r in refs:
            self._cb_variant.addItem(f"{r.name}  ({r.n_files})", r.path)
        # una cartella scelta a mano fuori dal progetto resta nell'elenco
        if keep_dir and os.path.isdir(keep_dir) and self._zone_of_variant(keep_dir) is None:
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

    def refresh_variants(self, select: str | None = None, force: bool = False) -> None:
        """Rilegge aree e varianti. `select` = nome o cartella da selezionare;
        `force` ricarica anche la variante gia' aperta."""
        if not HAS_MPL:
            return
        keep_dir = select or (self._store.path if self._store is not None else "")
        prev_zone = self._current_zone()
        self._zones = self._zone_list()
        active = self._viewer_zone
        # Se nel Profile Viewer e' cambiata l'area attiva il pannello la segue (e la geometria
        # in memoria del viewer non e' piu' quella dell'area che stava mostrando).
        active_changed = self._active_zone_seen is not None and active != self._active_zone_seen
        self._active_zone_seen = active
        want_zone = self._zone_of_variant(select) if select else None
        if want_zone is None:
            if active_changed:
                want_zone = active
            elif prev_zone is not None and self._zone_named(prev_zone["name"]) is not None:
                want_zone = prev_zone["name"]
            elif self._store is not None:
                want_zone = self._zone_of_variant(self._store.path)
        if active_changed:
            force = True
            keep_dir = select or ""
            if self._store is not None:
                self._store.invalidate_geometry()
        self._updating = True
        try:
            self._fill_zone_combo(want_zone)
            self._fill_variant_combo(keep_dir)
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
        """Apre la variante indicata (nome o cartella, in qualunque area), opzionalmente su un profilo/canale."""
        if not HAS_MPL:
            return False
        target = str(name_or_dir or "").strip()
        if not target:
            return False
        zone = self._zone_of_variant(target)
        if zone is None:
            self.refresh_variants(select=target)         # variante nuova o cartella scelta a mano
            zone = self._zone_of_variant(target)
        cur = self._current_zone()
        if zone is not None and (cur is None or cur["name"] != zone):
            self._updating = True
            try:
                self._fill_zone_combo(zone)
                self._fill_variant_combo(target)
            finally:
                self._updating = False
        for i in range(self._cb_variant.count()):
            d = str(self._cb_variant.itemData(i))
            if os.path.basename(d) == target or os.path.normcase(d) == os.path.normcase(os.path.normpath(target)):
                self._updating = True
                try:
                    self._cb_variant.setCurrentIndex(i)
                finally:
                    self._updating = False
                self._open_variant_dir(d)
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

    def _export_training(self) -> None:
        """Esporta l'intera variante aperta per la pipeline di Deep Learning esterna
        (vedi gpr_training_export.py)."""
        if self._store is None or len(self._store) == 0:
            self._show_message("Nessuna variante aperta da esportare.")
            return
        start = str(QSettings().value(_SETTINGS + "training_export_dir", "") or "") or os.path.expanduser("~")
        out = QFileDialog.getExistingDirectory(self, "Cartella di destinazione per l'export training (DL)", start)
        if not out:
            return
        QSettings().setValue(_SETTINGS + "training_export_dir", out)
        try:
            result = export_variant_for_training(self._store, out)
        except Exception as exc:
            self._show_message(f"Export training fallito: {exc}")
            return
        n_ok, n_stale, n_err = len(result["exported"]), len(result["stale_geometry"]), len(result["errors"])
        msg = f"Export training: {n_ok} canali scritti in {os.path.join(out, self._store.name)}"
        if n_stale:
            msg += f"  |  {n_stale} con geometria non aggiornata (riapri il Profile Viewer per il posizionamento corrente)"
        if n_err:
            msg += f"  |  {n_err} falliti"
        self._show_message(msg)

    def _on_zone_changed(self, _i: int) -> None:
        if self._updating:
            return
        self._updating = True
        try:
            self._fill_variant_combo("")
        finally:
            self._updating = False
        if self._cb_variant.count() == 0:
            self._set_store(None)
        else:
            self._open_variant_dir(str(self._cb_variant.currentData()))

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
        zone = self._current_zone()
        store = VariantStore(path, geometry_provider=self._make_geometry_provider(zone["name"] if zone else ""))
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
        # Un disegno in corso o un apice dell'iperbole si riferiscono al canale
        # precedente: mischiarli col nuovo produrrebbe una geometria sbagliata.
        self._draw_vertices = []
        self._hyper_apex = None
        self._hyper_v_mns = None
        self._hyper_locked = False
        self._btn_hyper_save.setEnabled(False)
        self._vel_apex = None
        self._cur = ch
        self._prepare_display(ch)
        self._spin_depth.blockSignals(True)
        self._spin_depth.setRange(0.0, max(0.05, ch.depth_max_m))
        if self._ref_z is not None:
            self._spin_depth.setValue(float(np.clip(self._ref_z, 0.0, ch.depth_max_m)))
        self._spin_depth.blockSignals(False)
        self._sync_combos()
        self._sync_velocity_spin()
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
        self._draw_preview_line = self._vel_curve_line = self._vel_apex_marker = None
        self._draw_preview_fill = None
        self._vel_popup = None
        self._draw_vertices = []
        self._hyper_apex = None
        self._hyper_v_mns = None
        self._hyper_locked = False
        self._vel_apex = None
        self._persisted_artists = []
        if getattr(self, "_btn_hyper_save", None) is not None:
            self._btn_hyper_save.setEnabled(False)
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
        self._draw_preview_fill = Polygon(
            [[0, 0], [0, 0], [0, 0]], closed=True, facecolor="#ff3d81", edgecolor="none",
            alpha=0.22, visible=False, zorder=5)
        ax.add_patch(self._draw_preview_fill)
        (self._draw_preview_line,) = ax.plot(
            [], [], color="#ff3d81", lw=1.6, marker="o", ms=3.5,
            mfc="#ff3d81", mec="#1f1f1f", mew=0.4, visible=False, zorder=6)
        (self._vel_curve_line,) = ax.plot(
            [], [], color="#ffd000", lw=1.4, alpha=0.95, visible=False, zorder=6)
        self._vel_apex_marker = ax.scatter(
            [], [], s=28, color="#ff8c00", edgecolor="#1f1f1f", linewidths=0.5,
            visible=False, zorder=7)
        self._vel_popup = ax.annotate(
            "", xy=(0, 0), xytext=(10, 10), textcoords="offset points", fontsize=8,
            color="#1f1f1f", visible=False, zorder=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff59d", ec="#c9a227", alpha=0.95))
        self._blit.set_artists([
            self._band_patch, self._ref_line, self._vline, self._hline,
            self._draw_preview_fill, self._draw_preview_line,
            self._vel_curve_line, self._vel_apex_marker, self._vel_popup,
        ])
        self._apply_reference()
        self._refresh_draw_preview(blit=False)
        self._refresh_hyper_overlay(blit=False)
        self._refresh_vel_popup(blit=False)
        self._refresh_persisted_features()
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

    def _catalog_timeslice_band(self, layer) -> dict | None:
        """Fascia di profondita' di un raster dal catalogo di progetto
        (project_catalog.json): serve per le time-slice senza `sliceset.json`
        accanto (es. generate/importate dal Project Manager, dove la cartella
        `processed/<variante>` puo' anche vivere in un albero di cartelle del
        tutto separato da quella delle time-slice - verificato su dati reali)."""
        root = self._workspace_root()
        if not root:
            return None
        try:
            from .project_catalog import load_catalog
            catalog = load_catalog(root)
        except Exception:
            return None
        path = os.path.normcase(os.path.normpath(str(layer.source() or "").split("|")[0].strip()))
        for rec in catalog.get("timeslices") or []:
            rec_path = str(rec.get("project_path") or "").strip()
            if not rec_path or os.path.normcase(os.path.normpath(rec_path)) != path:
                continue
            try:
                z0, z1 = float(rec["depth_from"]), float(rec["depth_to"])
            except (KeyError, TypeError, ValueError):
                continue
            lo, hi = (z0, z1) if z0 <= z1 else (z1, z0)
            return {"z_from": lo, "z_to": hi, "z_center": 0.5 * (lo + hi)}
        return None

    def _timeslice_band_for_layer(self, layer) -> dict | None:
        """Fascia di profondita' (z_from/z_to/z_center) di un raster time-slice,
        letta prima da `sliceset.json` (accanto al file) e, se assente, dal
        catalogo di progetto."""
        info = slice_info_for_path(layer.source())
        if info is not None:
            return {"z_from": info["z_from"], "z_to": info["z_to"], "z_center": info["z_center"]}
        return self._catalog_timeslice_band(layer)

    def _timeslice_layers_in_project(self) -> list:
        """Tutti i raster nel progetto QGIS con una fascia di profondita' nota
        (sliceset.json o catalogo), indipendentemente dalla variante aperta nel
        pannello: sul canvas c'e' di norma un solo gruppo di time-slice alla
        volta, e la cartella della variante processata puo' non avere alcuna
        relazione di parentela con quella delle time-slice. Ritorna
        ``[(layer, banda), ...]``."""
        out = []
        for layer in QgsProject.instance().mapLayers().values():
            if not isinstance(layer, QgsRasterLayer):
                continue
            band = self._timeslice_band_for_layer(layer)
            if band is not None:
                out.append((layer, band))
        return out

    def _visible_timeslice_band(self):
        """Fascia di profondita' della time-slice attualmente visibile sul canvas,
        o None se nessuna lo e' (usato dal clic sulla mappa, vedi
        `go_to_world_point`: il clic da' solo est/nord, la profondita' viene da
        qui)."""
        root = QgsProject.instance().layerTreeRoot()
        for layer, band in self._timeslice_layers_in_project():
            node = root.findLayer(layer.id())
            if node is not None and node.itemVisibilityChecked():
                return band
        return None

    def _activate_timeslice_at_depth(self, z: float) -> None:
        """Doppio clic sul radargramma: mostra sul canvas la time-slice la cui
        fascia di profondita' contiene z, nascondendo le sorelle nello stesso
        gruppo dell'albero dei livelli (stesso pattern di
        `gpr_viewer_timeslice._update_timeslice_visibility_in_canvas` /
        `catalog_group_mixin.update_visibility_with_dial`)."""
        candidates = self._timeslice_layers_in_project()
        if not candidates:
            self._show_message("Nessuna time-slice caricata nel progetto.")
            return
        match = next((lc for lc in candidates if lc[1]["z_from"] <= z <= lc[1]["z_to"]), None)
        if match is None:
            match = min(candidates, key=lambda lc: abs(lc[1]["z_center"] - z))
        layer, band = match
        root = QgsProject.instance().layerTreeRoot()
        node = root.findLayer(layer.id())
        if node is None:
            self._show_message("Time-slice trovata ma non presente nell'albero dei livelli.")
            return
        candidate_ids = {lyr.id() for lyr, _b in candidates}
        parent = node.parent() or root
        for sibling in parent.children():
            sib_layer = getattr(sibling, "layer", lambda: None)()
            if sib_layer is not None and sib_layer.id() in candidate_ids:
                sibling.setItemVisibilityChecked(sib_layer.id() == layer.id())
        if self.iface is not None:
            try:
                self.iface.setActiveLayer(layer)
            except Exception:
                pass
        self.set_reference_depth(band["z_center"], band=(band["z_from"], band["z_to"]))
        self._show_message(self._lbl_status.text() + "  |  time-slice attivata sul canvas")

    def _connect_tree_visibility_signal(self) -> None:
        try:
            root = QgsProject.instance().layerTreeRoot()
            root.visibilityChanged.connect(self._on_tree_visibility_changed)
            self._tree_signal = root.visibilityChanged
        except Exception:
            self._tree_signal = None

    def _on_tree_visibility_changed(self, node) -> None:
        """Un raster time-slice e' stato reso visibile sul canvas (slider/dial di
        navigazione, checkbox manuale, o il doppio clic sopra): se "Segui
        time-slice" e' attivo, la riga verde del radargramma segue."""
        if not HAS_MPL or not self._chk_follow.isChecked():
            return
        layer = getattr(node, "layer", lambda: None)()
        if layer is None or not isinstance(layer, QgsRasterLayer):
            return
        if not node.itemVisibilityChecked():
            return
        band = self._timeslice_band_for_layer(layer)
        if band is None:
            return
        self.set_reference_depth(band["z_center"], band=(band["z_from"], band["z_to"]))

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
        if self._draw_mode in ("line", "polygon") and self._draw_vertices:
            self._refresh_draw_preview(cursor_xz=(self._cursor_x, self._cursor_z), blit=False)
        if self._vel_mode_active:
            self._refresh_vel_popup(cursor_xz=(self._cursor_x, self._cursor_z), blit=False)
        self._update_status()
        self._blit.update()
        self._map_timer.start()

    def _on_axes_leave(self, _event) -> None:
        if self._vline is not None:
            self._vline.set_visible(False)
            self._hline.set_visible(False)
            if self._vel_popup is not None:
                self._vel_popup.set_visible(False)
            self._blit.update()

    def _on_mouse_press(self, event) -> None:
        if event.inaxes is not self._ax or self._cur is None or event.xdata is None:
            return
        self._canvas.setFocus()
        if event.button == 1:
            if self._draw_mode is not None:
                self._add_draw_vertex(float(event.xdata), float(event.ydata))
            elif self._hyper_mode_active:
                self._set_hyper_apex(float(event.xdata), float(event.ydata))
            elif self._vel_mode_active:
                self._set_vel_apex(float(event.xdata), float(event.ydata))
            elif getattr(event, "dblclick", False):
                self._activate_timeslice_at_depth(float(np.clip(event.ydata, 0.0, self._cur.depth_max_m)))
            else:
                # clic: la profondita' cliccata diventa il riferimento
                self._band = None
                self.set_reference_depth(float(np.clip(event.ydata, 0.0, self._cur.depth_max_m)))
        elif event.button == 3 and self._hyper_mode_active and self._hyper_apex is not None:
            self._hyper_locked = not self._hyper_locked
            self._refresh_hyper_overlay(blit=True)
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
        up = getattr(event, "step", 0) > 0 or event.button == "up"
        if self._hyper_mode_active and self._hyper_apex is not None and not self._hyper_locked:
            self._adjust_hyper_velocity(_HYPER_VEL_STEP_MNS if up else -_HYPER_VEL_STEP_MNS)
            return
        scale = 1.0 / 1.2 if up else 1.2
        x0, x1 = self._ax.get_xlim()
        y0, y1 = self._ax.get_ylim()
        x, y = float(event.xdata), float(event.ydata)
        self._set_view(x - (x - x0) * scale, x + (x1 - x) * scale,
                       y - (y - y0) * scale, y + (y1 - y) * scale)

    def _on_key(self, event) -> None:
        key = str(getattr(event, "key", "") or "")
        if self._draw_mode is not None:
            if key in ("enter", "return"):
                self._finalize_drawn_feature()
                return
            if key == "backspace":
                self._undo_draw_vertex()
                return
            if key == "escape":
                self._cancel_draw()
                return
        if self._hyper_mode_active and key == "escape":
            self._clear_hyper_apex()
            return
        if self._vel_mode_active and key == "escape":
            self._clear_vel_apex()
            return
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
    # Disegno di anomalie/interfacce direttamente sul radargramma
    #
    # Il disegno avviene in coordinate schermo del radargramma (distanza lungo
    # il profilo x profondita'), ma ogni vertice viene subito convertito nella
    # sua geometria reale: la traccia piu' vicina alla distanza cliccata da'
    # (est, nord) via ChannelData.position(), la profondita' e' quella cliccata
    # (stessa scala display del radargramma, cioe' gia' al netto del time-zero).
    # Il layer QGIS risultante e' quindi sempre in coordinate reali; lo schermo
    # 2D serve solo come superficie di digitalizzazione.
    # ------------------------------------------------------------------

    def _on_draw_mode_toggled(self, mode: str, on: bool) -> None:
        if self._updating:
            return
        if on:
            if self._btn_vel.isChecked():
                self._btn_vel.setChecked(False)      # esclusivo col puntatore di velocita'
            if self._btn_hyper.isChecked():
                self._btn_hyper.setChecked(False)    # esclusivo col fit iperbole
            for other_mode, btn in self._draw_mode_buttons.items():
                if other_mode != mode and btn.isChecked():
                    btn.blockSignals(True)
                    btn.setChecked(False)
                    btn.blockSignals(False)
            self._set_draw_mode(mode)
            if self._canvas is not None:
                self._canvas.setCursor(Qt.CrossCursor)
        elif self._draw_mode == mode:
            self._set_draw_mode(None)
            if self._canvas is not None:
                self._canvas.unsetCursor()

    def _update_tool_context(self, *_a) -> None:
        """Mostra i comandi di uno strumento solo mentre e' attivo (disegno, iperbole)."""
        draw_on = any(b.isChecked() for b in self._draw_mode_buttons.values())
        self._grp_draw_ctx.setVisible(draw_on)
        self._grp_hyper_ctx.setVisible(self._btn_hyper.isChecked())

    def _set_draw_mode(self, mode: str | None) -> None:
        self._draw_mode = mode
        self._draw_vertices = []
        self._refresh_draw_preview(blit=True)

    def _add_draw_vertex(self, x_m: float, z_m: float) -> None:
        ch = self._cur
        if ch is None:
            return
        trace = ch.trace_at_distance(x_m)
        pos = ch.position(trace)
        if pos is None:
            self._show_message("Geometria non disponibile per questo punto del profilo: scartato.")
            return
        east, north = pos
        depth = float(np.clip(z_m, 0.0, ch.depth_max_m))
        row = ch.row_at_depth(depth)
        amp = ch.amplitude_at(trace, row)
        self._draw_vertices.append({
            "x_m": float(x_m), "depth_m": depth, "trace": int(trace),
            "east": float(east), "north": float(north),
            "amp": amp if np.isfinite(amp) else None,
        })
        self._refresh_draw_preview(blit=True)
        if self._draw_mode == "point":
            self._finalize_drawn_feature()

    def _undo_draw_vertex(self) -> None:
        if self._draw_vertices:
            self._draw_vertices.pop()
            self._refresh_draw_preview(blit=True)

    def _cancel_draw(self) -> None:
        self._draw_vertices = []
        self._refresh_draw_preview(blit=True)
        for btn in self._draw_mode_buttons.values():
            if btn.isChecked():
                btn.blockSignals(True)
                btn.setChecked(False)
                btn.blockSignals(False)
        self._draw_mode = None
        if self._canvas is not None:
            self._canvas.unsetCursor()

    def _refresh_draw_preview(self, cursor_xz=None, blit: bool = True) -> None:
        line = self._draw_preview_line
        fill = self._draw_preview_fill
        if line is None:
            return
        verts = self._draw_vertices
        if not verts:
            line.set_data([], [])
            line.set_visible(False)
        else:
            xs = [v["x_m"] for v in verts]
            zs = [v["depth_m"] for v in verts]
            if cursor_xz is not None:
                xs = xs + [cursor_xz[0]]
                zs = zs + [cursor_xz[1]]
            elif self._draw_mode == "polygon" and len(verts) >= 2:
                xs = xs + [xs[0]]
                zs = zs + [zs[0]]
            line.set_data(xs, zs)
            line.set_visible(True)
        # Il poligono in corso va mostrato come tale (riempito), non solo come
        # contorno collega-punti - riempimento visibile solo in modalita' "Poligono"
        # con almeno 3 vertici (inclusa la posizione del cursore).
        if fill is not None:
            if self._draw_mode == "polygon" and len(verts) >= 2:
                pts = [(v["x_m"], v["depth_m"]) for v in verts]
                if cursor_xz is not None:
                    pts = pts + [(float(cursor_xz[0]), float(cursor_xz[1]))]
                if len(pts) >= 3:
                    fill.set_xy(pts)
                    fill.set_visible(True)
                else:
                    fill.set_visible(False)
            else:
                fill.set_visible(False)
        if blit and self._blit is not None:
            self._blit.update()

    def _finalize_drawn_feature(self) -> None:
        mode = self._draw_mode
        verts = list(self._draw_vertices)
        min_needed = {"point": 1, "line": 2, "polygon": 3}.get(mode or "", 2)
        if mode is None or len(verts) < min_needed:
            self._show_message(f"Servono almeno {min_needed} vertici per completare la geometria.")
            return
        if self._feature_sink is None:
            self._show_message("Salvataggio non disponibile: apri questo pannello da GeoSurvey Studio.")
        else:
            info = self._store.profiles[self._pi]
            preset = self._current_preset_velocity()
            try:
                self._feature_sink(
                    kind=mode,
                    variant_name=self._store.name,
                    profile_stem=info.stem,
                    channel=int(self._ci),
                    epsg=int(info.epsg or 0),
                    vertices=[
                        {"east": v["east"], "north": v["north"], "depth_m": v["depth_m"], "amp": v.get("amp")}
                        for v in verts
                    ],
                    velocity_m_ns=preset["velocity_m_ns"],
                    rdp=preset["rdp"],
                )
                self._show_message(f"{mode}: {len(verts)} vertici salvati nel layer.")
            except Exception as exc:
                self._show_message(f"Errore nel salvataggio della geometria: {exc}")
        self._draw_vertices = []
        self._refresh_draw_preview(blit=True)

    # ------------------------------------------------------------------
    # Velocita' preimpostata: popup informativo ("Misura velocita'", sola lettura)
    # + fit iperbole con salvataggio esplicito ("Calcola con iperbole")
    # ------------------------------------------------------------------

    def _reference_velocity_m_s(self) -> float:
        """Velocita' (m/s) coerente con l'asse di profondita' mostrato: usata SOLO
        dal fit iperbole, deve corrispondere alla velocita' che ha davvero generato
        la profondita' sotto l'apice (vedi VariantStore.effective_velocity_m_s) -
        non e' la stessa cosa del valore "preimpostato" mostrato dal popup."""
        try:
            return float(self._store.effective_velocity_m_s(self._pi))
        except Exception:
            return rdp_to_velocity_m_s(9.0)

    def _current_preset_velocity(self) -> dict:
        """Velocita' preimpostata (m/ns) per il profilo corrente: override salvato
        (manuale o da iperbole), altrimenti 0.1 m/ns - vedi
        VariantStore.reference_velocity_m_ns. Alimenta il campo editabile, il popup
        "Misura velocita'" e i campi auto-compilati delle geometrie disegnate."""
        if self._store is None or not (0 <= self._pi < len(self._store)):
            return {"velocity_m_ns": DEFAULT_VELOCITY_M_NS,
                    "rdp": velocity_to_rdp(m_ns_to_velocity_m_s(DEFAULT_VELOCITY_M_NS)), "source": "default"}
        return self._store.reference_velocity_m_ns(self._store.profiles[self._pi].stem)

    @staticmethod
    def _format_velocity_source(source: str) -> str:
        source = str(source or "")
        if source.startswith("preset:"):
            preset = material_preset(source.split(":", 1)[1])
            return preset["label"] if preset else source
        return {"default": "di base", "manual": "manuale", "hyperbola_fit": "da iperbole"}.get(source, source)

    def _sync_velocity_spin(self) -> None:
        """Precompila il campo velocita' col preimpostato del profilo corrente,
        senza scatenare un salvataggio (vedi _on_vel_spin_changed)."""
        preset = self._current_preset_velocity()
        self._spin_vel_mns.blockSignals(True)
        self._spin_vel_mns.setValue(float(preset["velocity_m_ns"]))
        self._spin_vel_mns.blockSignals(False)

    def _on_vel_spin_changed(self, value: float) -> None:
        if self._store is None or not (0 <= self._pi < len(self._store)):
            return
        stem = self._store.profiles[self._pi].stem
        write_velocity_override(self._store.path, profile_stem=stem, velocity_m_ns=float(value), source="manual")
        self._store.invalidate_geometry()   # l'asse di profondita' del canale dipende dalla velocita'
        self._show_message(f"Velocità preimpostata per {stem}: {value:.4f} m/ns (salvata).")
        self._load_current(reset_view=False, keep_y=True)

    def _on_vel_toggled(self, on: bool) -> None:
        self._vel_mode_active = bool(on)
        if on:
            if self._btn_hyper.isChecked():
                self._btn_hyper.setChecked(False)     # esclusivo col fit iperbole
            for btn in self._draw_mode_buttons.values():
                if btn.isChecked():
                    btn.setChecked(False)              # esclusivo col disegno
            if self._canvas is not None:
                self._canvas.setCursor(Qt.CrossCursor)
        else:
            if self._canvas is not None:
                self._canvas.unsetCursor()
            self._vel_apex = None
            if self._vel_popup is not None:
                self._vel_popup.set_visible(False)
            if self._vel_curve_line is not None:
                self._vel_curve_line.set_visible(False)
            if self._vel_apex_marker is not None:
                self._vel_apex_marker.set_visible(False)
            if self._blit is not None:
                self._blit.update()

    def _set_vel_apex(self, x_m: float, z_m: float) -> None:
        """Clic in modalita' "Misura velocita'": fissa l'apice da cui, muovendo il
        puntatore, si legge dal vivo v/RDP (vedi _refresh_vel_live_fit)."""
        self._vel_apex = (float(x_m), float(z_m))
        self._refresh_vel_popup(cursor_xz=self._vel_apex, blit=True)

    def _clear_vel_apex(self) -> None:
        """Esc in modalita' "Misura velocita'": torna alla lettura preimpostata statica."""
        self._vel_apex = None
        if self._vel_curve_line is not None:
            self._vel_curve_line.set_visible(False)
        if self._vel_apex_marker is not None:
            self._vel_apex_marker.set_visible(False)
        cursor_xz = ((self._cursor_x, self._cursor_z)
                     if self._cursor_x is not None and self._cursor_z is not None else None)
        self._refresh_vel_popup(cursor_xz=cursor_xz, blit=True)

    def _refresh_vel_popup(self, cursor_xz=None, blit: bool = True) -> None:
        popup = self._vel_popup
        if popup is None:
            return
        if not self._vel_mode_active or cursor_xz is None:
            popup.set_visible(False)
            if self._vel_apex is None:
                if self._vel_curve_line is not None:
                    self._vel_curve_line.set_visible(False)
                if self._vel_apex_marker is not None:
                    self._vel_apex_marker.set_visible(False)
            if blit and self._blit is not None:
                self._blit.update()
            return
        popup.xy = (float(cursor_xz[0]), float(cursor_xz[1]))
        if self._vel_apex is None:
            preset = self._current_preset_velocity()
            source_label = self._format_velocity_source(preset["source"])
            popup.set_text(f"v={preset['velocity_m_ns']:.3f} m/ns  εr={preset['rdp']:.2f}  ({source_label})")
        else:
            self._refresh_vel_live_fit(cursor_xz)
        popup.set_visible(True)
        if blit and self._blit is not None:
            self._blit.update()

    def _refresh_vel_live_fit(self, cursor_xz) -> None:
        """Con l'apice fissato (clic), aggiorna dal vivo testo del popup e curva
        d'anteprima in base alla posizione corrente del puntatore: v/RDP
        dell'iperbole con apice fissato che passa anche per il punto sotto il
        cursore (``velocity_from_two_points``) - stesso modello fisico di "Calcola
        con iperbole", letto pero' dalla posizione del mouse invece che dalla
        rotellina."""
        curve, marker = self._vel_curve_line, self._vel_apex_marker
        x0, z0 = self._vel_apex
        if marker is not None:
            marker.set_offsets([[x0, z0]])
            marker.set_visible(True)
        v_ref = self._reference_velocity_m_s()
        fit = velocity_from_two_points(x0, z0, float(cursor_xz[0]), float(cursor_xz[1]), v_ref)
        if fit is None:
            self._vel_popup.set_text("v=n/d  (sposta il puntatore sotto l'apice, lungo un'ala)")
            if curve is not None:
                curve.set_visible(False)
            return
        v_fit, rdp = fit
        v_mns = velocity_m_s_to_m_ns(v_fit)
        self._vel_popup.set_text(f"v={v_mns:.3f} m/ns  εr={rdp:.2f}  (dal vivo)")
        if curve is not None:
            ch = self._cur
            length = max(ch.length_m, 1e-6) if ch is not None else max(abs(x0) + 1.0, 1.0)
            depth_max = ch.depth_max_m if ch is not None else float(z0) * 2.0 + 1.0
            xs = np.linspace(0.0, length, 400, dtype=np.float64)
            t0 = (2.0 * float(z0)) / max(v_ref, 1e-9)
            zs = hyperbola_curve_depth(xs - float(x0), t0, v_fit, v_ref, 0.0)
            finite = np.isfinite(zs) & (zs >= 0.0) & (zs <= depth_max * 1.02)
            if finite.any():
                curve.set_data(xs[finite], zs[finite])
                curve.set_visible(True)
            else:
                curve.set_visible(False)

    def _on_hyper_toggled(self, on: bool) -> None:
        self._hyper_mode_active = bool(on)
        if on:
            if self._btn_vel.isChecked():
                self._btn_vel.setChecked(False)        # esclusivo con "Misura velocita'"
            for btn in self._draw_mode_buttons.values():
                if btn.isChecked():
                    btn.setChecked(False)              # esclusivo col disegno
            if self._canvas is not None:
                self._canvas.setCursor(Qt.CrossCursor)
        else:
            if self._canvas is not None:
                self._canvas.unsetCursor()
            self._clear_hyper_apex()

    def _set_hyper_apex(self, x_m: float, z_m: float) -> None:
        self._hyper_apex = (float(x_m), float(z_m))
        self._hyper_v_mns = float(self._current_preset_velocity()["velocity_m_ns"])
        self._hyper_locked = False
        self._refresh_hyper_overlay(blit=True)

    def _clear_hyper_apex(self) -> None:
        self._hyper_apex = None
        self._hyper_v_mns = None
        self._hyper_locked = False
        self._btn_hyper_save.setEnabled(False)
        self._lbl_vel.setText("v=n/d")
        self._refresh_hyper_overlay(blit=True)

    def _adjust_hyper_velocity(self, delta_mns: float) -> None:
        if self._hyper_v_mns is None:
            return
        self._hyper_v_mns = float(np.clip(self._hyper_v_mns + float(delta_mns), 0.01, 0.30))
        self._refresh_hyper_overlay(blit=True)

    def _save_hyper_fit_as_preset(self) -> None:
        if (self._hyper_v_mns is None or self._store is None
                or not (0 <= self._pi < len(self._store))):
            self._show_message(
                "Nessun valore da salvare: fissa prima l'apice dell'iperbole (clic sul radargramma).")
            return
        v_mns = float(self._hyper_v_mns)
        rdp = velocity_to_rdp(m_ns_to_velocity_m_s(v_mns))
        stem = self._store.profiles[self._pi].stem
        write_velocity_override(
            self._store.path, profile_stem=stem, velocity_m_ns=v_mns, rdp=rdp, source="hyperbola_fit")
        self._store.invalidate_geometry()
        self._show_message(f"Velocità salvata come preimpostato per {stem}: {v_mns:.4f} m/ns (εr={rdp:.2f}).")
        self._load_current(reset_view=False, keep_y=True)

    def _refresh_hyper_overlay(self, blit: bool = True) -> None:
        curve, marker = self._vel_curve_line, self._vel_apex_marker
        if curve is None or marker is None:
            return
        apex = self._hyper_apex
        if apex is None or self._hyper_v_mns is None:
            curve.set_visible(False)
            marker.set_visible(False)
            self._btn_hyper_save.setEnabled(False)
            if blit and self._blit is not None:
                self._blit.update()
            return
        x0, z0 = apex
        marker.set_offsets([[x0, z0]])
        marker.set_visible(True)
        v_ref = self._reference_velocity_m_s()
        v_fit = m_ns_to_velocity_m_s(float(self._hyper_v_mns))
        rdp = velocity_to_rdp(v_fit)
        self._btn_hyper_save.setEnabled(True)
        lock_txt = "  (bloccato)" if self._hyper_locked else ""
        self._lbl_vel.setText(f"v={self._hyper_v_mns:.4f} m/ns  εr={rdp:.2f}{lock_txt}")
        ch = self._cur
        length = max(ch.length_m, 1e-6) if ch is not None else max(float(x0) + 1.0, 1.0)
        depth_max = ch.depth_max_m if ch is not None else float(z0) * 2.0 + 1.0
        xs = np.linspace(0.0, length, 400, dtype=np.float64)
        t0 = (2.0 * float(z0)) / max(v_ref, 1e-9)
        zs = hyperbola_curve_depth(xs - float(x0), t0, v_fit, v_ref, 0.0)
        finite = np.isfinite(zs) & (zs >= 0.0) & (zs <= depth_max * 1.02)
        if finite.any():
            curve.set_data(xs[finite], zs[finite])
            curve.set_visible(True)
        else:
            curve.set_visible(False)
        if blit and self._blit is not None:
            self._blit.update()

    # ------------------------------------------------------------------
    # Geometrie gia' disegnate: overlay persistente sul radargramma
    # ------------------------------------------------------------------

    def _refresh_persisted_features(self) -> None:
        """Ridisegna come overlay statico (non blit-managed: si ricostruisce a ogni
        _redraw) le geometrie gia' salvate su questo profilo/canale, cosi' restano
        visibili anche fuori da una sessione di disegno attiva. Le coordinate reali
        (est/nord) di ogni vertice sono riproiettate sullo schermo trovando la
        traccia piu' vicina nel canale corrente - mai ricalcolate dalla geometria,
        la profondita' resta quella salvata al momento del disegno."""
        for artist in self._persisted_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._persisted_artists = []
        ch = self._cur
        if ch is None or self._ax is None or self._feature_query is None or self._store is None:
            return
        if not (0 <= self._pi < len(self._store)):
            return
        info = self._store.profiles[self._pi]
        try:
            features = self._feature_query(self._store.name, info.stem, int(self._ci)) or []
        except Exception:
            features = []
        if ch.east.size == 0:
            return
        for feat in features:
            kind = feat.get("kind")
            verts = feat.get("vertices") or []
            if not verts:
                continue
            xs, zs = [], []
            for v in verts:
                e, n = float(v.get("east", np.nan)), float(v.get("north", np.nan))
                d2 = (ch.east - e) ** 2 + (ch.north - n) ** 2
                if not np.isfinite(d2).any():
                    continue
                trace = int(np.nanargmin(d2))
                xs.append(float(ch.dist[trace]))
                zs.append(float(v.get("depth_m", 0.0)))
            if len(xs) != len(verts) or not xs:
                continue
            if kind == "point":
                artist = self._ax.scatter(
                    xs, zs, s=26, color="#00e5ff", edgecolor="#1f1f1f", linewidths=0.4, zorder=5)
            elif kind == "polygon":
                artist = Polygon(
                    list(zip(xs, zs)), closed=True, facecolor="#00e5ff", edgecolor="#00e5ff",
                    alpha=0.22, linewidth=1.3, zorder=4)
                self._ax.add_patch(artist)
            else:   # "line"
                (artist,) = self._ax.plot(xs, zs, color="#00e5ff", lw=1.4, alpha=0.9, zorder=5)
            self._persisted_artists.append(artist)

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
        zone = self._current_zone()
        where = f"{zone['name']} / {self._store.name}" if zone and zone["name"] and len(self._zones) > 1 else self._store.name
        head = f"Profilo {self._pi + 1}/{len(self._store)}  Ch {self._ci}  ({where})"
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
        # la profondita' del clic sulla mappa (2D) e' quella della time-slice che si
        # vede in quel momento sul canvas, non un vecchio riferimento eventualmente
        # rimasto da prima - vedi _visible_timeslice_band.
        visible_slice = self._visible_timeslice_band()
        if visible_slice is not None:
            self.set_reference_depth(visible_slice["z_center"],
                                     band=(visible_slice["z_from"], visible_slice["z_to"]))
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
            if self._tree_signal is not None:
                self._tree_signal.disconnect(self._on_tree_visibility_changed)
        except Exception:
            pass
        self._tree_signal = None
        try:
            self._map_timer.stop()
        except Exception:
            pass
        if self._map is not None:
            self._map.dispose()
