# -*- coding: utf-8 -*-
"""Finestra agganciabile "Vista 3D profili": radargrammi GPR e sezioni ERT come tende verticali.

Ogni profilo si disegna lungo la sua linea reale sulla mappa (coordinate per traccia del GPR,
georiferimento per l'ERT), quindi due profili che si incrociano si incrociano anche qui, nel punto
giusto. Profondita' sempre in verticale dal piano z = 0 (modalita' piana, come nel pannello 2D).

I profili si spuntano in un elenco: quelli del GPR sono i profili della variante aperta nel pannello
"Radargrammi processati", quelli ERT sono le sezioni importate nel progetto.
"""

from __future__ import annotations

import os
from collections import OrderedDict

import numpy as np
from qgis.PyQt.QtCore import QTimer, Qt
from qgis.PyQt.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from qgis.core import QgsProject

from .gpr_ert_georef import lonlat_transformer
from .gpr_ert_import import to_grid as ert_to_grid
from .gpr_ert_import_dialog import import_ert_section
from .gpr_ert_store import list_ert, load_ert
from .gpr_fence_slices import colormap_from_hex, discover_slice_groups, load_slice
from .radargram_store import VariantStore, discover_variants
from .gpr_fence_geometry import (
    crossings,
    curtain,
    downsample,
    plane,
    polyline_vertices,
    rgba_from_array,
    track_vertices,
)

try:
    from .gpr_viewer_common import normalize_display
    from .gpr_viewer_radargram import RadargramViewMixin
    _HAS_DISPLAY = True
except Exception:                                    # matplotlib/scipy mancanti
    _HAS_DISPLAY = False

_GPR_CMAPS = ("gray", "RdBu_r", "seismic", "bone")
_ERT_CMAPS = ("viridis", "turbo", "Spectral_r", "jet", "plasma")
_SLICE_CMAPS = ("viridis", "gray", "turbo", "RdBu_r", "Spectral_r")
_CACHE_MAX = 24
_ERT_CLIM_PERCENTILES = (2.0, 98.0)
_ROLE = Qt.UserRole
_SIDE_MIN_TOTAL_WIDTH = 640     # sotto questa larghezza il pannello dei comandi si nasconde da solo
_GPR_EXPAND_MAX = 15            # oltre questi profili il gruppo GPR parte chiuso (non copre ERT e time-slice)


def _load_pyvista():
    """(pv, QtInteractor) o (None, motivo) se pyvista/pyvistaqt non sono disponibili."""
    from . import gpr_3d_viewer_dialog as v3d

    if not v3d._bootstrap_pyvista_runtime():
        err = v3d._PYVISTA_IMPORT_ERROR or v3d._QTINTERACTOR_IMPORT_ERROR or "pyvista/pyvistaqt non trovati"
        return None, str(err)
    return v3d.pv, v3d.QtInteractor


class FenceDock(QDockWidget):
    """Vista 3D dei profili GPR e ERT del progetto."""

    def __init__(self, iface, workspace_provider=None, zones_provider=None, project_root_provider=None,
                 geometry_provider=None, parent=None, plotter_factory=None):
        """Provider come per ``RadargramDock`` (aree, cartella di lavoro, radice del progetto, geometria).

        ``plotter_factory(parent_widget) -> (plotter, widget)`` sostituisce il ``QtInteractor``
        (nei test non c'e' un contesto OpenGL nativo: si usa ``pv.Plotter(off_screen=True)``)."""
        super().__init__("Vista 3D profili", parent or (iface.mainWindow() if iface else None))
        self._plotter_factory = plotter_factory
        self.setObjectName("GeoSurveyFenceDock")
        self.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.iface = iface
        self._workspace_provider = workspace_provider
        self._zones_provider = zones_provider
        self._root_getter = project_root_provider
        self._geometry_provider = geometry_provider
        self._zones: list = []
        self._store = None
        self._gpr_cache: OrderedDict = OrderedDict()
        self._ert_cache: dict = {}
        self._slice_cache: dict = {}
        self._slice_info: dict = {}      # percorso raster -> riga del manifest
        self._origin = None
        self._side = None
        self._side_user = False           # l'utente ha scelto a mano: niente piu' nascondi/mostra automatico
        self._hint_on = False
        self._shown: dict = {}            # nome attore -> tipo ("gpr" | "ert" | "x")
        self._first_camera = True
        self._plotter = None
        self._pv = None
        self._updating = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self._rebuild)
        self._build_ui()
        if self._plotter is not None:
            self.refresh_variants()

    # --- interfaccia ---------------------------------------------------------------------------
    def _build_ui(self) -> None:
        host = QWidget(self)
        lay = QHBoxLayout(host)
        lay.setContentsMargins(2, 2, 2, 2)
        self.setWidget(host)
        self._pv, qt = _load_pyvista()
        if self._pv is None:
            lay.addWidget(QLabel("La vista 3D richiede pyvista e pyvistaqt nel Python di QGIS.\n"
                                 f"Dettaglio: {qt}\nInstalla con: python -m pip install pyvista pyvistaqt"))
            return
        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)
        lay.addWidget(split)

        side = QWidget()
        side.setMinimumWidth(240)
        self._side = side
        sl = QVBoxLayout(side)
        sl.setContentsMargins(0, 0, 0, 0)
        zrow = QHBoxLayout()
        self._lbl_zone = QLabel("Area")
        self._cb_zone = QComboBox()
        self._cb_zone.setToolTip("Area indagata del progetto: ognuna ha le sue varianti e i suoi profili.")
        self._cb_zone.currentIndexChanged.connect(self._on_zone_changed)
        zrow.addWidget(self._lbl_zone)
        zrow.addWidget(self._cb_zone, 1)
        sl.addLayout(zrow)
        vrow = QHBoxLayout()
        self._cb_variant = QComboBox()
        self._cb_variant.setToolTip("Variante processata di cui mostrare i profili GPR.")
        self._cb_variant.currentIndexChanged.connect(self._on_variant_changed)
        vrow.addWidget(QLabel("Variante"))
        vrow.addWidget(self._cb_variant, 1)
        sl.addLayout(vrow)
        self._ed_filter = QLineEdit()
        self._ed_filter.setPlaceholderText("Filtra i profili…")
        self._ed_filter.textChanged.connect(self._apply_filter)
        sl.addWidget(self._ed_filter)
        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.itemChanged.connect(self._on_item_changed)
        sl.addWidget(self._tree, 1)
        row = QHBoxLayout()
        for text, tip, slot in (("Tutti i GPR", "Spunta tutti i profili GPR visibili nel filtro.", lambda: self._check_group(0, True)),
                                ("Nessun GPR", "Toglie la spunta a tutti i profili GPR.", lambda: self._check_group(0, False)),
                                ("+ ERT", "Importa una sezione ERT invertita (CSV di ResIPY) e georiferiscila.", self._import_ert),
                                ("↻", "Rilegge aree, varianti, profili GPR e sezioni ERT.", self.refresh_variants)):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            row.addWidget(b)
        sl.addLayout(row)

        form = QWidget()
        fl = QVBoxLayout(form)
        fl.setContentsMargins(0, 0, 0, 0)

        def _line(label, widget):
            r = QHBoxLayout()
            r.addWidget(QLabel(label))
            r.addWidget(widget, 1)
            fl.addLayout(r)

        self._spin_ch = QSpinBox()
        self._spin_ch.setRange(0, 63)
        self._spin_ch.setToolTip("Canale del profilo GPR mostrato (uno per profilo).")
        self._spin_ch.valueChanged.connect(self._schedule)
        _line("Canale GPR", self._spin_ch)
        self._spin_gain = QDoubleSpinBox()
        self._spin_gain.setRange(0.2, 30.0)
        self._spin_gain.setSingleStep(0.5)
        self._spin_gain.setValue(1.7)
        self._spin_gain.valueChanged.connect(self._schedule)
        _line("Gain GPR", self._spin_gain)
        self._cb_gpr = QComboBox()
        self._cb_gpr.addItems(_GPR_CMAPS)
        self._cb_gpr.currentIndexChanged.connect(self._schedule)
        _line("Palette GPR", self._cb_gpr)
        self._cb_ert = QComboBox()
        self._cb_ert.addItems(_ERT_CMAPS)
        self._cb_ert.currentIndexChanged.connect(self._schedule)
        _line("Palette ERT", self._cb_ert)
        self._cb_slice = QComboBox()
        self._cb_slice.addItem("Stile del file", "")
        for name in _SLICE_CMAPS:
            self._cb_slice.addItem(name, name)
        self._cb_slice.setToolTip("Palette delle time-slice: lo stile salvato nel raster oppure una palette a scelta.")
        self._cb_slice.currentIndexChanged.connect(self._schedule)
        _line("Palette slice", self._cb_slice)
        self._spin_slice_op = QDoubleSpinBox()
        self._spin_slice_op.setRange(0.1, 1.0)
        self._spin_slice_op.setSingleStep(0.05)
        self._spin_slice_op.setValue(0.85)
        self._spin_slice_op.setToolTip("Opacità dei piani delle time-slice: più bassa, meglio si vedono i profili sotto.")
        self._spin_slice_op.valueChanged.connect(self._schedule)
        _line("Opacità slice", self._spin_slice_op)
        self._spin_depth = QDoubleSpinBox()
        self._spin_depth.setRange(0.0, 100.0)
        self._spin_depth.setSpecialValueText("tutta")
        self._spin_depth.setSuffix(" m")
        self._spin_depth.setSingleStep(0.5)
        self._spin_depth.setValue(0.0)
        self._spin_depth.setToolTip("Profondità massima mostrata, uguale per GPR ed ERT (0 = tutta).")
        self._spin_depth.valueChanged.connect(self._schedule)
        _line("Profondità max", self._spin_depth)
        self._spin_ve = QDoubleSpinBox()
        self._spin_ve.setRange(0.5, 20.0)
        self._spin_ve.setSingleStep(0.5)
        self._spin_ve.setValue(2.0)
        self._spin_ve.setToolTip("Esagerazione verticale: le profondità sono piccole rispetto alla lunghezza dei profili.")
        self._spin_ve.valueChanged.connect(self._apply_scale)
        _line("Esagerazione Z", self._spin_ve)
        self._chk_labels = QCheckBox("Nomi dei profili")
        self._chk_labels.setChecked(True)
        self._chk_labels.toggled.connect(self._schedule)
        fl.addWidget(self._chk_labels)
        self._chk_cross = QCheckBox("Incroci GPR × ERT")
        self._chk_cross.setChecked(True)
        self._chk_cross.toggled.connect(self._schedule)
        fl.addWidget(self._chk_cross)
        vrow = QHBoxLayout()
        for text, slot in (("Dall'alto", self._view_top), ("Reimposta", self._view_iso)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            vrow.addWidget(b)
        fl.addLayout(vrow)
        self._txt_cross = QPlainTextEdit()
        self._txt_cross.setReadOnly(True)
        self._txt_cross.setMaximumHeight(110)
        self._txt_cross.setPlaceholderText("Gli incroci tra profili GPR ed ERT spuntati compaiono qui.")
        fl.addWidget(self._txt_cross)
        sl.addWidget(form)
        split.addWidget(side)

        view = QWidget()
        vl = QVBoxLayout(view)
        vl.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        self._btn_side = QToolButton()
        self._btn_side.setText("\u2630 Comandi")
        self._btn_side.setCheckable(True)
        self._btn_side.setChecked(True)
        self._btn_side.setToolTip("Mostra o nasconde l'elenco dei profili e i comandi, per lasciare tutto lo spazio alla vista 3D.")
        self._btn_side.toggled.connect(lambda on: self._set_side_visible(on, user=True))
        bar.addWidget(self._btn_side)
        bar.addStretch(1)
        vl.addLayout(bar)
        if self._plotter_factory is not None:
            self._plotter, widget = self._plotter_factory(view)
        else:
            self._plotter = qt(view)
            widget = self._plotter.interactor if hasattr(self._plotter, "interactor") else self._plotter
        vl.addWidget(widget)
        self._plotter.set_background("white")
        self._plotter.add_axes()
        split.addWidget(view)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([260, 700])
        self._split = split

    def _set_side_visible(self, on: bool, user: bool = False) -> None:
        if self._side is None:
            return
        if user:
            self._side_user = True
        self._side.setVisible(bool(on))
        self._btn_side.blockSignals(True)
        self._btn_side.setChecked(bool(on))
        self._btn_side.blockSignals(False)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # agganciata in una colonna stretta il pannello dei comandi mangerebbe tutto lo spazio:
        # finche' l'utente non sceglie, lo si nasconde sotto una larghezza minima
        if self._side is None or self._side_user:
            return
        want = self.width() >= _SIDE_MIN_TOTAL_WIDTH
        if want == self._side.isHidden():
            self._set_side_visible(want, user=False)

    # --- elenchi -----------------------------------------------------------------------------------
    def _project_root(self) -> str:
        try:
            return str(self._root_getter() or "") if self._root_getter else ""
        except Exception:
            return ""

    def _workspace_root(self) -> str:
        try:
            return str(self._workspace_provider() or "") if self._workspace_provider else ""
        except Exception:
            return ""

    def _zone_list(self) -> list:
        """Aree del progetto come [{"name", "base", "active"}]; senza project.json, un'area unica senza nome."""
        zones = []
        if self._zones_provider is not None:
            try:
                for z in (self._zones_provider() or []):
                    zones.append({"name": str(z.get("name") or ""), "base": str(z.get("base") or ""),
                                  "active": bool(z.get("active"))})
            except Exception:
                zones = []
        if not zones:
            return [{"name": "", "base": self._workspace_root(), "active": True}]
        with_data = [z for z in zones if discover_variants(z["base"])]
        return with_data or zones

    def _current_zone(self):
        name = self._cb_zone.currentData()
        return next((z for z in self._zones if z["name"] == name), self._zones[0] if self._zones else None)

    def refresh_variants(self, *_a) -> None:
        """Rilegge aree e varianti mantenendo la scelta fatta; apre la variante selezionata."""
        if self._plotter is None:
            return
        prev_zone = self._cb_zone.currentData()
        prev_variant = str(self._cb_variant.currentData() or "")
        self._zones = self._zone_list()
        self._updating = True
        try:
            self._cb_zone.clear()
            for z in self._zones:
                self._cb_zone.addItem(z["name"] or "-", z["name"])
            idx = next((i for i, z in enumerate(self._zones) if z["name"] == prev_zone), -1)
            if idx < 0:
                idx = next((i for i, z in enumerate(self._zones) if z["active"]), 0)
            self._cb_zone.setCurrentIndex(idx)
            self._lbl_zone.setVisible(len(self._zones) > 1)
            self._cb_zone.setVisible(len(self._zones) > 1)
            self._fill_variants(prev_variant)
        finally:
            self._updating = False
        self._open_selected_variant()

    def _fill_variants(self, keep: str = "") -> None:
        zone = self._current_zone()
        refs = discover_variants(zone["base"]) if zone else []
        self._cb_variant.clear()
        for r in refs:
            self._cb_variant.addItem(f"{r.name}  ({r.n_files})", r.path)
        want = self._cb_variant.findData(keep) if keep else -1
        if want < 0 and self._cb_variant.count():
            names = [os.path.basename(str(self._cb_variant.itemData(i))) for i in range(self._cb_variant.count())]
            want = names.index("Lavoro") if "Lavoro" in names else 0
        self._cb_variant.setCurrentIndex(want)

    def _on_zone_changed(self, _i: int) -> None:
        if self._updating:
            return
        self._updating = True
        try:
            self._fill_variants()
        finally:
            self._updating = False
        self._open_selected_variant()

    def _on_variant_changed(self, _i: int) -> None:
        if not self._updating:
            self._open_selected_variant()

    def _make_geometry_provider(self, zone_name: str):
        """Il Profile Viewer ha in memoria solo i profili dell'area attiva: per le altre la geometria
        resta quella salvata nei .npz (stessa regola del pannello dei radargrammi)."""
        if self._geometry_provider is None:
            return None

        def provider(stem):
            z = next((z for z in self._zones if z["name"] == zone_name), None)
            if z is not None and not z["active"]:
                return None
            return self._geometry_provider(stem)
        return provider

    def _open_selected_variant(self) -> None:
        path = str(self._cb_variant.currentData() or "")
        cur = self._store
        if path and cur is not None and os.path.normcase(cur.path) == os.path.normcase(os.path.normpath(path))                 and not cur.is_stale():
            self.refresh_lists()                      # stessa variante: aggiorna solo gli elenchi (ERT nuove)
            return
        zone = self._current_zone()
        store = VariantStore(path, geometry_provider=self._make_geometry_provider(
            zone["name"] if zone else "")) if path else None
        self._set_store(store)

    def _set_store(self, store) -> None:
        if store is not self._store:                  # altra variante: cache e origine non valgono piu'
            self._gpr_cache.clear()
            self._origin = None
            self._first_camera = True
        self._store = store
        self.refresh_lists()

    def _import_ert(self, *_a) -> None:
        root = self._project_root()
        if not root:
            self._txt_cross.setPlainText("⚠ Apri un progetto di GeoSurvey Studio: la sezione ERT si salva nella "
                                         "cartella del progetto.")
            return
        canvas = self.iface.mapCanvas() if self.iface is not None else None
        name, error = import_ert_section(self, root, canvas)
        if error:
            self._txt_cross.setPlainText(f"⚠ {error}")
            return
        if name:
            self._ert_cache.pop(name, None)
            self.refresh_lists()
            self.set_checked("ert", [name] + [t for t, _ in self.checked("ert")])

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.refresh_variants()                       # varianti e sezioni ERT possono essere cambiate nel frattempo

    def refresh_lists(self) -> None:
        """Ricostruisce l'elenco dei profili GPR (variante aperta) e delle sezioni ERT, mantenendo le spunte."""
        if self._plotter is None:
            return
        store = self._store
        self._ert_cache.clear()
        checked = {self._path_key(it) for it in self._iter_items() if it.checkState(0) == Qt.Checked}
        expanded = {self._path_key(it): it.isExpanded() for it in self._iter_all()}
        self._slice_info.clear()
        self._updating = True
        try:
            self._tree.clear()
            n_gpr = len(store.profiles) if store is not None else 0
            gpr_title = f"Profili GPR ({store.name}) \u00b7 {n_gpr}" if store is not None else "Profili GPR"
            gpr = QTreeWidgetItem(self._tree, [gpr_title])
            ert = QTreeWidgetItem(self._tree, ["Sezioni ERT"])
            slc = QTreeWidgetItem(self._tree, ["Time-slice"])
            for top in (gpr, ert, slc):
                top.setFlags(Qt.ItemIsEnabled)
                top.setExpanded(expanded.get((top.text(0),), True))
            if store is not None:
                for pi, p in enumerate(store.profiles):
                    self._add_child(gpr, p.stem, ("gpr", pi), (gpr_title, p.stem) in checked)
            for name in list_ert(self._project_root()):
                self._add_child(ert, name, ("ert", name), (ert.text(0), name) in checked)
            for g in discover_slice_groups(self._slice_root()):
                title = f"{g['name']}  ({g['variant']})" if g["variant"] else g["name"]
                gi = QTreeWidgetItem(slc, [title])
                gi.setFlags(Qt.ItemIsEnabled)
                gi.setExpanded(expanded.get((slc.text(0), title),
                                            store is not None and g["variant"] == getattr(store, "name", None)))
                for row in g["slices"]:
                    label = f"{row['z_from']:.2f}–{row['z_to']:.2f} m"
                    self._slice_info[row["path"]] = dict(row, epsg=g["epsg"], group=g["name"])
                    self._add_child(gi, label, ("slice", row["path"]), (slc.text(0), title, label) in checked)
        finally:
            self._updating = False
        # tanti profili GPR coprirebbero ERT e time-slice sotto la piega dell'elenco: il gruppo parte chiuso
        # (a meno che l'utente l'abbia gia' aperto o ci sia qualcosa di spuntato)
        if (gpr_title,) not in expanded:
            gpr.setExpanded(n_gpr <= _GPR_EXPAND_MAX or any(k[:1] == (gpr_title,) for k in checked))
        self._apply_filter(self._ed_filter.text())
        self._schedule()

    @staticmethod
    def _add_child(parent, text, payload, checked: bool) -> None:
        it = QTreeWidgetItem(parent, [text])
        it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)
        it.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)
        it.setData(0, _ROLE, payload)

    def _iter_items(self):
        """Voci spuntabili (con un tipo di dati), a qualunque profondita' dell'albero."""
        stack = [self._tree.topLevelItem(t) for t in range(self._tree.topLevelItemCount())][::-1]
        while stack:
            it = stack.pop()
            if it.data(0, _ROLE):
                yield it
            stack.extend(it.child(c) for c in range(it.childCount() - 1, -1, -1))

    def _iter_all(self):
        """Tutte le voci dell'albero, gruppi compresi."""
        stack = [self._tree.topLevelItem(t) for t in range(self._tree.topLevelItemCount())][::-1]
        while stack:
            it = stack.pop()
            yield it
            stack.extend(it.child(c) for c in range(it.childCount() - 1, -1, -1))

    @staticmethod
    def _path_key(item) -> tuple:
        """Testi dalla radice alla voce: identifica una voce anche dopo la ricostruzione dell'albero."""
        parts = []
        while item is not None:
            parts.append(item.text(0))
            item = item.parent()
        return tuple(reversed(parts))

    def _slice_root(self) -> str:
        zone = self._current_zone()
        return str(zone["base"]) if zone else self._workspace_root()

    def _check_group(self, group: int, on: bool) -> None:
        self._updating = True
        try:
            top = self._tree.topLevelItem(group)
            for c in range(top.childCount()):
                it = top.child(c)
                if not it.isHidden():
                    it.setCheckState(0, Qt.Checked if on else Qt.Unchecked)
        finally:
            self._updating = False
        self._schedule()

    def _apply_filter(self, text: str) -> None:
        needle = str(text or "").strip().lower()
        for it in self._iter_items():
            it.setHidden(bool(needle) and needle not in it.text(0).lower())

    def _on_item_changed(self, _item, _col) -> None:
        if not self._updating:
            self._schedule()

    def checked(self, kind: str) -> list:
        """Payload (indice profilo GPR o nome ERT) degli elementi spuntati di un tipo."""
        out = []
        for it in self._iter_items():
            data = it.data(0, _ROLE)
            if data and data[0] == kind and it.checkState(0) == Qt.Checked:
                out.append((it.text(0), data[1]))
        return out

    def set_checked(self, kind: str, names) -> None:
        """Spunta solo i profili con questi nomi (per uso programmatico e test)."""
        wanted = set(names)
        self._updating = True
        try:
            for it in self._iter_items():
                data = it.data(0, _ROLE)
                if data and data[0] == kind:
                    names_of_item = {it.text(0), os.path.basename(str(data[1]))}
                    it.setCheckState(0, Qt.Checked if names_of_item & wanted else Qt.Unchecked)
        finally:
            self._updating = False
        self._schedule()

    # --- coordinate -----------------------------------------------------------------------------------
    @staticmethod
    def _common_crs() -> str:
        return QgsProject.instance().crs().authid() or ""

    def _to_common(self, xy, src_authid: str):
        dst = self._common_crs()
        if not dst or not src_authid or src_authid.upper() == dst.upper():
            return xy
        tr = lonlat_transformer(dst, src_authid)
        return np.asarray([tr(x, y) for x, y in xy], dtype=np.float64)

    def _set_origin_from(self, xy) -> None:
        if self._origin is None:
            self._origin = (float(xy[0][0]), float(xy[0][1]))

    # --- dati dei profili ------------------------------------------------------------------------------
    def _gpr_item(self, pi: int, ci: int):
        key = (id(self._store), pi, ci)
        hit = self._gpr_cache.get(key)
        if hit is not None:
            self._gpr_cache.move_to_end(key)
            return hit
        prof = self._store.profiles[pi]
        ci = int(min(ci, prof.n_channels - 1))
        ch = self._store.channel(pi, ci)
        base = normalize_display(ch.data).astype(np.float32, copy=False)
        base = RadargramViewMixin._resample_disp_to_uniform_grid(base, ch.dist)
        xy, d = track_vertices(ch.east, ch.north, ch.dist)
        epsg = int(getattr(prof, "epsg", 0) or 0)
        item = {"name": prof.stem, "base": downsample(base), "xy": self._to_common(xy, f"EPSG:{epsg}" if epsg else ""),
                "d": d, "span": (float(ch.dist[0]), float(ch.dist[-1])), "depth": float(ch.depth_max_m)}
        self._gpr_cache[key] = item
        while len(self._gpr_cache) > _CACHE_MAX:
            self._gpr_cache.popitem(last=False)
        return item

    def _slice_item(self, path: str):
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0.0
        hit = self._slice_cache.get(path)
        if hit is None or hit.get("_mtime") != mtime:      # raster rigenerato: si rilegge
            hit = dict(load_slice(path), _mtime=mtime)
            self._slice_cache[path] = hit
            while len(self._slice_cache) > _CACHE_MAX:
                self._slice_cache.pop(next(iter(self._slice_cache)))
        info = self._slice_info.get(path, {})
        epsg = hit["epsg"] or int(info.get("epsg") or 0)
        corners = self._to_common(np.asarray(hit["corners"], dtype=np.float64), f"EPSG:{epsg}" if epsg else "")
        return dict(hit, corners=np.asarray(corners), path=path, name=os.path.basename(path),
                    z_from=float(info.get("z_from", 0.0)), z_to=float(info.get("z_to", 0.0)),
                    z_center=float(info.get("z_center", 0.0)))

    def _ert_item(self, name: str):
        hit = self._ert_cache.get(name)
        if hit is None:
            entry = load_ert(self._project_root(), name)
            grid = ert_to_grid(entry.section)
            x0, x1 = float(grid.x_axis[0]), float(grid.x_axis[-1])
            xy, d = polyline_vertices(entry.georef, x0, x1)
            hit = {"name": name, "grid": grid, "xy": self._to_common(xy, entry.georef.crs_authid), "d": d,
                   "span": (x0, x1)}
            self._ert_cache[name] = hit
        return hit

    # --- scena -----------------------------------------------------------------------------------------
    def _schedule(self, *_a) -> None:
        if self._plotter is not None:
            self._timer.start()

    def _apply_scale(self, *_a) -> None:
        if self._plotter is not None:
            self._plotter.set_scale(zscale=float(self._spin_ve.value()))
            self._plotter.render()

    def _add_curtain(self, name: str, item: dict, rgba, depth_top: float, depth_bottom: float, kind: str):
        c = curtain(item["xy"], item["d"], item["span"], depth_top, depth_bottom, origin=self._origin)
        mesh = self._pv.PolyData(c["points"], c["faces"])
        mesh.active_texture_coordinates = c["tcoords"]
        self._plotter.add_mesh(mesh, texture=self._pv.Texture(rgba), lighting=False, name=name,
                               reset_camera=False, show_scalar_bar=False)
        self._shown[name] = kind

    def _rebuild(self) -> None:
        if self._plotter is None:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self._rebuild_scene()
        finally:
            QApplication.restoreOverrideCursor()

    def _rebuild_scene(self) -> None:
        pl = self._plotter
        for name in list(self._shown):
            pl.remove_actor(name, render=False)
        self._shown.clear()
        if self._hint_on:
            pl.remove_actor("hint", render=False)
            self._hint_on = False
        dmax = float(self._spin_depth.value())
        gpr_sel = self.checked("gpr") if self._store is not None else []
        ert_sel = self.checked("ert")
        errors, label_pts, label_txt = [], [], []

        gprs, erts = [], []
        for text, pi in gpr_sel:
            try:
                gprs.append(self._gpr_item(int(pi), int(self._spin_ch.value())))
            except Exception as exc:
                errors.append(f"{text}: {exc}")
        for text, name in ert_sel:
            try:
                erts.append(self._ert_item(str(name)))
            except Exception as exc:
                errors.append(f"{text}: {exc}")
        slices = []
        for text, path in self.checked("slice"):
            try:
                slices.append(self._slice_item(str(path)))
            except Exception as exc:
                errors.append(f"{text}: {exc}")
        for it in gprs + erts:
            self._set_origin_from(it["xy"])
        for it in slices:
            self._set_origin_from(it["corners"])

        gain = float(self._spin_gain.value())
        for it in gprs:
            base, depth = it["base"], it["depth"]
            rows = base.shape[0] if dmax <= 0 else max(2, int(np.ceil(base.shape[0] * min(1.0, dmax / depth))))
            shown_depth = depth * rows / base.shape[0]
            rgba = rgba_from_array(np.clip(base[:rows] * gain, -1.0, 1.0), self._cb_gpr.currentText(), -1.0, 1.0)
            self._add_curtain(f"gpr::{it['name']}", it, rgba, 0.0, shown_depth, "gpr")
            label_pts.append((*self._local(it["xy"][0]), 0.0))
            label_txt.append(it["name"])

        if erts:
            vals = np.concatenate([e["grid"].data[np.isfinite(e["grid"].data)] for e in erts])
            lo, hi = np.percentile(vals, _ERT_CLIM_PERCENTILES) if vals.size else (0.0, 1.0)
            for it in erts:
                g = it["grid"]
                depth = g.depth_axis
                keep = np.ones(len(depth), bool) if dmax <= 0 else depth <= dmax
                if keep.sum() < 2:                      # limite sotto il passo di griglia: restano le prime 2 righe
                    keep[:2] = True
                rgba = rgba_from_array(g.data[keep], self._cb_ert.currentText(), float(lo), float(hi))
                self._add_curtain(f"ert::{it['name']}", it, rgba, float(depth[keep][0]), float(depth[keep][-1]), "ert")
                label_pts.append((*self._local(it["xy"][0]), 0.0))
                label_txt.append(it["name"])

        style = str(self._cb_slice.currentData() or "")
        for it in slices:
            cmap = (colormap_from_hex(it["colors"]) if not style else None) or (style or "viridis")
            rgba = rgba_from_array(it["data"], cmap, it["vmin"], it["vmax"])
            pl_geo = plane(it["corners"], it["z_center"], origin=self._origin)
            mesh = self._pv.PolyData(pl_geo["points"], pl_geo["faces"])
            mesh.active_texture_coordinates = pl_geo["tcoords"]
            name = f"slice::{it['name']}"
            pl.add_mesh(mesh, texture=self._pv.Texture(rgba), lighting=False, name=name, reset_camera=False,
                        show_scalar_bar=False, opacity=float(self._spin_slice_op.value()), render=False)
            self._shown[name] = "slice"
            label_pts.append((*self._local(it["corners"][0]), -it["z_center"]))
            label_txt.append(f"{it['z_from']:.2f}–{it['z_to']:.2f} m")

        if self._chk_labels.isChecked() and label_pts:
            pl.add_point_labels(np.asarray(label_pts), label_txt, name="labels", font_size=11, shape=None,
                                show_points=False, always_visible=True, text_color="black", render=False)
            self._shown["labels"] = "x"
        self._draw_crossings(gprs, erts, dmax)
        if errors:
            self._txt_cross.appendPlainText("\n".join(f"⚠ {e}" for e in errors))
        if not self._shown:
            pl.add_text("Spunta profili GPR, sezioni ERT o time-slice nell'elenco (\u2630 Comandi)", name="hint",
                        position="upper_left", font_size=10, color="gray", render=False)
            self._hint_on = True
        self._apply_scale()
        if self._shown and self._first_camera:
            self._view_iso()
            self._first_camera = False
        pl.render()

    def _local(self, xy) -> tuple:
        return float(xy[0] - self._origin[0]), float(xy[1] - self._origin[1])

    def _draw_crossings(self, gprs, erts, dmax: float) -> None:
        lines = []
        if self._chk_cross.isChecked():
            for e in erts:
                depth_e = float(e["grid"].depth_axis[-1])
                for g in gprs:
                    for da, db, x, y in crossings(g["xy"], g["d"], e["xy"], e["d"]):
                        depth = min(g["depth"], depth_e) if dmax <= 0 else dmax
                        name = f"x::{g['name']}::{e['name']}::{da:.2f}"
                        line = self._pv.Line((x - self._origin[0], y - self._origin[1], 0.0),
                                             (x - self._origin[0], y - self._origin[1], -depth))
                        self._plotter.add_mesh(line, color="red", line_width=4, name=name,
                                               reset_camera=False, render=False)
                        self._shown[name] = "x"
                        lines.append(f"{e['name']} × {g['name']}:  ERT a {db:.1f} m,  GPR a {da:.1f} m")
        self._txt_cross.setPlainText("\n".join(lines))

    def _view_top(self) -> None:
        if self._plotter is not None:
            self._plotter.view_xy()
            self._plotter.reset_camera()
            self._plotter.render()

    def _view_iso(self) -> None:
        if self._plotter is not None:
            self._plotter.camera_position = "iso"
            self._plotter.reset_camera()
            self._plotter.render()

    # --- chiusura -----------------------------------------------------------------------------------------
    def cleanup(self) -> None:
        self._timer.stop()
        plotter, self._plotter = self._plotter, None
        if plotter is not None:
            try:
                plotter.close()
            except Exception:
                pass

    def closeEvent(self, event) -> None:
        self.cleanup()
        super().closeEvent(event)
