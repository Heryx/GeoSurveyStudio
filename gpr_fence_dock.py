# -*- coding: utf-8 -*-
"""Finestra agganciabile "Vista 3D profili": radargrammi GPR, sezioni ERT e time-slice nello spazio.

Ogni profilo si disegna lungo la sua linea reale sulla mappa (coordinate per traccia del GPR,
georiferimento per l'ERT), quindi due profili che si incrociano si incrociano anche qui, nel punto
giusto. Profondita' sempre in verticale dal piano z = 0 (modalita' piana, come nel pannello 2D). Le
time-slice sono piani orizzontali alla profondita' centrale della loro finestra.

Cosa si vede si decide in due passi:

- **Sorgenti GPR**: un elenco di zone -> varianti processate, spuntabili a piacere. Ogni variante
  spuntata porta nell'elenco dei profili un gruppo con i suoi profili; zone e processi diversi si
  possono quindi vedere insieme.
- **Elenco dei profili**: GPR (un gruppo per sorgente), sezioni ERT e time-slice (un gruppo per
  cartella). Si spunta cio' che si vuole vedere, oppure si usano i cursori sotto la vista: ognuno
  scorre un elemento alla volta *dentro il gruppo scelto* con il suo selettore, senza attraversare
  le altre zone.
"""

from __future__ import annotations

import os
from collections import Counter, OrderedDict

import numpy as np
from qgis.PyQt.QtCore import QTimer, Qt, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from qgis.core import QgsProject, QgsRasterLayer

from .gpr_ert_georef import lonlat_transformer
from .gpr_ert_import import to_grid as ert_to_grid
from .gpr_ert_import_dialog import import_ert_section
from .gpr_ert_store import list_ert, load_ert
from .gpr_fence_slices import colormap_from_hex, load_slice, project_slice_groups
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
_ROLE = Qt.UserRole                 # dati di una voce spuntabile: (tipo, payload)
_GROUP = Qt.UserRole + 1            # identificativo stabile di un gruppo (sorgente GPR o cartella di time-slice)
_SIDE_MIN_TOTAL_WIDTH = 640         # sotto questa larghezza il pannello dei comandi si nasconde da solo
_GPR_EXPAND_MAX = 15                # oltre questi profili il gruppo parte chiuso (non copre ERT e time-slice)
_SEP = "\x1f"


def _load_pyvista():
    """(pv, QtInteractor) o (None, motivo) se pyvista/pyvistaqt non sono disponibili."""
    from . import gpr_3d_viewer_dialog as v3d

    if not v3d._bootstrap_pyvista_runtime():
        err = v3d._PYVISTA_IMPORT_ERROR or v3d._QTINTERACTOR_IMPORT_ERROR or "pyvista/pyvistaqt non trovati"
        return None, str(err)
    return v3d.pv, v3d.QtInteractor


class _Scrubber(QWidget):
    """Due righe: [x] titolo + selettore del gruppo  /  ◀ cursore ▶ nome dell'elemento (n/N).

    Spuntata, la categoria mostra soltanto l'elemento sotto il cursore; non spuntata, quelli spuntati
    nell'elenco. Il cursore percorre gli elementi del *gruppo scelto* nel selettore, nell'ordine in cui
    compaiono; con un gruppo solo il selettore non si vede.
    """

    changed = pyqtSignal()

    def __init__(self, title: str, tip: str, parent=None):
        super().__init__(parent)
        lay = QGridLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setHorizontalSpacing(4)
        lay.setVerticalSpacing(0)
        self.chk = QCheckBox(title)
        self.chk.setToolTip(tip)
        self.cb_scope = QComboBox()
        self.cb_scope.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.cb_scope.setMinimumContentsLength(8)
        self.cb_scope.setToolTip("Gruppo su cui scorre il cursore (zona e processo, oppure cartella di time-slice).")
        self.cb_scope.setVisible(False)
        self.btn_prev = QToolButton()
        self.btn_prev.setText("◀")
        self.btn_prev.setAutoRaise(True)
        self.btn_next = QToolButton()
        self.btn_next.setText("▶")
        self.btn_next.setAutoRaise(True)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.setTracking(True)
        self.slider.setMinimumWidth(60)
        self.lbl = QLabel("")
        self.lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)   # il testo lungo si taglia, non allarga la finestra
        lay.addWidget(self.chk, 0, 0, 1, 2)
        lay.addWidget(self.cb_scope, 0, 2, 1, 3)
        lay.addWidget(self.btn_prev, 1, 0)
        lay.addWidget(self.slider, 1, 1, 1, 2)
        lay.addWidget(self.btn_next, 1, 3)
        lay.addWidget(self.lbl, 1, 4)
        lay.setColumnStretch(2, 1)
        lay.setColumnStretch(4, 2)
        self.chk.toggled.connect(lambda _on: self.changed.emit())
        self.slider.valueChanged.connect(lambda _v: self.changed.emit())
        self.cb_scope.currentIndexChanged.connect(self._on_scope)
        self.btn_prev.clicked.connect(lambda: self.slider.setValue(self.slider.value() - 1))
        self.btn_next.clicked.connect(lambda: self.slider.setValue(self.slider.value() + 1))
        self._scope_updating = False
        self.set_count(0)

    # --- gruppi ----------------------------------------------------------------------------------------
    def set_scopes(self, options) -> None:
        """``options`` = [(etichetta, chiave)]; resta scelto il gruppo di prima se esiste ancora."""
        keep = self.scope_key()
        self._scope_updating = True
        try:
            self.cb_scope.clear()
            for label, key in options:
                self.cb_scope.addItem(label, key)
            i = self.cb_scope.findData(keep) if keep is not None else -1
            self.cb_scope.setCurrentIndex(i if i >= 0 else (0 if options else -1))
        finally:
            self._scope_updating = False
        self.cb_scope.setVisible(len(options) > 1)

    def scope_key(self):
        return self.cb_scope.currentData() if self.cb_scope.count() else None

    def _on_scope(self, _i: int) -> None:
        if self._scope_updating:
            return
        self.slider.blockSignals(True)
        self.slider.setValue(0)                       # altro gruppo: si riparte dal primo elemento
        self.slider.blockSignals(False)
        self.changed.emit()

    # --- cursore -----------------------------------------------------------------------------------------
    def set_count(self, n: int) -> None:
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, n - 1))
        self.slider.setValue(min(self.slider.value(), max(0, n - 1)))
        self.slider.blockSignals(False)
        for w in (self.chk, self.btn_prev, self.slider, self.btn_next):
            w.setEnabled(n > 0)
        if n == 0:
            self.lbl.setText("nessun elemento")

    def count(self) -> int:
        return self.slider.maximum() + 1 if self.slider.isEnabled() else 0

    def index(self) -> int:
        return self.slider.value()

    def is_active(self) -> bool:
        return self.chk.isChecked() and self.count() > 0

    def set_active(self, on: bool) -> None:
        self.chk.blockSignals(True)
        self.chk.setChecked(bool(on))
        self.chk.blockSignals(False)

    def set_info(self, text: str) -> None:
        self.lbl.setText(text)
        self.lbl.setToolTip(text)


class FenceDock(QDockWidget):
    """Vista 3D dei profili GPR, delle sezioni ERT e delle time-slice del progetto."""

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
        self._stores: dict = {}           # id sorgente -> VariantStore (caricati solo per le sorgenti spuntate)
        self._sources_seeded = False      # la prima volta si spunta la variante di lavoro; poi decide l'utente
        self._gpr_cache: OrderedDict = OrderedDict()
        self._ert_cache: dict = {}
        self._slice_cache: dict = {}
        self._slice_info: dict = {}       # percorso raster -> riga del manifest
        self._origin = None
        self._side = None
        self._scrub: dict = {}
        self._side_user = False           # l'utente ha scelto a mano: niente piu' nascondi/mostra automatico
        self._hint_on = False
        self._shown: dict = {}            # nome attore -> tipo ("gpr" | "ert" | "slice" | "x")
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
        side.setMinimumWidth(180)
        self._side = side
        sl = QVBoxLayout(side)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.addWidget(QLabel("Zone e processi GPR da mostrare"))
        self._src_tree = QTreeWidget()
        self._src_tree.setHeaderHidden(True)
        self._src_tree.setMaximumHeight(130)
        self._src_tree.setToolTip("Spunta le varianti processate (zona › processo) di cui vuoi i profili: "
                                  "se ne possono vedere più insieme.")
        self._src_tree.itemChanged.connect(self._on_source_changed)
        sl.addWidget(self._src_tree)
        self._ed_filter = QLineEdit()
        self._ed_filter.setPlaceholderText("Filtra i profili…")
        self._ed_filter.textChanged.connect(self._apply_filter)
        sl.addWidget(self._ed_filter)
        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.itemChanged.connect(self._on_item_changed)
        sl.addWidget(self._tree, 1)
        row = QHBoxLayout()
        for text, tip, slot in (("Tutti i GPR", "Spunta tutti i profili GPR visibili nel filtro.", lambda: self._check_kind("gpr", True)),
                                ("Nessun GPR", "Toglie la spunta a tutti i profili GPR.", lambda: self._check_kind("gpr", False)),
                                ("+ ERT", "Importa una sezione ERT invertita (CSV di ResIPY) e georiferiscila.", self._import_ert),
                                ("↻", "Rilegge zone, varianti, profili GPR, sezioni ERT e time-slice.", self.refresh_variants)):
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
        self._btn_side.setText("☰ Comandi")
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
        widget.setMinimumSize(120, 120)
        vl.addWidget(widget, 1)
        for kind, title, tip in (
                ("gpr", "Profili GPR", "Spuntata: mostra solo il profilo GPR sotto il cursore, nel gruppo scelto."),
                ("ert", "Sezioni ERT", "Spuntata: mostra solo la sezione ERT sotto il cursore."),
                ("slice", "Time-slice", "Spuntata: mostra solo la time-slice sotto il cursore, dalla più superficiale alla più profonda.")):
            sc = _Scrubber(title, tip)
            sc.set_active(True)
            sc.changed.connect(self._on_scrub_changed)
            self._scrub[kind] = sc
            vl.addWidget(sc)
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

    # --- provider --------------------------------------------------------------------------------
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

    def _zone_named(self, name: str):
        return next((z for z in self._zones if z["name"] == name), None)

    def _zone_label_of_path(self, path: str) -> str:
        """Nome dell'area che contiene ``path`` (la piu' specifica), o stringa vuota."""
        p = os.path.normcase(os.path.normpath(str(path or "")))
        best = ""
        best_len = -1
        for z in self._zones:
            base = os.path.normcase(os.path.normpath(z["base"])) if z["base"] else ""
            if z["name"] and base and (p == base or p.startswith(base + os.sep)) and len(base) > best_len:
                best, best_len = z["name"], len(base)
        return best

    # --- sorgenti GPR: zone -> varianti ------------------------------------------------------------------
    @staticmethod
    def _source_id(zone: str, path: str) -> str:
        return f"{zone}{_SEP}{os.path.normcase(os.path.normpath(path))}"

    def refresh_variants(self, *_a) -> None:
        """Rilegge zone e varianti mantenendo le spunte; ricostruisce poi gli elenchi dei profili."""
        if self._plotter is None:
            return
        self._zones = self._zone_list()
        prev = {it.data(0, _GROUP) for it in self._src_leaves() if it.checkState(0) == Qt.Checked}
        self._updating = True
        try:
            self._src_tree.clear()
            single = len(self._zones) == 1 and not self._zones[0]["name"]
            for z in self._zones:
                refs = discover_variants(z["base"])
                if not refs:
                    continue
                parent = self._src_tree.invisibleRootItem()
                if not single:
                    parent = QTreeWidgetItem(self._src_tree, [z["name"] or "-"])
                    parent.setFlags(Qt.ItemIsEnabled)
                    parent.setExpanded(True)
                default = next((r for r in refs if r.name == "Lavoro"), refs[0])
                for r in refs:
                    it = QTreeWidgetItem(parent, [f"{r.name}  ({r.n_files})"])
                    it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                    sid = self._source_id(z["name"], r.path)
                    it.setData(0, _GROUP, sid)
                    it.setData(0, _ROLE, {"zone": z["name"], "path": r.path, "name": r.name, "id": sid})
                    # la prima volta si spunta la variante di lavoro dell'area attiva
                    seed = (not self._sources_seeded) and z["active"] and r is default
                    it.setCheckState(0, Qt.Checked if (sid in prev or seed) else Qt.Unchecked)
            self._sources_seeded = True
        finally:
            self._updating = False
        self.refresh_lists()

    def _src_leaves(self):
        stack = [self._src_tree.topLevelItem(t) for t in range(self._src_tree.topLevelItemCount())][::-1]
        while stack:
            it = stack.pop()
            if it.data(0, _ROLE):
                yield it
            stack.extend(it.child(c) for c in range(it.childCount() - 1, -1, -1))

    def checked_sources(self) -> list:
        """Sorgenti GPR spuntate: ``[{"zone", "path", "name", "id"}, ...]`` nell'ordine dell'elenco."""
        return [dict(it.data(0, _ROLE)) for it in self._src_leaves() if it.checkState(0) == Qt.Checked]

    def set_sources(self, names) -> None:
        """Spunta solo le varianti con questi nomi (o ``zona/variante``); per uso programmatico e test."""
        wanted = set(names)
        self._updating = True
        try:
            for it in self._src_leaves():
                d = it.data(0, _ROLE)
                ok = d["name"] in wanted or f"{d['zone']}/{d['name']}" in wanted
                it.setCheckState(0, Qt.Checked if ok else Qt.Unchecked)
        finally:
            self._updating = False
        self.refresh_lists()

    def _on_source_changed(self, _item, _col) -> None:
        if not self._updating:
            self.refresh_lists()

    def _make_geometry_provider(self, zone_name: str):
        """Il Profile Viewer ha in memoria solo i profili dell'area attiva: per le altre la geometria
        resta quella salvata nei .npz (stessa regola del pannello dei radargrammi)."""
        if self._geometry_provider is None:
            return None

        def provider(stem):
            z = self._zone_named(zone_name)
            if z is not None and not z["active"]:
                return None
            return self._geometry_provider(stem)
        return provider

    def _store_for(self, src: dict):
        """Archivio della sorgente: si crea al primo bisogno e si rifa se i file sono cambiati su disco."""
        store = self._stores.get(src["id"])
        if store is not None and store.is_stale():
            store = None
        if store is None:
            store = VariantStore(src["path"], geometry_provider=self._make_geometry_provider(src["zone"]))
            self._stores[src["id"]] = store
        return store

    @staticmethod
    def _source_title(src: dict, n: int) -> str:
        head = f"{src['zone']} › {src['name']}" if src["zone"] else src["name"]
        return f"{head}  ({n})"

    # --- importazione ERT --------------------------------------------------------------------------------
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

    # --- elenco dei profili -------------------------------------------------------------------------------
    def refresh_lists(self) -> None:
        """Ricostruisce l'elenco di profili GPR (uno per sorgente spuntata), sezioni ERT e time-slice."""
        if self._plotter is None:
            return
        self._ert_cache.clear()
        checked = {self._path_key(it) for it in self._iter_items() if it.checkState(0) == Qt.Checked}
        expanded = {self._path_key(it): it.isExpanded() for it in self._iter_all()}
        self._slice_info.clear()
        sources = self.checked_sources()
        live = {s["id"] for s in sources}
        for sid in [k for k in self._stores if k not in live]:      # sorgente tolta: si libera l'archivio
            del self._stores[sid]
        notes = []
        self._updating = True
        try:
            self._tree.clear()
            gpr = QTreeWidgetItem(self._tree, ["Profili GPR"])
            ert = QTreeWidgetItem(self._tree, ["Sezioni ERT"])
            slc = QTreeWidgetItem(self._tree, ["Time-slice"])
            for top in (gpr, ert, slc):
                top.setFlags(Qt.ItemIsEnabled)
                top.setExpanded(expanded.get((top.text(0),), True))
            for src in sources:
                try:
                    store = self._store_for(src)
                except Exception as exc:
                    notes.append(f"{src['name']}: {exc}")
                    continue
                n = len(store.profiles)
                gi = QTreeWidgetItem(gpr, [self._source_title(src, n)])
                gi.setFlags(Qt.ItemIsEnabled)
                gi.setData(0, _GROUP, src["id"])
                key = (gpr.text(0), src["id"])
                for pi, p in enumerate(store.profiles):
                    self._add_child(gi, p.stem, ("gpr", (src["id"], pi)), key + (p.stem,) in checked)
                # tanti profili coprirebbero ERT e time-slice sotto la piega dell'elenco: il gruppo parte chiuso,
                # a meno che l'utente l'abbia gia' aperto o ci sia qualcosa di spuntato
                gi.setExpanded(expanded.get(key, n <= _GPR_EXPAND_MAX or any(k[:2] == key for k in checked)))
            for name in list_ert(self._project_root()):
                self._add_child(ert, name, ("ert", name), (ert.text(0), name) in checked)
            variants = {s["name"] for s in sources}
            for g in project_slice_groups(self._slice_roots(), self._project_raster_paths()):
                zone = self._zone_label_of_path(g["dir"])
                head = f"{zone} › {g['name']}" if zone else g["name"]
                title = f"{head}  ({g['variant']})" if g["variant"] else head
                gi = QTreeWidgetItem(slc, [title])
                gi.setFlags(Qt.ItemIsEnabled)
                gi.setData(0, _GROUP, "slice" + _SEP + os.path.normcase(os.path.normpath(g["dir"])))
                key = (slc.text(0), gi.data(0, _GROUP))
                gi.setExpanded(expanded.get(key, g["variant"] in variants))
                for row in g["slices"]:
                    label = f"{row['z_from']:.2f}–{row['z_to']:.2f} m"
                    self._slice_info[row["path"]] = dict(row, epsg=g["epsg"], group=g["name"])
                    self._add_child(gi, label, ("slice", row["path"]), key + (label,) in checked)
        finally:
            self._updating = False
        if notes:
            self._txt_cross.setPlainText("\n".join(f"⚠ {n}" for n in notes))
        self._apply_filter(self._ed_filter.text())
        self._update_scrub()
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
        """Dalla radice alla voce: identifica una voce anche dopo la ricostruzione dell'albero.

        I gruppi si riconoscono dall'identificativo (stabile) e non dal titolo, che porta conteggi e nomi.
        """
        parts = []
        while item is not None:
            gid = item.data(0, _GROUP)
            parts.append(gid if gid else item.text(0))
            item = item.parent()
        return tuple(reversed(parts))

    def _slice_roots(self) -> list:
        """Radici in cui cercare le cartelle delle time-slice: tutte le aree, il progetto e la cartella di lavoro."""
        out = []
        for r in [z["base"] for z in self._zones] + [self._project_root(), self._workspace_root()]:
            if r and r not in out:
                out.append(str(r))
        return out

    @staticmethod
    def _project_raster_paths() -> list:
        """Raster del progetto QGIS: le time-slice si riconoscono dal sliceset.json accanto al file."""
        return [lyr.source().split("|")[0] for lyr in QgsProject.instance().mapLayers().values()
                if isinstance(lyr, QgsRasterLayer)]

    def _check_kind(self, kind: str, on: bool) -> None:
        self._updating = True
        try:
            for it in self._iter_items():
                if it.data(0, _ROLE)[0] == kind and not it.isHidden():
                    it.setCheckState(0, Qt.Checked if on else Qt.Unchecked)
        finally:
            self._updating = False
        if kind in self._scrub:
            self._scrub[kind].set_active(False)
        self._schedule()

    def _apply_filter(self, text: str) -> None:
        needle = str(text or "").strip().lower()
        for it in self._iter_items():
            it.setHidden(bool(needle) and needle not in it.text(0).lower())

    def _on_item_changed(self, item, _col) -> None:
        if self._updating:
            return
        data = item.data(0, _ROLE)
        if data and data[0] in self._scrub:          # spuntare a mano vuol dire "voglio questi": il cursore si spegne
            self._scrub[data[0]].set_active(False)
        self._schedule()

    # --- cursori -------------------------------------------------------------------------------------------------
    def _groups(self, kind: str) -> list:
        """Gruppi di una categoria, nell'ordine dell'elenco: ``[(titolo, chiave), ...]``."""
        top = {"gpr": 0, "slice": 2}.get(kind)
        if top is None or self._tree.topLevelItemCount() <= top:
            return []
        item = self._tree.topLevelItem(top)
        return [(item.child(c).text(0), item.child(c).data(0, _GROUP)) for c in range(item.childCount())]

    def _leaves(self, kind: str) -> list:
        """Elementi su cui scorre il cursore: quelli del gruppo scelto nel suo selettore."""
        leaves = [it for it in self._iter_items() if it.data(0, _ROLE)[0] == kind]
        key = self._scrub[kind].scope_key() if kind in self._scrub else None
        if key is None:
            return leaves
        return [it for it in leaves if it.parent() is not None and it.parent().data(0, _GROUP) == key]

    def _selected(self, kind: str) -> list:
        """Elementi da mostrare per una categoria: quello sotto il cursore, se attivo, o i spuntati."""
        sc = self._scrub.get(kind)
        if sc is not None and sc.is_active():
            leaves = self._leaves(kind)
            if leaves:
                it = leaves[min(sc.index(), len(leaves) - 1)]
                return [(it.text(0), it.data(0, _ROLE)[1])]
        return self.checked(kind)

    def _update_scrub(self) -> None:
        for kind, sc in self._scrub.items():
            sc.set_scopes(self._groups(kind))
            leaves = self._leaves(kind)
            sc.set_count(len(leaves))
            if not leaves:
                continue
            i = min(sc.index(), len(leaves) - 1)
            it = leaves[i]
            name = it.text(0)
            if it.parent() is not None and (kind == "slice" or sc.cb_scope.count() > 1):
                name = f"{it.parent().text(0)} · {name}"        # con piu' gruppi si dice in quale si e'
            sc.set_info(f"{name}   ({i + 1}/{len(leaves)})")

    def _on_scrub_changed(self) -> None:
        self._update_scrub()
        self._schedule()

    def checked(self, kind: str) -> list:
        """Payload (profilo GPR, nome ERT o percorso time-slice) degli elementi spuntati di un tipo."""
        out = []
        for it in self._iter_items():
            data = it.data(0, _ROLE)
            if data and data[0] == kind and it.checkState(0) == Qt.Checked:
                out.append((it.text(0), data[1]))
        return out

    def set_checked(self, kind: str, names) -> None:
        """Spunta solo i profili con questi nomi (per uso programmatico e test)."""
        wanted = set(names)
        if kind in self._scrub:
            self._scrub[kind].set_active(False)
        self._updating = True
        try:
            for it in self._iter_items():
                data = it.data(0, _ROLE)
                if data and data[0] == kind:
                    payload = data[1]
                    base = os.path.basename(str(payload)) if kind == "slice" else ""
                    names_of_item = {it.text(0), base}
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
    def _gpr_item(self, ref, ci: int):
        sid, pi = ref
        store = self._stores.get(sid)
        if store is None:
            raise RuntimeError("variante non caricata")
        key = (id(store), pi, ci)
        hit = self._gpr_cache.get(key)
        if hit is not None:
            self._gpr_cache.move_to_end(key)
            return hit
        prof = store.profiles[pi]
        ci = int(min(ci, prof.n_channels - 1))
        ch = store.channel(pi, ci)
        base = normalize_display(ch.data).astype(np.float32, copy=False)
        base = RadargramViewMixin._resample_disp_to_uniform_grid(base, ch.dist)
        xy, d = track_vertices(ch.east, ch.north, ch.dist)
        epsg = int(getattr(prof, "epsg", 0) or 0)
        zone = sid.split(_SEP)[0]
        item = {"name": prof.stem, "src": f"{zone} › {store.name}" if zone else store.name,
                "base": downsample(base), "xy": self._to_common(xy, f"EPSG:{epsg}" if epsg else ""),
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
        errors, label_pts, label_txt = [], [], []

        gprs, erts = [], []
        for text, ref in self._selected("gpr"):
            try:
                gprs.append(self._gpr_item(ref, int(self._spin_ch.value())))
            except Exception as exc:
                errors.append(f"{text}: {exc}")
        # lo stesso nome in due varianti: si distingue con la sorgente, altrimenti un profilo coprirebbe l'altro
        dup = Counter(g["name"] for g in gprs)
        gprs = [dict(g, label=f"{g['name']} ({g['src']})" if dup[g["name"]] > 1 else g["name"]) for g in gprs]
        for text, name in self._selected("ert"):
            try:
                erts.append(self._ert_item(str(name)))
            except Exception as exc:
                errors.append(f"{text}: {exc}")
        slices = []
        for text, path in self._selected("slice"):
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
            self._add_curtain(f"gpr::{it['label']}", it, rgba, 0.0, shown_depth, "gpr")
            label_pts.append((*self._local(it["xy"][0]), 0.0))
            label_txt.append(it["label"])

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
            self._origin = None                         # scena vuota: la prossima parte da zero (origine e camera)
            self._first_camera = True
            pl.add_text("Spunta profili GPR, sezioni ERT o time-slice nell'elenco (☰ Comandi)", name="hint",
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
                        name = f"x::{g['label']}::{e['name']}::{da:.2f}"
                        line = self._pv.Line((x - self._origin[0], y - self._origin[1], 0.0),
                                             (x - self._origin[0], y - self._origin[1], -depth))
                        self._plotter.add_mesh(line, color="red", line_width=4, name=name,
                                               reset_camera=False, render=False)
                        self._shown[name] = "x"
                        lines.append(f"{e['name']} × {g['label']}:  ERT a {db:.1f} m,  GPR a {da:.1f} m")
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
        self._stores.clear()
        if plotter is not None:
            try:
                plotter.close()
            except Exception:
                pass

    def closeEvent(self, event) -> None:
        self.cleanup()
        super().closeEvent(event)
