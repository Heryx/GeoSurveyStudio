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

import json
import time

import numpy as np
from qgis.PyQt.QtCore import QEvent, QSettings, QTimer, Qt, pyqtSignal
from qgis.PyQt.QtGui import QKeySequence
from qgis.PyQt.QtWidgets import (
    QAction,
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QInputDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QScrollArea,
    QShortcut,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStyle,
    QTabWidget,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)
from qgis.core import QgsProject, QgsRasterLayer

from .gpr_ert_georef import lonlat_transformer
from .gpr_ert_import import to_grid as ert_to_grid
from .gpr_ert_import_dialog import import_ert_section
from .gpr_ert_store import list_ert, load_ert
from .gpr_fence_slices import MAX_PLANE_PX, colormap_from_hex, load_slice, project_slice_groups
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
_CACHE_MAX = 24                     # time-slice in memoria (ognuna fino a 2048x2048 px)
_GPR_CACHE_MAX = 64                 # profili GPR gia' ridotti (pochi MB l'uno): "Tutti" non deve rileggere il disco
_MANY_SLICES = 8                    # oltre queste time-slice insieme i piani si riducono a _MANY_SLICES_PX
_MANY_SLICES_PX = 1024
_REFRESH_MIN_S = 60.0               # riaprendo la finestra l'elenco si rilegge solo se e' passato piu' tempo
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


_MODES = (("all", "Tutti"), ("one", "Uno alla volta"))
_MODE_TIPS = {
    "all": "Mostra tutto ciò che è spuntato nell'elenco.",
    "one": "Mostra un solo elemento alla volta, tra quelli spuntati: scorri con la barra, le frecce o il pulsante ▷.",
}
# all'apertura un solo radargramma (non tutto il gruppo); ERT e time-slice: le sezioni insieme, le slice una alla volta
_DEFAULT_MODE = {"gpr": "one", "ert": "all", "slice": "one"}
_CURSOR_MODES = ("one",)
_PLAY_MS = 800

# preset di scena: cosa si vede (modo per categoria) e come (parametri); chi non e' nominato resta com'e'
_BUILTIN_PRESETS = {
    "Panoramica": {"modes": {"gpr": "all", "ert": "all"}, "show": ["gpr", "ert"], "hide": ["slice"], "depth": 0.0,
                   "cross": True},
    "Un profilo e la sua slice": {"modes": {"gpr": "one", "slice": "one"}, "show": ["gpr", "slice"], "hide": ["ert"],
                                  "cross": False},
    "Confronto ERT / GPR": {"modes": {"gpr": "all", "ert": "all"}, "show": ["gpr", "ert"], "hide": ["slice"],
                            "cross": True, "labels": True},
    "Solo time-slice": {"modes": {"slice": "one"}, "show": ["slice"], "hide": ["gpr", "ert"]},
}
_OLD_MODES = {"pick": "all", "upto": "one"}       # modi delle scene salvate con le versioni precedenti
_PRESET_KEY = "GeoSurveyStudio/fence_presets"
_FOCUS_DIM = 0.12                    # opacita' degli elementi non isolati
_PICKABLE = ("gpr::", "ert::", "slice::")


class _Scrubber(QWidget):
    """Una categoria (profili GPR, sezioni ERT, time-slice) con un solo comando: *Tutti* o *Uno alla volta*.

    Cosa c'e' nel gruppo lo decidono le spunte dell'elenco (anche a gruppi, con la casella del gruppo).
    Il cursore ◀ barra ▶ ▷ c'e' sempre (il pulsante ▷ scorre da solo). *Uno alla volta* mostra solo l'elemento
    sotto il cursore. *Tutti* li mostra insieme: col cursore in fondo si vedono tutti; spostandolo indietro
    spariscono gli ultimi, cosi' si costruisce la sequenza (▷ la fa partire da sola).
    """

    changed = pyqtSignal()

    def __init__(self, title: str, tip: str, parent=None):
        super().__init__(parent)
        lay = QGridLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setHorizontalSpacing(4)
        lay.setVerticalSpacing(0)
        self.title = QLabel(title)
        self.title.setToolTip(tip)
        f = self.title.font()
        f.setBold(True)
        self.title.setFont(f)
        self.cb_mode = QComboBox()
        for key, label in _MODES:
            self.cb_mode.addItem(label, key)
            self.cb_mode.setItemData(self.cb_mode.count() - 1, _MODE_TIPS[key], Qt.ToolTipRole)
        self.cb_mode.setToolTip(tip)
        self.cb_mode.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.cb_mode.setMinimumContentsLength(6)
        self.btn_prev = QToolButton()
        self.btn_prev.setText("◀")
        self.btn_prev.setAutoRaise(True)
        self.btn_next = QToolButton()
        self.btn_next.setText("▶")
        self.btn_next.setAutoRaise(True)
        self.btn_play = QToolButton()
        self.btn_play.setText("▷")
        self.btn_play.setCheckable(True)
        self.btn_play.setAutoRaise(True)
        self.btn_play.setToolTip("Scorre da solo, uno dopo l'altro, e ricomincia dal primo.")
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(_PLAY_MS)
        self._play_timer.timeout.connect(self._play_tick)
        self.btn_play.toggled.connect(self._on_play)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.setTracking(True)
        self.slider.setMinimumWidth(60)
        self.lbl = QLabel("")
        self.lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)   # il testo lungo si taglia, non allarga la finestra
        lay.addWidget(self.title, 0, 0, 1, 2)
        lay.addWidget(self.cb_mode, 0, 2, 1, 4)
        lay.addWidget(self.btn_prev, 1, 0)
        lay.addWidget(self.slider, 1, 1, 1, 2)
        lay.addWidget(self.btn_next, 1, 3)
        lay.addWidget(self.btn_play, 1, 4)
        lay.addWidget(self.lbl, 1, 5)
        lay.setColumnStretch(2, 1)
        lay.setColumnStretch(5, 2)
        self.cb_mode.currentIndexChanged.connect(self._on_mode)
        self.slider.valueChanged.connect(lambda _v: self.changed.emit())
        self.btn_prev.clicked.connect(lambda: self.slider.setValue(self.slider.value() - 1))
        self.btn_next.clicked.connect(lambda: self.slider.setValue(self.slider.value() + 1))
        self.set_count(0)

    # --- modo -------------------------------------------------------------------------------------------
    def mode(self) -> str:
        return str(self.cb_mode.currentData())

    def set_mode(self, mode: str, emit: bool = True) -> None:
        mode = _OLD_MODES.get(mode, mode)
        i = self.cb_mode.findData(mode)
        if i < 0 or i == self.cb_mode.currentIndex():
            return
        self.cb_mode.blockSignals(True)
        self.cb_mode.setCurrentIndex(i)
        self.cb_mode.blockSignals(False)
        if mode == "all":
            self.to_end()
        if emit:
            self.changed.emit()

    def _on_mode(self, _i: int) -> None:
        if self.mode() == "all":
            self.to_end()                         # "Tutti": si parte da tutti visibili
        self.changed.emit()

    def to_end(self) -> None:
        """Cursore in fondo (senza emettere segnali): in "Tutti" significa vederli tutti."""
        self.slider.blockSignals(True)
        self.slider.setValue(self.slider.maximum())
        self.slider.blockSignals(False)

    # --- riproduzione automatica ------------------------------------------------------------------------
    def is_playing(self) -> bool:
        return self.btn_play.isChecked()

    def _on_play(self, on: bool) -> None:
        if on and self.count() > 1:
            self._play_timer.start()
        else:
            self._play_timer.stop()
            if on:
                self.btn_play.blockSignals(True)
                self.btn_play.setChecked(False)
                self.btn_play.blockSignals(False)

    def _play_tick(self) -> None:
        n = self.count()
        if n < 2:
            self.btn_play.setChecked(False)
            return
        self.slider.setValue(0 if self.slider.value() >= n - 1 else self.slider.value() + 1)

    # --- cursore -----------------------------------------------------------------------------------------
    def set_count(self, n: int, total: int = None) -> None:
        """``n`` = elementi spuntati (il cursore scorre su questi); ``total`` = tutti gli elementi della categoria."""
        total = n if total is None else total
        was_end = self.slider.value() >= self.slider.maximum()
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, n - 1))
        self.slider.setValue(min(self.slider.value(), max(0, n - 1)))
        if self.mode() == "all" and was_end:
            self.slider.setValue(self.slider.maximum())           # "Tutti" resta "tutti" quando le spunte cambiano
        self.slider.blockSignals(False)
        for w in (self.btn_prev, self.slider, self.btn_next):
            w.setEnabled(n > 0)
        self.btn_play.setEnabled(n > 1)
        if n < 2 and self.btn_play.isChecked():
            self.btn_play.setChecked(False)
        self.cb_mode.setEnabled(total > 0)
        if total == 0:
            self.lbl.setText("nessun elemento")

    def count(self) -> int:
        return self.slider.maximum() + 1 if self.slider.isEnabled() else 0

    def index(self) -> int:
        return self.slider.value()

    def is_active(self) -> bool:
        """Modo "Uno alla volta" con qualcosa su cui scorrere."""
        return self.mode() == "one" and self.count() > 0

    def set_active(self, on: bool) -> None:
        """Compatibilita': ``True`` = "Uno alla volta", ``False`` = "Tutti". Non emette segnali."""
        self.set_mode("one" if on else "all", emit=False)

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
        self._tick_memory: dict = {}      # voce (percorso nell'albero) -> spuntata?  Le voci nuove partono spuntate
        self._side_user = False           # l'utente ha scelto a mano: niente piu' nascondi/mostra automatico
        self._hint_on = False
        self._last_scene: dict = {}
        self._sig: dict = {}              # nome attore -> firma di come e' stato costruito (per non rifarlo)
        self._last_refresh = None
        self._focus = None                # nome dell'attore isolato con il doppio clic (gli altri si attenuano)
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
            self.topLevelChanged.connect(self._on_top_level_changed)
            sh = QShortcut(QKeySequence("F11"), self)
            sh.setContext(Qt.WidgetWithChildrenShortcut)
            sh.activated.connect(self.toggle_fullscreen)
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
        tabs = QTabWidget()
        self._tabs = tabs
        tabs.setUsesScrollButtons(False)
        sl.addWidget(tabs)

        # --- scheda "Dati": cosa caricare (1) e cosa mostrare (2) --------------------------------------------
        data = QWidget()
        dl = QVBoxLayout(data)
        dl.setContentsMargins(4, 6, 4, 4)
        dl.setSpacing(6)

        box_src = QGroupBox("1 · Zone e processi GPR")
        bl = QVBoxLayout(box_src)
        bl.addWidget(self._hint("Spunta i processi di cui vuoi i profili; puoi sceglierne più d'uno."))
        self._src_tree = QTreeWidget()
        self._src_tree.setHeaderHidden(True)
        self._src_tree.setMinimumHeight(60)
        self._src_tree.setMaximumHeight(150)
        self._src_tree.setToolTip("Varianti processate (zona › processo): ogni variante spuntata porta i suoi "
                                  "profili nell'elenco qui sotto.")
        self._src_tree.itemChanged.connect(self._on_source_changed)
        bl.addWidget(self._src_tree)
        dl.addWidget(box_src)

        box_items = QGroupBox("2 · Cosa mostrare")
        il = QVBoxLayout(box_items)
        il.addWidget(self._hint("Scorri con i cursori sotto la vista, oppure spunta a mano le voci."))
        frow = QHBoxLayout()
        self._ed_filter = QLineEdit()
        self._ed_filter.setPlaceholderText("Filtra…")
        self._ed_filter.setClearButtonEnabled(True)
        self._ed_filter.setMinimumWidth(40)
        self._ed_filter.textChanged.connect(self._apply_filter)
        frow.addWidget(self._ed_filter, 1)
        self._btn_pick = QToolButton()
        self._btn_pick.setText("Spunta ▾")
        self._btn_pick.setPopupMode(QToolButton.InstantPopup)
        self._btn_pick.setToolTip("Spunta o toglie la spunta a più voci insieme (solo quelle visibili nel filtro).")
        menu = QMenu(self._btn_pick)
        menu.addAction("Tutti i profili GPR", lambda: self._check_kind("gpr", True))
        menu.addAction("Nessun profilo GPR", lambda: self._check_kind("gpr", False))
        menu.addSeparator()
        menu.addAction("Togli tutte le spunte", lambda: [self._check_kind(k, False) for k in ("gpr", "ert", "slice")])
        self._btn_pick.setMenu(menu)
        frow.addWidget(self._btn_pick)
        b_ert = QToolButton()
        b_ert.setText("＋ ERT")
        b_ert.setToolTip("Importa una sezione ERT invertita (CSV di ResIPY) e georiferiscila.")
        b_ert.clicked.connect(self._import_ert)
        frow.addWidget(b_ert)
        il.addLayout(frow)
        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setToolTip("Doppio clic su una voce: la isola nella vista 3D.")
        self._tree.itemChanged.connect(self._on_item_changed)
        self._tree.itemDoubleClicked.connect(self._on_tree_double_click)
        il.addWidget(self._tree, 1)
        dl.addWidget(box_items, 1)
        tabs.addTab(data, "Dati")

        # --- scheda "Aspetto" -------------------------------------------------------------------------------
        self._spin_ch = QSpinBox()
        self._spin_ch.setRange(0, 63)
        self._spin_ch.setToolTip("Canale del profilo GPR mostrato (uno per profilo).")
        self._spin_ch.valueChanged.connect(self._schedule)
        self._spin_gain = QDoubleSpinBox()
        self._spin_gain.setRange(0.2, 30.0)
        self._spin_gain.setSingleStep(0.5)
        self._spin_gain.setValue(1.7)
        self._spin_gain.valueChanged.connect(self._schedule)
        self._cb_gpr = QComboBox()
        self._cb_gpr.addItems(_GPR_CMAPS)
        self._cb_gpr.currentIndexChanged.connect(self._schedule)
        self._cb_ert = QComboBox()
        self._cb_ert.addItems(_ERT_CMAPS)
        self._cb_ert.currentIndexChanged.connect(self._schedule)
        self._cb_slice = QComboBox()
        self._cb_slice.addItem("Stile del file", "")
        for name in _SLICE_CMAPS:
            self._cb_slice.addItem(name, name)
        self._cb_slice.setToolTip("Palette delle time-slice: lo stile salvato nel raster oppure una palette a scelta.")
        self._cb_slice.currentIndexChanged.connect(self._schedule)
        self._spin_slice_op = QDoubleSpinBox()
        self._spin_slice_op.setRange(0.1, 1.0)
        self._spin_slice_op.setSingleStep(0.05)
        self._spin_slice_op.setValue(0.85)
        self._spin_slice_op.setToolTip("Opacità dei piani delle time-slice: più bassa, meglio si vedono i profili sotto.")
        self._spin_slice_op.valueChanged.connect(self._schedule)
        self._spin_depth = QDoubleSpinBox()
        self._spin_depth.setRange(0.0, 100.0)
        self._spin_depth.setSpecialValueText("tutta")
        self._spin_depth.setSuffix(" m")
        self._spin_depth.setSingleStep(0.5)
        self._spin_depth.setValue(0.0)
        self._spin_depth.setToolTip("Profondità massima mostrata, uguale per GPR ed ERT (0 = tutta).")
        self._spin_depth.valueChanged.connect(self._schedule)
        self._spin_ve = QDoubleSpinBox()
        self._spin_ve.setRange(0.5, 20.0)
        self._spin_ve.setSingleStep(0.5)
        self._spin_ve.setValue(2.0)
        self._spin_ve.setToolTip("Esagerazione verticale: le profondità sono piccole rispetto alla lunghezza dei profili.")
        self._spin_ve.valueChanged.connect(self._apply_scale)
        self._chk_labels = QCheckBox("Legenda dei nomi (a lato)")
        self._chk_labels.setChecked(True)
        self._chk_labels.toggled.connect(self._schedule)

        look = QWidget()
        ll = QVBoxLayout(look)
        ll.setContentsMargins(4, 6, 4, 4)
        ll.setSpacing(6)
        ll.addWidget(self._form_box("Radargrammi GPR", [("Canale", self._spin_ch), ("Gain", self._spin_gain),
                                                        ("Palette", self._cb_gpr)]))
        ll.addWidget(self._form_box("Sezioni ERT", [("Palette", self._cb_ert)]))
        ll.addWidget(self._form_box("Time-slice", [("Palette", self._cb_slice), ("Opacità", self._spin_slice_op)]))
        ll.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)       # i campi si stringono, non scorrono di lato
        for cb in (self._cb_gpr, self._cb_ert, self._cb_slice):
            cb.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            cb.setMinimumContentsLength(6)
        scroll.setWidget(look)
        tabs.addTab(scroll, "Aspetto")

        # --- scheda "Incroci" (e messaggi) ---------------------------------------------------------------------
        cross = QWidget()
        cl = QVBoxLayout(cross)
        cl.setContentsMargins(4, 6, 4, 4)
        self._chk_cross = QCheckBox("Segna gli incroci GPR × ERT")
        self._chk_cross.setChecked(True)
        self._chk_cross.setToolTip("Marca nella scena i punti in cui un profilo GPR attraversa una sezione ERT.")
        self._chk_cross.toggled.connect(self._schedule)
        cl.addWidget(self._chk_cross)
        self._txt_cross = QPlainTextEdit()
        self._txt_cross.setReadOnly(True)
        self._txt_cross.setPlaceholderText("Gli incroci tra profili GPR ed ERT mostrati compaiono qui, "
                                           "con la distanza lungo ciascuno.")
        # un avviso (importazione fallita, archivio illeggibile...) non deve restare in una scheda nascosta
        self._txt_cross.textChanged.connect(
            lambda: tabs.setCurrentWidget(cross) if self._txt_cross.toPlainText().startswith("⚠") else None)
        cl.addWidget(self._txt_cross, 1)
        tabs.addTab(cross, "Incroci")
        split.addWidget(side)

        # --- vista 3D e cursori -------------------------------------------------------------------------------
        view = QWidget()
        vl = QVBoxLayout(view)
        vl.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        bar.setSpacing(2)
        self._btn_side = QToolButton()
        self._btn_side.setText("☰")
        self._btn_side.setCheckable(True)
        self._btn_side.setChecked(True)
        self._btn_side.setToolTip("Comandi: mostra o nasconde l'elenco dei profili e le impostazioni, per lasciare tutto lo spazio alla vista 3D.")
        self._btn_side.toggled.connect(lambda on: self._set_side_visible(on, user=True))
        bar.addWidget(self._btn_side)
        bar.addStretch(1)
        b_view = QToolButton()
        b_view.setText("Vista ▾")
        b_view.setPopupMode(QToolButton.InstantPopup)
        b_view.setToolTip("Punto di vista, profondità massima, esagerazione verticale e nomi dei profili.")
        vmenu = QMenu(b_view)
        vmenu.addAction("Isometrica (reimposta)", self._view_iso)
        vmenu.addAction("Dall'alto", self._view_top)
        vmenu.addAction("Mostra tutto (toglie l'isolamento)", lambda: self.focus_actor(None))
        st = self.style()
        self._act_img = QAction(st.standardIcon(QStyle.SP_DialogSaveButton), "Esporta immagine (PNG o JPG)…", self)
        self._act_img.setToolTip("Salva la vista 3D come immagine PNG o JPG.")
        self._act_img.triggered.connect(lambda _c=False: self._ask_export_image())
        self._act_kml = QAction(st.standardIcon(QStyle.SP_DriveNetIcon), "Esporta per Google Earth (KML o KMZ)…", self)
        self._act_kml.setToolTip("Esporta tracce e time-slice mostrate in KML o KMZ (Google Earth, QGIS).")
        self._act_kml.triggered.connect(lambda _c=False: self._ask_export_kml())
        self._act_full = QAction(st.standardIcon(QStyle.SP_TitleBarMaxButton), "Schermo intero (F11)", self)
        self._act_full.setCheckable(True)
        self._act_full.setToolTip("Schermo intero (F11): la finestra si stacca e occupa tutto lo schermo. "
                                  "Di nuovo per tornare com'era.")
        self._act_full.toggled.connect(self._set_fullscreen)
        self._act_close = QAction(st.standardIcon(QStyle.SP_TitleBarCloseButton), "Chiudi", self)
        self._act_close.setToolTip("Chiude la vista 3D.")
        self._act_close.triggered.connect(lambda _c=False: self._close_view())
        vmenu.addAction(self._act_full)
        emenu = vmenu.addMenu("Esporta")
        emenu.addAction(self._act_img)
        emenu.addAction(self._act_kml)
        vmenu.addSeparator()
        self._preset_menu = vmenu.addMenu("Scene salvate")
        self._preset_menu.setToolTip("Un clic imposta insieme modi, profondità, esagerazione e nomi.")
        self._preset_menu.aboutToShow.connect(self._fill_preset_menu)
        self._fill_preset_menu()
        vmenu.addSeparator()
        pop = QWidget()                                   # parametri della scena nello stesso menu: niente pulsante in piu'
        pform = QFormLayout(pop)
        pform.setContentsMargins(8, 4, 8, 4)
        pform.addRow("Profondità max", self._spin_depth)
        pform.addRow("Esagerazione Z", self._spin_ve)
        pform.addRow(self._chk_labels)
        act = QWidgetAction(vmenu)
        act.setDefaultWidget(pop)
        vmenu.addAction(act)
        b_view.setMenu(vmenu)
        for act in (self._act_img, self._act_kml):          # esportazione sempre a portata di mano
            bar.addWidget(self._icon_button(act))
        bar.addWidget(b_view)
        b_refresh = QToolButton()
        b_refresh.setIcon(st.standardIcon(QStyle.SP_BrowserReload))
        b_refresh.setToolTip("Rilegge zone, varianti, profili GPR, sezioni ERT e time-slice.")
        b_refresh.setAutoRaise(True)
        b_refresh.setFixedSize(24, 24)
        b_refresh.clicked.connect(self.refresh_variants)
        bar.addWidget(b_refresh)
        for act in (self._act_full, self._act_close):
            bar.addWidget(self._icon_button(act))
        vl.addLayout(bar)
        if self._plotter_factory is not None:
            self._plotter, widget = self._plotter_factory(view)
        else:
            self._plotter = qt(view)
            widget = self._plotter.interactor if hasattr(self._plotter, "interactor") else self._plotter
        widget.setMinimumSize(120, 200)
        widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        vl.addWidget(widget, 1)
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        vl.addWidget(line)
        for kind, title, tip in (
                ("gpr", "Profili GPR", "Profili GPR spuntati: tutti insieme (col cursore in fondo) oppure uno alla volta."),
                ("ert", "Sezioni ERT", "Sezioni ERT spuntate: tutte insieme oppure una alla volta."),
                ("slice", "Time-slice", "Time-slice spuntate: tutte insieme oppure una alla volta (scorri con la barra o col pulsante ▷).")):
            sc = _Scrubber(title, tip)
            sc.set_mode(_DEFAULT_MODE[kind], emit=False)
            sc.changed.connect(self._on_scrub_changed)
            self._scrub[kind] = sc
            sc.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)      # mai piu' alti del necessario
            vl.addWidget(sc, 0)
        self._plotter.set_background("white")
        self._plotter.add_axes()
        # doppio clic sul profilo o sulla slice nella vista 3D: lo isola; nel vuoto: mostra di nuovo tutto
        # (filtro d'eventi Qt: piu' affidabile dei doppi clic interni di VTK)
        self._view_widget = widget
        widget.installEventFilter(self)
        widget.setToolTip("Doppio clic su un profilo o una time-slice per isolarlo; doppio clic nel vuoto per mostrare tutto.")
        split.addWidget(view)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([280, 700])
        self._split = split

    @staticmethod
    def _hint(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)     # va a capo: non impone larghezza alla colonna
        lbl.setStyleSheet("color: gray;")
        return lbl

    @staticmethod
    def _form_box(title: str, rows, extra=None) -> QGroupBox:
        box = QGroupBox(title)
        form = QFormLayout(box)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        for label, widget in rows:
            form.addRow(label, widget)
        if extra is not None:
            form.addRow(extra)
        return box

    @staticmethod
    def _icon_button(action) -> QToolButton:
        """Pulsante a sola icona (larghezza fissa) legato a un'azione: stessa azione nel menu e nella barra."""
        b = QToolButton()
        b.setDefaultAction(action)
        b.setToolButtonStyle(Qt.ToolButtonIconOnly)
        b.setAutoRaise(True)
        b.setFixedSize(24, 24)
        return b

    def _close_view(self) -> None:
        if self._act_full.isChecked():
            self._act_full.setChecked(False)
        self.close()

    def _set_fullscreen(self, on: bool) -> None:
        """Stacca la finestra e la porta a tutto schermo; spenta, torna dov'era (agganciata o fluttuante)."""
        if on:
            self._was_floating = self.isFloating()
            if not self._was_floating:
                self.setFloating(True)
            self._act_full.blockSignals(True)           # setFloating emette topLevelChanged: non va letto come uscita
            self._act_full.setChecked(True)
            self._act_full.blockSignals(False)
            self.showMaximized()
        else:
            self.showNormal()
            if not getattr(self, "_was_floating", True):
                self.setFloating(False)

    def _on_top_level_changed(self, floating: bool) -> None:
        if not floating and self._act_full.isChecked():       # riagganciata a mano: lo schermo intero finisce
            self._act_full.blockSignals(True)
            self._act_full.setChecked(False)
            self._act_full.blockSignals(False)

    def toggle_fullscreen(self) -> None:
        if self._plotter is not None:
            self._act_full.setChecked(not self._act_full.isChecked())

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
        self._last_refresh = time.monotonic()
        self._zones = self._zone_list()
        prev = {it.data(0, _GROUP) for it in self._src_leaves() if it.checkState(0) == Qt.Checked}
        self._updating = True
        try:
            self._src_tree.clear()
            single = len(self._zones) == 1 and not self._zones[0]["name"]
            found = {id(z): discover_variants(z["base"]) for z in self._zones}
            with_data = [z for z in self._zones if found[id(z)]]
            # la prima volta si spunta la variante di lavoro dell'area attiva; se nessuna area e' attiva (o quella
            # attiva non ha dati) si parte dalla prima area che ne ha: la scena non deve aprirsi vuota
            seed_zone = next((z for z in with_data if z["active"]), with_data[0] if with_data else None)
            for z in self._zones:
                refs = found[id(z)]
                if not refs:
                    continue
                parent = self._src_tree.invisibleRootItem()
                if not single:
                    parent = QTreeWidgetItem(self._src_tree, [z["name"] or "-"])
                    parent.setFlags(Qt.ItemIsEnabled)
                default = next((r for r in refs if r.name == "Lavoro"), refs[0])
                for r in refs:
                    it = QTreeWidgetItem(parent, [f"{r.name}  ({r.n_files})"])
                    it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                    sid = self._source_id(z["name"], r.path)
                    it.setData(0, _GROUP, sid)
                    it.setData(0, _ROLE, {"zone": z["name"], "path": r.path, "name": r.name, "id": sid})
                    # la prima volta si spunta la variante di lavoro dell'area attiva
                    seed = (not self._sources_seeded) and z is seed_zone and r is default
                    it.setCheckState(0, Qt.Checked if (sid in prev or seed) else Qt.Unchecked)
                if not single:
                    parent.setExpanded(True)      # dopo i figli, altrimenti l'espansione si perde
            self._sources_seeded = self._sources_seeded or bool(with_data)
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
        now = time.monotonic()
        if self._last_refresh is None or now - self._last_refresh > _REFRESH_MIN_S:
            self.refresh_variants()                   # varianti e sezioni ERT possono essere cambiate nel frattempo

    # --- elenco dei profili -------------------------------------------------------------------------------
    def refresh_lists(self) -> None:
        """Ricostruisce l'elenco di profili GPR (uno per sorgente spuntata), sezioni ERT e time-slice."""
        if self._plotter is None:
            return
        self._ert_cache.clear()
        for it in self._iter_items():
            self._tick_memory[self._path_key(it)] = it.checkState(0) == Qt.Checked
        ticked = lambda key: self._tick_memory.get(key, True)         # una voce mai vista parte spuntata
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
                top.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsAutoTristate)
                top.setToolTip(0, "Spunta o togli la spunta a tutta la categoria.")
                top.setExpanded(expanded.get((top.text(0),), True))
            for src in sources:
                try:
                    store = self._store_for(src)
                except Exception as exc:
                    notes.append(f"{src['name']}: {exc}")
                    continue
                n = len(store.profiles)
                gi = QTreeWidgetItem(gpr, [self._source_title(src, n)])
                gi.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsAutoTristate)
                gi.setToolTip(0, "Spunta il gruppo per includere tutti i suoi profili.")
                gi.setData(0, _GROUP, src["id"])
                key = (gpr.text(0), src["id"])
                for pi, p in enumerate(store.profiles):
                    self._add_child(gi, p.stem, ("gpr", (src["id"], pi)), ticked(key + (p.stem,)))
                # tanti profili coprirebbero ERT e time-slice sotto la piega dell'elenco: il gruppo parte chiuso,
                # a meno che l'utente l'abbia gia' aperto
                gi.setExpanded(expanded.get(key, n <= _GPR_EXPAND_MAX))
            for name in list_ert(self._project_root()):
                self._add_child(ert, name, ("ert", name), ticked((ert.text(0), name)))
            variants = {s["name"] for s in sources}
            for g in project_slice_groups(self._slice_roots(), self._project_raster_paths()):
                zone = self._zone_label_of_path(g["dir"])
                head = f"{zone} › {g['name']}" if zone else g["name"]
                title = f"{head}  ({g['variant']})" if g["variant"] else head
                gi = QTreeWidgetItem(slc, [title])
                # flag sul gruppo: spunta o toglie in un colpo tutte le sue slice (parziale se solo alcune)
                gi.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsAutoTristate)
                gi.setToolTip(0, "Spunta il gruppo per includere tutte le sue time-slice.")
                gi.setData(0, _GROUP, "slice" + _SEP + os.path.normcase(os.path.normpath(g["dir"])))
                key = (slc.text(0), gi.data(0, _GROUP))
                gi.setExpanded(expanded.get(key, g["variant"] in variants))
                for row in g["slices"]:
                    label = f"{row['z_from']:.2f}–{row['z_to']:.2f} m"
                    self._slice_info[row["path"]] = dict(row, epsg=g["epsg"], group=g["name"])
                    self._add_child(gi, label, ("slice", row["path"]), ticked(key + (label,)))
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
        """Spunta o toglie la spunta a tutte le voci visibili (filtro compreso) di una categoria."""
        self._set_ticks(kind, on, visible_only=True)

    def _set_ticks(self, kind: str, on: bool, visible_only: bool = False) -> None:
        self._updating = True
        try:
            for it in self._iter_items():
                if it.data(0, _ROLE)[0] == kind and not (visible_only and it.isHidden()):
                    it.setCheckState(0, Qt.Checked if on else Qt.Unchecked)
        finally:
            self._updating = False
        self._update_scrub()
        self._schedule()

    def _apply_filter(self, text: str) -> None:
        """Il filtro serve a trovare le voci nell'elenco: non cambia cosa e' in scena."""
        needle = str(text or "").strip().lower()
        for it in self._iter_items():
            it.setHidden(bool(needle) and needle not in it.text(0).lower())

    def _on_item_changed(self, item, _col) -> None:
        if self._updating:
            return
        self._update_scrub()
        self._schedule()

    # --- cursori -------------------------------------------------------------------------------------------------
    def _leaves(self, kind: str) -> list:
        """Voci spuntate di una categoria, nell'ordine dell'elenco: e' su queste che scorre il cursore."""
        return [it for it in self._iter_items()
                if it.data(0, _ROLE)[0] == kind and it.checkState(0) == Qt.Checked]

    def _selected(self, kind: str) -> list:
        """Elementi da mostrare per una categoria: tutti quelli spuntati, o uno solo col cursore."""
        sc = self._scrub.get(kind)
        leaves = self._leaves(kind)
        if sc is not None and leaves:
            i = min(sc.index(), len(leaves) - 1)
            leaves = [leaves[i]] if sc.mode() == "one" else leaves[:i + 1]     # Tutti: dal primo fino al cursore
        return [(it.text(0), it.data(0, _ROLE)[1]) for it in leaves]

    def _update_scrub(self) -> None:
        for kind, sc in self._scrub.items():
            leaves = self._leaves(kind)
            total = sum(1 for it in self._iter_items() if it.data(0, _ROLE)[0] == kind)
            sc.set_count(len(leaves), total)
            if not total:
                continue
            if not leaves:
                sc.set_info("nessuna spuntata")
                continue
            i = min(sc.index(), len(leaves) - 1)
            if sc.mode() == "all" and i == len(leaves) - 1:
                sc.set_info(f"tutte ({len(leaves)})")
                continue
            it = leaves[i]
            name = it.text(0)
            groups = {id(x.parent()) for x in leaves}
            if it.parent() is not None and (kind == "slice" or len(groups) > 1):
                name = f"{it.parent().text(0)} · {name}"        # con piu' gruppi si dice in quale si e'
            sc.set_info(f"{name}   ({i + 1}/{len(leaves)})" + ("  – dal primo" if sc.mode() == "all" else ""))

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
            self._scrub[kind].set_mode("all", emit=False)       # scelta esplicita: si vedono tutte quelle indicate
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
        self._update_scrub()
        if kind in self._scrub:
            self._scrub[kind].to_end()                          # scelta esplicita: si vedono tutte quelle indicate
            self._update_scrub()
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
        while len(self._gpr_cache) > _GPR_CACHE_MAX:
            self._gpr_cache.popitem(last=False)
        return item

    def _slice_item(self, path: str, max_px: int = None):
        px = int(max_px or MAX_PLANE_PX)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0.0
        hit = self._slice_cache.get(path)
        if hit is None or hit.get("_mtime") != mtime or hit.get("_px") != px:     # rigenerato o altra risoluzione
            hit = dict(load_slice(path, px), _mtime=mtime, _px=px)
            self._slice_cache[path] = hit
            while len(self._slice_cache) > _CACHE_MAX:
                self._slice_cache.pop(next(iter(self._slice_cache)))
        info = self._slice_info.get(path, {})
        epsg = hit["epsg"] or int(info.get("epsg") or 0)
        crs = self._common_crs()
        cached = hit.get("_common")
        if cached is None or cached[0] != (epsg, crs):                            # il trasformatore e' lento: una volta sola
            corners = self._to_common(np.asarray(hit["corners"], dtype=np.float64), f"EPSG:{epsg}" if epsg else "")
            cached = ((epsg, crs), np.asarray(corners))
            hit["_common"] = cached
        return dict(hit, corners=cached[1], path=path, name=os.path.basename(path),
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

    def _slice_rgba(self, it: dict):
        """Texture di una time-slice con la palette scelta (stile del file o palette a scelta)."""
        style = str(self._cb_slice.currentData() or "")
        cmap = (colormap_from_hex(it["colors"]) if not style else None) or (style or "viridis")
        return rgba_from_array(it["data"], cmap, it["vmin"], it["vmax"])

    def _rebuild_scene(self) -> None:
        """Aggiorna la scena rifacendo solo cio' che e' cambiato: un attore con la stessa firma resta dov'e'.

        Spostare il cursore o cambiare una spunta non ricostruisce piu' le texture di tutti i profili e di tutte
        le slice gia' in scena (colori, mesh e caricamento sulla GPU), ma solo degli elementi nuovi.
        """
        pl = self._plotter
        old = {n: k for n, k in self._shown.items() if k != "x"}         # profili, sezioni e slice gia' in scena
        for name, kind in self._shown.items():
            if kind == "x":                                              # legenda, incroci, isolamento: si rifanno
                pl.remove_actor(name, render=False)
        self._shown = {}
        if self._hint_on:
            pl.remove_actor("hint", render=False)
            self._hint_on = False
        dmax = float(self._spin_depth.value())
        errors = []

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
        wanted_slices = self._selected("slice")
        px = MAX_PLANE_PX if len(wanted_slices) <= _MANY_SLICES else _MANY_SLICES_PX     # molte insieme: piani piu' leggeri
        slices = []
        for text, path in wanted_slices:
            try:
                slices.append(self._slice_item(str(path), px))
            except Exception as exc:
                errors.append(f"{text}: {exc}")
        for it in gprs + erts:
            self._set_origin_from(it["xy"])
        for it in slices:
            self._set_origin_from(it["corners"])

        def place(name, kind, sig, add):
            """Tiene l'attore se e' gia' in scena con la stessa firma, altrimenti lo (ri)costruisce."""
            if self._sig.get(name) == sig and old.pop(name, None) is not None:
                self._shown[name] = kind
                return
            old.pop(name, None)
            add()
            self._shown[name] = kind
            self._sig[name] = sig

        gain = float(self._spin_gain.value())
        gpr_cmap = self._cb_gpr.currentText()
        for it in gprs:
            base, depth = it["base"], it["depth"]
            rows = base.shape[0] if dmax <= 0 else max(2, int(np.ceil(base.shape[0] * min(1.0, dmax / depth))))
            shown_depth = depth * rows / base.shape[0]

            def add_gpr(it=it, base=base, rows=rows, shown_depth=shown_depth):
                rgba = rgba_from_array(np.clip(base[:rows] * gain, -1.0, 1.0), gpr_cmap, -1.0, 1.0)
                self._add_curtain(f"gpr::{it['label']}", it, rgba, 0.0, shown_depth, "gpr")
            place(f"gpr::{it['label']}", "gpr", ("gpr", id(base), rows, gain, gpr_cmap, self._origin), add_gpr)

        if erts:
            vals = np.concatenate([e["grid"].data[np.isfinite(e["grid"].data)] for e in erts])
            lo, hi = np.percentile(vals, _ERT_CLIM_PERCENTILES) if vals.size else (0.0, 1.0)
            ert_cmap = self._cb_ert.currentText()
            for it in erts:
                g = it["grid"]
                depth = g.depth_axis
                keep = np.ones(len(depth), bool) if dmax <= 0 else depth <= dmax
                if keep.sum() < 2:                      # limite sotto il passo di griglia: restano le prime 2 righe
                    keep[:2] = True

                def add_ert(it=it, g=g, keep=keep, depth=depth):
                    rgba = rgba_from_array(g.data[keep], ert_cmap, float(lo), float(hi))
                    self._add_curtain(f"ert::{it['name']}", it, rgba, float(depth[keep][0]), float(depth[keep][-1]), "ert")
                place(f"ert::{it['name']}", "ert",
                      ("ert", id(g), int(keep.sum()), ert_cmap, float(lo), float(hi), self._origin), add_ert)

        style = str(self._cb_slice.currentData() or "")
        for it in slices:
            def add_slice(it=it):
                rgba = self._slice_rgba(it)
                pl_geo = plane(it["corners"], it["z_center"], origin=self._origin)
                mesh = self._pv.PolyData(pl_geo["points"], pl_geo["faces"])
                mesh.active_texture_coordinates = pl_geo["tcoords"]
                pl.add_mesh(mesh, texture=self._pv.Texture(rgba), lighting=False, name=f"slice::{it['name']}",
                            reset_camera=False, show_scalar_bar=False, opacity=float(self._spin_slice_op.value()),
                            render=False)
            place(f"slice::{it['name']}", "slice",
                  ("slice", id(it["data"]), style, float(it["z_center"]), self._origin), add_slice)

        for name in old:                                # in scena prima, non piu' richiesti: via
            pl.remove_actor(name, render=False)
            self._sig.pop(name, None)

        if self._chk_labels.isChecked() and (gprs or erts or slices):
            pl.add_text(self._legend_text(gprs, erts, slices), name="legend", position="upper_right",
                        font_size=9, color="black", render=False)          # a lato della scena, mai dentro il modello
            self._shown["legend"] = "x"
        self._last_scene = {"gprs": gprs, "erts": erts, "slices": slices}
        self._draw_crossings(gprs, erts, dmax)
        if errors:
            self._txt_cross.appendPlainText("\n".join(f"⚠ {e}" for e in errors))
        if not self._shown:
            self._origin = None                         # scena vuota: la prossima parte da zero (origine e camera)
            self._first_camera = True
            self._sig.clear()
            pl.add_text("Niente da mostrare: spunta le voci nell'elenco (pulsante ☰) e scegli come vederle "
                        "sotto la vista (Tutti, Uno alla volta)", name="hint",
                        position="upper_left", font_size=10, color="gray", render=False)
            self._hint_on = True
        self._apply_focus(render=False)
        self._apply_scale()
        if self._shown and self._first_camera:
            self._view_iso()
            self._first_camera = False
        pl.render()

    @staticmethod
    def _legend_text(gprs, erts, slices) -> str:
        """Elenco dei nomi mostrati, per la legenda a lato (corto: oltre il limite si riassume)."""
        lines = []

        def block(title, names, limit):
            if not names:
                return
            lines.append(f"{title} ({len(names)})")
            lines.extend("  " + n for n in names[:limit])
            if len(names) > limit:
                lines.append(f"  … e altri {len(names) - limit}")

        block("Profili GPR", [g["label"] for g in gprs], 8)
        block("Sezioni ERT", [e["name"] for e in erts], 6)
        if slices:
            ranges = [f"{x['z_from']:.2f}–{x['z_to']:.2f} m" for x in slices]
            if len(ranges) > 6:
                lines.append(f"Time-slice ({len(ranges)})")
                lines.append(f"  da {slices[0]['z_from']:.2f} a {slices[-1]['z_to']:.2f} m")
            else:
                block("Time-slice", ranges, 6)
        return "\n".join(lines)

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

    # --- preset di scena ------------------------------------------------------------------------------------
    def scene_state(self) -> dict:
        """Cosa si vede e come, in forma salvabile."""
        total = {k: sum(1 for it in self._iter_items() if it.data(0, _ROLE)[0] == k) for k in self._scrub}
        shown = {k: len(self._leaves(k)) for k in self._scrub}
        return {"modes": {k: sc.mode() for k, sc in self._scrub.items()},
                "show": [k for k in self._scrub if total[k] and shown[k] == total[k]],
                "hide": [k for k in self._scrub if total[k] and not shown[k]],
                "depth": float(self._spin_depth.value()), "ve": float(self._spin_ve.value()),
                "labels": bool(self._chk_labels.isChecked()), "cross": bool(self._chk_cross.isChecked()),
                "slice_op": float(self._spin_slice_op.value())}

    def apply_scene(self, state: dict) -> None:
        """Applica un preset: le chiavi assenti non toccano nulla."""
        if self._plotter is None or not state:
            return
        self._updating = True
        try:
            for spin, key in ((self._spin_depth, "depth"), (self._spin_ve, "ve"), (self._spin_slice_op, "slice_op")):
                if key in state:
                    spin.setValue(float(state[key]))
            for chk, key in ((self._chk_labels, "labels"), (self._chk_cross, "cross")):
                if key in state:
                    chk.setChecked(bool(state[key]))
        finally:
            self._updating = False
        for kind, mode in (state.get("modes") or {}).items():
            if kind in self._scrub:
                if mode == "off":                      # scene salvate con il vecchio "Nascosto"
                    self._set_ticks(kind, False)
                else:
                    self._scrub[kind].set_mode(str(mode), emit=False)
        for kind in state.get("show") or []:
            if kind in self._scrub:
                self._set_ticks(kind, True)
        for kind in state.get("hide") or []:
            if kind in self._scrub:
                self._set_ticks(kind, False)
        self._on_scrub_changed()                       # etichette dei cursori e ricostruzione della scena

    @staticmethod
    def _load_user_presets() -> dict:
        try:
            data = json.loads(str(QSettings().value(_PRESET_KEY, "{}") or "{}"))
            return data if isinstance(data, dict) else {}
        except (ValueError, TypeError):
            return {}

    @staticmethod
    def _store_user_presets(data: dict) -> None:
        QSettings().setValue(_PRESET_KEY, json.dumps(data))

    def user_presets(self) -> list:
        return sorted(self._load_user_presets())

    def save_preset(self, name: str) -> bool:
        name = str(name or "").strip()
        if not name or name in _BUILTIN_PRESETS:
            return False
        data = self._load_user_presets()
        data[name] = self.scene_state()
        self._store_user_presets(data)
        return True

    def delete_preset(self, name: str) -> None:
        data = self._load_user_presets()
        if data.pop(name, None) is not None:
            self._store_user_presets(data)

    def apply_preset(self, name: str) -> bool:
        state = _BUILTIN_PRESETS.get(name) or self._load_user_presets().get(name)
        if not state:
            return False
        self.apply_scene(state)
        return True

    def _ask_save_preset(self) -> None:
        name, ok = QInputDialog.getText(self, "Salva scena", "Nome della scena:")
        if ok and name.strip():
            if not self.save_preset(name):
                self._txt_cross.setPlainText("⚠ Nome non valido: quello è un preset predefinito o il nome è vuoto.")

    def _fill_preset_menu(self) -> None:
        menu = self._preset_menu
        menu.clear()
        for name in _BUILTIN_PRESETS:
            menu.addAction(name, lambda n=name: self.apply_preset(n))
        mine = self.user_presets()
        if mine:
            menu.addSeparator()
            for name in mine:
                menu.addAction(name, lambda n=name: self.apply_preset(n))
        menu.addSeparator()
        menu.addAction("Salva la scena attuale…", self._ask_save_preset)
        if mine:
            sub = menu.addMenu("Elimina")
            for name in mine:
                sub.addAction(name, lambda n=name: self.delete_preset(n))

    # --- esportazione --------------------------------------------------------------------------------------
    def _export_dir(self) -> str:
        return self._project_root() or self._workspace_root() or os.path.expanduser("~")

    def _tell(self, text: str, ok: bool = True) -> None:
        try:
            bar = self.iface.messageBar()
            (bar.pushSuccess if ok else bar.pushWarning)("Vista 3D profili", text)
        except Exception:
            pass
        if not ok:
            self._txt_cross.setPlainText(f"⚠ {text}")

    def export_image(self, path: str) -> str:
        """Salva la vista 3D com'e' (PNG o JPG, dall'estensione). Restituisce il percorso scritto."""
        if self._plotter is None:
            raise RuntimeError("la vista 3D non e' disponibile")
        path = str(path)
        if os.path.splitext(path)[1].lower() not in (".png", ".jpg", ".jpeg"):
            path += ".png"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._plotter.screenshot(path)
        return path

    def export_kml(self, path: str) -> dict:
        """Esporta tracce e time-slice mostrate in KML/KMZ (WGS84). Restituisce i conteggi."""
        from .gpr_fence_kml import write_kml

        if not self._shown or not any(self._last_scene.get(k) for k in ("gprs", "erts", "slices")):
            raise ValueError("la scena è vuota: mostra qualcosa prima di esportare")
        common = self._common_crs()
        if not common:
            raise ValueError("il progetto QGIS non ha un sistema di riferimento")
        for it in self._last_scene.get("slices") or []:
            it["rgba"] = self._slice_rgba(it)
        tr = lonlat_transformer("EPSG:4326", common)
        path = str(path)
        if os.path.splitext(path)[1].lower() not in (".kml", ".kmz"):
            path += ".kmz"
        return write_kml(path, self._last_scene, tr, slice_opacity=float(self._spin_slice_op.value()))

    def _ask_export_image(self) -> None:
        path, _f = QFileDialog.getSaveFileName(self, "Esporta la vista 3D", os.path.join(self._export_dir(), "vista3d.png"),
                                               "PNG (*.png);;JPEG (*.jpg *.jpeg)")
        if not path:
            return
        try:
            self._tell(f"Immagine salvata: {self.export_image(path)}")
        except Exception as exc:
            self._tell(f"Immagine non salvata: {exc}", ok=False)

    def _ask_export_kml(self) -> None:
        path, _f = QFileDialog.getSaveFileName(self, "Esporta per Google Earth", os.path.join(self._export_dir(), "vista3d.kmz"),
                                               "KMZ (*.kmz);;KML (*.kml)")
        if not path:
            return
        try:
            n = self.export_kml(path)
            self._tell(f"Esportati {n['gpr']} profili GPR, {n['ert']} sezioni ERT e {n['slice']} time-slice. "
                       "Nel KML la profondità non esiste: le slice stanno sul terreno.")
        except Exception as exc:
            self._tell(f"KML non esportato: {exc}", ok=False)

    # --- isolamento con il doppio clic -----------------------------------------------------------------------
    def eventFilter(self, obj, event) -> bool:
        if (obj is getattr(self, "_view_widget", None) and event.type() == QEvent.MouseButtonDblClick
                and event.button() == Qt.LeftButton):
            ratio = obj.devicePixelRatioF() if hasattr(obj, "devicePixelRatioF") else 1.0
            pos = event.pos()
            self._on_double_click((pos.x() * ratio, (obj.height() - pos.y() - 1) * ratio))
        return False                                    # l'evento prosegue: la vista continua a funzionare

    def _actor_name_for_item(self, item):
        """Nome dell'attore in scena che corrisponde a una voce dell'elenco, o None se non e' mostrata."""
        data = item.data(0, _ROLE)
        if not data:
            return None
        kind, payload = data
        text = item.text(0)
        if kind == "slice":
            name = f"slice::{os.path.basename(str(payload))}"
            return name if name in self._shown else None
        for name in self._shown:
            if name.startswith(f"{kind}::") and (name == f"{kind}::{text}" or name.startswith(f"{kind}::{text} (")):
                return name
        return None

    def _on_tree_double_click(self, item, _col) -> None:
        name = self._actor_name_for_item(item)
        if name is None:
            if item.data(0, _ROLE):
                self._txt_cross.setPlainText("⚠ Questa voce non è nella scena: scegli un modo che la mostri "
                                             "(Tutti, Uno alla volta, Spuntati) e riprova.")
            return
        self.focus_actor(None if name == self._focus else name)

    def _on_double_click(self, pos) -> None:
        pl = self._plotter
        if pl is None:
            return
        name = None
        try:
            from vtkmodules.vtkRenderingCore import vtkPropPicker
            picker = vtkPropPicker()
            picker.Pick(float(pos[0]), float(pos[1]), 0.0, pl.renderer)
            actor = picker.GetActor()
            if actor is not None:
                name = self._name_of_actor(actor)
        except Exception:
            name = None
        self.focus_actor(name if name != self._focus else None)     # doppio clic sullo stesso: toglie l'isolamento

    def _name_of_actor(self, actor):
        for name, a in self._plotter.renderer.actors.items():
            if a is actor and name in self._shown and name.startswith(_PICKABLE):
                return name
        return None

    def focus_actor(self, name) -> None:
        """Isola un profilo, una sezione o una slice (gli altri si attenuano); ``None`` mostra di nuovo tutto."""
        if self._plotter is None:
            return
        self._focus = name if name in self._shown else None
        self._apply_focus()

    def _apply_focus(self, render: bool = True) -> None:
        pl = self._plotter
        if self._focus is not None and self._focus not in self._shown:
            self._focus = None                           # l'elemento isolato non e' piu' nella scena
        if "focus" in self._shown:
            pl.remove_actor("focus", render=False)
            del self._shown["focus"]
        for name in self._shown:
            if not name.startswith(_PICKABLE):
                continue
            actor = pl.renderer.actors.get(name)
            if actor is None:
                continue
            base = float(self._spin_slice_op.value()) if name.startswith("slice::") else 1.0
            actor.prop.opacity = base if self._focus in (None, name) else min(base, _FOCUS_DIM)
        if self._focus is not None:
            label = self._focus.split("::", 1)[1]
            pl.add_text(f"Isolato: {label}  (doppio clic nel vuoto o Vista ▸ Mostra tutto)", name="focus",
                        position="lower_left", font_size=9, color="gray", render=False)
            self._shown["focus"] = "x"
        if render:
            pl.render()

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
