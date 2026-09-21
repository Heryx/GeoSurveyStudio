# -*- coding: utf-8 -*-
"""
GPR Profile Viewer
==================
QMainWindow con radargram matplotlib integrato.
"""

from __future__ import annotations

from .gpr_viewer_common import (
    DEFAULT_CHAIN_ORDER,
    DEFAULT_CMAP,
    FILTER_REGISTRY,
    Figure,
    FigureCanvasQTAgg,
    FilterBlockWidget,
    FilterChainWidget,
    GprManualPositioningDialog,
    HAS_MPL,
    LOG_FILE_PATH,
    OgprProfile,
    Optional,
    QAction,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSettings,
    QSize,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QThread,
    QTimer,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
    Qgis,
    QgsMessageLog,
    QgsRubberBand,
    Qt,
    _BlitManager,
    _PipelineWorker,
    _PreviewSliceWorker,
    _RangeGainCurveDialog,
    _dbg,
    _gpr_palettes,
    _log_gpr,
    _populate_cmap_combo,
    _run_pipeline_job,
    auto_tune_params,
    check_profiles_have_gps,
    clear_ogpr_cache,
    detect_and_read,
    import_dialog_filter,
    json,
    log_exception,
    np,
    os,
    pyqtSignal,
    refresh_gpr_cmaps,
    sip,
)
from .gpr_viewer_radargram import RadargramViewMixin
from .gpr_viewer_timeslice import TimesliceViewMixin
from .gpr_viewer_variants import VariantsMixin
from .gpr_viewer_project import ProjectPersistenceMixin


class GprProfileViewer(
    RadargramViewMixin,
    TimesliceViewMixin,
    VariantsMixin,
    ProjectPersistenceMixin,
    QMainWindow,
):
    cursor_moved = pyqtSignal(float, float, float)  # easting, northing, depth

    def __init__(self, iface, plugin=None, parent=None):
        super().__init__(parent)
        self.iface  = iface
        self.plugin = plugin
        self.setWindowTitle("GPR Profile Viewer")
        self.resize(1200, 750)
        self.setDockNestingEnabled(True)

        self._profiles:    list[OgprProfile] = []
        self._prof_idx:    int = 0
        self._ch_idx:      int = 0
        self._raw_data:    Optional[np.ndarray] = None
        self._proc_data:   Optional[np.ndarray] = None
        self._disp_data:   Optional[np.ndarray] = None
        self._cursor_x:    Optional[float] = None
        self._cursor_z:    Optional[float] = None
        self._view_xlim:   Optional[tuple[float, float]] = None
        self._view_ylim:   Optional[tuple[float, float]] = None
        self._gpr_3d_viewer = None
        self._processing_dock = None
        self._timeslice_dock = None
        self._processing_panel_widget = None
        self._timeslice_panel_widget = None
        self._right_tabs: Optional[QTabWidget] = None
        self._cached_catalog = None
        self._cached_catalog_pr: Optional[str] = None
        self._cached_catalog_mtime: Optional[float] = None
        self._suppress_3d_depth_sync = False
        self._updating_range_gain_controls = False
        self._range_gain_breakpoints = np.asarray([(0.0, 1.0), (1.0, 6.0)], dtype=np.float64)
        self._hyper_apex_x: Optional[float] = None
        self._hyper_apex_depth: Optional[float] = None
        self._trim_start_traces = 0
        self._trim_end_traces = 0
        self._trace_source_indices: Optional[np.ndarray] = None
        self._stage_cache: dict = {}
        self._bp_figure = None
        self._bp_ax = None
        self._bp_canvas = None
        self._wiggle_trace_idx: Optional[int] = None
        self._wiggle_depth_line = None
        self._wiggle_raw_line = None
        self._wiggle_fil_line = None
        self._blit: Optional[_BlitManager] = None
        self._im_key: Optional[tuple] = None
        self._lod = None                 # _ImageLod: ritaglio/decimazione per profili lunghi
        self._lod_cid = None
        self._lod_resize_cid = None
        self._lod_busy = False
        self._im_full_extent: Optional[tuple] = None
        self._display_base_cache: Optional[tuple] = None
        self._proc_thread: Optional[QThread] = None
        self._proc_worker: Optional[_PipelineWorker] = None
        self._proc_job: Optional[dict] = None
        self._proc_job_seq: int = 0
        self._proc_rerun: bool = False
        self._proc_busy: bool = False
        self._fig_slice = None
        self._ax_slice = None
        self._canvas_slice = None
        self._slider_slice_depth = None
        self._slice_im = None
        self._slice_vline = None
        self._slice_hline = None
        self._slice_view_xlim: Optional[tuple[float, float]] = None
        self._slice_view_ylim: Optional[tuple[float, float]] = None
        self._slice_cache_path: Optional[str] = None
        self._slice_cache_mtime: Optional[float] = None
        self._slice_cache_arr: Optional[np.ndarray] = None
        self._slice_cache_extent: Optional[tuple[float, float, float, float]] = None
        self._slice_current_idx: Optional[int] = None
        self._slice_current_path: Optional[str] = None
        self._slice_current_extent: Optional[tuple[float, float, float, float]] = None
        self._slice_current_shape: Optional[tuple[int, int]] = None
        self._slice_current_cmap: str = ""
        self._slice_pan_anchor: Optional[tuple[float, float, tuple[float, float], tuple[float, float]]] = None
        self._slice_catalog: list[dict] = []
        self._slice_catalog_source_dir: str = ""
        self._slice_catalog_dz: float = 0.0
        self._slice_catalog_from_manifest: bool = False
        # Nome della variante da cui deriva la prossima variante salvata
        # (impostato da "Nuova variante da qui" nel pannello Varianti).
        self._variant_parent_ctx: Optional[str] = None
        self._updating_slice_nav: bool = False
        self._chk_slice_open_3d_on_complete = None
        self._show_wiggle: bool = True
        self._ax_main_bounds_default = None
        self._ax_wiggle_bounds_default = None
        self._last_pipeline_params: dict = {}
        # True se _proc_data contiene GIA' il range-gain (pre_agc_gain nella pipeline).
        # Descrive il dato, non i widget: con il fallback su dato grezzo i parametri
        # dicono 'applicato' ma il dato no.
        self._proc_pre_agc: bool = False
        self._filter_chain_widget = None
        self._filter_chain_blocks: dict[str, object] = {}
        self._phase1_controls: dict[str, dict] = {}
        self._syncing_filter_chain = False
        self._preview_mode_active: bool = False
        self._preview_slice_grid: Optional[np.ndarray] = None
        self._preview_slice_extent: Optional[tuple[float, float, float, float]] = None
        self._preview_slice_z_from: Optional[float] = None
        self._preview_slice_z_to: Optional[float] = None
        self._preview_signature: Optional[tuple] = None
        self._preview_busy: bool = False
        self._preview_request_id: int = 0
        self._preview_thread: Optional[QThread] = None
        self._preview_worker: Optional[_PreviewSliceWorker] = None
        self._preview_pending_zwindow: tuple[float, float] = (0.0, 0.0)
        self._profile_view_initialized: bool = True
        self._allow_close: bool = False

        self._rb_point: Optional[QgsRubberBand] = None
        self._rb_line:  Optional[QgsRubberBand] = None
        self._rb_crosshair_h: Optional[QgsRubberBand] = None
        self._rb_crosshair_v: Optional[QgsRubberBand] = None
        self._crosshair_size_m: float = 5.0

        self._canvas_timer = QTimer(self)
        self._canvas_timer.setSingleShot(True)
        self._canvas_timer.setInterval(50)
        self._canvas_timer.timeout.connect(self._flush_canvas_update)
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(300)
        self._preview_timer.timeout.connect(self._compute_and_show_preview)
        self._updating_xpan = False

        self._build_ui()
        try:
            self._update_cmap_previews()
        except Exception:
            pass
        # Auto-import profili da GPR_Processing/project.json se presente.
        try:
            QTimer.singleShot(0, self._try_import_saved_gpr_processing_project)
        except Exception:
            pass

    def _is_qt_alive(self, obj) -> bool:
        if obj is None:
            return False
        try:
            if sip is not None and hasattr(sip, "isdeleted") and sip.isdeleted(obj):
                return False
        except Exception:
            return False
        return True

    def _safe_set_text(self, widget, text: str):
        if not self._is_qt_alive(widget):
            return
        try:
            widget.setText(str(text))
        except Exception:
            pass

    def _safe_cb_clear(self, combo):
        if not self._is_qt_alive(combo):
            return
        try:
            combo.blockSignals(True)
            combo.clear()
        except Exception:
            pass
        finally:
            try:
                combo.blockSignals(False)
            except Exception:
                pass

    def _safe_draw_idle(self, canvas):
        if not self._is_qt_alive(canvas):
            return
        try:
            canvas.draw_idle()
        except Exception:
            pass

    def _note_swallowed(self, where: str, exc: BaseException) -> None:
        """Registra un errore che il codice sceglie di non propagare.

        Una sola volta per causa (dove + tipo + messaggio): sui percorsi caldi,
        come il movimento del mouse, un errore ripetuto non deve inondare il log.
        """
        seen = self.__dict__.setdefault("_swallowed_seen", set())
        key = (where, type(exc).__name__, str(exc)[:200])
        if key in seen:
            return
        seen.add(key)
        try:
            import traceback
            _log_gpr(
                f"[GPR] Errore ignorato in {where}: {type(exc).__name__}: {exc}\n"
                f"{traceback.format_exc(limit=6)}",
                level=Qgis.Warning,
            )
        except Exception:
            pass

    def _reset_session_state(self):
        self._active_variant_dir = ""
        try:
            self._canvas_timer.stop()
        except Exception:
            pass
        try:
            self._preview_timer.stop()
        except Exception:
            pass

        self._profiles = []
        self._prof_idx = 0
        self._ch_idx = 0
        self._raw_data = None
        self._proc_data = None
        self._proc_pre_agc = False
        self._disp_data = None
        self._cursor_x = None
        self._cursor_z = None
        self._view_xlim = None
        self._view_ylim = None
        self._trace_source_indices = None
        self._trim_start_traces = 0
        self._trim_end_traces = 0
        self._hyper_apex_x = None
        self._hyper_apex_depth = None
        self._wiggle_trace_idx = None
        self._wiggle_depth_line = None
        self._wiggle_raw_line = None
        self._wiggle_fil_line = None
        self._vline = None
        self._hline = None
        self._im = None
        self._im_key = None
        self._lod = None
        self._im_full_extent = None
        self._display_base_cache = None
        self._proc_job_seq += 1
        self._proc_rerun = False
        self._refresh_blit_artists()
        self._slice_pan_anchor = None

        self._slice_catalog = []
        self._slice_current_idx = None
        self._slice_current_path = None
        self._slice_current_extent = None
        self._slice_current_shape = None
        self._slice_current_cmap = ""
        self._slice_view_xlim = None
        self._slice_view_ylim = None
        self._slice_cache_path = None
        self._slice_cache_mtime = None
        self._slice_cache_arr = None
        self._slice_cache_extent = None
        self._preview_mode_active = False
        self._preview_slice_grid = None
        self._preview_slice_extent = None
        self._preview_slice_z_from = None
        self._preview_slice_z_to = None
        self._preview_signature = None

        self._safe_set_text(self._lbl_profile, "\u2014")
        self._safe_set_text(self._lbl_status, "Importa un file .ogpr per iniziare.")
        self._safe_set_text(self._lbl_xpan, "Full")
        self._safe_cb_clear(self._cb_channel)
        if self._is_qt_alive(getattr(self, "_xpan_slider", None)):
            try:
                self._xpan_slider.setEnabled(False)
                self._xpan_slider.setValue(0)
            except Exception:
                pass
        if self._is_qt_alive(getattr(self, "_slider_slice_depth", None)):
            try:
                self._slider_slice_depth.setRange(0, 0)
                self._slider_slice_depth.setValue(0)
                self._slider_slice_depth.setEnabled(False)
            except Exception:
                pass
        if self._is_qt_alive(getattr(self, "_btn_slice_up", None)):
            try:
                self._btn_slice_up.setEnabled(False)
            except Exception:
                pass
        if self._is_qt_alive(getattr(self, "_btn_slice_down", None)):
            try:
                self._btn_slice_down.setEnabled(False)
            except Exception:
                pass

        try:
            if self._ax is not None:
                self._ax.clear()
            if self._ax_wiggle is not None:
                self._ax_wiggle.clear()
            if self._canvas_mpl is not None:
                self._safe_draw_idle(self._canvas_mpl)
        except Exception:
            pass

        try:
            if self._ax_slice is not None:
                self._ax_slice.clear()
            if self._canvas_slice is not None:
                self._safe_draw_idle(self._canvas_slice)
        except Exception:
            pass

        self._update_slice_navigator()
        try:
            self._clear_timeslice_crosshair()
        except Exception:
            pass

    def _channel_signal_score(self, data: np.ndarray) -> float:
        arr = np.asarray(data, dtype=np.float64)
        if arr.size == 0:
            return -1.0
        # Sampling leggero per evitare costi elevati su profili lunghi.
        if arr.size > 800_000:
            step = max(1, arr.size // 200_000)
            arr = arr.ravel()[::step]
        finite = np.isfinite(arr)
        if not finite.any():
            return -1.0
        arr = arr[finite]
        if arr.size == 0:
            return -1.0
        nz_ratio = float(np.count_nonzero(arr)) / float(arr.size)
        std = float(np.std(arr))
        amp = float(np.nanpercentile(np.abs(arr), 95)) if arr.size else 0.0
        return nz_ratio * 10.0 + std + amp * 1e-3

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _update_cmap_previews(self, *_):
        """Aggiorna le strisce di anteprima della colormap (radargramma + slice)."""
        if _gpr_palettes is None:
            return
        pairs = (
            ("_lbl_cmap_preview", "_cb_cmap"),
            ("_lbl_slice_cmap_preview", "_cb_slice_cmap"),
        )
        for lbl_attr, cb_attr in pairs:
            lbl = getattr(self, lbl_attr, None)
            cb = getattr(self, cb_attr, None)
            if not self._is_qt_alive(lbl) or not self._is_qt_alive(cb):
                continue
            name = cb.currentText().strip()
            if not name:
                continue
            try:
                pm = _gpr_palettes.colormap_preview_pixmap(
                    name, max(16, lbl.width()), max(8, lbl.height())
                )
                lbl.setPixmap(pm)
            except Exception:
                lbl.clear()

    def _on_import_palette(self):
        """Importa uno o piu' file palette e li rende disponibili in tutte le combo."""
        if _gpr_palettes is None:
            QMessageBox.warning(self, "Palette", "Modulo palette non disponibile.")
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Importa palette",
            "",
            "Palette (*.svg *.ggr *.cpt);;Tutti i file (*)",
        )
        if not paths:
            return

        added, errors = [], []
        for p in paths:
            try:
                added.append(_gpr_palettes.import_palette_file(p))
            except Exception as exc:
                errors.append("%s: %s" % (os.path.basename(p), exc))

        refresh_gpr_cmaps()
        for cb_attr in ("_cb_cmap", "_cb_slice_cmap"):
            cb = getattr(self, cb_attr, None)
            if self._is_qt_alive(cb):
                _populate_cmap_combo(cb)
        if added and self._is_qt_alive(getattr(self, "_cb_cmap", None)):
            self._cb_cmap.setCurrentText(added[-1])
        self._update_cmap_previews()

        msg = []
        if added:
            msg.append("Aggiunte: " + ", ".join(added))
        if errors:
            msg.append("Errori:\n" + "\n".join(errors))
        if msg:
            QMessageBox.information(self, "Import palette", "\n\n".join(msg))

    def _build_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        tb = QToolBar()
        act_open = QAction("\U0001f4c2  Importa profili", self)
        act_open.triggered.connect(self.import_files)
        tb.addAction(act_open)

        act_prev = QAction("\u25c4 Profilo", self)
        act_prev.triggered.connect(self._prev_profile)
        tb.addAction(act_prev)

        self._lbl_profile = QLabel("\u2014")
        self._lbl_profile.setMinimumWidth(200)
        tb.addWidget(self._lbl_profile)

        act_next = QAction("Profilo \u25ba", self)
        act_next.triggered.connect(self._next_profile)
        tb.addAction(act_next)

        tb.addSeparator()
        tb.addWidget(QLabel(" Canale:"))
        self._cb_channel = QComboBox()
        self._cb_channel.setMinimumWidth(80)
        self._cb_channel.currentIndexChanged.connect(self._on_channel_changed)
        tb.addWidget(self._cb_channel)

        tb.addSeparator()
        tb.addWidget(QLabel(" Colormap:"))
        self._cb_cmap = QComboBox()
        self._cb_cmap.setIconSize(QSize(64, 14))
        _populate_cmap_combo(self._cb_cmap, current=DEFAULT_CMAP)
        self._cb_cmap.currentTextChanged.connect(self._redraw)
        self._cb_cmap.currentTextChanged.connect(self._update_cmap_previews)
        tb.addWidget(self._cb_cmap)

        self._lbl_cmap_preview = QLabel()
        self._lbl_cmap_preview.setFixedSize(140, 16)
        self._lbl_cmap_preview.setToolTip("Anteprima della scala di colori selezionata.")
        tb.addWidget(self._lbl_cmap_preview)

        self._btn_import_palette = QToolButton()
        self._btn_import_palette.setText("➕")
        self._btn_import_palette.setToolTip(
            "Importa palette (cpt-city .svg, GIMP .ggr, GMT .cpt)."
        )
        self._btn_import_palette.clicked.connect(self._on_import_palette)
        tb.addWidget(self._btn_import_palette)

        tb.addSeparator()
        tb.addWidget(QLabel(" Render:"))
        self._cb_interp = QComboBox()
        self._cb_interp.addItem("Sharp", "nearest")
        self._cb_interp.addItem("Smooth", "bilinear")
        self._cb_interp.addItem("Fine", "bicubic")
        self._cb_interp.setCurrentIndex(1)  # Smooth by default
        self._cb_interp.setToolTip(
            "Sharp: pixel netti.\n"
            "Smooth: riduce effetto pixel rettangolari.\n"
            "Fine: resa piu' continua (piu' morbida)."
        )
        self._cb_interp.currentIndexChanged.connect(self._redraw)
        tb.addWidget(self._cb_interp)

        tb.addSeparator()
        self._chk_real_aspect = QCheckBox("Scala reale")
        self._chk_real_aspect.setChecked(True)
        self._chk_real_aspect.setToolTip(
            "Mantiene proporzioni metriche reali distanza/profondita'.\n"
            "Disattiva per adattare il profilo alla finestra."
        )
        self._chk_real_aspect.toggled.connect(self._on_real_aspect_toggled)
        tb.addWidget(self._chk_real_aspect)
        tb.addWidget(QLabel(" VE:"))
        self._spin_vertical_exag = QDoubleSpinBox()
        self._spin_vertical_exag.setRange(0.5, 20.0)
        self._spin_vertical_exag.setSingleStep(0.5)
        self._spin_vertical_exag.setValue(5.0)
        self._spin_vertical_exag.setDecimals(1)
        self._spin_vertical_exag.setEnabled(bool(self._chk_real_aspect.isChecked()))
        self._spin_vertical_exag.setToolTip(
            "Vertical Exaggeration (VE): 1.0 = scala reale.\n"
            "Valori > 1 aumentano l'enfasi verticale del radargramma."
        )
        self._spin_vertical_exag.valueChanged.connect(lambda _v: self._redraw())
        tb.addWidget(self._spin_vertical_exag)

        self._chk_show_wiggle = QCheckBox("Mostra Wiggle")
        self._chk_show_wiggle.setChecked(True)
        self._chk_show_wiggle.toggled.connect(self._on_toggle_wiggle)
        tb.addWidget(self._chk_show_wiggle)

        act_reset_zoom = QAction("Reset Zoom", self)
        act_reset_zoom.setToolTip("Ripristina l'estensione completa del profilo.")
        act_reset_zoom.triggered.connect(self._reset_zoom)
        tb.addAction(act_reset_zoom)

        act_export = QAction("Export Radargram", self)
        act_export.setToolTip("Esporta il radargramma corrente in PNG/TIFF con DPI configurabile.")
        act_export.triggered.connect(self._export_radargram_image)
        tb.addAction(act_export)

        tb.addSeparator()
        act_processing = QAction("Processing", self)
        act_processing.setToolTip("Apri il pannello laterale e mostra il tab Elaborazione.")
        act_processing.triggered.connect(self._open_processing_window)
        tb.addAction(act_processing)

        act_timeslice = QAction("Timeslice", self)
        act_timeslice.setToolTip("Apri il pannello laterale e mostra il tab Slicing.")
        act_timeslice.triggered.connect(self._open_timeslice_window)
        tb.addAction(act_timeslice)

        act_view3d = QAction("3D Viewer", self)
        act_view3d.setToolTip("Apri il viewer 3D delle timeslice.")
        act_view3d.triggered.connect(self._open_3d_viewer)
        tb.addAction(act_view3d)

        tb.addSeparator()
        act_manual_pos = QAction("\U0001f4cd Posizionamento Griglia", self)
        act_manual_pos.setToolTip(
            "Apri il dialog di posizionamento manuale della griglia di rilievo\n"
            "(profili senza GPS: assegna coordinate, gestisci zig-zag, rinomina)."
        )
        act_manual_pos.triggered.connect(self._open_manual_positioning_dialog)
        tb.addAction(act_manual_pos)

        act_coord_log = QAction("📋 Log coordinate", self)
        act_coord_log.setToolTip(
            "Mostra il log di debug di lettura/applicazione coordinate,\n"
            "con pulsanti per pulire il log e la cache dei profili."
        )
        act_coord_log.triggered.connect(self._show_coord_log_dialog)
        tb.addAction(act_coord_log)

        act_clear_cache = QAction("🧹 Pulisci cache", self)
        act_clear_cache.setToolTip(
            "Svuota la cache di lettura dei file .ogpr (i prossimi caricamenti\n"
            "rileggono i file da disco, con le coordinate originali)."
        )
        act_clear_cache.triggered.connect(self._clear_profile_cache)
        tb.addAction(act_clear_cache)

        root.addWidget(tb)

        center = QHBoxLayout()
        self._xpan_slider = None
        self._lbl_xpan = None

        if HAS_MPL:
            # Upper panel: active timeslice 2D view.
            self._fig_slice = Figure(figsize=(9, 3), tight_layout=True)
            self._ax_slice = self._fig_slice.add_subplot(111)
            self._canvas_slice = FigureCanvasQTAgg(self._fig_slice)
            self._canvas_slice.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self._canvas_slice.mpl_connect("button_press_event", self._on_slice_press)
            self._canvas_slice.mpl_connect("button_release_event", self._on_slice_release)
            self._canvas_slice.mpl_connect("motion_notify_event", self._on_slice_mouse_move)
            self._canvas_slice.mpl_connect("scroll_event", self._on_slice_scroll_zoom)
            self._slider_slice_depth = QSlider(Qt.Vertical)
            self._slider_slice_depth.setRange(0, 0)
            self._slider_slice_depth.setValue(0)
            self._slider_slice_depth.setEnabled(False)
            self._slider_slice_depth.setInvertedAppearance(True)
            self._slider_slice_depth.setInvertedControls(True)
            self._slider_slice_depth.setToolTip(
                "Scorri tra le slice calcolate.\nSu = superficiale | Giu = profondo"
            )
            self._slider_slice_depth.valueChanged.connect(self._on_slice_depth_slider)
            self._btn_slice_up = QToolButton()
            self._btn_slice_up.setArrowType(Qt.UpArrow)
            self._btn_slice_up.setFixedSize(20, 20)
            self._btn_slice_up.setAutoRepeat(True)
            self._btn_slice_up.setAutoRepeatDelay(250)
            self._btn_slice_up.setAutoRepeatInterval(150)
            self._btn_slice_up.setToolTip("Slice precedente")
            self._btn_slice_up.clicked.connect(lambda: self._step_slice(-1))
            self._btn_slice_down = QToolButton()
            self._btn_slice_down.setArrowType(Qt.DownArrow)
            self._btn_slice_down.setFixedSize(20, 20)
            self._btn_slice_down.setAutoRepeat(True)
            self._btn_slice_down.setAutoRepeatDelay(250)
            self._btn_slice_down.setAutoRepeatInterval(150)
            self._btn_slice_down.setToolTip("Slice successiva")
            self._btn_slice_down.clicked.connect(lambda: self._step_slice(+1))

            nav_col = QWidget()
            nav_col_lay = QVBoxLayout(nav_col)
            nav_col_lay.setContentsMargins(0, 0, 0, 0)
            nav_col_lay.setSpacing(2)
            nav_col_lay.setAlignment(Qt.AlignHCenter)
            nav_col_lay.addWidget(self._btn_slice_up, 0, Qt.AlignHCenter)
            nav_col_lay.addWidget(self._slider_slice_depth, 1, Qt.AlignHCenter)
            nav_col_lay.addWidget(self._btn_slice_down, 0, Qt.AlignHCenter)
            slice_host = QWidget()
            slice_lay = QHBoxLayout(slice_host)
            slice_lay.setContentsMargins(0, 0, 0, 0)
            slice_lay.setSpacing(4)
            slice_lay.addWidget(self._canvas_slice, stretch=1)
            slice_lay.addWidget(nav_col, stretch=0)

            # Lower panel: radargram + wiggle + pan slider.
            self._fig = Figure(figsize=(9, 3), tight_layout=False)
            gs = self._fig.add_gridspec(
                1,
                2,
                width_ratios=[4.0, 0.6],
                wspace=0.02,
                left=0.07,
                right=0.98,
                top=0.94,
                bottom=0.11,
            )
            self._ax         = self._fig.add_subplot(gs[0, 0])
            self._ax_wiggle  = self._fig.add_subplot(gs[0, 1], sharey=self._ax)
            self._ax_wiggle.tick_params(axis="y", left=False, labelleft=False)
            self._ax_main_bounds_default = list(self._ax.get_position().bounds)
            self._ax_wiggle_bounds_default = list(self._ax_wiggle.get_position().bounds)
            self._canvas_mpl = FigureCanvasQTAgg(self._fig)
            self._canvas_mpl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self._canvas_mpl.mpl_connect("motion_notify_event", self._on_mouse_move)
            self._canvas_mpl.mpl_connect("button_press_event",  self._on_mouse_press)
            self._canvas_mpl.mpl_connect("axes_leave_event",    self._on_axes_leave)
            self._canvas_mpl.mpl_connect("scroll_event",        self._on_scroll_zoom)
            self._blit = _BlitManager(self._canvas_mpl)

            left_host = QWidget()
            left_lay = QVBoxLayout(left_host)
            left_lay.setContentsMargins(0, 0, 0, 0)
            left_lay.setSpacing(4)
            left_lay.addWidget(self._canvas_mpl, stretch=1)

            pan_row = QWidget()
            pan_row_l = QHBoxLayout(pan_row)
            pan_row_l.setContentsMargins(4, 0, 4, 0)
            pan_row_l.setSpacing(6)
            pan_row_l.addWidget(QLabel("Scorri profilo:"))
            self._xpan_slider = QSlider(Qt.Horizontal)
            self._xpan_slider.setRange(0, 1000)
            self._xpan_slider.setValue(0)
            self._xpan_slider.setEnabled(False)
            self._xpan_slider.setToolTip(
                "Slider orizzontale sotto il profilo per navigare profili molto lunghi.\n"
                "Si attiva automaticamente quando lo zoom X e' < 100%."
            )
            self._xpan_slider.valueChanged.connect(self._on_xpan_slider_changed)
            pan_row_l.addWidget(self._xpan_slider, 1)
            self._lbl_xpan = QLabel("Full")
            self._lbl_xpan.setMinimumWidth(90)
            self._lbl_xpan.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            pan_row_l.addWidget(self._lbl_xpan, 0)
            left_lay.addWidget(pan_row, stretch=0)

            splitter = QSplitter(Qt.Vertical)
            splitter.addWidget(slice_host)
            splitter.addWidget(left_host)
            splitter.setStretchFactor(0, 2)
            splitter.setStretchFactor(1, 1)
            center.addWidget(splitter, stretch=4)
            self._im    = None
            self._vline = None
            self._hline = None
            self._wiggle_line = None
        else:
            lbl = QLabel("matplotlib non trovato.\nInstalla: pip install matplotlib")
            lbl.setAlignment(Qt.AlignCenter)
            center.addWidget(lbl, stretch=4)

        self._processing_panel_widget, self._timeslice_panel_widget = self._build_processing_panel()
        self._init_aux_windows()
        self._build_menu()
        root.addLayout(center)

        # Barra non modale "ripristinato l'ultimo stato" (con annulla).
        self._restore_bar = QWidget()
        rb_lay = QHBoxLayout(self._restore_bar)
        rb_lay.setContentsMargins(6, 3, 6, 3)
        self._lbl_restore_bar = QLabel("")
        self._btn_restore_undo = QPushButton("Annulla ripristino")
        self._btn_restore_undo.clicked.connect(self._undo_restore)
        self._btn_restore_close = QToolButton()
        self._btn_restore_close.setText("×")
        self._btn_restore_close.clicked.connect(self._restore_bar.hide)
        rb_lay.addWidget(self._lbl_restore_bar, 1)
        rb_lay.addWidget(self._btn_restore_undo)
        rb_lay.addWidget(self._btn_restore_close)
        self._restore_bar.setStyleSheet(
            "QWidget { background: #fff4c2; color: #333; }")
        self._restore_bar.hide()
        self._pre_restore_state = None
        root.addWidget(self._restore_bar)

        status_row = QHBoxLayout()
        self._lbl_status = QLabel("Importa un file .ogpr per iniziare.")
        # Il testo cambia a ogni movimento del mouse: senza questo la larghezza
        # minima segue il testo, la finestra si allarga/restringe e il canvas
        # viene ridimensionato (= ridisegnato per intero) a ogni movimento.
        self._lbl_status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._lbl_status.setMinimumWidth(0)
        self._lbl_recipe_state = QLabel("")
        self._lbl_recipe_state.setToolTip(
            "Processo: salvato = scritto automaticamente in project.json e "
            "ripristinato alla riapertura. Le varianti (radargrammi filtrati "
            "su disco) si salvano solo su richiesta esplicita.")
        status_row.addWidget(self._lbl_status, 1)
        status_row.addWidget(self._lbl_recipe_state)
        root.addLayout(status_row)

    def _dock_settings(self):
        st = getattr(self.plugin, "settings", None)
        if st is None:
            st = QSettings()
        return st

    @staticmethod
    def _dock_key(dock_id: str, suffix: str) -> str:
        return f"GeoSurveyStudio/gpr_profile_viewer/docks/{dock_id}/{suffix}"

    def _save_dock_state(self, dock: QDockWidget, dock_id: str):
        if dock is None:
            return
        settings = self._dock_settings()
        try:
            area = int(self.dockWidgetArea(dock))
        except Exception:
            area = int(Qt.RightDockWidgetArea)
        settings.setValue(self._dock_key(dock_id, "area"), area)
        settings.setValue(self._dock_key(dock_id, "floating"), bool(dock.isFloating()))
        settings.setValue(self._dock_key(dock_id, "visible"), bool(dock.isVisible()))
        try:
            settings.setValue(self._dock_key(dock_id, "geometry"), dock.saveGeometry())
        except Exception:
            pass

    def _restore_dock_state(self, dock: QDockWidget, dock_id: str, default_area=Qt.RightDockWidgetArea):
        if dock is None:
            return
        settings = self._dock_settings()
        area_val = settings.value(self._dock_key(dock_id, "area"), int(default_area), type=int)
        try:
            area = Qt.DockWidgetArea(int(area_val))
        except Exception:
            area = default_area
        try:
            self.addDockWidget(area, dock)
        except Exception:
            self.addDockWidget(default_area, dock)
        floating = bool(settings.value(self._dock_key(dock_id, "floating"), False, type=bool))
        try:
            dock.setFloating(floating)
        except Exception:
            pass
        geom = settings.value(self._dock_key(dock_id, "geometry"), None)
        if geom is not None:
            try:
                dock.restoreGeometry(geom)
            except Exception:
                pass
        visible = bool(settings.value(self._dock_key(dock_id, "visible"), True, type=bool))
        dock.setVisible(visible)

    def _create_aux_dock(self, title: str, object_name: str, content: QWidget):
        dock = QDockWidget(title, self)
        dock.setObjectName(object_name)
        dock.setAllowedAreas(
            Qt.LeftDockWidgetArea
            | Qt.RightDockWidgetArea
            | Qt.BottomDockWidgetArea
            | Qt.TopDockWidgetArea
        )
        dock.setFeatures(
            QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
            | QDockWidget.DockWidgetClosable
        )
        dock.setWidget(content)
        return dock

    def _init_aux_windows(self):
        created_proc = False
        created_slice = False
        if self._processing_dock is None and self._processing_panel_widget is not None:
            proc_title = (
                "GPR - Process/Slicing"
                if self._timeslice_panel_widget is None and self._is_qt_alive(self._right_tabs)
                else "GPR - Processing"
            )
            self._processing_dock = self._create_aux_dock(
                proc_title, "GprProcessingDock", self._processing_panel_widget
            )
            self._restore_dock_state(self._processing_dock, "processing", Qt.RightDockWidgetArea)
            created_proc = True
        if self._timeslice_dock is None and self._timeslice_panel_widget is not None:
            self._timeslice_dock = self._create_aux_dock(
                "GPR - Timeslice", "GprTimesliceDock", self._timeslice_panel_widget
            )
            self._restore_dock_state(self._timeslice_dock, "timeslice", Qt.RightDockWidgetArea)
            created_slice = True

        if (created_proc or created_slice) and self._processing_dock is not None and self._timeslice_dock is not None:
            try:
                self.tabifyDockWidget(self._processing_dock, self._timeslice_dock)
            except Exception:
                pass

    def _open_processing_window(self):
        if self._processing_dock is None:
            self._init_aux_windows()
        if self._processing_dock is None:
            return
        self._processing_dock.show()
        self._processing_dock.raise_()
        if self._is_qt_alive(self._right_tabs):
            try:
                self._right_tabs.setCurrentIndex(0)
            except Exception:
                pass

    def _open_timeslice_window(self):
        # Preferred path: unified Process/Slicing tabs in the same dock.
        if self._processing_dock is None:
            self._init_aux_windows()
        if self._processing_dock is not None and self._is_qt_alive(self._right_tabs):
            try:
                self._processing_dock.show()
                self._processing_dock.raise_()
                self._right_tabs.setCurrentIndex(1)
                return
            except Exception:
                pass
        # Backward-compatible fallback (old separate Timeslice dock).
        if self._timeslice_dock is None:
            return
        self._timeslice_dock.show()
        self._timeslice_dock.raise_()

    def _toggle_wiggle_panel(self, checked: bool):
        try:
            self._chk_show_wiggle.setChecked(bool(checked))
        except Exception:
            self._show_wiggle = bool(checked)
            self._apply_wiggle_visibility_layout()
            self._redraw()

    def _on_real_aspect_toggled(self, checked: bool):
        try:
            if hasattr(self, "_spin_vertical_exag") and self._spin_vertical_exag is not None:
                self._spin_vertical_exag.setEnabled(bool(checked))
        except Exception:
            pass
        self._redraw()

    def _build_menu(self):
        mb = self.menuBar()
        if mb is None:
            return
        try:
            mb.clear()
        except Exception:
            pass

        m_file = mb.addMenu("File")
        act_new_proj = QAction("Nuovo progetto GPR...", self)
        act_new_proj.setShortcut("Ctrl+N")
        act_new_proj.triggered.connect(self._new_gpr_project)
        m_file.addAction(act_new_proj)

        act_open_proj = QAction("Apri progetto GPR...", self)
        act_open_proj.triggered.connect(self._open_gpr_project)
        m_file.addAction(act_open_proj)
        m_file.addSeparator()

        act_import = QAction("Importa profili...", self)
        act_import.setShortcut("Ctrl+O")
        act_import.triggered.connect(self.import_files)
        m_file.addAction(act_import)

        act_import_saved = QAction("Ricarica ultimo progetto GPR", self)
        act_import_saved.triggered.connect(lambda: self._try_import_saved_gpr_processing_project(force=True))
        m_file.addAction(act_import_saved)

        act_open_pm = QAction("Apri Project Manager...", self)
        act_open_pm.setShortcut("Ctrl+Shift+O")
        act_open_pm.triggered.connect(self._open_project_manager)
        m_file.addAction(act_open_pm)

        act_save_proj = QAction("Salva progetto", self)
        act_save_proj.setShortcut("Ctrl+S")
        act_save_proj.triggered.connect(self._save_project_file)
        m_file.addAction(act_save_proj)

        m_file.addSeparator()
        act_export = QAction("Export Radargram...", self)
        act_export.triggered.connect(self._export_radargram_image)
        m_file.addAction(act_export)

        act_import_slice = QAction("Import Timeslices to Canvas", self)
        act_import_slice.triggered.connect(self._import_slices_to_canvas)
        m_file.addAction(act_import_slice)

        m_file.addSeparator()
        act_close = QAction("Chiudi", self)
        act_close.setShortcut("Ctrl+W")
        act_close.triggered.connect(self.close)
        m_file.addAction(act_close)

        m_view = mb.addMenu("View")
        if self._processing_dock is not None:
            act_proc = QAction("Processing", self)
            act_proc.setCheckable(True)
            act_proc.setChecked(bool(self._processing_dock.isVisible()))
            act_proc.toggled.connect(self._processing_dock.setVisible)
            self._processing_dock.visibilityChanged.connect(act_proc.setChecked)
            m_view.addAction(act_proc)

        if self._timeslice_dock is not None:
            act_slice = QAction("Timeslice", self)
            act_slice.setCheckable(True)
            act_slice.setChecked(bool(self._timeslice_dock.isVisible()))
            act_slice.toggled.connect(self._timeslice_dock.setVisible)
            self._timeslice_dock.visibilityChanged.connect(act_slice.setChecked)
            m_view.addAction(act_slice)
        elif self._is_qt_alive(self._right_tabs):
            act_slice_tab = QAction("Vai a Slicing", self)
            act_slice_tab.triggered.connect(self._open_timeslice_window)
            m_view.addAction(act_slice_tab)

        act_wiggle = QAction("Wiggle", self)
        act_wiggle.setCheckable(True)
        act_wiggle.setChecked(bool(self._show_wiggle))
        act_wiggle.toggled.connect(self._toggle_wiggle_panel)
        m_view.addAction(act_wiggle)

        m_view.addSeparator()
        act_reset = QAction("Reset Zoom", self)
        act_reset.setShortcut("Ctrl+0")
        act_reset.triggered.connect(self._reset_zoom)
        m_view.addAction(act_reset)

        act_real_scale = QAction("Scala reale", self)
        act_real_scale.setCheckable(True)
        act_real_scale.setChecked(bool(self._chk_real_aspect.isChecked()))
        act_real_scale.toggled.connect(self._chk_real_aspect.setChecked)
        self._chk_real_aspect.toggled.connect(act_real_scale.setChecked)
        m_view.addAction(act_real_scale)

        m_proc = mb.addMenu("Processing")
        act_apply = QAction("Applica pipeline", self)
        act_apply.setShortcut("F5")
        act_apply.triggered.connect(self._apply_processing_async)
        m_proc.addAction(act_apply)

        act_gain = QAction("Aggiorna gain", self)
        act_gain.setShortcut("F6")
        act_gain.triggered.connect(self._apply_gain_only)
        m_proc.addAction(act_gain)

        m_proc.addSeparator()
        act_save_variant = QAction("Salva variante processata (.npz)...", self)
        act_save_variant.triggered.connect(self._save_processed_variant)
        m_proc.addAction(act_save_variant)

        act_save_working = QAction("Aggiorna variante di lavoro (sovrascrive 'Lavoro')", self)
        act_save_working.setToolTip(
            "Scrive i radargrammi filtrati nell'unica variante di lavoro, sempre\n"
            "con lo stesso nome. Viene aggiornata in automatico alla chiusura."
        )
        act_save_working.triggered.connect(lambda *_: self._save_working_variant())
        m_proc.addAction(act_save_working)

        act_slice_variant = QAction("Time-slice da variante processata...", self)
        act_slice_variant.triggered.connect(self._build_slices_from_variant)
        m_proc.addAction(act_slice_variant)

        act_blend = QAction("Combina multi-frequenza → variante...", self)
        act_blend.setToolTip(
            "Fonde un'antenna dual/triple frequenza in un unico radargramma\n"
            "(somma pesata + allineamento tempo-zero). Sorgenti: i canali di un\n"
            "file multicanale OPPURE piu' profili caricati (un file per frequenza\n"
            "dello stesso tracciato). La frequenza nominale non e' richiesta."
        )
        act_blend.triggered.connect(self._blend_multifreq_to_variant)
        m_proc.addAction(act_blend)

        act_grid_filter = QAction("Filtra time-slice (dominio griglia)...", self)
        act_grid_filter.setToolTip(
            "Applica un filtro 2D (passa-basso, laplaciano, background di linea,\n"
            "notch FFT direzionale) a tutte le GeoTIFF di un set di time-slice,\n"
            "scrivendo un nuovo set filtrato."
        )
        act_grid_filter.triggered.connect(self._filter_timeslices_grid)
        m_proc.addAction(act_grid_filter)

        act_merge_sets = QAction("Unisci set di time-slice...", self)
        act_merge_sets.setToolTip(
            "Fonde due o piu' cartelle di GeoTIFF (stessa cell size, stessi\n"
            "indici z) in un unico set: per ogni z pareggia la giunzione\n"
            "(edge-match opzionale) e appende le griglie (media sulle\n"
            "sovrapposizioni). Utile per survey a piu' griglie adiacenti."
        )
        act_merge_sets.triggered.connect(self._merge_timeslice_sets)
        m_proc.addAction(act_merge_sets)

        act_rt_set = QAction("Ruota/trasla set di time-slice...", self)
        act_rt_set.setToolTip(
            "Ruota (CCW) e trasla ogni GeoTIFF di un set verso un'origine\n"
            "comune, con ricampionamento bilineare. Scrive un nuovo set _rt."
        )
        act_rt_set.triggered.connect(self._rotate_translate_timeslice_set)
        m_proc.addAction(act_rt_set)
        m_proc.addSeparator()

        act_save_preset = QAction("Salva preset gain/hyperbola...", self)
        act_save_preset.triggered.connect(self._save_gain_hyper_preset)
        m_proc.addAction(act_save_preset)

        act_load_preset = QAction("Carica preset gain/hyperbola...", self)
        act_load_preset.triggered.connect(self._load_gain_hyper_preset)
        m_proc.addAction(act_load_preset)

    def _on_toggle_wiggle(self, checked: bool):
        self._show_wiggle = bool(checked)
        self._apply_wiggle_visibility_layout()
        self._redraw()

    def _pipeline_chain_help_text(self, step_id: str) -> str:
        txt = {
            "timezero": "Corregge il time-zero prima degli altri filtri.",
            "dewow": "Rimuove la componente lenta/DC della traccia.",
            "bandpass": "Applica filtro passa-banda in frequenza.",
            "bg_removal": "Sottrae il clutter orizzontale (background).",
            "migration": "Migrazione 2D F-K (Stolt) per collassare le iperboli di diffrazione.",
            "pre_agc_gain": "Range gain applicato prima dell'AGC.",
            "agc": "Equalizza dinamicamente l'ampiezza con finestra locale.",
            "envelope": "Converte il segnale in inviluppo Hilbert (solo ampiezza).",
            "dcshift": "Sottrae da ogni traccia la sua media (valore costante, non mobile).",
            "boxcar": "Lisciatura boxcar 2D (media non pesata) su tracce e/o campioni.",
            "notch": "Attenua una banda di frequenza stretta centrata su freq (MHz).",
            "whiten": "Whitening spettrale: |X(f)|=1 nella banda, fase preservata.",
            "amp_eq": "Equalizzazione d'ampiezza: gain = scala_ref / RMS(finestra mobile).",
            "rad_threshold": "Azzera i campioni la cui ampiezza sta dentro le soglie.",
            "power_gain": "Range gain analitico: potenza t^a o esponenziale exp(a*t).",
            "eigen_filter": "Filtro a immagini proprie (SVD): rimuove il clutter orizzontale coerente.",
            "median2d": "Mediana 2D scorrevole: elimina spike isolati preservando i bordi.",
            "fk_dip": "Filtro FK a ventaglio: rimuove/isola eventi per velocita' apparente (dip).",
            "kirchhoff": "Migrazione Kirchhoff 2D (diffraction stack) con obliquita' + spreading.",
            "inst_attr": "Attributo istantaneo dal segnale analitico: fase / cos(fase) / frequenza / inviluppo.",
        }
        return txt.get(str(step_id), "")

    # Spec dei controlli per i filtri Fase 1 (costruiti in modo generico).
    # Ogni voce: (param_key, etichetta, tipo, *args)
    #   int   -> (min, max, default)
    #   float -> (min, max, default, decimali, step)
    #   choice-> (lista_opzioni, default)
    _MAIN_FILTER_IDS = (
        "dewow", "timezero", "bg_removal", "bandpass", "migration", "pre_agc_gain", "agc",
    )

    def _connect_param_widgets(self):
        """Ogni parametro dei filtri ricalcola il radargramma (con debounce), come
        gia' facevano solo passa-banda e AGC max/da-campione."""
        self._param_timer = QTimer(self)
        self._param_timer.setSingleShot(True)
        self._param_timer.setInterval(300)

        def _fire():
            if self._raw_data is not None and not self._syncing_filter_chain:
                self._apply_processing_async()

        self._param_timer.timeout.connect(_fire)
        kick = lambda *_a: self._param_timer.start()
        spins = ["_spin_dewow", "_spin_tz_threshold", "_spin_tz_backup", "_spin_bg_window",
                 "_spin_bg_sample_start", "_spin_bg_sample_end", "_spin_agc", "_spin_agc_lateral",
                 "_spin_mig_vel", "_spin_mig_dx", "_spin_gain_surface", "_spin_gain_deep"]
        combos = ["_cb_tz_method", "_cb_tz_mode", "_cb_bg_method", "_cb_bg_mode"]
        for name in spins:
            w = getattr(self, name, None)
            if w is not None:
                try:
                    w.valueChanged.connect(kick)
                except Exception:
                    pass
        for name in combos:
            w = getattr(self, name, None)
            if w is not None:
                try:
                    w.currentIndexChanged.connect(kick)
                except Exception:
                    pass
        w = getattr(self, "_chk_bg_auto", None)
        if w is not None:
            w.toggled.connect(kick)
        for store in (getattr(self, "_phase1_controls", {}) or {}).values():
            for w in (store or {}).values():
                try:
                    if isinstance(w, QComboBox):
                        w.currentIndexChanged.connect(kick)
                    elif isinstance(w, QCheckBox):
                        w.toggled.connect(kick)
                    elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                        w.valueChanged.connect(kick)
                except Exception:
                    pass

    def _reveal_enabled_hidden_blocks(self):
        blocks = getattr(self, "_filter_chain_blocks", None) or {}
        chain = getattr(self, "_filter_chain_widget", None)
        if not blocks or not self._is_qt_alive(chain):
            return
        try:
            enabled = chain.get_enabled_map()
        except Exception:
            return
        for sid, blk in blocks.items():
            if enabled.get(sid) and not blk.isVisibleTo(chain):
                blk.setVisible(True)

    def _build_add_filter_menu(self):
        btn = QToolButton()
        btn.setText("+ Aggiungi filtro")
        btn.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(btn)
        blocks = getattr(self, "_filter_chain_blocks", None) or {}
        chain = self._filter_chain_widget

        def _toggle(sid, act):
            blk = blocks.get(sid)
            if blk is None:
                return
            if act.isChecked():
                blk.setVisible(True)
            elif blk.is_enabled():
                act.setChecked(True)
                QMessageBox.information(self, "Filtro", "Disattiva prima il filtro per toglierlo dalla catena.")
            else:
                blk.setVisible(False)

        for sid in DEFAULT_CHAIN_ORDER:
            if sid in self._MAIN_FILTER_IDS or sid not in blocks:
                continue
            act = menu.addAction(str(FILTER_REGISTRY.get(sid, {}).get("label", sid)))
            act.setCheckable(True)
            act.triggered.connect(lambda _c=False, s=sid, a=act: _toggle(s, a))
        btn.setMenu(menu)
        self._btn_add_filter = btn
        return btn

    def _open_diagnosis_dialog(self):
        from .gpr_diagnosis_dialog import DiagnosisDialog
        if self._raw_data is None:
            QMessageBox.information(self, "Diagnosi", "Nessun profilo caricato.")
            return
        dock = getattr(self, "_diagnosis_dock", None)
        if dock is not None and self._is_qt_alive(dock):
            dock.show()
            dock.raise_()
            try:
                dock.widget().run_diagnosis()
                dock.widget().refresh_log()
            except Exception:
                pass
            return
        # Dock a sinistra della finestra del profilo (non sul canvas di QGIS).
        host = self
        panel = DiagnosisDialog(self)
        dock = QDockWidget("GPR Diagnosi / Log / AI", host)
        dock.setObjectName("RasterLinkerGprDiagnosisDock")
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea
                             | Qt.TopDockWidgetArea | Qt.BottomDockWidgetArea)
        dock.setWidget(panel)
        dock.setAttribute(Qt.WA_DeleteOnClose, True)
        host.addDockWidget(Qt.LeftDockWidgetArea, dock)
        dock.destroyed.connect(lambda *_: setattr(self, "_diagnosis_dock", None))
        self._diagnosis_dock = dock
        self.destroyed.connect(dock.close)
        dock.show()
        dock.raise_()

    _PHASE1_STEP_IDS = (
        "dcshift", "boxcar", "notch", "whiten", "amp_eq", "rad_threshold", "power_gain",
        "eigen_filter", "median2d", "fk_dip", "kirchhoff", "inst_attr",
    )
    _PHASE1_SPECS = {
        "dcshift": [
            ("dcshift_sample_start", "Da campione:", "int", 0, 99999, 0),
            ("dcshift_sample_end", "A campione (0=fine):", "int", 0, 99999, 0),
        ],
        "boxcar": [
            ("boxcar_width", "Semi-ampiezza tracce:", "int", 0, 64, 3),
            ("boxcar_height", "Semi-ampiezza campioni:", "int", 0, 256, 3),
        ],
        "notch": [
            ("notch_freq_mhz", "Freq centrale (MHz):", "float", 0.0, 3000.0, 0.0, 1, 5.0),
            ("notch_bw_mhz", "Larghezza banda (MHz):", "float", 1.0, 2000.0, 50.0, 1, 5.0),
        ],
        "whiten": [
            ("whiten_low_mhz", "Low (MHz, 0=auto):", "float", 0.0, 3000.0, 0.0, 1, 5.0),
            ("whiten_high_mhz", "High (MHz, 0=auto):", "float", 0.0, 3000.0, 0.0, 1, 5.0),
            ("whiten_gain", "Gain (divisore):", "float", 0.1, 1000.0, 20.0, 2, 1.0),
        ],
        "amp_eq": [
            ("amp_eq_window", "Finestra (campioni):", "int", 4, 8192, 120),
            ("amp_eq_gain", "Gain:", "float", 0.01, 100.0, 1.0, 2, 0.1),
        ],
        "rad_threshold": [
            ("rad_thr_neg", "Soglia neg (0..1):", "float", 0.0, 1.0, 0.0, 3, 0.01),
            ("rad_thr_pos", "Soglia pos (0..1):", "float", 0.0, 1.0, 0.0, 3, 0.01),
        ],
        "power_gain": [
            ("power_gain_mode", "Modo:", "choice", ["power", "exp"], "power"),
            ("power_gain_exp", "Esponente a:", "float", -5.0, 5.0, 1.0, 2, 0.1),
            ("power_gain_t0_ns", "t0 (ns):", "float", 0.0, 5000.0, 0.0, 1, 1.0),
            ("power_gain_scale", "Scala:", "float", 0.01, 100.0, 1.0, 2, 0.1),
        ],
        "eigen_filter": [
            ("eigen_drop", "Componenti da azzerare:", "int", 0, 64, 1),
            ("eigen_keep_last", "Tieni fino a (0=tutte):", "int", 0, 64, 0),
        ],
        "median2d": [
            ("median2d_w_trace", "Semi-ampiezza tracce:", "int", 0, 32, 1),
            ("median2d_w_sample", "Semi-ampiezza campioni:", "int", 0, 64, 1),
        ],
        "fk_dip": [
            ("fk_dip_mode", "Modo:", "choice", ["reject", "pass"], "reject"),
            ("fk_dip_vmin", "v min (m/s):", "float", 0.0, 3.0e8, 0.0, 0, 1.0e6),
            ("fk_dip_vmax", "v max (m/s, 0=inf):", "float", 0.0, 3.0e8, 0.0, 0, 1.0e6),
            ("fk_dip_dx_m", "dx tracce (m):", "float", 0.001, 10.0, 0.05, 3, 0.01),
            ("fk_dip_taper", "Taper:", "float", 0.0, 0.5, 0.15, 2, 0.05),
        ],
        "kirchhoff": [
            ("kirchhoff_velocity_m_s", "Velocita' (m/s):", "float", 1.0e6, 3.0e8, 1.0e8, 0, 1.0e6),
            ("kirchhoff_dx_m", "dx tracce (m):", "float", 0.001, 10.0, 0.05, 3, 0.01),
            ("kirchhoff_aperture_m", "Apertura (m, 0=auto):", "float", 0.0, 200.0, 0.0, 2, 0.5),
            ("kirchhoff_txrx_m", "Offset Tx-Rx (m):", "float", 0.0, 3.0, 0.0, 3, 0.02),
            ("kirchhoff_topo", "Correzione topografica (usa quote):", "bool", False),
        ],
        "inst_attr": [
            ("inst_attr_kind", "Attributo:", "choice",
             ["phase", "cos_phase", "frequency", "envelope"], "phase"),
        ],
    }

    def _on_phase1_control_changed(self, step_id: str):
        """Riapplica la pipeline quando cambia un parametro di un filtro Fase 1."""
        if getattr(self, "_syncing_filter_chain", False):
            return
        enabled = {}
        if self._is_qt_alive(getattr(self, "_filter_chain_widget", None)):
            try:
                enabled = dict(self._filter_chain_widget.get_enabled_map() or {})
            except Exception:
                enabled = {}
        if bool(enabled.get(str(step_id), False)) and self._raw_data is not None:
            self._apply_processing_async()
            self._request_preview_update(immediate=False)

    def _build_phase1_step_controls(self, step_id: str):
        sid = str(step_id or "").strip().lower()
        spec = self._PHASE1_SPECS.get(sid)
        host = QWidget()
        fl = QFormLayout(host)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setSpacing(4)
        if not spec:
            fl.addRow(QLabel("Nessun comando disponibile."))
            return host
        store = self._phase1_controls.setdefault(sid, {})
        for row in spec:
            key, label, kind = row[0], row[1], row[2]
            w = store.get(key)
            if w is None or not self._is_qt_alive(w):
                if kind == "int":
                    _mn, _mx, _dv = row[3], row[4], row[5]
                    w = QSpinBox()
                    w.setRange(int(_mn), int(_mx))
                    w.setValue(int(_dv))
                    w.setKeyboardTracking(False)
                    w.valueChanged.connect(
                        lambda _v, s=sid: self._on_phase1_control_changed(s)
                    )
                elif kind == "float":
                    _mn, _mx, _dv, _dec, _step = row[3], row[4], row[5], row[6], row[7]
                    w = QDoubleSpinBox()
                    w.setRange(float(_mn), float(_mx))
                    w.setDecimals(int(_dec))
                    w.setSingleStep(float(_step))
                    w.setValue(float(_dv))
                    w.setKeyboardTracking(False)
                    w.valueChanged.connect(
                        lambda _v, s=sid: self._on_phase1_control_changed(s)
                    )
                elif kind == "choice":
                    _opts, _dv = row[3], row[4]
                    w = QComboBox()
                    for opt in _opts:
                        w.addItem(str(opt), str(opt))
                    idx = w.findData(str(_dv))
                    w.setCurrentIndex(idx if idx >= 0 else 0)
                    w.currentIndexChanged.connect(
                        lambda _i, s=sid: self._on_phase1_control_changed(s)
                    )
                elif kind == "bool":
                    _dv = bool(row[3]) if len(row) > 3 else False
                    w = QCheckBox()
                    w.setChecked(_dv)
                    w.toggled.connect(
                        lambda _v, s=sid: self._on_phase1_control_changed(s)
                    )
                else:
                    continue
                store[key] = w
            fl.addRow(label, w)
        return host

    def _phase1_params(self) -> dict:
        """Valori correnti dei controlli Fase 1 (solo i parametri, non gli enable)."""
        out: dict = {}
        for sid, store in self._phase1_controls.items():
            for key, w in store.items():
                if not self._is_qt_alive(w):
                    continue
                if isinstance(w, QComboBox):
                    out[key] = str(w.currentData() or w.currentText())
                elif isinstance(w, QCheckBox):
                    out[key] = bool(w.isChecked())
                elif isinstance(w, QSpinBox):
                    out[key] = int(w.value())
                elif isinstance(w, QDoubleSpinBox):
                    out[key] = float(w.value())
        return out

    def _build_pipeline_step_controls(self, step_id: str):
        sid = str(step_id or "").strip().lower()
        host = QWidget()
        fl = QFormLayout(host)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setSpacing(4)

        if sid == "timezero":
            fl.addRow("Metodo:", self._cb_tz_method)
            fl.addRow("Mode:", self._cb_tz_mode)
            fl.addRow("Soglia:", self._spin_tz_threshold)
            fl.addRow("Backup N:", self._spin_tz_backup)
            return host

        if sid == "dewow":
            fl.addRow("Finestra:", self._spin_dewow)
            return host

        if sid == "bg_removal":
            fl.addRow("Metodo:", self._cb_bg_method)
            fl.addRow("Modo:", self._cb_bg_mode)
            fl.addRow("Auto:", self._chk_bg_auto)
            fl.addRow("Finestra:", self._spin_bg_window)
            fl.addRow("Da campione:", self._spin_bg_sample_start)
            fl.addRow("A campione (0=fine):", self._spin_bg_sample_end)
            return host

        if sid == "bandpass":
            fl.addRow("Low (MHz):", self._spin_bp_lo)
            fl.addRow("High (MHz):", self._spin_bp_hi)
            fl.addRow("Taper (roll-off):", self._spin_bp_taper)
            if HAS_MPL and self._is_qt_alive(getattr(self, "_bp_canvas", None)):
                fl.addRow(QLabel("Spettro:"))
                fl.addRow(self._bp_canvas)
            return host

        if sid == "migration":
            fl.addRow("Velocita' (m/s):", self._spin_mig_vel)
            fl.addRow("RDP (\u03b5r):", self._spin_mig_rdp)
            fl.addRow("dx tracce (m):", self._spin_mig_dx)
            fl.addRow(self._btn_sync_mig_vel)
            return host

        if sid == "agc":
            fl.addRow("Finestra:", self._spin_agc)
            fl.addRow("Max Gain (0=nessun limite):", self._spin_agc_max_gain)
            fl.addRow("Tracce laterali (0=singola):", self._spin_agc_lateral)
            agc_s0_row = QHBoxLayout()
            agc_s0_row.addWidget(self._spin_agc_sample_start)
            agc_s0_row.addWidget(self._btn_pick_agc_s0)
            fl.addRow("Da campione:", agc_s0_row)
            return host

        if sid == "envelope":
            lbl = QLabel("Nessun comando aggiuntivo.")
            lbl.setWordWrap(True)
            fl.addRow(lbl)
            return host

        if sid == "pre_agc_gain":
            fl.addRow("Gain min:", self._spin_gain_surface)
            fl.addRow("Gain max:", self._spin_gain_deep)
            fl.addRow("Breakpoints:", self._btn_range_gain_curve)
            return host

        if sid == "manual_gain":
            fl.addRow("Tipo:", self._cb_manual_gain_type)
            fl.addRow("Gain superficie:", self._spin_manual_gain_start)
            fl.addRow("Gain profondita':", self._spin_manual_gain_end)
            fl.addRow("Da campione:", self._spin_manual_gain_sample_start)
            fl.addRow("A campione (0=fine):", self._spin_manual_gain_sample_end)
            return host

        if sid in self._PHASE1_SPECS:
            return self._build_phase1_step_controls(sid)

        lbl = QLabel("Nessun comando disponibile.")
        lbl.setWordWrap(True)
        fl.addRow(lbl)
        return host

    def _build_filter_chain_controls(self):
        if FilterChainWidget is None or FilterBlockWidget is None:
            return None
        try:
            chain = FilterChainWidget(self)
            blocks = {}
            for sid in list(DEFAULT_CHAIN_ORDER):
                meta = FILTER_REGISTRY.get(sid, {})
                label = str(meta.get("label", sid))
                block = FilterBlockWidget(
                    sid,
                    label,
                    help_text=self._pipeline_chain_help_text(sid),
                    parent=chain,
                )
                if hasattr(block, "set_content_widget"):
                    try:
                        block.set_content_widget(self._build_pipeline_step_controls(sid))
                    except Exception:
                        pass
                chain.add_block(block)
                blocks[sid] = block
            # I filtri Fase 1 partono disattivati: il resto della UI non ha
            # un _chk_* corrispondente, quindi la casella del blocco stesso e'
            # l'unica sorgente di verita' del loro stato enable.
            try:
                chain.set_enabled_map(
                    {sid: False for sid in self._PHASE1_STEP_IDS}, emit_signal=False
                )
            except Exception:
                pass
            chain.chain_changed.connect(self._on_filter_chain_widget_changed)
            self._filter_chain_widget = chain
            self._filter_chain_blocks = blocks
            for _sid, _blk in blocks.items():
                if _sid not in self._MAIN_FILTER_IDS:
                    _blk.setVisible(False)
            return chain
        except Exception:
            self._filter_chain_widget = None
            self._filter_chain_blocks = {}
            return None

    def _pipeline_enabled_map_from_controls(self) -> dict:
        chk_rg = bool(getattr(self, "_chk_range_gain", None) and self._chk_range_gain.isChecked())
        return {
            "dewow": bool(getattr(self, "_chk_dewow", None) and self._chk_dewow.isChecked()),
            "timezero": bool(getattr(self, "_chk_timezero", None) and self._chk_timezero.isChecked()),
            "bg_removal": bool(getattr(self, "_chk_bg", None) and self._chk_bg.isChecked()),
            "pre_agc_gain": bool(chk_rg),
            "agc": bool(getattr(self, "_chk_agc", None) and self._chk_agc.isChecked()),
            "bandpass": bool(getattr(self, "_chk_bp", None) and self._chk_bp.isChecked()),
            "envelope": bool(getattr(self, "_chk_envelope", None) and self._chk_envelope.isChecked()),
            "manual_gain": bool(getattr(self, "_chk_manual_gain", None) and self._chk_manual_gain.isChecked()),
            "migration": bool(getattr(self, "_chk_mig", None) and self._chk_mig.isChecked()),
        }

    def _current_chain_order(self) -> list[str]:
        if self._is_qt_alive(self._filter_chain_widget):
            try:
                ids = list(self._filter_chain_widget.get_ordered_chain_ids())
                if ids:
                    return ids
            except Exception:
                pass
        return list(DEFAULT_CHAIN_ORDER)

    def _sync_filter_chain_from_controls(self):
        if self._syncing_filter_chain:
            return
        if not self._is_qt_alive(self._filter_chain_widget):
            return
        self._syncing_filter_chain = True
        try:
            self._filter_chain_widget.set_enabled_map(
                self._pipeline_enabled_map_from_controls(),
                emit_signal=False,
            )
        except Exception:
            pass
        finally:
            self._syncing_filter_chain = False

    def _on_filter_chain_widget_changed(self):
        if self._syncing_filter_chain:
            return
        if not self._is_qt_alive(self._filter_chain_widget):
            return
        try:
            enabled = dict(self._filter_chain_widget.get_enabled_map() or {})
        except Exception:
            return

        self._syncing_filter_chain = True
        try:
            if self._is_qt_alive(getattr(self, "_chk_dewow", None)):
                self._chk_dewow.setChecked(bool(enabled.get("dewow", self._chk_dewow.isChecked())))
            if self._is_qt_alive(getattr(self, "_chk_timezero", None)):
                self._chk_timezero.setChecked(bool(enabled.get("timezero", self._chk_timezero.isChecked())))
            if self._is_qt_alive(getattr(self, "_chk_bg", None)):
                self._chk_bg.setChecked(bool(enabled.get("bg_removal", self._chk_bg.isChecked())))
            if self._is_qt_alive(getattr(self, "_chk_agc", None)):
                self._chk_agc.setChecked(bool(enabled.get("agc", self._chk_agc.isChecked())))
            if self._is_qt_alive(getattr(self, "_chk_bp", None)):
                self._chk_bp.setChecked(bool(enabled.get("bandpass", self._chk_bp.isChecked())))
            if self._is_qt_alive(getattr(self, "_chk_envelope", None)):
                self._chk_envelope.setChecked(bool(enabled.get("envelope", self._chk_envelope.isChecked())))

            if self._is_qt_alive(getattr(self, "_chk_mig", None)):
                self._chk_mig.setChecked(bool(enabled.get("migration", self._chk_mig.isChecked())))

            pre_agc_enabled = bool(enabled.get("pre_agc_gain", False))
            if self._is_qt_alive(getattr(self, "_chk_range_gain", None)):
                self._chk_range_gain.setChecked(pre_agc_enabled)
            if self._is_qt_alive(getattr(self, "_chk_manual_gain", None)):
                self._chk_manual_gain.setChecked(bool(enabled.get("manual_gain", self._chk_manual_gain.isChecked())))
        finally:
            self._syncing_filter_chain = False

        if self._raw_data is not None:
            self._apply_processing_async()
            self._request_preview_update(immediate=False)

    def _on_filter_chain_source_toggled(self, _value=None):
        self._sync_filter_chain_from_controls()

    def _build_processing_panel(self):
        grp = QGroupBox("Processing")
        fl  = QFormLayout(grp)

        # Auto-tune
        self._btn_auto_tune = QPushButton("Auto-tune parametri")
        self._btn_auto_tune.setToolTip(
            "Analizza i profili caricati e suggerisce i parametri ottimali.\n"
            "Mostra un'anteprima con i valori calcolati prima di applicarli."
        )
        self._btn_auto_tune.clicked.connect(self._on_auto_tune)
        fl.addRow(self._btn_auto_tune)
        self._btn_diagnosis = QPushButton("Diagnosi automatica / Log / AI...")
        self._btn_diagnosis.setToolTip(
            "Analizza il radargramma, propone sequenza e parametri dei filtri,\n"
            "mostra il log e permette di chiedere un parere a un assistente AI."
        )
        self._btn_diagnosis.clicked.connect(self._open_diagnosis_dialog)
        fl.addRow(self._btn_diagnosis)

        # Dewow
        self._chk_dewow  = QCheckBox(); self._chk_dewow.setChecked(False)
        self._spin_dewow = QSpinBox();  self._spin_dewow.setRange(4, 256); self._spin_dewow.setValue(16)

        # Time-zero
        self._chk_timezero = QCheckBox()
        self._chk_timezero.setChecked(False)
        self._chk_timezero.setToolTip(
            "Tronca i campioni pre-onda diretta scan-by-scan o line-by-line.\n"
            "NON usa shift circolare: il radargram viene fisicamente troncato."
        )
        self._cb_tz_method = QComboBox()
        self._cb_tz_method.addItems(["peak", "threshold", "zero_crossing", "dominant", "surface", "coppens"])
        self._cb_tz_method.setCurrentText("peak")
        self._cb_tz_method.setEnabled(False)
        self._cb_tz_mode = QComboBox()
        self._cb_tz_mode.addItems(["line_by_line", "scan_by_scan", "individual"])
        self._cb_tz_mode.setCurrentText("line_by_line")
        self._cb_tz_mode.setEnabled(False)
        self._spin_tz_threshold = QDoubleSpinBox()
        self._spin_tz_threshold.setRange(0.05, 0.95); self._spin_tz_threshold.setSingleStep(0.05)
        self._spin_tz_threshold.setValue(0.20); self._spin_tz_threshold.setEnabled(False)
        self._spin_tz_backup = QSpinBox()
        self._spin_tz_backup.setRange(0, 32); self._spin_tz_backup.setValue(4)
        self._spin_tz_backup.setEnabled(False)

        def _toggle_tz(checked):
            for w in (self._cb_tz_method, self._cb_tz_mode,
                      self._spin_tz_threshold, self._spin_tz_backup):
                w.setEnabled(checked)
        self._chk_timezero.toggled.connect(_toggle_tz)

        # ------------------------------------------------------------------
        # BG removal
        # ------------------------------------------------------------------
        self._chk_bg = QCheckBox()
        self._chk_bg.setChecked(False)
        self._chk_bg.setToolTip(
            "Background Removal: sottrae la traccia media o mediana per eliminare banding orizzontale.\n\n"
            "ATTENZIONE: rimuove anche riflessi reali paralleli al profilo."
        )

        self._cb_bg_method = QComboBox()
        self._cb_bg_method.addItem("Media (standard)", "mean")
        self._cb_bg_method.addItem("Mediana (robusta)", "median")
        self._cb_bg_method.setCurrentIndex(0)
        self._cb_bg_method.setEnabled(False)
        self._cb_bg_method.setToolTip(
            "Media   : sottrae la media aritmetica (standard).\n"
            "Mediana : sottrae la mediana (robusta: evita ombre scure intorno a forti bersagli)."
        )

        self._cb_bg_mode = QComboBox()
        self._cb_bg_mode.addItem("line_by_line", "line_by_line")
        self._cb_bg_mode.addItem("grid_by_grid", "grid_by_grid")
        self._cb_bg_mode.setCurrentIndex(0)
        self._cb_bg_mode.setEnabled(False)
        self._cb_bg_mode.setToolTip(
            "line_by_line : media/mediana calcolata su ogni singolo radargram.\n"
            "grid_by_grid : media/mediana calcolata su tutti i radargram del grid\n"
            "               (due passate: pre-BG + sottrazione globale)."
        )

        self._chk_bg_auto = QCheckBox("Auto")
        self._chk_bg_auto.setChecked(True)
        self._chk_bg_auto.setEnabled(False)
        self._chk_bg_auto.setToolTip(
            "Auto: sottrae la media/mediana dell'intero profilo (filter_length=99000).\n"
            "Disabilita per usare una finestra scorrevole personalizzata."
        )

        self._spin_bg_window = QSpinBox()
        self._spin_bg_window.setRange(8, 99000)
        self._spin_bg_window.setValue(200)
        self._spin_bg_window.setEnabled(False)
        self._spin_bg_window.setToolTip(
            "Lunghezza del filtro in tracce (finestra scorrevole).\n"
            "Attivo solo se Auto e' disabilitato."
        )

        # Finestra temporale
        self._spin_bg_sample_start = QSpinBox()
        self._spin_bg_sample_start.setRange(0, 9999)
        self._spin_bg_sample_start.setValue(10)
        self._spin_bg_sample_start.setEnabled(False)
        self._spin_bg_sample_start.setToolTip(
            "Primo campione incluso nel BG removal (0 = dall'inizio).\n\n"
            "Imposta > 0 per ESCLUDERE i primi N campioni dalla sottrazione.\n"
            "Effetto: preserva il ground coupling / onda diretta nei primi\n"
            "campioni, permettendo all'AGC di amplificarli correttamente.\n"
            "Esempio: per 600 MHz con dt=0.117 ns, 0.5 m ~ 28 campioni."
        )

        self._spin_bg_sample_end = QSpinBox()
        self._spin_bg_sample_end.setRange(0, 9999)
        self._spin_bg_sample_end.setValue(0)
        self._spin_bg_sample_end.setEnabled(False)
        self._spin_bg_sample_end.setToolTip(
            "Ultimo campione escluso dal BG removal (0 = fine traccia).\n"
            "Usare per limitare la rimozione a una zona di profondita' specifica."
        )

        def _toggle_bg_auto(auto_checked):
            self._spin_bg_window.setEnabled(not auto_checked)

        def _toggle_bg(bg_checked):
            self._cb_bg_method.setEnabled(bg_checked)
            self._cb_bg_mode.setEnabled(bg_checked)
            self._chk_bg_auto.setEnabled(bg_checked)
            self._spin_bg_sample_start.setEnabled(bg_checked)
            self._spin_bg_sample_end.setEnabled(bg_checked)
            if bg_checked:
                _toggle_bg_auto(self._chk_bg_auto.isChecked())
            else:
                self._spin_bg_window.setEnabled(False)

        self._chk_bg.toggled.connect(_toggle_bg)
        self._chk_bg_auto.toggled.connect(_toggle_bg_auto)
        self._cb_bg_method.currentIndexChanged.connect(
            lambda _i: self._apply_processing_async() if self._chk_bg.isChecked() and self._raw_data is not None else None
        )

        # ------------------------------------------------------------------
        # Migration 2D F-K (Stolt)
        # ------------------------------------------------------------------
        self._chk_mig = QCheckBox()
        self._chk_mig.setChecked(False)
        self._chk_mig.setToolTip("Migrazione 2D F-K (Stolt) per collassare le iperboli di diffrazione.")

        self._spin_mig_vel = QDoubleSpinBox()
        self._spin_mig_vel.setRange(1e6, 3e8)
        self._spin_mig_vel.setSingleStep(1e6)
        self._spin_mig_vel.setValue(1e8)
        self._spin_mig_vel.setDecimals(0)
        self._spin_mig_vel.setEnabled(False)
        self._spin_mig_vel.setToolTip("Velocita' di propagazione nel mezzo in m/s (es. 100,000,000 = 0.10 m/ns).")

        self._spin_mig_rdp = QDoubleSpinBox()
        self._spin_mig_rdp.setRange(1.0, 81.0)
        self._spin_mig_rdp.setSingleStep(0.5)
        self._spin_mig_rdp.setValue(9.0)
        self._spin_mig_rdp.setDecimals(1)
        self._spin_mig_rdp.setEnabled(False)
        self._spin_mig_rdp.setToolTip("Costante dielettrica relativa (RDP \u03b5r). Sincronizzata con la velocita'.")

        self._spin_mig_dx = QDoubleSpinBox()
        self._spin_mig_dx.setRange(0.001, 10.0)
        self._spin_mig_dx.setSingleStep(0.01)
        self._spin_mig_dx.setValue(0.05)
        self._spin_mig_dx.setDecimals(3)
        self._spin_mig_dx.setEnabled(False)
        self._spin_mig_dx.setToolTip("Spaziatura orizzontale media tra le tracce in metri.")

        self._btn_sync_mig_vel = QPushButton("Usa velocita' iperbole")
        self._btn_sync_mig_vel.setEnabled(False)
        self._btn_sync_mig_vel.setToolTip("Copia il valore di RDP/velocita' ricavato dal fit iperbolico.")

        def _on_mig_vel_changed(val: float):
            rdp = (299792458.0 / max(float(val), 1e3)) ** 2
            self._spin_mig_rdp.blockSignals(True)
            try:
                self._spin_mig_rdp.setValue(float(np.clip(rdp, 1.0, 81.0)))
            finally:
                self._spin_mig_rdp.blockSignals(False)
            if self._chk_mig.isChecked() and self._raw_data is not None:
                self._apply_processing_async()

        def _on_mig_rdp_changed(val: float):
            v = 299792458.0 / np.sqrt(max(float(val), 1.0))
            self._spin_mig_vel.blockSignals(True)
            try:
                self._spin_mig_vel.setValue(float(np.clip(v, 1e6, 3e8)))
            finally:
                self._spin_mig_vel.blockSignals(False)
            if self._chk_mig.isChecked() and self._raw_data is not None:
                self._apply_processing_async()

        def _sync_mig_vel():
            try:
                rdp_fit = float(self._spin_hyperbola_rdp.value())
                self._spin_mig_rdp.setValue(rdp_fit)
            except Exception:
                pass

        def _toggle_mig(mig_checked):
            self._spin_mig_vel.setEnabled(mig_checked)
            self._spin_mig_rdp.setEnabled(mig_checked)
            self._spin_mig_dx.setEnabled(mig_checked)
            self._btn_sync_mig_vel.setEnabled(mig_checked)
            if self._raw_data is not None:
                self._apply_processing_async()

        self._spin_mig_vel.valueChanged.connect(_on_mig_vel_changed)
        self._spin_mig_rdp.valueChanged.connect(_on_mig_rdp_changed)
        self._spin_mig_dx.valueChanged.connect(
            lambda _v: self._apply_processing_async() if self._chk_mig.isChecked() and self._raw_data is not None else None
        )
        self._btn_sync_mig_vel.clicked.connect(_sync_mig_vel)
        self._chk_mig.toggled.connect(_toggle_mig)

        # ------------------------------------------------------------------
        # AGC
        # ------------------------------------------------------------------
        self._chk_agc  = QCheckBox(); self._chk_agc.setChecked(False)
        self._spin_agc = QSpinBox();  self._spin_agc.setRange(8, 512); self._spin_agc.setValue(128)
        self._spin_agc_max_gain = QDoubleSpinBox()
        self._spin_agc_max_gain.setRange(0.0, 1000.0)
        self._spin_agc_max_gain.setValue(0.0)
        self._spin_agc_max_gain.setSingleStep(5.0)
        self._spin_agc_max_gain.setEnabled(False)
        self._spin_agc_max_gain.setToolTip(
            "Limite massimo fattore di guadagno (0 = illimitato/automatico).\n"
            "Previene l'amplificazione eccessiva del rumore di fondo nelle zone profonde."
        )
        self._spin_agc_lateral = QSpinBox()
        self._spin_agc_lateral.setRange(0, 999)
        self._spin_agc_lateral.setValue(0)
        self._spin_agc_lateral.setEnabled(False)
        self._spin_agc_lateral.setToolTip(
            "0 = guadagno calcolato su ogni singola traccia (appiattisce le anomalie).\n"
            "N = energia mediata su N tracce vicine: il guadagno compensa il decadimento in\n"
            "profondita' ma conserva le differenze di ampiezza tra tracce (consigliato 15-31)."
        )
        self._spin_agc_sample_start = QSpinBox()
        self._spin_agc_sample_start.setRange(0, 9999)
        self._spin_agc_sample_start.setValue(0)
        self._spin_agc_sample_start.setEnabled(False)
        self._spin_agc_sample_start.setToolTip(
            "Primo campione su cui applicare l'AGC (0 = dall'inizio).\n"
            "I campioni precedenti (burst) non entrano nel calcolo del guadagno e vengono\n"
            "riportati alla scala AGC con un fattore unico."
        )
        # Stato pick-mode: nome del target attivo (es. 'agc_sample_start') o None
        self._pick_mode_target: str | None = None

        # Bottone pick creato una volta sola come singleton (come gli spinner)
        self._btn_pick_agc_s0 = QPushButton("Seleziona")
        self._btn_pick_agc_s0.setCheckable(True)
        self._btn_pick_agc_s0.setEnabled(False)
        self._btn_pick_agc_s0.setToolTip(
            "Clicca qui, poi clicca sul profilo per impostare\n"
            "il campione di inizio AGC dalla posizione del cursore."
        )
        self._btn_pick_agc_s0.toggled.connect(
            lambda checked: self._set_pick_mode("agc_sample_start" if checked else None,
                                                self._btn_pick_agc_s0)
        )

        def _toggle_agc(agc_checked):
            self._spin_agc.setEnabled(agc_checked)
            self._spin_agc_max_gain.setEnabled(agc_checked)
            self._spin_agc_lateral.setEnabled(agc_checked)
            self._spin_agc_sample_start.setEnabled(agc_checked)
            if self._is_qt_alive(self._btn_pick_agc_s0):
                self._btn_pick_agc_s0.setEnabled(agc_checked)
            if not agc_checked and self._pick_mode_target == "agc_sample_start":
                self._set_pick_mode(None, None)

        self._chk_agc.toggled.connect(_toggle_agc)
        self._spin_agc_max_gain.valueChanged.connect(
            lambda _v: self._apply_processing_async() if self._chk_agc.isChecked() and self._raw_data is not None else None
        )
        self._spin_agc_sample_start.valueChanged.connect(
            lambda _v: self._apply_processing_async() if self._chk_agc.isChecked() and self._raw_data is not None else None
        )

        # Bandpass
        self._chk_bp     = QCheckBox(); self._chk_bp.setChecked(False)
        self._spin_bp_lo = QDoubleSpinBox(); self._spin_bp_lo.setRange(1, 3000); self._spin_bp_lo.setValue(200)
        self._spin_bp_hi = QDoubleSpinBox(); self._spin_bp_hi.setRange(1, 3000); self._spin_bp_hi.setValue(1200)
        self._spin_bp_taper = QDoubleSpinBox()
        self._spin_bp_taper.setRange(0.0, 50.0)
        self._spin_bp_taper.setSingleStep(1.0)
        self._spin_bp_taper.setValue(12.0)
        self._spin_bp_taper.setSuffix(" %")
        self._spin_bp_taper.setToolTip(
            "Roll-off coseno per lato, in % della banda.\n"
            "0 % = taglio netto (piu' ringing), 12 % = default storico,\n"
            "50 % = finestra di Hann completa (transizione morbida)."
        )
        # Avoid transient low/high resets while typing (e.g. "1200" passes through 1->12->120).
        self._spin_bp_lo.setKeyboardTracking(False)
        self._spin_bp_hi.setKeyboardTracking(False)
        self._spin_bp_taper.setKeyboardTracking(False)
        self._chk_bp.toggled.connect(self._on_bp_controls_changed)
        self._spin_bp_lo.valueChanged.connect(self._on_bp_spin_changed)
        self._spin_bp_hi.valueChanged.connect(self._on_bp_spin_changed)
        self._spin_bp_taper.valueChanged.connect(self._on_bp_spin_changed)

        # Envelope
        self._chk_envelope = QCheckBox()
        self._chk_envelope.setChecked(False)
        self._chk_envelope.setToolTip(
            "Converte il segnale in inviluppo Hilbert all'interno della pipeline."
        )

        # Manual gain (depth-varying, Linear / Exponential)
        self._chk_manual_gain = QCheckBox()
        self._chk_manual_gain.setChecked(False)
        self._chk_manual_gain.setToolTip(
            "Gain manuale dipendente dalla profondita' (lineare o esponenziale).\n"
            "Step display: applicato dopo la normalizzazione, prima della visualizzazione.\n"
            "I campioni profondi vengono saturati a bianco/nero se il gain e' elevato."
        )
        self._cb_manual_gain_type = QComboBox()
        self._cb_manual_gain_type.addItem("Lineare", "linear")
        self._cb_manual_gain_type.addItem("Esponenziale", "exponential")
        self._cb_manual_gain_type.setEnabled(False)
        self._cb_manual_gain_type.setToolTip(
            "Tipo di curva di gain:\n"
            "  Lineare:      g(t) aumenta linearmente da gain_start a gain_end\n"
            "  Esponenziale: g(t) = gain_start * exp(log(gain_end/gain_start)*t)"
        )
        self._spin_manual_gain_start = QDoubleSpinBox()
        self._spin_manual_gain_start.setRange(0.0, 1000.0)
        self._spin_manual_gain_start.setSingleStep(0.5)
        self._spin_manual_gain_start.setValue(1.0)
        self._spin_manual_gain_start.setEnabled(False)
        self._spin_manual_gain_start.setToolTip("Amplificazione al campione 0 (superficie).")
        self._spin_manual_gain_end = QDoubleSpinBox()
        self._spin_manual_gain_end.setRange(0.0, 1000.0)
        self._spin_manual_gain_end.setSingleStep(1.0)
        self._spin_manual_gain_end.setValue(20.0)
        self._spin_manual_gain_end.setEnabled(False)
        self._spin_manual_gain_end.setToolTip("Amplificazione all'ultimo campione (profondita' max).")
        self._spin_manual_gain_sample_start = QSpinBox()
        self._spin_manual_gain_sample_start.setRange(0, 9999)
        self._spin_manual_gain_sample_start.setValue(0)
        self._spin_manual_gain_sample_start.setEnabled(False)
        self._spin_manual_gain_sample_start.setToolTip("Primo campione su cui applicare il gain (0 = dall'inizio).")
        self._spin_manual_gain_sample_end = QSpinBox()
        self._spin_manual_gain_sample_end.setRange(0, 9999)
        self._spin_manual_gain_sample_end.setValue(0)
        self._spin_manual_gain_sample_end.setEnabled(False)
        self._spin_manual_gain_sample_end.setToolTip("Ultimo campione escluso (0 = fino alla fine).")

        def _toggle_manual_gain(checked):
            self._cb_manual_gain_type.setEnabled(checked)
            self._spin_manual_gain_start.setEnabled(checked)
            self._spin_manual_gain_end.setEnabled(checked)
            self._spin_manual_gain_sample_start.setEnabled(checked)
            self._spin_manual_gain_sample_end.setEnabled(checked)
            # manual_gain is a display-only step: no pipeline re-run needed.
            self._apply_gain_only()

        self._chk_manual_gain.toggled.connect(_toggle_manual_gain)
        self._cb_manual_gain_type.currentIndexChanged.connect(
            lambda _i: self._apply_gain_only() if self._chk_manual_gain.isChecked() else None
        )
        self._spin_manual_gain_start.valueChanged.connect(
            lambda _v: self._apply_gain_only() if self._chk_manual_gain.isChecked() else None
        )
        self._spin_manual_gain_end.valueChanged.connect(
            lambda _v: self._apply_gain_only() if self._chk_manual_gain.isChecked() else None
        )
        self._spin_manual_gain_sample_start.valueChanged.connect(
            lambda _v: self._apply_gain_only() if self._chk_manual_gain.isChecked() else None
        )
        self._spin_manual_gain_sample_end.valueChanged.connect(
            lambda _v: self._apply_gain_only() if self._chk_manual_gain.isChecked() else None
        )

        # Display
        self._spin_gain = QDoubleSpinBox()
        self._spin_gain.setRange(0.1, 20.0); self._spin_gain.setSingleStep(0.5)
        self._spin_gain.setValue(1.0)
        self._spin_gain.setToolTip("Moltiplicatore display post-normalize.")

        self._chk_flip_profile = QCheckBox()
        self._chk_flip_profile.setChecked(False)
        self._chk_flip_profile.setToolTip(
            "Inverti orizzontalmente il radargramma corrente (direzione tracce)."
        )
        self._chk_flip_profile.toggled.connect(lambda _v: self._reload_data(deferred=True))

        self._spin_trim_start = QSpinBox()
        self._spin_trim_start.setRange(0, 0)
        self._spin_trim_start.setValue(0)
        self._spin_trim_start.setToolTip("Numero tracce da rimuovere all'inizio del profilo.")
        self._spin_trim_end = QSpinBox()
        self._spin_trim_end.setRange(0, 0)
        self._spin_trim_end.setValue(0)
        self._spin_trim_end.setToolTip("Numero tracce da rimuovere alla fine del profilo.")
        self._btn_apply_trim = QPushButton("Apply Trim")
        self._btn_apply_trim.setToolTip("Applica il ritaglio tracce al profilo corrente.")
        self._btn_apply_trim.clicked.connect(self._reload_data)
        self._spin_trim_start.valueChanged.connect(self._on_trim_traces_changed)
        self._spin_trim_end.valueChanged.connect(self._on_trim_traces_changed)

        self._chk_range_gain = QCheckBox()
        self._chk_range_gain.setChecked(False)
        self._chk_range_gain.setToolTip(
            "Gain in funzione della profondita' per recuperare riflessioni profonde."
        )
        self._spin_gain_surface = QDoubleSpinBox()
        self._spin_gain_surface.setRange(0.0, 100.0)
        self._spin_gain_surface.setSingleStep(0.1)
        self._spin_gain_surface.setValue(0.3)
        self._spin_gain_surface.setEnabled(False)
        self._spin_gain_surface.setToolTip(
            "Gain minimo della curva TVG (profondita' superficiale)."
        )
        self._spin_gain_deep = QDoubleSpinBox()
        self._spin_gain_deep.setRange(0.0, 100.0)
        self._spin_gain_deep.setSingleStep(0.5)
        self._spin_gain_deep.setValue(6.0)
        self._spin_gain_deep.setEnabled(False)
        self._spin_gain_deep.setToolTip(
            "Gain massimo della curva TVG (profondita' profonda)."
        )
        self._btn_range_gain_curve = QPushButton("Edit curve...")
        self._btn_range_gain_curve.setEnabled(False)
        self._btn_range_gain_curve.clicked.connect(self._open_range_gain_curve_editor)

        def _toggle_range_gain(checked):
            self._spin_gain_surface.setEnabled(checked)
            self._spin_gain_deep.setEnabled(checked)
            self._btn_range_gain_curve.setEnabled(checked)
            self._apply_range_gain_change()

        self._chk_range_gain.toggled.connect(_toggle_range_gain)
        self._spin_gain_surface.valueChanged.connect(self._on_range_gain_surface_changed)
        self._spin_gain_deep.valueChanged.connect(self._on_range_gain_deep_changed)
        for _chk in (
            self._chk_dewow,
            self._chk_timezero,
            self._chk_bg,
            self._chk_agc,
            self._chk_bp,
            self._chk_envelope,
            self._chk_range_gain,
            self._chk_manual_gain,
        ):
            try:
                _chk.toggled.connect(self._on_filter_chain_source_toggled)
            except Exception:
                pass

        self._chk_hyperbola = QCheckBox()
        self._chk_hyperbola.setChecked(False)
        self._chk_hyperbola.setToolTip(
            "Overlay iperbole modello per stima RDP/velocita'."
        )
        self._spin_hyperbola_rdp = QDoubleSpinBox()
        self._spin_hyperbola_rdp.setRange(1.0, 40.0)
        self._spin_hyperbola_rdp.setSingleStep(0.1)
        self._spin_hyperbola_rdp.setValue(9.0)
        self._spin_hyperbola_rdp.setEnabled(False)
        self._spin_hyperbola_txrx = QDoubleSpinBox()
        self._spin_hyperbola_txrx.setRange(0.0, 3.0)
        self._spin_hyperbola_txrx.setSingleStep(0.02)
        self._spin_hyperbola_txrx.setDecimals(3)
        self._spin_hyperbola_txrx.setValue(0.0)
        self._spin_hyperbola_txrx.setSuffix(" m")
        self._spin_hyperbola_txrx.setEnabled(False)
        self._spin_hyperbola_txrx.setKeyboardTracking(False)
        self._spin_hyperbola_txrx.setToolTip(
            "Separazione fissa antenna Tx-Rx (offset bistatico).\n"
            "0 = antenna monostatica. Migliora la stima di velocita' su\n"
            "iperboli poco profonde con antenne a offset noto."
        )
        self._lbl_hyperbola_vel = QLabel("v=n/a")
        self._lbl_hyperbola_depth = QLabel("d=n/a")
        self._btn_set_hyper_apex = QPushButton("Set apex from cursor")
        self._btn_set_hyper_apex.setEnabled(False)
        self._btn_hyper_auto_fit = QPushButton("Auto-fit RDP")
        self._btn_hyper_auto_fit.setEnabled(False)
        self._btn_clear_hyper = QPushButton("Clear apex")
        self._btn_clear_hyper.setEnabled(False)

        def _toggle_hyperbola_controls(checked):
            self._spin_hyperbola_rdp.setEnabled(checked)
            self._spin_hyperbola_txrx.setEnabled(checked)
            self._btn_set_hyper_apex.setEnabled(checked)
            self._btn_hyper_auto_fit.setEnabled(checked)
            self._btn_clear_hyper.setEnabled(checked)
            self._redraw()

        self._chk_hyperbola.toggled.connect(_toggle_hyperbola_controls)
        self._spin_hyperbola_rdp.valueChanged.connect(self._on_hyperbola_rdp_changed)
        self._spin_hyperbola_txrx.valueChanged.connect(self._on_hyperbola_rdp_changed)
        self._btn_set_hyper_apex.clicked.connect(self._set_hyperbola_apex_from_cursor)
        self._btn_hyper_auto_fit.clicked.connect(self._auto_fit_hyperbola_rdp)
        self._btn_clear_hyper.clicked.connect(self._clear_hyperbola_apex)
        self._on_hyperbola_rdp_changed(self._spin_hyperbola_rdp.value())
        self._btn_save_gain_preset = QPushButton("Save Gain/Hyper Preset")
        self._btn_load_gain_preset = QPushButton("Load Gain/Hyper Preset")
        self._btn_save_gain_preset.clicked.connect(self._save_gain_hyper_preset)
        self._btn_load_gain_preset.clicked.connect(self._load_gain_hyper_preset)

        if HAS_MPL:
            self._bp_figure = Figure(figsize=(5.2, 3.0), tight_layout=True)
            self._bp_ax = self._bp_figure.add_subplot(111)
            self._bp_canvas = FigureCanvasQTAgg(self._bp_figure)
            self._bp_canvas.setMinimumHeight(280)
            self._bp_canvas.setMinimumWidth(360)
            self._bp_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self._bp_canvas.setToolTip(
                "Bandpass spectrum:\n"
                "click sinistro = low (MHz)\n"
                "click destro = high (MHz)\n"
                "click centrale = handle piu' vicino"
            )
            try:
                self._bp_canvas.mpl_connect("button_press_event", self._on_bp_hist_click)
            except Exception:
                pass
        else:
            self._bp_figure = None
            self._bp_ax = None
            self._bp_canvas = None

        self._filter_chain_widget = self._build_filter_chain_controls()
        self._connect_param_widgets()

        btn_apply = QPushButton("Applica")
        btn_apply.clicked.connect(self._apply_processing_async)
        btn_gain = QPushButton("Aggiorna gain")
        btn_gain.setToolTip("Applica solo Gain display senza rieseguire i filtri.")
        btn_gain.clicked.connect(self._apply_gain_only)
        btn_bar = QWidget()
        btn_bar_lay = QHBoxLayout(btn_bar)
        btn_bar_lay.setContentsMargins(0, 0, 0, 0)
        btn_bar_lay.setSpacing(6)
        btn_bar_lay.addWidget(btn_apply)
        btn_bar_lay.addWidget(btn_gain)
        fl.addRow(btn_bar)

        if self._is_qt_alive(self._filter_chain_widget):
            fl.addRow("Pipeline chain:", self._filter_chain_widget)
            fl.addRow("", self._build_add_filter_menu())

        fl.addRow("Trim start traces:",  self._spin_trim_start)
        fl.addRow("Trim end traces:",    self._spin_trim_end)
        fl.addRow("  apply trim:",       self._btn_apply_trim)
        fl.addRow("Flip profile X:",     self._chk_flip_profile)
        fl.addRow("Gain display:",       self._spin_gain)
        # TVG controls are exposed directly inside the pipeline step "Range gain (TVG)".
        # Do not duplicate those widgets here, otherwise Qt reparents them and the
        # controls disappear from the pipeline block.
        fl.addRow("Hyperbola fit:",      self._chk_hyperbola)
        fl.addRow("  RDP (εr):",     self._spin_hyperbola_rdp)
        fl.addRow("  offset Tx-Rx:",     self._spin_hyperbola_txrx)
        fl.addRow("  velocity:",         self._lbl_hyperbola_vel)
        fl.addRow("  depth target:",     self._lbl_hyperbola_depth)
        fl.addRow("  apex:",             self._btn_set_hyper_apex)
        fl.addRow("  auto-fit:",         self._btn_hyper_auto_fit)
        fl.addRow("  clear apex:",       self._btn_clear_hyper)
        fl.addRow(self._btn_save_gain_preset)
        fl.addRow(self._btn_load_gain_preset)

        # Timeslice
        grp_slice = QGroupBox("Timeslice")
        fl_slice  = QFormLayout(grp_slice)

        self._lbl_slice_profiles = QLabel("Nessun profilo caricato")
        self._le_slice_outdir = QLineEdit()
        self._le_slice_outdir.setPlaceholderText("Cartella output GeoTIFF timeslice")
        self._btn_slice_outdir = QPushButton("...")
        self._btn_slice_outdir.setToolTip("Seleziona cartella output per importazione slice.")
        self._btn_slice_outdir.clicked.connect(self._choose_slice_outdir)
        out_row = QWidget()
        out_lay = QHBoxLayout(out_row)
        out_lay.setContentsMargins(0, 0, 0, 0)
        out_lay.setSpacing(4)
        out_lay.addWidget(self._le_slice_outdir, 1)
        out_lay.addWidget(self._btn_slice_outdir, 0)

        self._spin_slice_thickness = QDoubleSpinBox()
        self._spin_slice_thickness.setRange(0.01, 2.0)
        self._spin_slice_thickness.setSingleStep(0.05)
        self._spin_slice_thickness.setDecimals(2)
        self._spin_slice_thickness.setValue(0.10)
        self._spin_slice_thickness.setSuffix(" m")
        self._spin_slice_thickness.setToolTip(
            "Spessore verticale di ogni timeslice (dz)."
        )
        self._chk_slice_thickness_locked = QCheckBox("Blocca spessore")
        self._chk_slice_thickness_locked.setChecked(False)
        self._chk_slice_thickness_locked.setToolTip(
            "Blocca il valore di spessore finche' non vuoi rigenerare le slice."
        )

        def _on_thickness_lock_toggled(locked: bool):
            self._spin_slice_thickness.setEnabled(not bool(locked))

        self._chk_slice_thickness_locked.toggled.connect(_on_thickness_lock_toggled)
        self._spin_slice_thickness.valueChanged.connect(lambda _v: self._request_preview_update(immediate=False))

        self._cb_slice_cmap = QComboBox()
        self._cb_slice_cmap.setIconSize(QSize(64, 14))
        _populate_cmap_combo(self._cb_slice_cmap, current=DEFAULT_CMAP)
        self._cb_slice_cmap.setToolTip("Colormap dedicata alla vista timeslice (indipendente dal radargramma).")
        self._lbl_slice_cmap_preview = QLabel()
        self._lbl_slice_cmap_preview.setFixedSize(160, 16)
        self._spin_slice_vmin_pct = QDoubleSpinBox()
        self._spin_slice_vmin_pct.setRange(0.0, 49.0)
        self._spin_slice_vmin_pct.setSingleStep(0.5)
        self._spin_slice_vmin_pct.setValue(2.0)
        self._spin_slice_vmin_pct.setToolTip("Percentile minimo per contrasto timeslice.")
        self._spin_slice_vmax_pct = QDoubleSpinBox()
        self._spin_slice_vmax_pct.setRange(51.0, 100.0)
        self._spin_slice_vmax_pct.setSingleStep(0.5)
        self._spin_slice_vmax_pct.setValue(98.0)
        self._spin_slice_vmax_pct.setToolTip("Percentile massimo per contrasto timeslice.")
        self._cb_slice_cmap.currentTextChanged.connect(lambda _v: self._redraw_slice_view(force=True))
        self._cb_slice_cmap.currentTextChanged.connect(self._update_cmap_previews)
        self._spin_slice_vmin_pct.valueChanged.connect(self._on_slice_clip_changed)
        self._spin_slice_vmax_pct.valueChanged.connect(self._on_slice_clip_changed)

        self._cb_slice_preset = QComboBox()
        self._cb_slice_preset.addItem("Base (default)", "base")
        self._cb_slice_preset.addItem("Qualita'", "quality")
        self._cb_slice_preset.addItem("Aggressivo", "aggressive")
        self._cb_slice_preset.setCurrentIndex(0)
        self._cb_slice_preset.setToolTip(
            "Preset rapido parametri timeslice.\n"
            "Base: stabile e veloce.\n"
            "Qualita': migliore continuita' e contrasto.\n"
            "Aggressivo: massimo recupero, piu' smoothing."
        )
        self._btn_slice_preset_apply = QPushButton("Apply preset")
        self._btn_slice_preset_apply.clicked.connect(self._apply_selected_timeslice_preset)

        self._cb_slice_extraction = QComboBox()
        self._cb_slice_extraction.addItem("LAS-like (abs)", "las_like")
        self._cb_slice_extraction.addItem("Envelope (Hilbert)", "envelope")
        self._cb_slice_extraction.addItem("Signed amplitude", "signed")
        self._cb_slice_extraction.setCurrentIndex(0)
        self._cb_slice_extraction.setToolTip(
            "Metodo di estrazione ampiezza per creare le timeslice.\n"
            "LAS-like usa abs(dato) e si comporta in modo piu' simile al flusso LAS."
        )

        self._chk_slice_use_processing = QCheckBox()
        self._chk_slice_use_processing.setChecked(True)
        self._chk_slice_use_processing.setToolTip(
            "Applica dewow/time-zero/bg/agc/bandpass prima della creazione slice.\n"
            "Disattivato = comportamento piu' vicino al LAS."
        )

        self._chk_slice_bg = QCheckBox()
        self._chk_slice_bg.setChecked(True)
        self._chk_slice_bg.setToolTip(
            "Background removal diretto prima dell'estrazione ampiezza.\n"
            "Se Processing pre-slice e' OFF, aiuta a rimuovere banding orizzontale."
        )
        self._cb_slice_bg_mode = QComboBox()
        self._cb_slice_bg_mode.addItem("line_by_line", "line_by_line")
        self._cb_slice_bg_mode.addItem("grid_by_grid", "grid_by_grid")
        self._cb_slice_bg_mode.setCurrentIndex(0)
        self._chk_slice_bg_auto = QCheckBox("Auto")
        self._chk_slice_bg_auto.setChecked(True)
        self._spin_slice_bg_window = QSpinBox()
        self._spin_slice_bg_window.setRange(8, 99000)
        self._spin_slice_bg_window.setValue(200)
        self._spin_slice_bg_window.setEnabled(False)
        self._spin_slice_bg_sample_start = QSpinBox()
        self._spin_slice_bg_sample_start.setRange(0, 9999)
        self._spin_slice_bg_sample_start.setValue(0)
        self._spin_slice_bg_sample_end = QSpinBox()
        self._spin_slice_bg_sample_end.setRange(0, 9999)
        self._spin_slice_bg_sample_end.setValue(0)

        def _toggle_slice_bg_auto(auto_checked):
            self._spin_slice_bg_window.setEnabled(bool(self._chk_slice_bg.isChecked()) and (not auto_checked))

        def _toggle_slice_bg(bg_checked):
            self._cb_slice_bg_mode.setEnabled(bg_checked)
            self._chk_slice_bg_auto.setEnabled(bg_checked)
            self._spin_slice_bg_sample_start.setEnabled(bg_checked)
            self._spin_slice_bg_sample_end.setEnabled(bg_checked)
            _toggle_slice_bg_auto(self._chk_slice_bg_auto.isChecked())

        self._chk_slice_bg.toggled.connect(_toggle_slice_bg)
        self._chk_slice_bg_auto.toggled.connect(_toggle_slice_bg_auto)
        _toggle_slice_bg(self._chk_slice_bg.isChecked())

        self._spin_slice_stack_n = QSpinBox()
        self._spin_slice_stack_n.setRange(1, 31)
        self._spin_slice_stack_n.setValue(1)
        self._spin_slice_stack_n.setToolTip(
            "Numero tracce per stacking laterale (1 = disattivato)."
        )
        self._cb_slice_stack_kernel = QComboBox()
        self._cb_slice_stack_kernel.addItem("Boxcar", "boxcar")
        self._cb_slice_stack_kernel.addItem("Triangolare", "triangular")
        self._cb_slice_stack_kernel.setCurrentIndex(0)

        self._cb_slice_flip_mode = QComboBox()
        self._cb_slice_flip_mode.addItem("None", "none")
        self._cb_slice_flip_mode.addItem("Flip all profiles", "all")
        self._cb_slice_flip_mode.addItem("Flip odd profiles", "odd")
        self._cb_slice_flip_mode.addItem("Flip even profiles", "even")
        self._cb_slice_flip_mode.setCurrentIndex(0)
        self._cb_slice_flip_mode.setToolTip(
            "Inverte la direzione tracce prima del calcolo timeslice.\n"
            "Utile per uniformare profili acquisiti avanti/indietro."
        )

        self._chk_slice_topographic = QCheckBox()
        self._chk_slice_topographic.setChecked(False)
        self._chk_slice_topographic.setToolTip(
            "Correzione topografica: allinea la finestra depth per quota traccia."
        )
        self._cb_slice_topo_ref_mode = QComboBox()
        self._cb_slice_topo_ref_mode.addItem("Median surface", "median")
        self._cb_slice_topo_ref_mode.addItem("Mean surface", "mean")
        self._cb_slice_topo_ref_mode.addItem("Min surface", "min")
        self._cb_slice_topo_ref_mode.addItem("Max surface", "max")
        self._cb_slice_topo_ref_mode.addItem("Custom elevation", "custom")
        self._cb_slice_topo_ref_mode.setCurrentIndex(0)
        self._cb_slice_topo_ref_mode.setEnabled(False)
        self._spin_slice_topo_ref_custom = QDoubleSpinBox()
        self._spin_slice_topo_ref_custom.setDecimals(3)
        self._spin_slice_topo_ref_custom.setRange(-10000.0, 100000.0)
        self._spin_slice_topo_ref_custom.setSingleStep(0.1)
        self._spin_slice_topo_ref_custom.setValue(0.0)
        self._spin_slice_topo_ref_custom.setEnabled(False)

        def _toggle_topo_controls():
            on = bool(self._chk_slice_topographic.isChecked())
            self._cb_slice_topo_ref_mode.setEnabled(on)
            is_custom = str(self._cb_slice_topo_ref_mode.currentData() or "") == "custom"
            self._spin_slice_topo_ref_custom.setEnabled(on and is_custom)

        self._chk_slice_topographic.toggled.connect(lambda _v: _toggle_topo_controls())
        self._cb_slice_topo_ref_mode.currentIndexChanged.connect(lambda _v: _toggle_topo_controls())
        _toggle_topo_controls()

        self._chk_normalize_ch = QCheckBox(); self._chk_normalize_ch.setChecked(False)
        self._chk_normalize_ch.setToolTip("Bilancia ampiezza inter-canale (LAS-like: normalmente OFF).")

        self._chk_amplitude_filter = QCheckBox(); self._chk_amplitude_filter.setChecked(False)
        self._spin_amplitude_sigma = QDoubleSpinBox()
        self._spin_amplitude_sigma.setRange(1.0, 10.0); self._spin_amplitude_sigma.setSingleStep(0.5)
        self._spin_amplitude_sigma.setValue(3.0); self._spin_amplitude_sigma.setEnabled(False)
        self._chk_amplitude_filter.toggled.connect(self._spin_amplitude_sigma.setEnabled)

        self._cb_slice_idw_mode = QComboBox()
        self._cb_slice_idw_mode.addItem("Fast (kNN)", "fast")
        self._cb_slice_idw_mode.addItem("Quality (radius)", "quality")
        self._cb_slice_idw_mode.setCurrentIndex(1)
        self._cb_slice_idw_mode.setToolTip(
            "Fast: molto piu' veloce su griglie grandi (kNN vettorizzato).\n"
            "Quality: ricerca entro raggio per cella (piu' lenta ma piu' fedele localmente)."
        )
        self._spin_slice_search_radius = QDoubleSpinBox()
        self._spin_slice_search_radius.setRange(0.0, 100.0)
        self._spin_slice_search_radius.setDecimals(3)
        self._spin_slice_search_radius.setSingleStep(0.05)
        self._spin_slice_search_radius.setValue(0.0)
        self._spin_slice_search_radius.setSuffix(" m")
        self._spin_slice_search_radius.setToolTip(
            "Search radius IDW in metri.\n"
            "0 = automatico (stimato dai dati)."
        )
        self._cb_slice_pipeline_source = QComboBox()
        self._cb_slice_pipeline_source.addItem("Auto (Hilbert -> ultimo attivo)", "auto")
        self._cb_slice_pipeline_source.addItem("Hilbert (Envelope)", "envelope")
        self._cb_slice_pipeline_source.addItem("AGC", "agc")
        self._cb_slice_pipeline_source.addItem("Bandpass", "bandpass")
        self._cb_slice_pipeline_source.addItem("BG removal", "bg_removal")
        self._cb_slice_pipeline_source.addItem("Time-zero", "timezero")
        self._cb_slice_pipeline_source.addItem("Dewow", "dewow")
        self._cb_slice_pipeline_source.addItem("RAW (nessun filtro)", "raw")
        self._cb_slice_pipeline_source.addItem("Manuale (usa opzioni avanzate)", "manual")
        self._cb_slice_pipeline_source.setCurrentIndex(0)
        self._cb_slice_pipeline_source.setToolTip(
            "Sorgente dati per timeslice basata sulla pipeline di Elaborazione.\n"
            "Auto: usa Hilbert se attivo, altrimenti l'ultimo step attivo."
        )
        self._cb_slice_pipeline_source.currentIndexChanged.connect(
            lambda _v: self._sync_slice_source_mode()
        )
        self._chk_slice_use_hilbert = QCheckBox()
        self._chk_slice_use_hilbert.setChecked(True)
        self._chk_slice_use_hilbert.setToolTip(
            "Usa l'envelope Hilbert come ampiezza di default per stabilizzare le timeslice."
        )
        self._spin_slice_overlap_pct = QSpinBox()
        self._spin_slice_overlap_pct.setRange(0, 90)
        self._spin_slice_overlap_pct.setSingleStep(5)
        self._spin_slice_overlap_pct.setValue(50)
        self._spin_slice_overlap_pct.setSuffix(" %")
        self._spin_slice_overlap_pct.setToolTip(
            "Overlap tra slice consecutive. 50% migliora la continuita' verticale."
        )
        self._spin_slice_blanking_m = QDoubleSpinBox()
        self._spin_slice_blanking_m.setRange(0.0, 100.0)
        self._spin_slice_blanking_m.setDecimals(3)
        self._spin_slice_blanking_m.setSingleStep(0.05)
        self._spin_slice_blanking_m.setValue(0.30)
        self._spin_slice_blanking_m.setSuffix(" m")
        self._spin_slice_blanking_m.setToolTip(
            "Celle oltre questa distanza dai dati reali diventano NoData (0=disattiva).\n"
            "E' indipendente dal Search radius e ha sempre priorita': se e' piu' piccolo\n"
            "del radius, limita comunque la copertura anche aumentando il radius."
        )
        self._spin_slice_idw_power = QSpinBox()
        self._spin_slice_idw_power.setRange(1, 4)
        self._spin_slice_idw_power.setValue(2)
        self._spin_slice_idw_power.setToolTip(
            "Esponente potenza IDW (p). 2 standard, 3-4 enfatizza dettagli locali."
        )

        self._chk_anisotropic_idw = QCheckBox(); self._chk_anisotropic_idw.setChecked(False)
        self._chk_auto_radius     = QCheckBox(); self._chk_auto_radius.setChecked(True)
        self._chk_fill_nodata     = QCheckBox(); self._chk_fill_nodata.setChecked(True)
        self._chk_slice_balance_profiles = QCheckBox(); self._chk_slice_balance_profiles.setChecked(True)
        self._chk_slice_balance_profiles.setToolTip(
            "Bilancia l'ampiezza media tra profili prima dell'IDW per ridurre le strisce."
        )
        self._spin_slice_depth_radius_factor = QDoubleSpinBox()
        self._spin_slice_depth_radius_factor.setRange(0.0, 2.0)
        self._spin_slice_depth_radius_factor.setSingleStep(0.1)
        self._spin_slice_depth_radius_factor.setValue(0.6)
        self._spin_slice_depth_radius_factor.setToolTip(
            "Aumenta progressivamente il raggio IDW con la profondita'. 0 = disattivo."
        )
        self._spin_slice_min_points = QSpinBox()
        self._spin_slice_min_points.setRange(1, 12)
        self._spin_slice_min_points.setValue(1)
        self._spin_slice_min_points.setToolTip(
            "Punti minimi per stimare una cella IDW (LAS default = 1)."
        )

        self._chk_slice_parallel_profiles = QCheckBox()
        self._chk_slice_parallel_profiles.setChecked(True)
        self._chk_slice_parallel_profiles.setToolTip(
            "Elabora i profili in parallelo durante la creazione timeslice/volume."
        )
        self._spin_slice_profile_workers = QSpinBox()
        self._spin_slice_profile_workers.setRange(0, max(0, int(os.cpu_count() or 32)))
        self._spin_slice_profile_workers.setValue(0)
        self._spin_slice_profile_workers.setToolTip(
            "Numero worker per profili (0 = automatico)."
        )
        self._spin_slice_profile_workers.setEnabled(True)

        def _toggle_profile_workers(checked):
            self._spin_slice_profile_workers.setEnabled(bool(checked))

        self._chk_slice_parallel_profiles.toggled.connect(_toggle_profile_workers)
        _toggle_profile_workers(self._chk_slice_parallel_profiles.isChecked())

        self._chk_smooth = QCheckBox(); self._chk_smooth.setChecked(True)
        self._spin_smooth_sigma = QDoubleSpinBox()
        self._spin_smooth_sigma.setRange(0.5, 10.0); self._spin_smooth_sigma.setSingleStep(0.1)
        self._spin_smooth_sigma.setValue(0.8); self._spin_smooth_sigma.setEnabled(True)
        self._chk_smooth.toggled.connect(self._spin_smooth_sigma.setEnabled)

        fl_slice.addRow("Profili:",            self._lbl_slice_profiles)
        fl_slice.addRow("Output folder:",      out_row)
        fl_slice.addRow("Spessore slice:",     self._spin_slice_thickness)
        fl_slice.addRow("  lock:",             self._chk_slice_thickness_locked)
        fl_slice.addRow("Interpolazione:",     self._cb_slice_idw_mode)
        fl_slice.addRow("Search radius (m):",  self._spin_slice_search_radius)
        fl_slice.addRow("Sorgente pipeline:",  self._cb_slice_pipeline_source)

        self._btn_slice_advanced = QToolButton()
        self._btn_slice_advanced.setCheckable(True)
        self._btn_slice_advanced.setChecked(False)
        self._btn_slice_advanced.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self._btn_slice_advanced.setText("▶ Avanzato")

        # "raggio auto" non ha piu' effetto pratico (il raggio IDW viene gia'
        # stimato dai dati quando Search radius = 0): widget tenuto in vita per
        # compatibilita' con i preset, ma non piu' mostrato in UI.
        self._chk_auto_radius.setVisible(False)

        self._slice_advanced_host = QWidget()
        fl_slice_adv = QFormLayout(self._slice_advanced_host)
        fl_slice_adv.setContentsMargins(0, 0, 0, 0)
        fl_slice_adv.setSpacing(4)
        fl_slice_adv.addRow("Preset:",             self._cb_slice_preset)
        fl_slice_adv.addRow("  azione:",           self._btn_slice_preset_apply)
        fl_slice_adv.addRow("Slice colormap:",     self._cb_slice_cmap)
        fl_slice_adv.addRow("  anteprima:",        self._lbl_slice_cmap_preview)
        fl_slice_adv.addRow("Slice clip min %:",   self._spin_slice_vmin_pct)
        fl_slice_adv.addRow("Slice clip max %:",   self._spin_slice_vmax_pct)
        fl_slice_adv.addRow("Potenza IDW (p):",    self._spin_slice_idw_power)
        fl_slice_adv.addRow("Overlap slice:",      self._spin_slice_overlap_pct)
        fl_slice_adv.addRow("Blanking (m):",       self._spin_slice_blanking_m)
        fl_slice_adv.addRow("BG pre-slice:",       self._chk_slice_bg)
        fl_slice_adv.addRow("Bilancia profili:",   self._chk_slice_balance_profiles)
        fl_slice_adv.addRow("Fill NoData:",        self._chk_fill_nodata)
        fl_slice_adv.addRow("Smooth gaussiano:",   self._chk_smooth)
        fl_slice_adv.addRow("  sigma:",            self._spin_smooth_sigma)

        # --- Esperto: parametri di fine-tuning usati raramente. Restano allo
        # stesso valore di default (= comportamento attuale) finche' questa
        # sezione non viene esplicitamente abilitata; i preset "Qualita'" e
        # "Aggressivo" la abilitano automaticamente perche' la richiedono.
        self._btn_slice_expert = QToolButton()
        self._btn_slice_expert.setCheckable(True)
        self._btn_slice_expert.setChecked(False)
        self._btn_slice_expert.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self._btn_slice_expert.setText("▶ Esperto")
        self._btn_slice_expert.setToolTip(
            "Parametri di fine-tuning raramente necessari.\n"
            "Finche' non e' attivo, questi campi vengono ignorati e si usa il\n"
            "comportamento di default (= come se questa sezione non esistesse)."
        )

        self._slice_expert_host = QWidget()
        fl_slice_expert = QFormLayout(self._slice_expert_host)
        fl_slice_expert.setContentsMargins(0, 0, 0, 0)
        fl_slice_expert.setSpacing(4)
        fl_slice_expert.addRow("Estrazione:",         self._cb_slice_extraction)
        fl_slice_expert.addRow("Usa Hilbert:",        self._chk_slice_use_hilbert)
        fl_slice_expert.addRow("Processing pre-slice:", self._chk_slice_use_processing)
        fl_slice_expert.addRow("  modo BG:",          self._cb_slice_bg_mode)
        fl_slice_expert.addRow("  auto BG:",          self._chk_slice_bg_auto)
        fl_slice_expert.addRow("  finestra BG:",      self._spin_slice_bg_window)
        fl_slice_expert.addRow("  BG da campione:",   self._spin_slice_bg_sample_start)
        fl_slice_expert.addRow("  BG a campione:",    self._spin_slice_bg_sample_end)
        fl_slice_expert.addRow("Trace stacking:",     self._spin_slice_stack_n)
        fl_slice_expert.addRow("  kernel:",           self._cb_slice_stack_kernel)
        fl_slice_expert.addRow("Flip pre-slice:",     self._cb_slice_flip_mode)
        fl_slice_expert.addRow("Topographic corr.:",  self._chk_slice_topographic)
        fl_slice_expert.addRow("  topo reference:",   self._cb_slice_topo_ref_mode)
        fl_slice_expert.addRow("  custom elev:",      self._spin_slice_topo_ref_custom)
        fl_slice_expert.addRow("Normalizza canali:",  self._chk_normalize_ch)
        fl_slice_expert.addRow("Filtro ampiezza:",    self._chk_amplitude_filter)
        fl_slice_expert.addRow("  sigma:",            self._spin_amplitude_sigma)
        fl_slice_expert.addRow("IDW anisotropo:",     self._chk_anisotropic_idw)
        fl_slice_expert.addRow("Raggio vs profondita':", self._spin_slice_depth_radius_factor)
        fl_slice_expert.addRow("  min points:",       self._spin_slice_min_points)
        fl_slice_expert.addRow("Profili paralleli:",  self._chk_slice_parallel_profiles)
        fl_slice_expert.addRow("  workers (0=auto):", self._spin_slice_profile_workers)
        self._slice_expert_host.setVisible(False)

        def _toggle_slice_expert(checked: bool):
            self._slice_expert_host.setVisible(bool(checked))
            self._btn_slice_expert.setText(("▼ Esperto" if checked else "▶ Esperto"))

        self._btn_slice_expert.toggled.connect(_toggle_slice_expert)
        fl_slice_adv.addRow(self._btn_slice_expert)
        fl_slice_adv.addRow(self._slice_expert_host)
        self._slice_advanced_host.setVisible(False)

        def _toggle_slice_advanced(checked: bool):
            self._slice_advanced_host.setVisible(bool(checked))
            self._btn_slice_advanced.setText(("▼ Avanzato" if checked else "▶ Avanzato"))

        self._btn_slice_advanced.toggled.connect(_toggle_slice_advanced)
        fl_slice.addRow(self._btn_slice_advanced)
        fl_slice.addRow(self._slice_advanced_host)

        # Ogni controllo che influenza il calcolo della timeslice deve
        # aggiornare il preview live: senza questo, cambiare es. Blanking o
        # Search radius non ha alcun effetto visibile finche' non si
        # disattiva/riattiva il Preview a mano.
        _preview_value_widgets = (
            self._cb_slice_idw_mode, self._spin_slice_search_radius, self._cb_slice_pipeline_source,
            self._spin_slice_idw_power, self._spin_slice_blanking_m,
            self._chk_slice_bg, self._chk_slice_balance_profiles, self._chk_fill_nodata,
            self._chk_smooth, self._spin_smooth_sigma, self._btn_slice_expert,
            self._cb_slice_extraction, self._chk_slice_use_hilbert, self._chk_slice_use_processing,
            self._cb_slice_bg_mode, self._chk_slice_bg_auto, self._spin_slice_bg_window,
            self._spin_slice_bg_sample_start, self._spin_slice_bg_sample_end,
            self._spin_slice_stack_n, self._cb_slice_stack_kernel, self._cb_slice_flip_mode,
            self._chk_slice_topographic, self._cb_slice_topo_ref_mode, self._spin_slice_topo_ref_custom,
            self._chk_normalize_ch, self._chk_amplitude_filter, self._spin_amplitude_sigma,
            self._chk_anisotropic_idw, self._spin_slice_depth_radius_factor, self._spin_slice_min_points,
            self._chk_slice_parallel_profiles, self._spin_slice_profile_workers,
        )
        for _w in _preview_value_widgets:
            if isinstance(_w, (QSpinBox, QDoubleSpinBox)):
                _w.valueChanged.connect(lambda _v: self._request_preview_update(immediate=False))
            elif isinstance(_w, QComboBox):
                _w.currentIndexChanged.connect(lambda _v: self._request_preview_update(immediate=False))
            elif isinstance(_w, (QCheckBox, QToolButton)):
                _w.toggled.connect(lambda _v: self._request_preview_update(immediate=False))

        self._cb_preview_quality = QComboBox()
        self._cb_preview_quality.addItem("Bassa", "low")
        self._cb_preview_quality.addItem("Media", "medium")
        self._cb_preview_quality.addItem("Alta", "high")
        self._cb_preview_quality.setCurrentIndex(1)
        self._cb_preview_quality.setToolTip(
            "Qualita' preview RAM: bassa=veloce, alta=piu' dettaglio."
        )
        self._cb_preview_quality.currentIndexChanged.connect(lambda _v: self._request_preview_update(immediate=False))
        fl_slice.addRow("Qualita' preview:", self._cb_preview_quality)

        btn_preview = QPushButton("Preview")
        btn_preview.setCheckable(True)
        btn_preview.setChecked(False)
        btn_preview.setToolTip(
            "Anteprima in RAM della slice corrente (non crea file, non aggiorna catalogo)."
        )
        btn_preview.toggled.connect(self._on_toggle_preview_mode)
        self._btn_preview_slice = btn_preview
        fl_slice.addRow(btn_preview)

        self._lbl_preview_status = QLabel("")
        self._lbl_preview_status.setVisible(False)
        self._progress_preview = QProgressBar()
        self._progress_preview.setRange(0, 0)  # indeterminato: nessuna % reale disponibile
        self._progress_preview.setMaximumHeight(10)
        self._progress_preview.setTextVisible(False)
        self._progress_preview.setVisible(False)
        fl_slice.addRow(self._lbl_preview_status)
        fl_slice.addRow(self._progress_preview)

        btn_slice = QPushButton("Salva gruppo")
        btn_slice.setToolTip(
            "Esegue il calcolo completo su disco e registra il gruppo timeslice nel catalogo."
        )
        btn_slice.clicked.connect(self._open_slice_dialog)
        fl_slice.addRow(btn_slice)

        self._lbl_slice_nav = QLabel("\u2014 nessuna slice \u2014")
        self._lbl_slice_nav.setAlignment(Qt.AlignCenter)
        self._lbl_slice_depth_top = QLabel("0.00 m")
        self._lbl_slice_depth_bottom = QLabel("\u2014")
        self._lbl_slice_depth_top.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self._lbl_slice_depth_bottom.setAlignment(Qt.AlignLeft | Qt.AlignBottom)
        depth_row = QWidget()
        depth_layout = QHBoxLayout(depth_row)
        depth_layout.setContentsMargins(0, 0, 0, 0)
        depth_layout.setSpacing(8)
        depth_layout.addWidget(self._lbl_slice_depth_top)
        depth_layout.addStretch(1)
        depth_layout.addWidget(self._lbl_slice_depth_bottom)

        fl_slice.addRow("Navigator:", self._lbl_slice_nav)
        fl_slice.addRow(depth_row)

        btn_slice_import = QPushButton("Import to Canvas")
        btn_slice_import.setToolTip(
            "Importa i GeoTIFF timeslice dalla cartella output nel gruppo 'GPR Timeslices'."
        )
        btn_slice_import.clicked.connect(self._import_slices_to_canvas)
        fl_slice.addRow(btn_slice_import)

        btn_view3d = QPushButton("Apri Viewer 3D...")
        btn_view3d.setToolTip(
            "Costruisce un volume 3D dalle timeslice interpolate e apre il viewer PyVista."
        )
        btn_view3d.clicked.connect(self._open_3d_viewer)
        fl_slice.addRow(btn_view3d)

        self._chk_slice_open_3d_on_complete = QCheckBox()
        self._chk_slice_open_3d_on_complete.setChecked(False)
        self._chk_slice_open_3d_on_complete.setToolTip(
            "Se attivo, apre automaticamente il viewer 3D al termine di 'Salva gruppo'."
        )
        fl_slice.addRow("Apri 3D al termine:", self._chk_slice_open_3d_on_complete)

        self._chk_timeslice_sync = QCheckBox()
        self._chk_timeslice_sync.setChecked(False)
        self._chk_timeslice_sync.setToolTip(
            "Sincronizza la timeslice visibile nel canvas QGIS con la profondita' "
            "del cursore sul radargramma e mostra un crosshair rosso X/Y."
        )
        self._spin_crosshair_size = QDoubleSpinBox()
        self._spin_crosshair_size.setRange(0.5, 100.0)
        self._spin_crosshair_size.setSingleStep(0.5)
        self._spin_crosshair_size.setValue(5.0)
        self._spin_crosshair_size.setEnabled(False)
        self._spin_crosshair_size.setToolTip("Dimensione del crosshair (metri, semi-lunghezza).")

        def _on_toggle_timeslice_sync(checked):
            self._spin_crosshair_size.setEnabled(bool(checked))
            if checked:
                self._canvas_timer.start()
            else:
                self._clear_timeslice_crosshair()

        self._chk_timeslice_sync.toggled.connect(_on_toggle_timeslice_sync)
        self._spin_crosshair_size.valueChanged.connect(
            lambda v: setattr(self, "_crosshair_size_m", float(v))
        )

        fl_slice.addRow("Sync timeslice canvas:", self._chk_timeslice_sync)
        fl_slice.addRow("  crosshair size (m):", self._spin_crosshair_size)

        # --- Fase D: single SliceSpec form ---------------------------------
        # One always-visible form (slice_spec_panel.SliceSpecPanel) drives every
        # generation parameter. The scattered legacy knobs stay constructed (so
        # presets / persistence keep working) but are hidden; get_slice_params()
        # now reads the panel. Remove the hidden rows once the migration settles.
        self._slice_spec_panel = None
        try:
            from .slice_spec_panel import SliceSpecPanel

            self._slice_spec_panel = SliceSpecPanel()
            self._slice_spec_panel.spec_changed.connect(
                lambda: self._request_preview_update(immediate=False)
            )
            fl_slice.insertRow(2, "Parametri slice:", self._slice_spec_panel)
            for _w in (
                self._spin_slice_thickness, self._chk_slice_thickness_locked,
                self._cb_slice_idw_mode, self._spin_slice_search_radius,
                self._cb_slice_pipeline_source, self._btn_slice_advanced,
                self._slice_advanced_host,
            ):
                if _w is None:
                    continue
                _lbl = fl_slice.labelForField(_w)
                if _lbl is not None:
                    _lbl.hide()
                _w.hide()
        except Exception as _exc:
            self._slice_spec_panel = None
            print(f"[GPR viewer][WARN] SliceSpecPanel non inizializzato: {_exc}")

        proc_content = QWidget()
        proc_layout = QVBoxLayout(proc_content)
        proc_layout.setContentsMargins(0, 0, 0, 0)
        proc_layout.addWidget(grp)
        proc_layout.addStretch()

        processing_scroll = QScrollArea()
        processing_scroll.setWidgetResizable(True)
        processing_scroll.setWidget(proc_content)
        processing_scroll.setMinimumWidth(250)

        slice_content = QWidget()
        slice_layout = QVBoxLayout(slice_content)
        slice_layout.setContentsMargins(0, 0, 0, 0)
        slice_layout.addWidget(grp_slice)
        slice_layout.addStretch()

        timeslice_scroll = QScrollArea()
        timeslice_scroll.setWidgetResizable(True)
        timeslice_scroll.setWidget(slice_content)
        timeslice_scroll.setMinimumWidth(270)

        # Unified right panel: Process + Slicing + Variants tabs in a single dock.
        tabs = QTabWidget()
        tabs.addTab(processing_scroll, "Elaborazione")
        tabs.addTab(timeslice_scroll, "Slicing")
        try:
            variants_scroll = QScrollArea()
            variants_scroll.setWidgetResizable(True)
            variants_scroll.setWidget(self._build_variants_tab())
            variants_scroll.setMinimumWidth(250)
            tabs.addTab(variants_scroll, "Varianti")

            def _on_right_tab_changed(i):
                txt = tabs.tabText(i)
                if txt == "Varianti":
                    self._refresh_variants_list()
                elif txt == "Slicing":
                    self._refresh_slice_variant_choices()

            tabs.currentChanged.connect(_on_right_tab_changed)
        except Exception:
            pass
        tabs.setCurrentIndex(0)
        self._right_tabs = tabs
        try:
            default_out = self._default_slice_output_dir()
            if default_out:
                self._le_slice_outdir.setText(default_out)
        except Exception:
            pass
        self._update_slice_profile_info()
        try:
            dz0 = float(self._spin_slice_thickness.value())
        except Exception:
            dz0 = 0.10
        self._refresh_slice_catalog(
            str(self._le_slice_outdir.text() or "").strip(),
            dz=dz0,
            reset_index=True,
            redraw=False,
        )
        self._sync_slice_source_mode()
        self._sync_filter_chain_from_controls()
        # Return single widget for one dock; keep second None for backward compatibility.
        return tabs, None

    def _current_dt_ns(self) -> float:
        if not self._profiles:
            return 0.117
        try:
            dt_ns = float(getattr(self._profiles[self._prof_idx], "dt_ns", 0.117) or 0.117)
        except Exception:
            dt_ns = 0.117
        if not np.isfinite(dt_ns) or dt_ns <= 0.0:
            dt_ns = 0.117
        return dt_ns

    def _current_bp_source_data(self) -> Optional[np.ndarray]:
        src = self._raw_data if self._raw_data is not None else self._proc_data
        if src is None:
            return None
        arr = np.asarray(src, dtype=np.float64)
        if arr.ndim != 2 or arr.size <= 0:
            return None
        if not np.isfinite(arr).any():
            return None
        return arr

    def _bandpass_limits(self) -> tuple[float, float]:
        lo = float(self._spin_bp_lo.value())
        hi = float(self._spin_bp_hi.value())
        if not np.isfinite(lo):
            lo = 1.0
        if not np.isfinite(hi):
            hi = lo + 1.0
        if hi <= lo:
            hi = lo + 1.0
        return lo, hi

    def _refresh_bp_histogram(self):
        if (not HAS_MPL) or self._bp_ax is None or self._bp_canvas is None:
            return
        ax = self._bp_ax
        ax.clear()

        arr = self._current_bp_source_data()
        if arr is None:
            ax.text(0.5, 0.5, "Nessun dato", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            self._safe_draw_idle(self._bp_canvas)
            return

        n_s, n_t = int(arr.shape[0]), int(arr.shape[1])
        if n_s <= 4 or n_t <= 0:
            ax.text(0.5, 0.5, "Dati insufficienti", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            self._safe_draw_idle(self._bp_canvas)
            return

        dt_s = self._current_dt_ns() * 1e-9
        if dt_s <= 0:
            dt_s = 0.117e-9

        # Limit trace count for responsiveness on long profiles.
        if n_t > 64:
            idx = np.linspace(0, n_t - 1, 64, dtype=np.int64)
            arr_fft = arr[:, idx]
        else:
            arr_fft = arr

        freqs_mhz = np.fft.rfftfreq(n_s, d=dt_s) / 1e6
        spec = np.mean(np.abs(np.fft.rfft(arr_fft, axis=0)), axis=1)
        if not np.isfinite(spec).any():
            spec = np.zeros_like(freqs_mhz)
        else:
            spec = np.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)

        max_mhz = float(self._spin_bp_hi.maximum())
        finite_f = np.isfinite(freqs_mhz)
        valid = finite_f & (freqs_mhz >= 0.0) & (freqs_mhz <= max_mhz)
        if not valid.any():
            ax.text(0.5, 0.5, "Spettro non disponibile", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            self._safe_draw_idle(self._bp_canvas)
            return

        fx = freqs_mhz[valid]
        sy = spec[valid]
        ax.plot(fx, sy, color="#3d7ab5", lw=1.0)
        ax.fill_between(fx, sy, color="#3d7ab5", alpha=0.18)

        lo, hi = self._bandpass_limits()
        is_on = bool(self._chk_bp.isChecked())
        c_lo = "#e74c3c" if is_on else "#9aa0a6"
        c_hi = "#e74c3c" if is_on else "#9aa0a6"
        c_band = "#f39c12" if is_on else "#c8ccd2"
        ax.axvline(lo, color=c_lo, lw=1.4, ls="--")
        ax.axvline(hi, color=c_hi, lw=1.4, ls="--")
        ax.axvspan(lo, hi, color=c_band, alpha=0.12)

        ax.set_xlim(float(np.nanmin(fx)), float(np.nanmax(fx)))
        ax.set_xlabel("Freq (MHz)")
        ax.set_ylabel("Amp")
        ax.set_title("Bandpass")
        ax.grid(True, ls=":", lw=0.4, alpha=0.6)
        try:
            self._safe_draw_idle(self._bp_canvas)
        except Exception:
            pass

    def _on_bp_controls_changed(self, _checked: bool):
        self._refresh_bp_histogram()
        if self._raw_data is not None:
            self._apply_processing_async()

    def _on_bp_spin_changed(self, _value: float):
        lo = float(self._spin_bp_lo.value())
        hi = float(self._spin_bp_hi.value())
        if hi <= lo:
            if self.sender() is self._spin_bp_lo:
                hi = lo + 1.0
                self._spin_bp_hi.blockSignals(True)
                try:
                    self._spin_bp_hi.setValue(hi)
                finally:
                    self._spin_bp_hi.blockSignals(False)
            else:
                # Keep LOW stable when user edits HIGH; clamp HIGH above LOW.
                hi = lo + 1.0
                self._spin_bp_hi.blockSignals(True)
                try:
                    self._spin_bp_hi.setValue(min(float(self._spin_bp_hi.maximum()), hi))
                finally:
                    self._spin_bp_hi.blockSignals(False)
        self._refresh_bp_histogram()
        if self._raw_data is not None and bool(self._chk_bp.isChecked()):
            self._apply_processing_async()

    def _on_bp_hist_click(self, event):
        if event is None or event.xdata is None:
            return
        if self._bp_ax is None or event.inaxes != self._bp_ax:
            return

        freq = float(event.xdata)
        fmin = float(self._spin_bp_lo.minimum())
        fmax = float(self._spin_bp_hi.maximum())
        freq = float(np.clip(freq, fmin, fmax))
        lo, hi = self._bandpass_limits()
        btn = int(getattr(event, "button", 0) or 0)

        if btn == 1:
            lo = min(freq, hi - 1.0)
        elif btn == 3:
            hi = max(freq, lo + 1.0)
        elif btn == 2:
            if abs(freq - lo) <= abs(freq - hi):
                lo = min(freq, hi - 1.0)
            else:
                hi = max(freq, lo + 1.0)
        else:
            return

        self._spin_bp_lo.blockSignals(True)
        self._spin_bp_hi.blockSignals(True)
        try:
            self._spin_bp_lo.setValue(float(np.clip(lo, fmin, fmax)))
            self._spin_bp_hi.setValue(float(np.clip(hi, fmin, fmax)))
        finally:
            self._spin_bp_lo.blockSignals(False)
            self._spin_bp_hi.blockSignals(False)

        self._refresh_bp_histogram()
        if self._raw_data is not None:
            self._apply_processing_async()

    def import_files(self):
        # Il progetto (con GPR_Processing e cartelle Project Manager) esiste gia':
        # l'import non chiede piu' dove creare le cartelle di lavoro.
        if not self._gpr_processing_paths():
            ret = QMessageBox.question(
                self, "Importa profili GPR",
                "Nessun progetto GPR attivo. Vuoi crearne uno nuovo adesso?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
            )
            if ret != QMessageBox.Yes or not self._new_gpr_project():
                return
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Importa profili GPR", "",
            import_dialog_filter(),
        )
        if not paths:
            return
        self._ensure_gpr_processing_layout()
        staged_paths = [self._stage_profile_in_project_raw(p) for p in paths]
        # Nuova sessione import: reset completo stato+UI per evitare riferimenti stale a widget Qt.
        self._reset_session_state()
        errors = []
        imported_warnings = []
        newly_imported = []
        for p in staged_paths:
            md5_failed = False
            try:
                prof = detect_and_read(p, verify_md5=True)
                self._profiles.append(prof)
                newly_imported.append(prof)
            except Exception as e:
                msg = str(e)
                if "MD5" in msg:
                    try:
                        prof = detect_and_read(p, verify_md5=False)
                        self._profiles.append(prof)
                        newly_imported.append(prof)
                        md5_failed = True
                        imported_warnings.append(
                            f"{os.path.basename(p)}: verifica MD5 fallita, importato comunque "
                            "(possibile schema MD5 non standard o file alterato)."
                        )
                    except Exception as e2:
                        errors.append(f"{os.path.basename(p)}: {e2}")
                        continue
                else:
                    errors.append(f"{os.path.basename(p)}: {msg}")
                    continue
            for w in list(getattr(prof, "parse_warnings", []) or []):
                # Evita doppioni: il warning MD5 esteso e' gia' mostrato sopra.
                if md5_failed and "MD5 mismatch" in str(w):
                    continue
                imported_warnings.append(f"{os.path.basename(p)}: {w}")
        if errors:
            QMessageBox.warning(self, "Errori import", "\n".join(errors))
        if imported_warnings:
            dedup = list(dict.fromkeys(imported_warnings))
            QMessageBox.information(
                self,
                "Warning import OGPR",
                "\n".join(dedup[:12]) + (
                    f"\n... altri {len(dedup) - 12} warning"
                    if len(dedup) > 12
                    else ""
                ),
            )
        self._update_slice_profile_info()
        if self._profiles:
            self._prof_idx = len(self._profiles) - 1
            self._load_current_profile()
            self._save_gpr_processing_project_meta()
        # --- Rilevamento assenza GPS: proponi posizionamento manuale ---
        if newly_imported and not check_profiles_have_gps(newly_imported):
            ret = QMessageBox.question(
                self,
                "Profili senza GPS rilevati",
                f"<b>{len(newly_imported)} profilo/i importato/i non contiene coordinate GPS/GEC.</b><br><br>"
                "I dati verranno trattati come acquisiti senza georeferenziazione GPS.<br>"
                "Puoi assegnare manualmente la geometria della griglia di rilievo adesso.<br><br>"
                "<b>Aprire ora il dialog di Posizionamento Griglia Manuale?</b>",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if ret == QMessageBox.Yes:
                self._open_manual_positioning_dialog(profiles_override=newly_imported)


    # ------------------------------------------------------------------
    # Posizionamento Manuale Griglia (Profili senza GPS)
    # ------------------------------------------------------------------

    def _open_manual_positioning_dialog(self, profiles_override: list | None = None):
        """Apre il dialog di posizionamento manuale della griglia GPR.

        Può essere chiamato:
        - Automaticamente dopo l'import di profili privi di GPS.
        - Manualmente dall'utente tramite il pulsante toolbar '📍 Posizionamento Griglia'.
        """
        profiles = list(profiles_override or self._profiles or [])
        if not profiles:
            QMessageBox.information(
                self,
                "Posizionamento Griglia",
                "Importa almeno un profilo .ogpr prima di aprire il posizionamento manuale.",
            )
            return

        # Recupera parametri salvati in precedenza (se disponibili) dal progetto
        saved_params = getattr(self, "_grid_positioning_params", None)
        if saved_params is None:
            try:
                meta = self._load_gpr_processing_project_meta()
                saved_params = meta.get("grid_positioning")
            except Exception:
                saved_params = None

        # Ottieni il canvas QGIS per la funzione "punta su mappa"
        canvas = None
        try:
            from qgis.utils import iface
            canvas = iface.mapCanvas() if iface else None
        except Exception:
            pass

        dlg = GprManualPositioningDialog(
            profiles=profiles,
            project_crs=None,
            canvas=canvas,
            parent=self,
        )

        # Pre-popola con i parametri salvati in precedenza
        if isinstance(saved_params, dict):
            try:
                gb = saved_params.get("grid_bounds", {})
                if gb:
                    dlg._spin_x0.setValue(float(gb.get("x0", 0.0)))
                    dlg._spin_x1.setValue(float(gb.get("x1", 10.0)))
                    dlg._spin_y0.setValue(float(gb.get("y0", 0.0)))
                    dlg._spin_y1.setValue(float(gb.get("y1", 10.0)))
                gr = saved_params.get("georef", {})
                if gr and gr.get("enabled"):
                    dlg._grp_geo.setChecked(True)
                    dlg._spin_e0.setValue(float(gr.get("origin_e", 0.0)))
                    dlg._spin_n0.setValue(float(gr.get("origin_n", 0.0)))
                    dlg._spin_azimuth.setValue(float(gr.get("azimuth_deg", 0.0)))
                nav = saved_params.get("navigation", {})
                if isinstance(nav, dict) and nav:
                    if hasattr(dlg, "_chk_real_length"):
                        dlg._chk_real_length.setChecked(bool(nav.get("use_real_length", False)))
                    if hasattr(dlg, "_spin_stagger"):
                        dlg._spin_stagger.setValue(float(nav.get("stagger_m", 0.0)))
                    if hasattr(dlg, "_cb_elev_mode"):
                        want = str(nav.get("elev_mode", "none") or "none")
                        j = dlg._cb_elev_mode.findData(want)
                        if j >= 0:
                            dlg._cb_elev_mode.setCurrentIndex(j)
                        dlg._spin_z_start.setValue(float(nav.get("z_start", 0.0) or 0.0))
                        dlg._spin_z_end.setValue(float(nav.get("z_end", 0.0) or 0.0))
                        if nav.get("elev_file"):
                            dlg._le_elev_file.setText(str(nav.get("elev_file")))
                        k = dlg._cb_elev_kind.findData(str(nav.get("elev_ref_kind", "x") or "x"))
                        if k >= 0:
                            dlg._cb_elev_kind.setCurrentIndex(k)
                    if hasattr(dlg, "_cb_pos_mode"):
                        pm = dlg._cb_pos_mode.findData(str(nav.get("pos_mode", "uniform") or "uniform"))
                        if pm >= 0:
                            dlg._cb_pos_mode.setCurrentIndex(pm)
                        try:
                            dlg._spin_pos_param.setValue(int(nav.get("pos_param", 0) or 0))
                        except (TypeError, ValueError):
                            pass
                    if hasattr(dlg, "_chk_reverse_far"):
                        dlg._chk_reverse_far.setChecked(
                            str(nav.get("reverse_anchor", "near") or "near").lower().startswith("far")
                        )

                # Ripristina lo stato per-profilo (verso zig-zag, asse, offset,
                # nome, lunghezza) salvato in precedenza — altrimenti si perde
                # a ogni riapertura del dialog.
                saved_profs = saved_params.get("profiles") or []
                items = getattr(dlg, "_profile_items", None)
                if saved_profs and isinstance(items, list) and items:
                    by_path = {}
                    for sp in saved_profs:
                        key = os.path.basename(str(sp.get("path", "") or ""))
                        if key:
                            by_path.setdefault(key, sp)
                    for i, it in enumerate(items):
                        key = os.path.basename(str(it.get("prof") and getattr(it["prof"], "path", "") or ""))
                        sp = by_path.get(key) or (saved_profs[i] if i < len(saved_profs) else None)
                        if not isinstance(sp, dict):
                            continue
                        if "reversed" in sp:
                            it["reversed"] = bool(sp["reversed"])
                        if sp.get("axis") in ("X", "Y"):
                            it["axis"] = sp["axis"]
                        if sp.get("offset") is not None:
                            it["offset"] = float(sp["offset"])
                        if sp.get("name"):
                            it["name"] = str(sp["name"])
                        try:
                            if float(sp.get("length", 0.0)) > 0.0:
                                it["length"] = float(sp["length"])
                        except (TypeError, ValueError):
                            pass
                    try:
                        dlg._update_table_from_items()
                        dlg._draw_preview()
                    except Exception:
                        pass
            except Exception:
                pass

        def _on_positioning_applied(result: dict):
            _dbg.log("VIEWER", f"posizionamento applicato a {len(result.get('profiles', []))} profilo/i; "
                     f"georef={result.get('georef')} nav={result.get('navigation')}")
            # la geometria e' cambiata: invalida le anteprime da variante in cache
            self._geometry_rev = int(getattr(self, "_geometry_rev", 0)) + 1
            self._preview_variant_cache = None
            # Salva parametri per il progetto
            self._grid_positioning_params = result
            try:
                self._save_gpr_processing_project_meta()
            except Exception:
                pass
            # Ricarica il profilo corrente per aggiornare le coordinate nel viewer
            try:
                self._load_current_profile()
                self._update_slice_profile_info()
            except Exception:
                pass
            self._safe_set_text(
                self._lbl_status,
                f"Posizionamento manuale applicato a {len(result.get('profiles', []))} profilo/i. "
                "Le coordinate sono ora disponibili per il viewer e lo slicing 3D.",
            )

        dlg.positioning_applied.connect(_on_positioning_applied)
        dlg.exec_()

    # ------------------------------------------------------------------
    # Navigazione
    # ------------------------------------------------------------------

    def _prev_profile(self):
        if self._profiles:
            self._prof_idx = max(0, self._prof_idx - 1)
            self._load_current_profile(deferred=True)

    def _next_profile(self):
        if self._profiles:
            self._prof_idx = min(len(self._profiles) - 1, self._prof_idx + 1)
            self._load_current_profile(deferred=True)

    def keyPressEvent(self, event):
        key = int(event.key()) if event is not None else 0
        if key == int(Qt.Key_Up):
            self._step_slice(-1)
            event.accept()
            return
        if key == int(Qt.Key_Down):
            self._step_slice(+1)
            event.accept()
            return
        super().keyPressEvent(event)

    def _on_channel_changed(self, idx):
        if idx >= 0:
            self._ch_idx = idx
            self._reload_data(deferred=True)
            # Auto reset zoom on channel change (same behavior as Reset Zoom button).
            self._reset_zoom()

    def _current_channel_elevation(self):
        """Quote per traccia del profilo/canale corrente, allineate al numero
        di tracce del dato grezzo. None se assenti."""
        try:
            prof = self._profiles[self._prof_idx]
            ch = prof.channel(self._ch_idx)
            alt = np.asarray(getattr(ch, "altitude", []), dtype=np.float64)
            if alt.size == 0 or not np.isfinite(alt).any():
                return None
            n_t = None
            if self._raw_data is not None and getattr(self._raw_data, "ndim", 0) == 2:
                n_t = int(self._raw_data.shape[1])
            if n_t:
                alt = self._channel_trace_vector(alt, n_t)
            return np.asarray(alt, dtype=np.float64).tolist()
        except Exception:
            return None

    @staticmethod
    def _profile_channel_elevation(prof, ci: int):
        try:
            alt = np.asarray(getattr(prof.channel(ci), "altitude", []), dtype=np.float64)
            if alt.size and np.isfinite(alt).any():
                return alt.tolist()
        except Exception:
            pass
        return None

    def _update_trim_controls_for_channel(self, n_traces: int):
        n_t = max(0, int(n_traces))
        max_trim = max(0, n_t - 1)
        cur_start = int(np.clip(self._spin_trim_start.value(), 0, max_trim))
        cur_end = int(np.clip(self._spin_trim_end.value(), 0, max_trim))
        if cur_start + cur_end > max_trim:
            cur_end = max(0, max_trim - cur_start)
        self._spin_trim_start.blockSignals(True)
        self._spin_trim_end.blockSignals(True)
        try:
            self._spin_trim_start.setRange(0, max_trim)
            self._spin_trim_end.setRange(0, max_trim)
            self._spin_trim_start.setValue(cur_start)
            self._spin_trim_end.setValue(cur_end)
        finally:
            self._spin_trim_start.blockSignals(False)
            self._spin_trim_end.blockSignals(False)
        self._trim_start_traces = int(cur_start)
        self._trim_end_traces = int(cur_end)

    def _on_trim_traces_changed(self, _value):
        if not self._profiles:
            return
        start = int(self._spin_trim_start.value())
        end = int(self._spin_trim_end.value())
        max_trim = max(0, int(self._spin_trim_start.maximum()))
        if start + end > max_trim:
            if self.sender() is self._spin_trim_start:
                end = max(0, max_trim - start)
                self._spin_trim_end.blockSignals(True)
                try:
                    self._spin_trim_end.setValue(end)
                finally:
                    self._spin_trim_end.blockSignals(False)
            else:
                start = max(0, max_trim - end)
                self._spin_trim_start.blockSignals(True)
                try:
                    self._spin_trim_start.setValue(start)
                finally:
                    self._spin_trim_start.blockSignals(False)
        self._trim_start_traces = int(start)
        self._trim_end_traces = int(end)
        self._reload_data(deferred=True)

    def _load_current_profile(self, deferred: bool = False):
        if not self._profiles:
            self._update_slice_profile_info()
            return
        prof = self._profiles[self._prof_idx]
        # Reset zoom when changing profile to avoid carrying tiny previous windows.
        self._view_xlim = None
        self._view_ylim = None
        self._profile_view_initialized = False
        self._update_slice_profile_info()
        self._safe_set_text(
            self._lbl_profile,
            f"{self._prof_idx + 1}/{len(self._profiles)}  "
            f"{os.path.basename(prof.path)}"
        )
        if not self._is_qt_alive(self._cb_channel):
            return
        self._cb_channel.blockSignals(True)
        self._safe_cb_clear(self._cb_channel)
        for i in range(prof.n_channels):
            self._cb_channel.addItem(f"Ch {i}")

        # Seleziona automaticamente il canale con segnale migliore.
        best_idx = 0
        best_score = -1.0
        for i in range(prof.n_channels):
            try:
                score = self._channel_signal_score(prof.channel(i).data)
            except Exception as exc:
                self._note_swallowed(f"punteggio segnale del canale {i}", exc)
                score = -1.0
            if score > best_score:
                best_score = score
                best_idx = i
        self._ch_idx = int(best_idx)
        self._cb_channel.setCurrentIndex(self._ch_idx)
        try:
            self._cb_channel.blockSignals(False)
        except Exception:
            pass
        try:
            self._update_trim_controls_for_channel(int(prof.channel(self._ch_idx).data.shape[1]))
        except Exception as exc:
            self._note_swallowed("controlli di trim del canale", exc)
            self._update_trim_controls_for_channel(0)
        try:
            v_prof = self._profile_reference_velocity(prof)
            rdp_prof = float(np.clip(self._velocity_to_rdp(v_prof), 1.0, 40.0))
            self._spin_hyperbola_rdp.setValue(rdp_prof)
        except Exception:
            pass
        self._reload_data(deferred=deferred)
        # Auto reset zoom on profile change (same behavior as Reset Zoom button).
        self._reset_zoom()
        if self._ch_idx != 0:
            self._safe_set_text(
                self._lbl_status,
                f"Canale auto-selezionato: Ch {self._ch_idx} (segnale migliore)."
            )
        self._draw_profile_line_on_canvas()

    def _trace_indices_for(self, n_t: int) -> np.ndarray:
        """Indici delle tracce visibili (trim + flip) per un profilo di n_t tracce."""
        start = int(np.clip(self._trim_start_traces, 0, max(0, n_t - 1)))
        end_trim = int(np.clip(self._trim_end_traces, 0, max(0, n_t - 1)))
        end = min(max(start + 1, n_t - end_trim), n_t)
        idx = np.arange(start, end, dtype=np.int64)
        if bool(getattr(self, "_chk_flip_profile", None) and self._chk_flip_profile.isChecked()):
            idx = idx[::-1]
        if idx.size <= 0:
            idx = np.arange(0, min(1, n_t), dtype=np.int64)
        return idx

    def _grid_bg_reference(self, params: dict, channel_idx: int):
        """Traccia media globale per il background grid_by_grid (due passate).

        Prima passata: dewow + time-zero su tutti i profili, poi media di tutte le
        tracce (stessa logica dello slicer). Ritorna None se non serve."""
        if not (params.get("bg_removal") and str(params.get("bg_mode")) == "grid_by_grid"):
            return None
        if not self._profiles:
            return None
        pre_keys = ("dewow", "dewow_win", "timezero", "tz_method", "tz_mode", "tz_threshold",
                    "tz_backup_nsamp", "tz_fixed_sample")
        key = (
            tuple(str(getattr(p, "path", "")) for p in self._profiles), int(channel_idx),
            int(self._trim_start_traces), int(self._trim_end_traces),
            bool(getattr(self, "_chk_flip_profile", None) and self._chk_flip_profile.isChecked()),
            tuple((k, params.get(k)) for k in pre_keys),
        )
        cache = getattr(self, "_grid_bg_ref_cache", None)
        if cache and cache[0] == key:
            return cache[1]
        from .gpr_processing import apply_pre_bg_pipeline
        pres = []
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for prof in self._profiles:
                if channel_idx >= int(getattr(prof, "n_channels", 0) or 0):
                    continue
                d = np.asarray(prof.channel(channel_idx).data, dtype=np.float32)
                if d.ndim != 2 or d.size == 0:
                    continue
                d = d[:, self._trace_indices_for(d.shape[1])]
                try:
                    pre = apply_pre_bg_pipeline(d, params, dt_ns=getattr(prof, "dt_ns", 0.117))
                except Exception as exc:
                    # Il profilo esce dalla media del background: il riferimento
                    # grid_by_grid si basa su meno profili senza che si veda.
                    self._note_swallowed(
                        f"background grid_by_grid, profilo escluso "
                        f"({os.path.basename(str(getattr(prof, 'path', '?')))})",
                        exc,
                    )
                    continue
                if pre is not None and pre.ndim == 2 and pre.shape[0] > 0 and pre.shape[1] > 0:
                    pres.append(pre)
        finally:
            QApplication.restoreOverrideCursor()
        ref = None
        if pres:
            min_ns = min(t.shape[0] for t in pres)
            ref = np.concatenate([t[:min_ns, :] for t in pres], axis=1).mean(axis=1).astype(np.float64)
        self._grid_bg_ref_cache = (key, ref)
        return ref

    def _reload_data(self, deferred: bool = False):
        """Ricarica i dati grezzi del profilo/canale e rielabora.

        deferred=True: la pipeline gira in un thread (comandi interattivi: cambio
        canale, flip, trim, profilo precedente/successivo) e fino al risultato il
        dato mostrato e' considerato assente. deferred=False: sincrono, per chi
        legge `_proc_data` subito dopo (import, ripristino progetto, preset).
        """
        if not self._profiles:
            return
        if not self._is_qt_alive(self):
            return
        for w in (
            getattr(self, "_spin_trim_start", None),
            getattr(self, "_spin_trim_end", None),
            getattr(self, "_chk_flip_profile", None),
        ):
            if not self._is_qt_alive(w):
                return
        prof = self._profiles[self._prof_idx]
        ch   = prof.channel(self._ch_idx)
        full = ch.data.copy()
        n_t = int(full.shape[1]) if full.ndim == 2 else 0
        self._update_trim_controls_for_channel(n_t)
        idx = self._trace_indices_for(n_t)
        if full.ndim == 2 and idx.size > 0:
            self._raw_data = full[:, idx].copy()
        else:
            self._raw_data = full.copy()
        self._trace_source_indices = idx.astype(np.int64, copy=False)
        self._proc_data = None
        self._proc_pre_agc = False
        self._stage_cache = {}
        self._hyper_apex_x = None
        self._hyper_apex_depth = None
        if deferred:
            # `_disp_data` appartiene al profilo/canale precedente: cursore, A-scan e
            # pick lo trattano come 'nessun dato' finche' il nuovo non e' pronto.
            self._disp_data = None
            self._apply_processing_async()
        else:
            self._apply_processing()

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def _current_processing_params(self) -> dict:
        """Return current processing controls as pipeline params."""
        bg_auto = bool(self._chk_bg_auto.isChecked())
        bg_window = 0 if bg_auto else int(self._spin_bg_window.value())
        bg_sample_start = int(self._spin_bg_sample_start.value())
        bg_sample_end = int(self._spin_bg_sample_end.value())
        # Defensive UX: if end <= start, interpret as "to end" instead of empty range.
        if bg_sample_end > 0 and bg_sample_end <= bg_sample_start:
            bg_sample_end = 0
        bp_lo, bp_hi = self._bandpass_limits()
        rg_points = self._sanitize_range_gain_breakpoints(self._range_gain_breakpoints)
        # Filtri Fase 1: enable dalla casella del blocco chain, valori dai
        # controlli generici. Assenti -> restano ai default (OFF) di DEFAULT_PIPELINE.
        chain_enabled: dict = {}
        if self._is_qt_alive(getattr(self, "_filter_chain_widget", None)):
            try:
                chain_enabled = dict(self._filter_chain_widget.get_enabled_map() or {})
            except Exception:
                chain_enabled = {}
        phase1: dict = dict(self._phase1_params())
        for _sid in self._PHASE1_STEP_IDS:
            phase1[_sid] = bool(chain_enabled.get(_sid, False))
        # Migrazione Kirchhoff topografica: inietta le quote per traccia del
        # profilo/canale corrente (per il salvataggio variante ogni profilo
        # riceve le proprie quote in _save_processed_variant).
        if phase1.get("kirchhoff") and phase1.get("kirchhoff_topo"):
            phase1["kirchhoff_elev_m"] = self._current_channel_elevation()
        return {
            **phase1,
            "dewow": bool(self._chk_dewow.isChecked()),
            "dewow_win": int(self._spin_dewow.value()),
            "timezero": bool(self._chk_timezero.isChecked()),
            "tz_method": str(self._cb_tz_method.currentText()),
            "tz_mode": str(self._cb_tz_mode.currentText()),
            "tz_threshold": float(self._spin_tz_threshold.value()),
            "tz_backup_nsamp": int(self._spin_tz_backup.value()),
            "bg_removal": bool(self._chk_bg.isChecked()),
            "bg_mode": str(self._cb_bg_mode.currentData() or "line_by_line"),
            "bg_method": str(self._cb_bg_method.currentData() or "mean") if hasattr(self, "_cb_bg_method") else "mean",
            "bg_window": int(bg_window),
            "bg_sample_start": int(bg_sample_start),
            "bg_sample_end": int(bg_sample_end),
            "migration": bool(getattr(self, "_chk_mig", None) and self._chk_mig.isChecked()),
            "mig_velocity_m_s": float(self._spin_mig_vel.value()) if hasattr(self, "_spin_mig_vel") else 1e8,
            "mig_dx_m": float(self._spin_mig_dx.value()) if hasattr(self, "_spin_mig_dx") else 0.05,
            "agc": bool(self._chk_agc.isChecked()),
            "agc_win": int(self._spin_agc.value()),
            "agc_max_gain": float(self._spin_agc_max_gain.value()) if hasattr(self, "_spin_agc_max_gain") else 0.0,
            "agc_sample_start": int(self._spin_agc_sample_start.value()) if getattr(self, "_spin_agc_sample_start", None) else 0,
            "agc_lateral": int(self._spin_agc_lateral.value()) if getattr(self, "_spin_agc_lateral", None) else 0,
            "pre_agc_gain": bool(self._chk_range_gain.isChecked()),
            "pre_agc_surface_gain": float(self._spin_gain_surface.value()),
            "pre_agc_deep_gain": float(self._spin_gain_deep.value()),
            "pre_agc_breakpoints": [[float(x), float(y)] for x, y in rg_points.tolist()],
            "bandpass": bool(self._chk_bp.isChecked()),
            "envelope": bool(getattr(self, "_chk_envelope", None) and self._chk_envelope.isChecked()),
            "bp_low_mhz": float(bp_lo),
            "bp_high_mhz": float(bp_hi),
            "bp_taper_frac": (
                float(self._spin_bp_taper.value()) / 100.0
                if self._is_qt_alive(getattr(self, "_spin_bp_taper", None)) else 0.12
            ),
            # manual_gain is a display-only step applied in _compose_display_data,
            # not in the processing pipeline. Keep the key so has_active_filters
            # can trigger a pipeline re-run when manual_gain is the only active control.
            "manual_gain": bool(getattr(self, "_chk_manual_gain", None) and self._chk_manual_gain.isChecked()),
            "chain_order": self._current_chain_order(),
        }

    WORKING_VARIANT_NAME = "Lavoro"

    _PIPELINE_FILTER_KEYS = (
        "timezero", "dewow", "bg_removal", "bandpass", "migration", "agc",
        "envelope", "pre_agc_gain", "manual_gain", "dcshift", "boxcar", "notch",
        "whiten", "amp_eq", "rad_threshold", "power_gain", "eigen_filter",
        "median2d", "fk_dip", "kirchhoff", "inst_attr",
    )

    def _begin_processing(self, deferred: bool):
        """Prologo comune di _apply_processing / _apply_processing_async.

        Ritorna None se non c'e' nulla da eseguire (nessun dato, caso RAW,
        variante gia' su disco); altrimenti il job da passare ad apply_pipeline.
        `deferred=True` = il risultato arrivera' piu' tardi da un thread.
        """
        if self._raw_data is None:
            return None
        if not self._is_qt_alive(self):
            return None
        for w in (
            getattr(self, "_chk_dewow", None),
            getattr(self, "_spin_dewow", None),
            getattr(self, "_chk_timezero", None),
            getattr(self, "_chk_bg", None),
            getattr(self, "_chk_mig", None),
            getattr(self, "_chk_agc", None),
            getattr(self, "_spin_agc", None),
            getattr(self, "_chk_envelope", None),
        ):
            if not self._is_qt_alive(w):
                return None
        self._reveal_enabled_hidden_blocks()
        params = self._current_processing_params()
        self._schedule_recipe_autosave()
        # Ogni nuova richiesta rende obsoleto un eventuale risultato in volo.
        self._proc_job_seq += 1
        has_active_filters = any(bool(params.get(k, False)) for k in self._PIPELINE_FILTER_KEYS)
        if not has_active_filters:
            self._last_pipeline_params = dict(params)
            self._proc_data = np.array(self._raw_data, dtype=np.float32, copy=True)
            self._proc_pre_agc = False
            self._refresh_bp_histogram()
            self._apply_gain_only()
            self._safe_set_text(
                self._lbl_status,
                "Modalita' RAW: nessun filtro pipeline attivo.",
            )
            self._request_preview_update(immediate=False)
            return None
        if not self._profiles or self._prof_idx < 0 or self._prof_idx >= len(self._profiles):
            return None
        prof = self._profiles[self._prof_idx]
        dt_ns_val = float(getattr(prof, "dt_ns", 0.117) or 0.117)
        cached = self._load_variant_channel(
            prof, int(self._ch_idx),
            self._recipe_signature(params, self._view_geom_sig()),
            getattr(self._raw_data, "shape", None),
        )
        if cached is not None:
            self._last_pipeline_params = dict(params)
            self._proc_data = cached
            self._proc_pre_agc = bool(params.get("pre_agc_gain", False))
            self._refresh_bp_histogram()
            self._apply_gain_only()
            self._safe_set_text(
                self._lbl_status,
                "Dato processato caricato dalla variante salvata su disco.",
            )
            self._request_preview_update(immediate=False)
            return None
        return {
            "seq": int(self._proc_job_seq),
            "raw": self._raw_data,
            "params": params,
            "dt_ns": dt_ns_val,
            "bg_reference_trace": self._grid_bg_reference(params, int(self._ch_idx)),
            "chain_order": params.get("chain_order"),
            "stage_cache": self._stage_cache,
        }

    def _finish_processing(self, params: dict, result: np.ndarray) -> None:
        """Coda comune: valida il risultato e aggiorna vista e anteprima."""
        self._last_pipeline_params = dict(params)
        self._proc_data = result
        self._proc_pre_agc = bool(params.get("pre_agc_gain", False))

        # Fallback robusto: evita profilo "vuoto" quando il risultato e' quasi nullo o non finito.
        proc = np.asarray(self._proc_data)
        finite = np.isfinite(proc)
        finite_ratio = float(finite.mean()) if proc.size else 0.0
        spread = float(np.nanmax(proc) - np.nanmin(proc)) if finite.any() else 0.0
        fallback_msg = None
        if finite_ratio < 0.5 or spread < 1e-6:
            try:
                self._proc_data = np.asarray(self._raw_data, dtype=np.float32)
                self._proc_pre_agc = False  # dato grezzo: il range-gain NON e' incorporato
                fallback_msg = (
                    f"Processing inconcludente "
                    f"(finite={finite_ratio:.0%}, spread={spread:.2e}): "
                    "visualizzazione fallback su dato grezzo."
                )
                try:
                    QgsMessageLog.logMessage(
                        f"[GPR] _apply_processing fallback: {fallback_msg}",
                        "RasterLinker",
                        Qgis.Warning,
                    )
                except Exception:
                    print(f"[GPR] _apply_processing fallback: {fallback_msg}")
            except Exception:
                pass

        self._refresh_bp_histogram()
        self._apply_gain_only()
        if fallback_msg:
            # _apply_gain_only() ha appena sovrascritto lo stato con le
            # statistiche "proc: min/max..."; senza questo, l'avviso che la
            # pipeline e' stata scartata (fallback su dato grezzo) sparisce
            # subito e diventa impossibile capire perche' un filtro non ha
            # alcun effetto visibile.
            self._safe_set_text(self._lbl_status, fallback_msg)
        self._request_preview_update(immediate=False)

    def _show_processing_error(self, exc) -> None:
        log_exception("apply_pipeline", exc)
        QMessageBox.critical(
            self, "Errore processing",
            f"{exc}\n\nDettagli in: {LOG_FILE_PATH}",
        )

    def _apply_processing(self):
        """Esegue la pipeline nel thread della GUI (chi chiama si aspetta
        `_proc_data` pronto al ritorno). Per i comandi interattivi usare
        `_apply_processing_async`."""
        self._wait_processing_job()
        job = self._begin_processing(deferred=False)
        if job is None:
            return
        try:
            result = _run_pipeline_job(job)
        except Exception as e:
            self._show_processing_error(e)
            return
        self._finish_processing(job["params"], result)

    # -- versione asincrona (comandi interattivi) ------------------------

    def _apply_processing_async(self, *_args):
        """Come _apply_processing ma la pipeline gira in un QThread.

        La GUI resta libera durante filtri pesanti (kirchhoff, fk, eigen...).
        Se un job e' gia' in corso non lo si interrompe: se ne ricorda solo la
        richiesta e, a job finito, si riparte con lo stato piu' recente dei
        controlli scartando il risultato ormai vecchio.
        """
        if self._raw_data is None:
            return
        if self._proc_thread_running():
            self._proc_rerun = True
            return
        job = self._begin_processing(deferred=True)
        if job is None:
            return
        self._start_processing_thread(job)

    def _proc_thread_running(self) -> bool:
        thread = getattr(self, "_proc_thread", None)
        return bool(self._is_qt_alive(thread) and thread.isRunning())

    def _start_processing_thread(self, job: dict) -> None:
        self._set_processing_busy(True)
        thread = QThread(self)
        worker = _PipelineWorker(job)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_processing_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._proc_thread = thread
        self._proc_worker = worker
        self._proc_job = job
        thread.start()

    def _on_processing_finished(self, seq: int, result, error: str) -> None:
        job = self._proc_job
        self._proc_job = None
        self._set_processing_busy(False)
        rerun = bool(self._proc_rerun)
        self._proc_rerun = False
        if not self._is_qt_alive(self):
            return
        if rerun:
            # Nel frattempo i parametri sono cambiati: il risultato e' vecchio.
            # Il thread sta chiudendo (worker.finished -> quit): va atteso, se no
            # _apply_processing_async lo vedrebbe ancora attivo e perderebbe la richiesta.
            thread = self._proc_thread
            if self._is_qt_alive(thread) and thread.isRunning():
                thread.quit()
                thread.wait(5000)
            self._apply_processing_async()
            return
        if job is None or int(seq) != int(self._proc_job_seq):
            return  # cambiato profilo/canale/dato: risultato non piu' valido
        if error or result is None:
            self._show_processing_error(RuntimeError(error or "risultato vuoto"))
            return
        self._finish_processing(job["params"], result)

    def _wait_processing_job(self, timeout_ms: int = 60000) -> None:
        """Attende un job in volo e ne scarta il risultato."""
        self._proc_job_seq += 1
        self._proc_rerun = False
        thread = getattr(self, "_proc_thread", None)
        if self._is_qt_alive(thread) and thread.isRunning():
            try:
                thread.quit()
                thread.wait(int(timeout_ms))
            except Exception:
                pass
        if self._proc_job is not None:
            self._proc_job = None
            self._set_processing_busy(False)

    def _set_processing_busy(self, busy: bool) -> None:
        self._proc_busy = bool(busy)
        title = "GPR Profile Viewer"
        self.setWindowTitle(title + ("  —  elaborazione in corso..." if busy else ""))
        if busy:
            self._safe_set_text(self._lbl_status, "Elaborazione pipeline in corso...")

    # ------------------------------------------------------------------
    # Auto-tune
    # ------------------------------------------------------------------

    def _on_auto_tune(self):
        """Calcola i parametri ottimali e mostra il dialog di anteprima."""
        if not self._profiles:
            QMessageBox.information(self, "Auto-tune", "Nessun profilo caricato.")
            return

        # Raccoglie i raw data di tutti i profili caricati
        # (usa il profilo corrente come unico raw_data disponibile in memoria;
        #  i restanti vengono letti direttamente dal file .ogpr se necessario)
        from PyQt5.QtWidgets import QProgressDialog
        from PyQt5.QtCore import Qt as _Qt

        n_profs = len(self._profiles)
        raw_list = []
        dt_list = []
        freq_list = []

        prog = QProgressDialog("Analisi profili in corso...", "Annulla", 0, n_profs, self)
        prog.setWindowModality(_Qt.WindowModal)
        prog.setMinimumDuration(0)
        prog.setValue(0)

        for i, prof in enumerate(self._profiles):
            if prog.wasCanceled():
                break
            prog.setValue(i)
            try:
                if i == self._prof_idx and self._raw_data is not None:
                    raw_list.append(np.asarray(self._raw_data, dtype=np.float32))
                else:
                    ch_idx = min(self._ch_idx, len(prof.channels) - 1) if prof.channels else 0
                    cached = detect_and_read(prof.path)
                    if cached and cached.channels and ch_idx < len(cached.channels):
                        raw_list.append(np.asarray(cached.channels[ch_idx].data, dtype=np.float32))
                    else:
                        log_exception(
                            "auto_tune raccolta profili",
                            RuntimeError(f"profilo {prof.path!r} senza canali validi (ch_idx={ch_idx})"),
                        )
                        continue
                dt_list.append(float(getattr(prof, "dt_ns", 0.117) or 0.117))
                freq_list.append(float(getattr(prof, "frequency_mhz", 0.0) or 0.0))
            except Exception as e:
                log_exception(f"auto_tune raccolta profilo {getattr(prof, 'path', i)!r}", e)
                continue
        prog.setValue(n_profs)

        if not raw_list:
            QMessageBox.warning(
                self, "Auto-tune",
                f"Nessun dato disponibile per l'analisi.\n\nDettagli in: {LOG_FILE_PATH}",
            )
            return

        # Frequenza nominale antenna (mediana dei profili campionati): usata
        # come "prior" per ancorare la stima automatica del bandpass, cosi'
        # che rumore a banda larga non finisca per dominare la stima su
        # antenne a bassa frequenza (es. 100 MHz).
        valid_freqs = [f for f in freq_list if f > 10.0]
        antenna_freq_mhz = float(np.median(valid_freqs)) if valid_freqs else 0.0

        try:
            suggested = auto_tune_params(raw_list, dt_list, antenna_freq_mhz=antenna_freq_mhz)
        except Exception as e:
            log_exception("auto_tune_params", e)
            QMessageBox.critical(
                self, "Auto-tune",
                f"Errore nel calcolo dei parametri:\n{e}\n\nDettagli in: {LOG_FILE_PATH}",
            )
            return

        self._show_auto_tune_preview(suggested)

    def _show_auto_tune_preview(self, suggested: dict):
        """Mostra il dialog di anteprima con i parametri suggeriti."""
        from PyQt5.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QDialogButtonBox,
            QGroupBox, QFormLayout, QSpinBox as _QSpinBox,
            QDoubleSpinBox as _QDoubleSpinBox, QCheckBox as _QCheckBox,
            QLabel as _QLabel,
        )

        dlg = QDialog(self)
        dlg.setWindowTitle("Auto-tune — Anteprima parametri suggeriti")
        dlg.setMinimumWidth(380)
        vlay = QVBoxLayout(dlg)

        info = QLabel(
            "Parametri calcolati automaticamente sui profili caricati.\n"
            "Modifica i valori se necessario, poi clicca Applica."
        )
        info.setWordWrap(True)
        vlay.addWidget(info)

        grp = QGroupBox("Parametri suggeriti")
        fl = QFormLayout(grp)

        def _spin(val, lo, hi, step=1):
            w = QSpinBox()
            w.setRange(lo, hi)
            w.setValue(int(val))
            w.setSingleStep(step)
            return w

        def _dspin(val, lo, hi, step=1.0, dec=1):
            w = QDoubleSpinBox()
            w.setRange(lo, hi)
            w.setValue(float(val))
            w.setSingleStep(step)
            w.setDecimals(dec)
            return w

        def _chk(val, label=""):
            w = QCheckBox(label)
            w.setChecked(bool(val))
            return w

        chk_dewow   = _chk(suggested.get("dewow", True), "Abilitato")
        spin_dewow  = _spin(suggested.get("dewow_win", 16), 4, 256)
        chk_tz      = _chk(suggested.get("timezero", True), "Abilitato")
        chk_bg      = _chk(suggested.get("bg_removal", True), "Abilitato")
        spin_bg_s0  = _spin(suggested.get("bg_sample_start", 0), 0, 9999)
        chk_agc     = _chk(suggested.get("agc", True), "Abilitato")
        spin_agc_w  = _spin(suggested.get("agc_win", 128), 8, 512)
        spin_agc_s0 = _spin(suggested.get("agc_sample_start", 0), 0, 9999)
        chk_bp      = _chk(suggested.get("bandpass", False), "Abilitato")
        spin_bp_lo  = _dspin(suggested.get("bp_low_mhz", 100.0), 1, 3000, 10.0)
        spin_bp_hi  = _dspin(suggested.get("bp_high_mhz", 1200.0), 1, 3000, 10.0)

        fl.addRow("Dewow:", chk_dewow)
        fl.addRow("  Finestra:", spin_dewow)
        fl.addRow("Time-zero:", chk_tz)
        fl.addRow("BG removal:", chk_bg)
        fl.addRow("  Da campione:", spin_bg_s0)
        fl.addRow("AGC:", chk_agc)
        fl.addRow("  Finestra:", spin_agc_w)
        fl.addRow("  Da campione:", spin_agc_s0)
        fl.addRow("Bandpass:", chk_bp)
        fl.addRow("  Low (MHz):", spin_bp_lo)
        fl.addRow("  High (MHz):", spin_bp_hi)

        vlay.addWidget(grp)

        btn_box = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Cancel)
        btn_apply = btn_box.button(QDialogButtonBox.Apply)
        btn_apply.setText("Applica")
        btn_box.rejected.connect(dlg.reject)

        def _do_apply():
            params = {
                "dewow":            chk_dewow.isChecked(),
                "dewow_win":        spin_dewow.value(),
                "timezero":         chk_tz.isChecked(),
                "bg_removal":       chk_bg.isChecked(),
                "bg_sample_start":  spin_bg_s0.value(),
                "agc":              chk_agc.isChecked(),
                "agc_win":          spin_agc_w.value(),
                "agc_sample_start": spin_agc_s0.value(),
                "bandpass":         chk_bp.isChecked(),
                "bp_low_mhz":       spin_bp_lo.value(),
                "bp_high_mhz":      spin_bp_hi.value(),
            }
            self._apply_auto_tune_params(params)
            dlg.accept()

        btn_apply.clicked.connect(_do_apply)
        vlay.addWidget(btn_box)
        dlg.exec_()

    def _apply_auto_tune_params(self, params: dict):
        """Applica i parametri auto-tune ai widget del pannello processing."""
        def _safe_set_spin(attr, val):
            w = getattr(self, attr, None)
            if w is not None and self._is_qt_alive(w):
                was_enabled = w.isEnabled()
                w.setEnabled(True)   # force-enable so setValue works even if disabled
                w.blockSignals(True)
                w.setValue(val)
                w.blockSignals(False)
                if not was_enabled:
                    w.setEnabled(False)  # restore disabled state

        # Setta i valori degli spinner prima (con segnali bloccati)
        _safe_set_spin("_spin_dewow",            params.get("dewow_win", 16))
        _safe_set_spin("_spin_bg_sample_start",  params.get("bg_sample_start", 0))
        _safe_set_spin("_spin_agc",              params.get("agc_win", 128))
        _safe_set_spin("_spin_agc_sample_start", params.get("agc_sample_start", 0))
        _safe_set_spin("_spin_bp_lo",            params.get("bp_low_mhz", 100.0))
        _safe_set_spin("_spin_bp_hi",            params.get("bp_high_mhz", 1200.0))

        # Aggiorna i breakpoints del range gain (TVG) se forniti dall'auto-tune
        bkpts = params.get("pre_agc_breakpoints")
        if bkpts is not None:
            bkpts_arr = np.asarray(bkpts, dtype=np.float64).reshape(-1, 2)
            if bkpts_arr.shape[0] >= 2:
                self._range_gain_breakpoints = bkpts_arr
                self._sync_range_gain_spins_from_breakpoints()

        # Setta i checkbox emettendo i segnali — così i toggle abilitano/disabilitano
        # gli spinner dipendenti e sincronizzano la filter chain.
        for attr, val in (
            ("_chk_dewow",      params.get("dewow",      True)),
            ("_chk_timezero",   params.get("timezero",   True)),
            ("_chk_bg",         params.get("bg_removal", True)),
            ("_chk_range_gain", params.get("pre_agc_gain", False)),
            ("_chk_agc",        params.get("agc",        True)),
            ("_chk_bp",         params.get("bandpass",   False)),
        ):
            w = getattr(self, attr, None)
            if w is not None and self._is_qt_alive(w):
                w.setChecked(bool(val))

        self._sync_filter_chain_from_controls()
        if self._raw_data is not None:
            self._apply_processing()
            self._request_preview_update(immediate=False)

    def _sanitize_range_gain_breakpoints(self, points):
        arr = np.asarray(points if points is not None else [], dtype=np.float64).reshape(-1, 2)
        gain_max = float(
            getattr(self, "_spin_gain_deep", None).value()
            if hasattr(self, "_spin_gain_deep")
            else 100.0
        )
        gain_max = float(np.clip(gain_max, 1.0, 100.0))
        rows = []
        for x, y in arr:
            if np.isfinite(x) and np.isfinite(y):
                rows.append((float(np.clip(x, 0.0, 1.0)), float(np.clip(y, 0.0, gain_max))))
        if not rows:
            rows = [(0.0, float(self._spin_gain_surface.value())), (1.0, float(self._spin_gain_deep.value()))]
        rows.sort(key=lambda p: p[0])
        out = [(0.0, float(rows[0][1]))]
        for x, y in rows[1:-1]:
            if x <= 0.0 or x >= 1.0:
                continue
            if abs(x - out[-1][0]) < 1e-6:
                continue
            out.append((x, y))
        out.append((1.0, float(rows[-1][1])))
        max_pts = 16
        if len(out) > max_pts:
            inner = out[1:-1]
            take = max(0, max_pts - 2)
            idx = np.linspace(0, len(inner) - 1, take, dtype=np.int64) if take > 0 else np.zeros(0, dtype=np.int64)
            inner = [inner[i] for i in idx] if take > 0 else []
            out = [out[0]] + inner + [out[-1]]
        return np.asarray(out, dtype=np.float64)

    def _sync_range_gain_spins_from_breakpoints(self):
        if self._range_gain_breakpoints is None or self._range_gain_breakpoints.shape[0] < 2:
            return
        self._updating_range_gain_controls = True
        try:
            self._spin_gain_surface.setValue(float(self._range_gain_breakpoints[0, 1]))
            self._spin_gain_deep.setValue(float(self._range_gain_breakpoints[-1, 1]))
        finally:
            self._updating_range_gain_controls = False

    def _sync_range_gain_breakpoints_from_spins(self):
        if self._range_gain_breakpoints is None or self._range_gain_breakpoints.shape[0] < 2:
            self._range_gain_breakpoints = np.asarray(
                [
                    (0.0, float(self._spin_gain_surface.value())),
                    (1.0, float(self._spin_gain_deep.value())),
                ],
                dtype=np.float64,
            )
        else:
            self._range_gain_breakpoints[0, 0] = 0.0
            self._range_gain_breakpoints[-1, 0] = 1.0
            self._range_gain_breakpoints[0, 1] = float(self._spin_gain_surface.value())
            self._range_gain_breakpoints[-1, 1] = float(self._spin_gain_deep.value())
            self._range_gain_breakpoints = self._sanitize_range_gain_breakpoints(self._range_gain_breakpoints)

    def _apply_range_gain_change(self):
        if bool(getattr(self, "_chk_range_gain", None) and self._chk_range_gain.isChecked()):
            self._apply_processing_async()
        else:
            # Gain disattivato: _compose_display_data legge _proc_pre_agc dal
            # dato, quindi basta ridisegnare.
            self._apply_gain_only()

    def _on_range_gain_surface_changed(self, _value: float):
        if self._updating_range_gain_controls:
            return
        if self._spin_gain_surface.value() > self._spin_gain_deep.value():
            self._updating_range_gain_controls = True
            try:
                self._spin_gain_deep.setValue(self._spin_gain_surface.value())
            finally:
                self._updating_range_gain_controls = False
        self._sync_range_gain_breakpoints_from_spins()
        self._apply_range_gain_change()

    def _on_range_gain_deep_changed(self, _value: float):
        if self._updating_range_gain_controls:
            return
        if self._spin_gain_deep.value() < self._spin_gain_surface.value():
            self._updating_range_gain_controls = True
            try:
                self._spin_gain_surface.setValue(self._spin_gain_deep.value())
            finally:
                self._updating_range_gain_controls = False
        self._sync_range_gain_breakpoints_from_spins()
        self._apply_range_gain_change()

    def _on_range_gain_curve_live_changed(self, points, gain_max=None):
        if gain_max is not None and hasattr(self, "_spin_gain_deep"):
            self._updating_range_gain_controls = True
            try:
                self._spin_gain_deep.setValue(float(np.clip(float(gain_max), 1.0, 100.0)))
            finally:
                self._updating_range_gain_controls = False
        self._range_gain_breakpoints = self._sanitize_range_gain_breakpoints(points)
        self._sync_range_gain_spins_from_breakpoints()
        if self._chk_range_gain.isChecked():
            self._apply_range_gain_change()

    def _open_range_gain_curve_editor(self):
        if not HAS_MPL:
            QMessageBox.information(self, "Range Gain", "Editor curva non disponibile (matplotlib mancante).")
            return
        original = np.asarray(self._range_gain_breakpoints, dtype=np.float64).copy()
        dlg = _RangeGainCurveDialog(
            points=original,
            max_points=16,
            gain_max=float(np.clip(self._spin_gain_deep.value(), 1.0, 100.0)),
            on_curve_changed=self._on_range_gain_curve_live_changed,
            parent=self,
        )
        if dlg.exec_() == QDialog.Accepted:
            self._range_gain_breakpoints = self._sanitize_range_gain_breakpoints(dlg.points())
            self._sync_range_gain_spins_from_breakpoints()
            if self._chk_range_gain.isChecked():
                self._apply_range_gain_change()
            return
        # Revert previewed edits on cancel.
        self._range_gain_breakpoints = self._sanitize_range_gain_breakpoints(original)
        self._sync_range_gain_spins_from_breakpoints()
        if self._chk_range_gain.isChecked():
            self._apply_range_gain_change()

    @staticmethod
    def _rdp_to_velocity_m_s(rdp: float) -> float:
        c = 299792458.0
        rr = max(float(rdp), 1e-6)
        return float(c / np.sqrt(rr))

    @staticmethod
    def _velocity_to_rdp(v_m_s: float) -> float:
        c = 299792458.0
        v = max(float(v_m_s), 1e-9)
        return float((c / v) ** 2)

    def _profile_reference_velocity(self, prof) -> float:
        try:
            v = float(getattr(prof, "velocity_m_s", 0.0) or 0.0)
        except Exception:
            v = 0.0
        if not np.isfinite(v) or v <= 0:
            # Default medium equivalent ~RDP 9.
            v = self._rdp_to_velocity_m_s(9.0)
        return float(v)

    def _hyperbola_txrx_offset(self) -> float:
        w = getattr(self, "_spin_hyperbola_txrx", None)
        if self._is_qt_alive(w):
            try:
                return max(0.0, float(w.value()))
            except Exception:
                return 0.0
        return 0.0

    def _hyperbola_curve_depth(self, x_rel, t0, v_fit, v_ref, s):
        """Profondita' display della curva iperbolica per offset orizzontale `x_rel`.

        Modello bistatico: riflettore puntiforme a profondita' d_fit (nel mezzo
        a velocita' v_fit), antenne Tx-Rx separate da `s`. Con s=0 collassa
        nella forma monostatica t=sqrt(t0^2+(2*dx/v)^2).
        """
        x_rel = np.asarray(x_rel, dtype=np.float64)
        d_fit = np.sqrt(max(0.0, (0.5 * float(v_fit) * float(t0)) ** 2 - (0.5 * float(s)) ** 2))
        half_s = 0.5 * float(s)
        t = (
            np.hypot(x_rel - half_s, d_fit) + np.hypot(x_rel + half_s, d_fit)
        ) / max(float(v_fit), 1e-9)
        return 0.5 * float(v_ref) * t

    def _update_hyperbola_readout(self):
        rdp = float(self._spin_hyperbola_rdp.value())
        v = self._rdp_to_velocity_m_s(rdp)
        v_cmns = v * 1e-7  # m/s -> cm/ns
        self._safe_set_text(
            self._lbl_hyperbola_vel,
            f"v={v:.3e} m/s  ({v_cmns:.2f} cm/ns)  εr={rdp:.2f}",
        )
        s = self._hyperbola_txrx_offset()
        depth_txt = "d=n/a (imposta apice)"
        if self._hyper_apex_depth is not None and self._profiles:
            try:
                prof = self._profiles[self._prof_idx]
                v_ref = self._profile_reference_velocity(prof)
                t0 = (2.0 * float(self._hyper_apex_depth)) / max(v_ref, 1e-9)
                d = np.sqrt(max(0.0, (0.5 * v * t0) ** 2 - (0.5 * s) ** 2))
                depth_txt = f"d={d:.3f} m  (t0={t0 * 1e9:.2f} ns)"
            except Exception:
                pass
        self._safe_set_text(self._lbl_hyperbola_depth, depth_txt)

    def _on_hyperbola_rdp_changed(self, _value: float):
        self._update_hyperbola_readout()
        if self._chk_hyperbola.isChecked():
            self._redraw()

    def _set_hyperbola_apex_from_cursor(self):
        if self._cursor_x is None or self._cursor_z is None:
            QMessageBox.information(self, "Hyperbola Fit", "Muovi il cursore sul radargramma e riprova.")
            return
        self._hyper_apex_x = float(self._cursor_x)
        self._hyper_apex_depth = float(self._cursor_z)
        self._update_hyperbola_readout()
        if self._chk_hyperbola.isChecked():
            self._redraw()

    def _clear_hyperbola_apex(self):
        self._hyper_apex_x = None
        self._hyper_apex_depth = None
        self._update_hyperbola_readout()
        if self._chk_hyperbola.isChecked():
            self._redraw()

    def _auto_fit_hyperbola_rdp(self):
        if self._disp_data is None or not self._profiles:
            QMessageBox.information(self, "Hyperbola Fit", "Nessun profilo visualizzato.")
            return
        if self._hyper_apex_x is None or self._hyper_apex_depth is None:
            if self._cursor_x is not None and self._cursor_z is not None:
                self._set_hyperbola_apex_from_cursor()
            else:
                QMessageBox.information(
                    self,
                    "Hyperbola Fit",
                    "Imposta prima l'apice (Set apex from cursor o doppio click).",
                )
                return
        prof = self._profiles[self._prof_idx]
        arr = np.asarray(self._disp_data, dtype=np.float64)
        if arr.ndim != 2 or arr.size == 0:
            QMessageBox.information(self, "Hyperbola Fit", "Dati radargramma non disponibili.")
            return
        n_s, n_t = arr.shape
        depth_max = self._depth_max_display(prof)
        if not np.isfinite(depth_max) or depth_max <= 0:
            QMessageBox.information(self, "Hyperbola Fit", "Scala profondita' non valida.")
            return

        x0 = float(self._hyper_apex_x)
        z0 = float(self._hyper_apex_depth)
        if not (np.isfinite(x0) and np.isfinite(z0) and z0 >= 0.0):
            QMessageBox.information(self, "Hyperbola Fit", "Apice non valido.")
            return

        distances = self._display_distance_axis(prof, n_t)
        v_ref = self._profile_reference_velocity(prof)
        t0 = (2.0 * z0) / max(v_ref, 1e-9)
        if not np.isfinite(t0) or t0 < 0:
            QMessageBox.information(self, "Hyperbola Fit", "Tempo apice non valido.")
            return

        abs_arr = np.abs(arr)
        if not np.isfinite(abs_arr).any():
            QMessageBox.information(self, "Hyperbola Fit", "Segnale non valido per il fit.")
            return

        # Focus near apex: robust span around current profile.
        dist_span = float(np.nanmax(distances) - np.nanmin(distances)) if distances.size > 1 else 0.0
        focus_half = max(1.5, 0.30 * max(dist_span, 1.0))
        near = np.abs(distances - x0) <= focus_half
        if int(np.count_nonzero(near)) < 8:
            near = np.ones_like(distances, dtype=bool)

        corridor = max(1, int(round(0.006 * n_s)))
        candidates = np.linspace(1.0, 40.0, 157, dtype=np.float64)
        s_txrx = self._hyperbola_txrx_offset()
        best_rdp = None
        best_score = -np.inf
        best_count = 0

        for rdp in candidates:
            v_fit = self._rdp_to_velocity_m_s(float(rdp))
            dx = distances - x0
            z_disp = self._hyperbola_curve_depth(dx, t0, v_fit, v_ref, s_txrx)
            idx_f = (z_disp / max(depth_max, 1e-9)) * max(0, n_s - 1)
            valid = near & np.isfinite(idx_f) & (idx_f >= 0.0) & (idx_f <= (n_s - 1))
            count = int(np.count_nonzero(valid))
            if count < 12:
                continue
            jj = np.where(valid)[0]
            ii = np.rint(idx_f[valid]).astype(np.int64)
            # Corridor amplitude around modeled curve.
            if corridor > 0:
                vals = []
                for k in range(-corridor, corridor + 1):
                    si = np.clip(ii + k, 0, n_s - 1)
                    vals.append(abs_arr[si, jj])
                amp = np.max(np.vstack(vals), axis=0)
            else:
                amp = abs_arr[ii, jj]
            dxv = np.abs(distances[valid] - x0)
            span = max(1e-6, float(np.nanpercentile(dxv, 90))) if dxv.size > 0 else 1.0
            w = 1.0 / (1.0 + (dxv / span) ** 2)
            if not np.isfinite(amp).any():
                continue
            score = float(np.average(np.nan_to_num(amp, nan=0.0), weights=np.nan_to_num(w, nan=0.0)))
            if np.isfinite(score) and score > best_score:
                best_score = score
                best_rdp = float(rdp)
                best_count = count

        if best_rdp is None:
            QMessageBox.information(self, "Hyperbola Fit", "Impossibile stimare RDP con i dati correnti.")
            return

        self._spin_hyperbola_rdp.setValue(float(best_rdp))
        self._safe_set_text(
            self._lbl_status,
            f"Hyperbola auto-fit: RDP={best_rdp:.2f}  score={best_score:.4g}  traces={best_count}"
        )
        if self._chk_hyperbola.isChecked():
            self._redraw()

    def _collect_gain_hyper_preset(self) -> dict:
        points = self._sanitize_range_gain_breakpoints(self._range_gain_breakpoints)
        chain_order = list(self._current_chain_order() or [])
        chain_enabled = {}
        if self._is_qt_alive(self._filter_chain_widget):
            try:
                chain_enabled = dict(self._filter_chain_widget.get_enabled_map() or {})
            except Exception:
                chain_enabled = {}
        preset = {
            "format": "gpr_gain_hyper_preset",
            "version": 2,
            "display_gain": float(self._spin_gain.value()),
            "pipeline_chain": {
                "order": [str(x) for x in chain_order],
                "enabled": {str(k): bool(v) for k, v in chain_enabled.items()},
            },
            "range_gain": {
                "enabled": bool(self._chk_range_gain.isChecked()),
                "gain_min": float(self._spin_gain_surface.value()),
                "gain_max": float(self._spin_gain_deep.value()),
                "breakpoints": [[float(x), float(y)] for x, y in points.tolist()],
            },
            "manual_gain": {
                "enabled": bool(getattr(self, "_chk_manual_gain", None) and self._chk_manual_gain.isChecked()),
                "gain_type": str(self._cb_manual_gain_type.currentData() or "linear") if getattr(self, "_cb_manual_gain_type", None) else "linear",
                "gain_start": float(self._spin_manual_gain_start.value()) if getattr(self, "_spin_manual_gain_start", None) else 1.0,
                "gain_end": float(self._spin_manual_gain_end.value()) if getattr(self, "_spin_manual_gain_end", None) else 20.0,
                "sample_start": int(self._spin_manual_gain_sample_start.value()) if getattr(self, "_spin_manual_gain_sample_start", None) else 0,
                "sample_end": int(self._spin_manual_gain_sample_end.value()) if getattr(self, "_spin_manual_gain_sample_end", None) else 0,
            },
            "hyperbola": {
                "enabled": bool(self._chk_hyperbola.isChecked()),
                "rdp": float(self._spin_hyperbola_rdp.value()),
                "txrx_offset_m": self._hyperbola_txrx_offset(),
                "apex_x": (float(self._hyper_apex_x) if self._hyper_apex_x is not None else None),
                "apex_depth": (float(self._hyper_apex_depth) if self._hyper_apex_depth is not None else None),
            },
        }
        return preset

    # ------------------------------------------------------------------
    # Persistenza dell'INTERO stato di processing nel progetto
    # ------------------------------------------------------------------
    def _collect_full_processing_state(self) -> dict:
        """Ricetta completa della pipeline (parametri per-step + catena +
        gain + iperbole), serializzabile in ``project.json``."""
        try:
            params = dict(self._current_processing_params())
        except Exception:
            params = {}
        # niente array numpy (es. kirchhoff_elev_m: iniettato a runtime)
        params = {k: v for k, v in params.items() if not isinstance(v, np.ndarray)}
        try:
            gain_hyper = self._collect_gain_hyper_preset()
        except Exception:
            gain_hyper = {}
        n_samples = 0
        try:
            if self._raw_data is not None and self._raw_data.ndim == 2:
                n_samples = int(self._raw_data.shape[0])
            elif self._profiles:
                n_samples = int(getattr(self._profiles[0], "n_samples", 0) or 0)
        except Exception:
            n_samples = 0
        return {"version": 1, "params": params, "gain_hyper": gain_hyper,
                "n_samples": n_samples}

    # Parametri legati alla geometria del dato (unita' = campioni o
    # dipendenti dalla posizione del burst): NON vanno ripristinati se il
    # numero di campioni e' cambiato (es. dopo una correzione del reader).
    _SAMPLE_TIED_PARAM_KEYS = frozenset({
        "dewow_win", "tz_backup_nsamp",
        "bg_window", "bg_sample_start", "bg_sample_end",
        "agc_win", "agc_sample_start",
        "pre_agc_breakpoints",
        "dcshift_sample_start", "dcshift_sample_end",
        "amp_eq_window", "boxcar_width", "boxcar_height",
        "median2d_w_trace", "median2d_w_sample",
        "eigen_drop", "eigen_keep_last",
        "manual_gain_sample_start", "manual_gain_sample_end",
    })

    @staticmethod
    def _set_widget_value(w, value) -> None:
        if w is None or value is None:
            return
        try:
            blocker = w.blockSignals(True)
        except Exception:
            blocker = None
        try:
            if isinstance(w, QComboBox):
                idx = w.findData(value)
                if idx < 0:
                    idx = w.findText(str(value))
                if idx >= 0:
                    w.setCurrentIndex(idx)
            elif isinstance(w, QCheckBox):
                w.setChecked(bool(value))
            elif isinstance(w, QSpinBox):
                w.setValue(int(round(float(value))))
            elif isinstance(w, QDoubleSpinBox):
                w.setValue(float(value))
        except Exception:
            pass
        finally:
            if blocker is not None:
                try:
                    w.blockSignals(blocker)
                except Exception:
                    pass

    def _apply_processing_widget_params(self, params: dict) -> None:
        """Reimposta i valori numerici/scelta dei singoli step da un dict
        piatto (inverso di ``_current_processing_params``). Gli *enable* non
        vengono toccati qui: li ripristina la catena filtri."""
        if not isinstance(params, dict) or not params:
            return
        # Controlli per-step Fase 1
        for _sid, store in (getattr(self, "_phase1_controls", {}) or {}).items():
            for key, w in (store or {}).items():
                if key in params and self._is_qt_alive(w):
                    self._set_widget_value(w, params[key])
        # Step "pipeline" classici
        mapping = {
            "dewow_win": getattr(self, "_spin_dewow", None),
            "tz_method": getattr(self, "_cb_tz_method", None),
            "tz_mode": getattr(self, "_cb_tz_mode", None),
            "tz_threshold": getattr(self, "_spin_tz_threshold", None),
            "tz_backup_nsamp": getattr(self, "_spin_tz_backup", None),
            "bg_mode": getattr(self, "_cb_bg_mode", None),
            "bg_method": getattr(self, "_cb_bg_method", None),
            "bg_window": getattr(self, "_spin_bg_window", None),
            "bg_sample_start": getattr(self, "_spin_bg_sample_start", None),
            "bg_sample_end": getattr(self, "_spin_bg_sample_end", None),
            "mig_velocity_m_s": getattr(self, "_spin_mig_vel", None),
            "mig_dx_m": getattr(self, "_spin_mig_dx", None),
            "agc_win": getattr(self, "_spin_agc", None),
            "agc_max_gain": getattr(self, "_spin_agc_max_gain", None),
            "agc_sample_start": getattr(self, "_spin_agc_sample_start", None),
            "agc_lateral": getattr(self, "_spin_agc_lateral", None),
            "bp_low_mhz": getattr(self, "_spin_bp_lo", None),
            "bp_high_mhz": getattr(self, "_spin_bp_hi", None),
        }
        for key, w in mapping.items():
            if key in params and self._is_qt_alive(w):
                self._set_widget_value(w, params[key])
        # taper bandpass: memorizzato come frazione, il widget è in percentuale
        if "bp_taper_frac" in params and self._is_qt_alive(getattr(self, "_spin_bp_taper", None)):
            try:
                self._set_widget_value(self._spin_bp_taper, float(params["bp_taper_frac"]) * 100.0)
            except Exception:
                pass

    # Coppie (id step catena) -> (chiave bool in _current_processing_params).
    # I 12 step Fase 1 usano il proprio id come chiave; qui solo le eccezioni.
    _CHAIN_ENABLE_KEY = {
        "bg_removal": "bg_removal",
        "pre_agc_gain": "pre_agc_gain",
    }

    def _last_enabled_step(self, params: dict, chain_order: list | None) -> str:
        order = [str(s) for s in (chain_order or params.get("chain_order") or [])]
        for sid in reversed(order):
            key = self._CHAIN_ENABLE_KEY.get(sid, sid)
            if bool(params.get(key, False)):
                return sid
        return ""

    def _apply_pipeline_params_dict(self, params: dict, chain_order: list | None = None) -> None:
        """Carica nei controlli di processing una ricetta salvata come dict
        piatto (``variant.json['pipeline_params']``): ordine + enable della
        catena dai flag booleani, poi i valori numerici dei singoli step,
        infine un solo riprocessamento."""
        if not isinstance(params, dict) or not params:
            return
        order = [str(s) for s in (chain_order or params.get("chain_order") or [])]
        enabled = {}
        for sid in order:
            key = self._CHAIN_ENABLE_KEY.get(sid, sid)
            enabled[sid] = bool(params.get(key, False))
        self._syncing_filter_chain = True
        try:
            fcw = getattr(self, "_filter_chain_widget", None)
            if self._is_qt_alive(fcw):
                try:
                    if order:
                        fcw.set_order([str(s) for s in order], emit_signal=False)
                    if enabled:
                        fcw.set_enabled_map({str(k): bool(v) for k, v in enabled.items()},
                                            emit_signal=False)
                    self._on_filter_chain_widget_changed()
                except Exception as exc:
                    print(f"[GPR viewer][WARN] catena filtri: {exc}")
            self._apply_processing_widget_params(params)
        finally:
            self._syncing_filter_chain = False
        try:
            if self._raw_data is not None:
                self._apply_processing()
                self._request_preview_update(immediate=True)
        except Exception:
            pass

    def _clear_profile_cache(self):
        clear_ogpr_cache()
        self._preview_variant_cache = None
        self._safe_set_text(
            self._lbl_status,
            "Cache profili svuotata: i prossimi import rileggono i file da disco.",
        )

    def _show_coord_log_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Log coordinate (debug)")
        dlg.resize(980, 520)
        lay = QVBoxLayout(dlg)
        from qgis.PyQt.QtWidgets import QPlainTextEdit, QPushButton
        box = QPlainTextEdit()
        box.setReadOnly(True)
        box.setLineWrapMode(QPlainTextEdit.NoWrap)
        box.setStyleSheet("font-family: Consolas, monospace; font-size: 9pt;")
        lay.addWidget(box, 1)

        def refresh():
            box.setPlainText(_dbg.get_text() or "(log vuoto)")
            sb = box.verticalScrollBar()
            sb.setValue(sb.maximum())

        def clear_log():
            _dbg.clear()
            refresh()

        def clear_cache():
            clear_ogpr_cache()
            refresh()

        def copy_all():
            QApplication.clipboard().setText(_dbg.get_text())

        row = QHBoxLayout()
        for text, fn in (
            ("Aggiorna", refresh),
            ("Copia", copy_all),
            ("Pulisci log", clear_log),
            ("Pulisci cache", clear_cache),
        ):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch(1)
        b_close = QPushButton("Chiudi")
        b_close.clicked.connect(dlg.accept)
        row.addWidget(b_close)
        lay.addLayout(row)
        refresh()
        dlg.exec_()

    def _apply_full_processing_state(self, state: dict) -> None:
        """Ripristina l'intero stato di processing salvato nel progetto e
        riprocessa una sola volta."""
        if not isinstance(state, dict):
            return
        params = dict(state.get("params") if isinstance(state.get("params"), dict) else {})
        gain_hyper = dict(state.get("gain_hyper") if isinstance(state.get("gain_hyper"), dict) else {})

        # Se il numero di campioni e' cambiato dall'ultimo salvataggio (es.
        # dopo una correzione del reader), i parametri legati alla geometria
        # non sono piu' validi: scartali e invita a rifare l'auto-tune.
        cur_ns = 0
        try:
            if self._raw_data is not None and self._raw_data.ndim == 2:
                cur_ns = int(self._raw_data.shape[0])
            elif self._profiles:
                cur_ns = int(getattr(self._profiles[0], "n_samples", 0) or 0)
        except Exception:
            cur_ns = 0
        saved_ns = int(state.get("n_samples", 0) or 0)
        geometry_changed = bool(saved_ns and cur_ns and saved_ns != cur_ns)
        if geometry_changed:
            for k in self._SAMPLE_TIED_PARAM_KEYS:
                params.pop(k, None)
            rg = gain_hyper.get("range_gain")
            if isinstance(rg, dict):
                rg.pop("breakpoints", None)
            mg = gain_hyper.get("manual_gain")
            if isinstance(mg, dict):
                for k in ("sample_start", "sample_end"):
                    mg.pop(k, None)

        self._syncing_filter_chain = True
        try:
            if gain_hyper:
                try:
                    self._apply_gain_hyper_preset(gain_hyper)
                except Exception as exc:
                    print(f"[GPR viewer][WARN] ripristino gain/iperbole fallito: {exc}")
            self._apply_processing_widget_params(params)
        finally:
            self._syncing_filter_chain = False

        if geometry_changed:
            try:
                self._safe_set_text(
                    self._lbl_status,
                    f"Geometria dati cambiata ({saved_ns}→{cur_ns} campioni): "
                    "parametri legati ai campioni non ripristinati. "
                    "Riesegui 'Auto-tune parametri'.",
                )
            except Exception:
                pass
        try:
            if self._raw_data is not None:
                self._apply_processing()
                self._request_preview_update(immediate=True)
        except Exception:
            pass

    def _apply_gain_hyper_preset(self, payload: dict):
        if not isinstance(payload, dict):
            raise ValueError("Preset non valido: struttura JSON non riconosciuta.")
        if payload.get("format") not in {"gpr_gain_hyper_preset", None}:
            raise ValueError("Preset non valido: campo format non supportato.")

        chain_cfg = payload.get("pipeline_chain", {}) if isinstance(payload.get("pipeline_chain"), dict) else {}
        rg = payload.get("range_gain", {}) if isinstance(payload.get("range_gain"), dict) else {}
        hy = payload.get("hyperbola", {}) if isinstance(payload.get("hyperbola"), dict) else {}

        # Basic controls first.
        self._spin_gain.setValue(float(payload.get("display_gain", self._spin_gain.value())))

        self._updating_range_gain_controls = True
        try:
            self._spin_gain_surface.setValue(
                float(
                    rg.get(
                        "gain_min",
                        rg.get("surface_gain", self._spin_gain_surface.value()),
                    )
                )
            )
            self._spin_gain_deep.setValue(
                float(
                    rg.get(
                        "gain_max",
                        rg.get("deep_gain", self._spin_gain_deep.value()),
                    )
                )
            )
        finally:
            self._updating_range_gain_controls = False

        # Breakpoint curve.
        pts = rg.get("breakpoints")
        if isinstance(pts, list) and pts:
            self._range_gain_breakpoints = self._sanitize_range_gain_breakpoints(pts)
            self._sync_range_gain_spins_from_breakpoints()
        else:
            self._sync_range_gain_breakpoints_from_spins()

        self._chk_range_gain.setChecked(bool(rg.get("enabled", self._chk_range_gain.isChecked())))
        self._btn_range_gain_curve.setEnabled(bool(self._chk_range_gain.isChecked()))

        # Manual gain.
        mg = payload.get("manual_gain", {}) if isinstance(payload.get("manual_gain"), dict) else {}
        if mg and getattr(self, "_chk_manual_gain", None):
            if getattr(self, "_spin_manual_gain_start", None):
                self._spin_manual_gain_start.setValue(float(mg.get("gain_start", self._spin_manual_gain_start.value())))
            if getattr(self, "_spin_manual_gain_end", None):
                self._spin_manual_gain_end.setValue(float(mg.get("gain_end", self._spin_manual_gain_end.value())))
            if getattr(self, "_spin_manual_gain_sample_start", None):
                self._spin_manual_gain_sample_start.setValue(int(mg.get("sample_start", self._spin_manual_gain_sample_start.value())))
            if getattr(self, "_spin_manual_gain_sample_end", None):
                self._spin_manual_gain_sample_end.setValue(int(mg.get("sample_end", self._spin_manual_gain_sample_end.value())))
            if getattr(self, "_cb_manual_gain_type", None):
                gain_type = str(mg.get("gain_type", "linear"))
                idx = self._cb_manual_gain_type.findData(gain_type)
                if idx >= 0:
                    self._cb_manual_gain_type.setCurrentIndex(idx)
            self._chk_manual_gain.setChecked(bool(mg.get("enabled", self._chk_manual_gain.isChecked())))

        # Hyperbola.
        self._spin_hyperbola_rdp.setValue(float(hy.get("rdp", self._spin_hyperbola_rdp.value())))
        if self._is_qt_alive(getattr(self, "_spin_hyperbola_txrx", None)):
            self._spin_hyperbola_txrx.setValue(
                float(hy.get("txrx_offset_m", self._spin_hyperbola_txrx.value()))
            )
        self._chk_hyperbola.setChecked(bool(hy.get("enabled", self._chk_hyperbola.isChecked())))
        apex_x = hy.get("apex_x")
        apex_d = hy.get("apex_depth")
        self._hyper_apex_x = float(apex_x) if apex_x is not None else None
        self._hyper_apex_depth = float(apex_d) if apex_d is not None else None
        self._update_hyperbola_readout()

        if self._is_qt_alive(self._filter_chain_widget):
            order = chain_cfg.get("order")
            enabled_map = chain_cfg.get("enabled")
            chain_updated = False
            try:
                if isinstance(order, list) and order:
                    self._filter_chain_widget.set_order(
                        [str(x) for x in order],
                        emit_signal=False,
                    )
                    chain_updated = True
            except Exception:
                pass
            try:
                if isinstance(enabled_map, dict) and enabled_map:
                    self._filter_chain_widget.set_enabled_map(
                        {str(k): bool(v) for k, v in enabled_map.items()},
                        emit_signal=False,
                    )
                    self._on_filter_chain_widget_changed()
                    chain_updated = True
            except Exception:
                pass
            if (
                chain_updated
                and self._raw_data is not None
                and not (isinstance(enabled_map, dict) and bool(enabled_map))
            ):
                self._apply_processing()

        self._apply_gain_only()
        if self._chk_hyperbola.isChecked():
            self._redraw()

    def _save_gain_hyper_preset(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Gain/Hyper Preset",
            "gpr_gain_hyper_preset.json",
            "JSON (*.json)",
        )
        if not path:
            return
        if not str(path).lower().endswith(".json"):
            path = f"{path}.json"
        payload = self._collect_gain_hyper_preset()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=True)
            self._safe_set_text(self._lbl_status, f"Preset salvato: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save Preset", f"Errore salvataggio preset:\n{exc}")

    def _load_gain_hyper_preset(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Gain/Hyper Preset",
            "",
            "JSON (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self._apply_gain_hyper_preset(payload)
            self._safe_set_text(self._lbl_status, f"Preset caricato: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Load Preset", f"Errore caricamento preset:\n{exc}")

    def _open_3d_viewer(self):
        if not self._profiles:
            QMessageBox.information(
                self,
                "Nessun profilo",
                "Importa almeno un file .ogpr prima di aprire il viewer 3D.",
            )
            return

        try:
            from .gpr_3d_viewer_dialog import Gpr3dViewerDialog
            from .gpr_ogpr_slicer import compute_ogpr_slice_grids
            from .gpr_volume_3d import build_3d_volume
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Viewer 3D",
                f"Impossibile inizializzare il viewer 3D:\n{exc}",
            )
            return

        depth_max = max(float(getattr(p, "depth_max_m", 0.0) or 0.0) for p in self._profiles)
        n_channels = max(int(getattr(p, "n_channels", 1) or 1) for p in self._profiles)

        extra = self.get_slice_params()
        # Fase D: spec-native path - one SliceSpec drives the whole volume, no dialog.
        _panel = getattr(self, "_slice_spec_panel", None)
        _use_spec = _panel is not None and self._is_qt_alive(_panel)

        if _use_spec:
            try:
                from .slice_params_compat import spec_to_build_params
                slice_profiles, _variant_src = self._profiles_for_slicing()
                if not slice_profiles:
                    slice_profiles = list(self._profiles or [])
                spec = _panel.get_spec().resolve(slice_profiles)
                params = spec_to_build_params(
                    spec, channel=-1, combine_method="mean",
                    group_name="gpr_volume3d_preview",
                )
                grids, meta = compute_ogpr_slice_grids(
                    profiles=slice_profiles, channel=-1, combine_method="mean",
                    spec=spec, emit_diagnostics=True,
                )
            except Exception as exc:
                QMessageBox.critical(
                    self, "Viewer 3D",
                    f"Errore durante il calcolo volume 3D:\n{exc}",
                )
                return
        else:
            params = None
            if self.plugin is not None and hasattr(self.plugin, "_ask_ogpr_slice_params"):
                try:
                    params = self.plugin._ask_ogpr_slice_params(
                        profiles=self._profiles,
                        n_channels=n_channels,
                        depth_max_m=depth_max,
                        default_group="gpr_volume3d_preview",
                        saved=None,
                        slice_params=self.get_slice_params(),
                    )
                except Exception as exc:
                    QMessageBox.warning(
                        self, "Viewer 3D",
                        f"Errore nel dialog parametri 3D:\n{exc}",
                    )
                    return
                if params is None:
                    return
            else:
                params = {
                    "channel": int(self._ch_idx),
                    "combine_method": "mean",
                    "z_min": 0.0,
                    "z_max": max(depth_max, 0.1),
                    "z_step": 0.05,
                    "resolution": 0.10,
                    "radius": 0.10 * (2.0 ** 0.5),
                    "group_name": "gpr_volume3d_preview",
                }
            try:
                grids, meta = compute_ogpr_slice_grids(
                    profiles=self._profiles,
                    channel=int(params["channel"]),
                    combine_method=str(params["combine_method"]),
                    resolution=float(params["resolution"]),
                    z_step=float(params["z_step"]),
                    z_min=float(params["z_min"]),
                    z_max=float(params["z_max"]),
                    radius=float(params["radius"]),
                    pipeline_params=extra.get("pipeline_params"),
                    normalize_channels=bool(extra.get("normalize_channels", False)),
                    extraction_mode=str(extra.get("extraction_mode", "las_like") or "las_like"),
                    use_hilbert=bool(extra.get("use_hilbert", True)),
                    use_processing=bool(extra.get("use_processing", False)),
                    amplitude_sigma=extra.get("amplitude_sigma"),
                    use_anisotropic_idw=bool(extra.get("use_anisotropic_idw", False)),
                    idw_mode=str(extra.get("idw_mode", "quality") or "quality"),
                    idw_power=float(extra.get("idw_power", 2.0) or 2.0),
                    auto_radius=bool(extra.get("auto_radius", True)),
                    min_points=int(extra.get("min_points", 1) or 1),
                    fill_nodata=bool(extra.get("fill_nodata", True)),
                    overlap_fraction=float(extra.get("overlap_fraction", 0.5) or 0.0),
                    blanking_distance=float(extra.get("blanking_distance", 0.0) or 0.0),
                    smooth_sigma=float(extra.get("smooth_sigma", 0.8) or 0.0),
                    depth_radius_factor=float(extra.get("depth_radius_factor", 0.6) or 0.0),
                    balance_profiles=bool(extra.get("balance_profiles", True)),
                    pre_slice_bg_removal=bool(extra.get("pre_slice_bg_removal", False)),
                    pre_slice_bg_mode=str(extra.get("pre_slice_bg_mode", "line_by_line") or "line_by_line"),
                    pre_slice_bg_window=int(extra.get("pre_slice_bg_window", 0) or 0),
                    pre_slice_bg_sample_start=int(extra.get("pre_slice_bg_sample_start", 0) or 0),
                    pre_slice_bg_sample_end=int(extra.get("pre_slice_bg_sample_end", 0) or 0),
                    stack_n=int(extra.get("stack_n", 1) or 1),
                    stack_kernel=str(extra.get("stack_kernel", "boxcar") or "boxcar"),
                    flip_traces_mode=str(extra.get("flip_traces_mode", "none") or "none"),
                    topographic_correction=bool(extra.get("topographic_correction", False)),
                    topo_reference_mode=str(extra.get("topo_reference_mode", "median") or "median"),
                    topo_reference_elevation=extra.get("topo_reference_elevation"),
                    parallel_profiles=bool(extra.get("parallel_profiles", True)),
                    profile_workers=int(extra.get("profile_workers", 0) or 0),
                    emit_diagnostics=True,
                )
            except Exception as exc:
                QMessageBox.critical(
                    self, "Viewer 3D",
                    f"Errore durante il calcolo volume 3D:\n{exc}",
                )
                return

        if not grids:
            QMessageBox.warning(
                self,
                "Viewer 3D",
                "Nessuna slice disponibile per costruire il volume 3D.",
            )
            return

        try:
            volume = build_3d_volume(grids, meta)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Viewer 3D",
                f"Impossibile costruire il volume 3D:\n{exc}",
            )
            return

        z_levels = []
        for g in grids:
            try:
                z_levels.append(float(g.get("z_lev")))
            except Exception:
                continue
        z_levels = [z for z in z_levels if np.isfinite(z)]
        z_min_view = float(min(z_levels)) if z_levels else float(params["z_min"])
        z_max_view = float(max(z_levels)) if z_levels else float(params["z_max"])

        meta_3d = dict(meta or {})
        meta_3d.update({
            "n_z": int(volume.shape[0]),
            "n_y": int(volume.shape[1]),
            "n_x": int(volume.shape[2]),
            "z_step": float(params["z_step"]),
            "z_min": z_min_view,
            "z_max": z_max_view,
            "z_levels": [float(z) for z in z_levels] if z_levels else None,
            "channel": int(params["channel"]),
            "use_processing": bool(extra.get("use_processing", False)),
            "extraction_mode": str(extra.get("extraction_mode", "las_like") or "las_like"),
            "pipeline_params": dict(extra.get("pipeline_params") or {}),
        })

        if self._gpr_3d_viewer is not None:
            try:
                self.cursor_moved.disconnect(self._gpr_3d_viewer.update_cursor_position)
            except Exception:
                pass
            try:
                self._gpr_3d_viewer.depth_changed.disconnect(self._on_3d_depth_changed)
            except Exception:
                pass
            try:
                self._gpr_3d_viewer.close()
            except Exception:
                pass
            self._gpr_3d_viewer = None

        try:
            viewer = Gpr3dViewerDialog(
                volume=volume,
                meta=meta_3d,
                profiles=self._profiles,
                grids=grids,
                parent=self,
            )
        except Exception as exc:
            import traceback
            tb = traceback.format_exc(limit=8)
            QMessageBox.warning(
                self,
                "Viewer 3D",
                f"Impossibile aprire il viewer 3D:\n{exc}\n\n{tb}",
            )
            return

        self._gpr_3d_viewer = viewer
        try:
            self.cursor_moved.connect(viewer.update_cursor_position)
        except Exception:
            pass
        try:
            viewer.depth_changed.connect(self._on_3d_depth_changed)
        except Exception:
            pass
        self._emit_cursor_moved()
        viewer.show()
        viewer.raise_()
        viewer.activateWindow()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def request_close(self):
        """Force real close/destruction (used by plugin unload)."""
        self._allow_close = True
        try:
            self.close()
        finally:
            self._allow_close = False

    def _close_3d_viewer(self):
        if self._gpr_3d_viewer is None:
            return
        try:
            self.cursor_moved.disconnect(self._gpr_3d_viewer.update_cursor_position)
        except Exception:
            pass
        try:
            self._gpr_3d_viewer.depth_changed.disconnect(self._on_3d_depth_changed)
        except Exception:
            pass
        try:
            self._gpr_3d_viewer.close()
        except Exception:
            pass
        self._gpr_3d_viewer = None

    def closeEvent(self, event):
        # Default UX: closing hides the window to preserve child widgets/docks.
        # This avoids stale Python refs to deleted Qt C++ objects on next reopen.
        if not bool(getattr(self, "_allow_close", False)):
            if not self._confirm_save_before_close():
                event.ignore()
                return
            try:
                self._save_gpr_processing_project_meta()
            except Exception:
                pass
            try:
                if self._processing_dock is not None:
                    self._save_dock_state(self._processing_dock, "processing")
                if self._timeslice_dock is not None:
                    self._save_dock_state(self._timeslice_dock, "timeslice")
            except Exception:
                pass
            try:
                self._close_3d_viewer()
            except Exception:
                pass
            self.hide()
            event.ignore()
            return

        # Real close path (plugin unload).
        self._wait_processing_job(timeout_ms=15000)
        self._stop_preview_worker_blocking()
        try:
            self._save_gpr_processing_project_meta()
        except Exception:
            pass
        self._close_3d_viewer()
        if self._processing_dock is not None:
            try:
                self._save_dock_state(self._processing_dock, "processing")
            except Exception:
                pass
            try:
                self.removeDockWidget(self._processing_dock)
            except Exception:
                pass
            try:
                self._processing_dock.deleteLater()
            except Exception:
                pass
            self._processing_dock = None
        if self._timeslice_dock is not None:
            try:
                self._save_dock_state(self._timeslice_dock, "timeslice")
            except Exception:
                pass
            try:
                self.removeDockWidget(self._timeslice_dock)
            except Exception:
                pass
            try:
                self._timeslice_dock.deleteLater()
            except Exception:
                pass
            self._timeslice_dock = None
        for rb in (self._rb_point, self._rb_line, self._rb_crosshair_h, self._rb_crosshair_v):
            if rb is not None:
                try:
                    rb.reset()
                except Exception:
                    pass
        super().closeEvent(event)


