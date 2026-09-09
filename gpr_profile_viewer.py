# -*- coding: utf-8 -*-
"""
GPR Profile Viewer
==================
QMainWindow con radargram matplotlib integrato.
"""

from __future__ import annotations

import json
import os
import glob
import re
import shutil
from datetime import datetime
from typing import Optional

import numpy as np

from qgis.PyQt.QtCore import Qt, QTimer, pyqtSignal, QSettings, QThread, QObject
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import (
    QDialog, QMainWindow, QVBoxLayout, QHBoxLayout, QToolBar,
    QAction, QLabel, QComboBox,
    QCheckBox, QDoubleSpinBox, QSpinBox,
    QSlider,
    QSplitter,
    QToolButton,
    QTabWidget,
    QWidget,
    QInputDialog,
    QFileDialog, QMessageBox, QSizePolicy,
    QGroupBox, QFormLayout, QPushButton, QDialogButtonBox, QScrollArea, QDockWidget, QLineEdit,
    QProgressBar, QListWidget, QListWidgetItem,
    QTreeWidget, QTreeWidgetItem, QMenu,
)
from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsMessageLog,
    QgsPointXY,
    QgsProject,
    QgsWkbTypes,
    QgsRasterLayer,
)
from qgis.gui import QgsRubberBand

try:
    from qgis.PyQt import sip  # type: ignore
except Exception:
    try:
        import sip  # type: ignore
    except Exception:
        sip = None  # type: ignore

try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from .gpr_ogpr_reader import read_ogpr_cached, OgprProfile, OgprChannel
from .gpr_raw_format_registry import detect_and_read, sidecar_paths, import_dialog_filter
from .gpr_manual_positioning_dialog import GprManualPositioningDialog, check_profiles_have_gps
from .gpr_processing import (
    apply_pipeline,
    DEFAULT_PIPELINE,
    DEFAULT_CHAIN_ORDER,
    FILTER_REGISTRY,
    normalize_display,
    _normalize_display_vmax,
    manual_gain,
    auto_tune_params,
    multifreq_blend,
    log_exception,
    LOG_FILE_PATH,
)
from .gpr_npz_io import (
    save_processed_profile_npz,
    load_processed_profile_npz,
    write_variant_manifest,
    read_variant_manifest,
    list_variants,
)

try:
    from .filter_chain_widget import FilterChainWidget, FilterBlockWidget
except Exception:
    FilterChainWidget = None
    FilterBlockWidget = None


GPR_CMAPS    = [
    "RdBu_r",
    "seismic",
    "gray",
    "bwr",
    "Greys_r",
    "viridis",
    "plasma",
    "magma",
    "cividis",
    "turbo",
    "Spectral_r",
]
DEFAULT_CMAP = "RdBu_r"


class _RangeGainCurveDialog(QDialog):
    """Interactive breakpoint editor for range gain curve."""

    def __init__(self, points, max_points=16, gain_max=100.0, on_curve_changed=None, parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("Range Gain Curve")
        self.resize(620, 420)
        self._max_points = int(max(2, min(32, max_points)))
        self._on_curve_changed = on_curve_changed
        self._gain_max = float(np.clip(float(gain_max), 1.0, 100.0))
        self._drag_idx = None
        self._selected_idx = None
        self._did_drag = False
        self._points = self._sanitize(points)

        root = QVBoxLayout(self)

        if HAS_MPL:
            self._fig = Figure(figsize=(6.2, 3.2), tight_layout=True)
            self._ax = self._fig.add_subplot(111)
            self._canvas = FigureCanvasQTAgg(self._fig)
            root.addWidget(self._canvas, 1)
            self._canvas.mpl_connect("button_press_event", self._on_press)
            self._canvas.mpl_connect("button_release_event", self._on_release)
            self._canvas.mpl_connect("motion_notify_event", self._on_move)
        else:
            self._fig = None
            self._ax = None
            self._canvas = None
            lbl = QLabel("matplotlib non disponibile.")
            lbl.setAlignment(Qt.AlignCenter)
            root.addWidget(lbl, 1)

        help_lbl = QLabel(
            "X=Gain, Y=Depth (0 in alto) | Left click: seleziona/drag | "
            "Double-click: aggiungi | Right-click: elimina punto interno"
        )
        help_lbl.setWordWrap(True)
        root.addWidget(help_lbl)

        gain_row = QHBoxLayout()
        gain_row.addWidget(QLabel("Gain max:"))
        self._spin_gain_max = QDoubleSpinBox()
        self._spin_gain_max.setRange(1.0, 100.0)
        self._spin_gain_max.setSingleStep(1.0)
        self._spin_gain_max.setDecimals(1)
        self._spin_gain_max.setValue(self._gain_max)
        self._spin_gain_max.valueChanged.connect(self._on_gain_max_changed)
        gain_row.addWidget(self._spin_gain_max)
        gain_row.addStretch(1)
        root.addLayout(gain_row)

        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Step:"))
        self._spin_step = QDoubleSpinBox()
        self._spin_step.setRange(0.01, 1.0)
        self._spin_step.setSingleStep(0.01)
        self._spin_step.setValue(0.05)
        self._spin_step.setDecimals(2)
        step_row.addWidget(self._spin_step)
        self._btn_step_up = QPushButton("▲ Gain +")
        self._btn_step_down = QPushButton("▼ Gain -")
        self._btn_step_up.clicked.connect(lambda: self._move_selected(+float(self._spin_step.value())))
        self._btn_step_down.clicked.connect(lambda: self._move_selected(-float(self._spin_step.value())))
        step_row.addWidget(self._btn_step_up)
        step_row.addWidget(self._btn_step_down)
        step_row.addStretch(1)
        root.addLayout(step_row)

        btns_row = QHBoxLayout()
        self._btn_reset = QPushButton("Reset")
        self._btn_reset.clicked.connect(self._on_reset)
        btns_row.addWidget(self._btn_reset)
        btns_row.addStretch(1)
        root.addLayout(btns_row)

        dbb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        dbb.accepted.connect(self.accept)
        dbb.rejected.connect(self.reject)
        root.addWidget(dbb)

        self._redraw()

    def _sanitize(self, points):
        arr = np.asarray(points if points is not None else [], dtype=np.float64).reshape(-1, 2)
        clean = []
        for x, y in arr:
            if np.isfinite(x) and np.isfinite(y):
                clean.append(
                    (
                        float(np.clip(x, 0.0, 1.0)),
                        float(np.clip(y, 0.0, self._gain_max)),
                    )
                )
        if not clean:
            clean = [(0.0, 1.0), (1.0, 6.0)]
        clean.sort(key=lambda p: p[0])
        # Force endpoints and uniqueness.
        x0, y0 = clean[0]
        x1, y1 = clean[-1]
        pts = [(0.0, float(y0))]
        for x, y in clean[1:-1]:
            if x <= 0.0 or x >= 1.0:
                continue
            if abs(x - pts[-1][0]) < 1e-6:
                continue
            pts.append((x, y))
        if abs(1.0 - pts[-1][0]) < 1e-6:
            pts[-1] = (1.0, float(y1))
        else:
            pts.append((1.0, float(y1)))
        if len(pts) > self._max_points:
            inner = pts[1:-1]
            take = max(0, self._max_points - 2)
            if len(inner) > take:
                idx = np.linspace(0, len(inner) - 1, take, dtype=np.int64)
                inner = [inner[i] for i in idx]
            pts = [pts[0]] + inner + [pts[-1]]
        return np.asarray(pts, dtype=np.float64)

    def points(self):
        return self._points.astype(np.float64, copy=True)

    def gain_max(self) -> float:
        return float(self._gain_max)

    def _emit_changed(self):
        if callable(self._on_curve_changed):
            try:
                self._on_curve_changed(self.points(), float(self._gain_max))
            except Exception:
                pass

    def _redraw(self):
        if not HAS_MPL or self._ax is None or self._canvas is None:
            return
        self._ax.clear()
        p = self._points
        # X=gain, Y=depth fraction (0 top, 1 bottom).
        self._ax.plot(p[:, 1], p[:, 0], color="#2d6ba3", lw=1.4)
        self._ax.scatter(p[:, 1], p[:, 0], s=40, color="#c0392b", zorder=3)
        self._ax.set_xlim(0.0, float(self._gain_max))
        self._ax.set_ylim(1.0, 0.0)
        self._ax.grid(True, ls=":", lw=0.5, alpha=0.7)
        self._ax.set_xlabel("Gain")
        self._ax.set_ylabel("Depth fraction")
        self._ax.set_title(f"Breakpoints: {len(p)}/{self._max_points}")
        if self._selected_idx is not None:
            try:
                i = int(self._selected_idx)
                if 0 <= i < len(p):
                    sel = p[i]
                    self._ax.scatter(
                        [float(sel[1])],
                        [float(sel[0])],
                        s=80,
                        color="#ffd000",
                        edgecolor="#1f1f1f",
                        linewidths=1.0,
                        zorder=5,
                    )
            except Exception:
                pass
        self._canvas.draw_idle()

    def _nearest_idx(self, x, y):
        p = self._points
        if p.shape[0] <= 0:
            return None, 1e9
        # x=gain, y=depth fraction
        x_span = max(1e-6, float(np.nanmax(p[:, 1]) - np.nanmin(p[:, 1])))
        dx = (p[:, 1] - float(x)) / x_span
        dy = (p[:, 0] - float(y))
        d = np.sqrt(dx * dx + dy * dy)
        idx = int(np.argmin(d))
        return idx, float(d[idx])

    def _on_press(self, event):
        if event is None or event.inaxes != self._ax or event.xdata is None or event.ydata is None:
            return
        gain = float(np.clip(event.xdata, 0.0, self._gain_max))
        depth = float(np.clip(event.ydata, 0.0, 1.0))
        idx, dist = self._nearest_idx(gain, depth)

        # Right click: remove internal point.
        if int(getattr(event, "button", 0) or 0) == 3:
            if idx is not None and dist < 0.05 and idx not in (0, len(self._points) - 1):
                self._points = np.delete(self._points, idx, axis=0)
                self._points = self._sanitize(self._points)
                self._selected_idx = None
                self._emit_changed()
                self._redraw()
            return

        # Double left click: add point.
        if bool(getattr(event, "dblclick", False)) and int(getattr(event, "button", 0) or 0) == 1:
            if len(self._points) < self._max_points and (idx is None or dist >= 0.02):
                self._points = np.vstack([self._points, np.asarray([[depth, gain]], dtype=np.float64)])
                self._points = self._sanitize(self._points)
                idx_new, _ = self._nearest_idx(gain, depth)
                self._selected_idx = idx_new
                self._emit_changed()
                self._redraw()
            return

        # Single left click: select nearest point and enable drag.
        if int(getattr(event, "button", 0) or 0) == 1 and idx is not None and dist < 0.08:
            self._selected_idx = int(idx)
            self._drag_idx = int(idx)
            self._did_drag = False
            self._redraw()

    def _on_move(self, event):
        if self._drag_idx is None:
            return
        if event is None or event.inaxes != self._ax or event.xdata is None or event.ydata is None:
            return
        i = int(self._drag_idx)
        gain = float(np.clip(event.xdata, 0.0, self._gain_max))
        depth = float(np.clip(event.ydata, 0.0, 1.0))
        p = self._points.copy()
        if i == 0:
            depth = 0.0
        elif i == (len(p) - 1):
            depth = 1.0
        else:
            lo = float(p[i - 1, 0] + 1e-4)
            hi = float(p[i + 1, 0] - 1e-4)
            if hi <= lo:
                depth = float(p[i, 0])
            else:
                depth = float(np.clip(depth, lo, hi))
        p[i, 0] = depth
        p[i, 1] = gain
        self._points = self._sanitize(p)
        self._did_drag = True
        self._emit_changed()
        self._redraw()

    def _on_release(self, _event):
        self._did_drag = False
        self._drag_idx = None

    def _move_selected(self, delta: float):
        if self._selected_idx is None:
            return
        i = int(self._selected_idx)
        p = self._points.copy()
        if i < 0 or i >= len(p):
            return
        p[i, 1] = float(np.clip(p[i, 1] + float(delta), 0.0, self._gain_max))
        self._points = self._sanitize(p)
        self._emit_changed()
        self._redraw()

    def _on_gain_max_changed(self, value: float):
        self._gain_max = float(np.clip(float(value), 1.0, 100.0))
        self._points = self._sanitize(self._points)
        self._emit_changed()
        self._redraw()

    def _on_reset(self):
        self._points = np.asarray([(0.0, 1.0), (1.0, 6.0)], dtype=np.float64)
        self._selected_idx = None
        self._drag_idx = None
        self._emit_changed()
        self._redraw()


class _PreviewSliceWorker(QObject):
    """Esegue compute_timeslice_preview() su un thread separato dalla GUI.

    Evita il blocco del thread principale (che produce il cursore "in
    caricamento" permanente di Windows/Qt) durante il preview live delle
    timeslice.
    """

    finished = pyqtSignal(int, object, object, str)  # request_id, grid, extent, error

    def __init__(self, request_id: int, profiles, z_from, z_to, resolution, chain_order, kwargs,
                 spec=None):
        super().__init__()
        self._request_id = int(request_id)
        self._profiles = profiles
        self._z_from = z_from
        self._z_to = z_to
        self._resolution = resolution
        self._chain_order = chain_order
        self._kwargs = kwargs
        self._spec = spec

    def run(self):
        try:
            if self._spec is not None:
                # Fase D: spec-native preview (single window), same code path
                # as the final generation but with cheap knobs: fast kNN IDW,
                # no NoData fill, no diagnostics. The final "Salva gruppo" run
                # still uses the quality settings from the spec.
                from .gpr_ogpr_slicer import compute_slice_preview
                grid, extent = compute_slice_preview(
                    self._profiles, self._spec, self._z_from, self._z_to,
                    resolution=self._resolution,
                    idw_mode="fast",
                    fill_nodata=False,
                    emit_diagnostics=False,
                )
            else:
                from .gpr_ogpr_slicer import compute_timeslice_preview
                grid, extent = compute_timeslice_preview(
                    profiles=self._profiles,
                    z_from=self._z_from,
                    z_to=self._z_to,
                    resolution=self._resolution,
                    chain_order=self._chain_order,
                    **self._kwargs,
                )
            self.finished.emit(self._request_id, grid, extent, "")
        except Exception as exc:
            self.finished.emit(self._request_id, None, None, str(exc))


class GprProfileViewer(QMainWindow):
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

    def _reset_session_state(self):
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
        self._cb_cmap.addItems(GPR_CMAPS)
        self._cb_cmap.setCurrentText(DEFAULT_CMAP)
        self._cb_cmap.currentTextChanged.connect(self._redraw)
        tb.addWidget(self._cb_cmap)

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

        self._lbl_status = QLabel("Importa un file .ogpr per iniziare.")
        root.addWidget(self._lbl_status)

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

    def _open_project_manager(self):
        if self.plugin is not None and hasattr(self.plugin, "open_project_manager"):
            try:
                self.plugin.open_project_manager()
                return
            except Exception:
                pass
        QMessageBox.information(
            self,
            "Project Manager",
            "Project Manager non disponibile in questo contesto.",
        )

    def _save_project_file(self):
        try:
            ok = bool(QgsProject.instance().write())
        except Exception:
            ok = False
        meta_ok = False
        try:
            meta_ok = bool(self._save_gpr_processing_project_meta())
        except Exception:
            meta_ok = False
        if ok and meta_ok:
            self._safe_set_text(self._lbl_status, "Progetto QGIS + stato GPR salvati.")
        elif ok and (not meta_ok):
            self._safe_set_text(self._lbl_status, "Progetto QGIS salvato, stato GPR non salvato.")
        else:
            QMessageBox.warning(
                self,
                "Salva progetto",
                "Impossibile salvare il progetto QGIS corrente.",
            )

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
        act_apply.triggered.connect(self._apply_processing)
        m_proc.addAction(act_apply)

        act_gain = QAction("Aggiorna gain", self)
        act_gain.setShortcut("F6")
        act_gain.triggered.connect(self._apply_gain_only)
        m_proc.addAction(act_gain)

        m_proc.addSeparator()
        act_save_variant = QAction("Salva variante processata (.npz)...", self)
        act_save_variant.triggered.connect(self._save_processed_variant)
        m_proc.addAction(act_save_variant)

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

    def _apply_wiggle_visibility_layout(self):
        if not HAS_MPL or not hasattr(self, "_ax") or not hasattr(self, "_ax_wiggle"):
            return
        if self._ax is None or self._ax_wiggle is None:
            return
        if self._ax_main_bounds_default is None or self._ax_wiggle_bounds_default is None:
            try:
                self._ax_main_bounds_default = list(self._ax.get_position().bounds)
                self._ax_wiggle_bounds_default = list(self._ax_wiggle.get_position().bounds)
            except Exception:
                return

        main = list(self._ax_main_bounds_default)
        wig = list(self._ax_wiggle_bounds_default)
        if bool(getattr(self, "_show_wiggle", True)):
            self._ax_wiggle.set_visible(True)
            self._ax.set_position(main)
            self._ax_wiggle.set_position(wig)
        else:
            self._ax_wiggle.set_visible(False)
            right = float(wig[0] + wig[2])
            self._ax.set_position([float(main[0]), float(main[1]), right - float(main[0]), float(main[3])])

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
            self._apply_processing()
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
            fl.addRow("Max Gain (0=auto):", self._spin_agc_max_gain)
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

            pre_agc_enabled = bool(enabled.get("pre_agc_gain", False))
            if self._is_qt_alive(getattr(self, "_chk_range_gain", None)):
                self._chk_range_gain.setChecked(pre_agc_enabled)
            if self._is_qt_alive(getattr(self, "_chk_manual_gain", None)):
                self._chk_manual_gain.setChecked(bool(enabled.get("manual_gain", self._chk_manual_gain.isChecked())))
        finally:
            self._syncing_filter_chain = False

        if self._raw_data is not None:
            self._apply_processing()
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
        self._cb_tz_method.addItems(["peak", "threshold", "zero_crossing"])
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
            lambda _i: self._apply_processing() if self._chk_bg.isChecked() and self._raw_data is not None else None
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
                self._apply_processing()

        def _on_mig_rdp_changed(val: float):
            v = 299792458.0 / np.sqrt(max(float(val), 1.0))
            self._spin_mig_vel.blockSignals(True)
            try:
                self._spin_mig_vel.setValue(float(np.clip(v, 1e6, 3e8)))
            finally:
                self._spin_mig_vel.blockSignals(False)
            if self._chk_mig.isChecked() and self._raw_data is not None:
                self._apply_processing()

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
                self._apply_processing()

        self._spin_mig_vel.valueChanged.connect(_on_mig_vel_changed)
        self._spin_mig_rdp.valueChanged.connect(_on_mig_rdp_changed)
        self._spin_mig_dx.valueChanged.connect(
            lambda _v: self._apply_processing() if self._chk_mig.isChecked() and self._raw_data is not None else None
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
        self._spin_agc_sample_start = QSpinBox()
        self._spin_agc_sample_start.setRange(0, 9999)
        self._spin_agc_sample_start.setValue(0)
        self._spin_agc_sample_start.setEnabled(False)
        self._spin_agc_sample_start.setToolTip(
            "Primo campione su cui applicare l'AGC (0 = dall'inizio).\n"
            "I campioni precedenti vengono copiati invariati (gain=1).\n"
            "Utile per escludere il burst superficiale (coupling antenna)."
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
            self._spin_agc_sample_start.setEnabled(agc_checked)
            if self._is_qt_alive(self._btn_pick_agc_s0):
                self._btn_pick_agc_s0.setEnabled(agc_checked)
            if not agc_checked and self._pick_mode_target == "agc_sample_start":
                self._set_pick_mode(None, None)

        self._chk_agc.toggled.connect(_toggle_agc)
        self._spin_agc_max_gain.valueChanged.connect(
            lambda _v: self._apply_processing() if self._chk_agc.isChecked() and self._raw_data is not None else None
        )
        self._spin_agc_sample_start.valueChanged.connect(
            lambda _v: self._apply_processing() if self._chk_agc.isChecked() and self._raw_data is not None else None
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
        self._chk_flip_profile.toggled.connect(lambda _v: self._reload_data())

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

        btn_apply = QPushButton("Applica")
        btn_apply.clicked.connect(self._apply_processing)
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
        self._cb_slice_cmap.addItems(GPR_CMAPS)
        self._cb_slice_cmap.setCurrentText(DEFAULT_CMAP)
        self._cb_slice_cmap.setToolTip("Colormap dedicata alla vista timeslice (indipendente dal radargramma).")
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
            self._apply_processing()

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
            self._apply_processing()

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
            self._apply_processing()

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------

    def _prompt_workspace_dir(self) -> None:
        """Chiede all'utente dove salvare il workspace GPR (raw/processed/timeslices).

        Imposta self._gpr_workspace_override sulla cartella scelta. Se l'utente
        annulla, mantiene l'eventuale workspace precedente (o il default
        automatico sotto il progetto attivo).
        """
        start = str(getattr(self, "_gpr_workspace_override", "") or "").strip()
        if not start:
            root = self._active_project_root()
            start = (
                os.path.join(root, "GPR_Processing") if root else os.path.expanduser("~")
            )
        folder = QFileDialog.getExistingDirectory(
            self, "Cartella di lavoro GPR (dove salvare raw/processed/timeslices)", start
        )
        if folder:
            self._set_active_workspace(folder)

    def _new_gpr_project(self):
        """Crea un nuovo progetto GPR in una cartella scelta e azzera la sessione."""
        start = str(getattr(self, "_gpr_workspace_override", "") or "").strip() \
            or os.path.expanduser("~")
        folder = QFileDialog.getExistingDirectory(
            self, "Nuovo progetto GPR — scegli la cartella", start
        )
        if not folder:
            return
        self._set_active_workspace(folder)
        self._reset_session_state()
        self._ensure_gpr_processing_layout()
        try:
            self._save_gpr_processing_project_meta()
        except Exception:
            pass
        self._safe_set_text(
            self._lbl_status,
            f"Nuovo progetto GPR: {os.path.normpath(folder)}. Importa i .ogpr per iniziare.",
        )

    def _open_gpr_project(self):
        """Apre un progetto GPR esistente da una cartella scelta."""
        start = str(getattr(self, "_gpr_workspace_override", "") or "").strip() \
            or os.path.expanduser("~")
        folder = QFileDialog.getExistingDirectory(
            self, "Apri progetto GPR — scegli la cartella del progetto", start
        )
        if not folder:
            return
        folder = os.path.normpath(folder)
        # Accetta sia la cartella-progetto (contiene GPR_Processing) sia la
        # cartella GPR_Processing stessa (in tal caso usa la cartella superiore).
        if (not os.path.isfile(os.path.join(folder, "GPR_Processing", "project.json"))
                and os.path.isfile(os.path.join(folder, "project.json"))):
            folder = os.path.dirname(folder)
        self._set_active_workspace(folder)
        if not self._try_import_saved_gpr_processing_project(force=True):
            QMessageBox.information(
                self, "Apri progetto GPR",
                "Nessun profilo trovato nel progetto selezionato.\n"
                "Verifica che la cartella contenga GPR_Processing/project.json.",
            )

    def import_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Importa profili GPR", "",
            import_dialog_filter(),
        )
        if not paths:
            return
        # L'utente sceglie dove salvare i file di lavoro a ogni import.
        self._prompt_workspace_dir()
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
            self._load_current_profile()

    def _next_profile(self):
        if self._profiles:
            self._prof_idx = min(len(self._profiles) - 1, self._prof_idx + 1)
            self._load_current_profile()

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
            self._reload_data()
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
        self._reload_data()

    def _load_current_profile(self):
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
            except Exception:
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
        except Exception:
            self._update_trim_controls_for_channel(0)
        try:
            v_prof = self._profile_reference_velocity(prof)
            rdp_prof = float(np.clip(self._velocity_to_rdp(v_prof), 1.0, 40.0))
            self._spin_hyperbola_rdp.setValue(rdp_prof)
        except Exception:
            pass
        self._reload_data()
        # Auto reset zoom on profile change (same behavior as Reset Zoom button).
        self._reset_zoom()
        if self._ch_idx != 0:
            self._safe_set_text(
                self._lbl_status,
                f"Canale auto-selezionato: Ch {self._ch_idx} (segnale migliore)."
            )
        self._draw_profile_line_on_canvas()

    def _reload_data(self):
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
        start = int(np.clip(self._trim_start_traces, 0, max(0, n_t - 1)))
        end_trim = int(np.clip(self._trim_end_traces, 0, max(0, n_t - 1)))
        end = max(start + 1, n_t - end_trim)
        end = min(end, n_t)
        idx = np.arange(start, end, dtype=np.int64)
        if bool(getattr(self, "_chk_flip_profile", None) and self._chk_flip_profile.isChecked()):
            idx = idx[::-1]
        if idx.size <= 0:
            idx = np.arange(0, min(1, n_t), dtype=np.int64)
        if full.ndim == 2 and idx.size > 0:
            self._raw_data = full[:, idx].copy()
        else:
            self._raw_data = full.copy()
        self._trace_source_indices = idx.astype(np.int64, copy=False)
        self._proc_data = None
        self._stage_cache = {}
        self._hyper_apex_x = None
        self._hyper_apex_depth = None
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

    def _apply_processing(self):
        if self._raw_data is None:
            return
        if not self._is_qt_alive(self):
            return
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
                return
        params = self._current_processing_params()
        self._last_pipeline_params = dict(params)
        has_active_filters = any(
            bool(params.get(k, False))
            for k in (
                "timezero",
                "dewow",
                "bg_removal",
                "bandpass",
                "migration",
                "agc",
                "envelope",
                "pre_agc_gain",
                "manual_gain",
                "dcshift",
                "boxcar",
                "notch",
                "whiten",
                "amp_eq",
                "rad_threshold",
                "power_gain",
                "eigen_filter",
                "median2d",
                "fk_dip",
                "kirchhoff",
                "inst_attr",
            )
        )
        if not has_active_filters:
            self._proc_data = np.array(self._raw_data, dtype=np.float32, copy=True)
            self._refresh_bp_histogram()
            self._apply_gain_only()
            self._safe_set_text(
                self._lbl_status,
                "Modalita' RAW: nessun filtro pipeline attivo.",
            )
            self._request_preview_update(immediate=False)
            return
        if not self._profiles or self._prof_idx < 0 or self._prof_idx >= len(self._profiles):
            return
        prof = self._profiles[self._prof_idx]
        dt_ns_val = float(getattr(prof, "dt_ns", 0.117) or 0.117)
        try:
            self._proc_data = apply_pipeline(
                self._raw_data,
                params,
                dt_ns=dt_ns_val,
                normalize_output=False,
                chain_order=params.get("chain_order"),
                stage_cache=self._stage_cache,
            )
        except Exception as e:
            log_exception("apply_pipeline", e)
            QMessageBox.critical(
                self, "Errore processing",
                f"{e}\n\nDettagli in: {LOG_FILE_PATH}",
            )
            return

        # Fallback robusto: evita profilo "vuoto" quando il risultato e' quasi nullo o non finito.
        proc = np.asarray(self._proc_data)
        finite = np.isfinite(proc)
        finite_ratio = float(finite.mean()) if proc.size else 0.0
        spread = float(np.nanmax(proc) - np.nanmin(proc)) if finite.any() else 0.0
        fallback_msg = None
        if finite_ratio < 0.5 or spread < 1e-6:
            try:
                self._proc_data = np.asarray(self._raw_data, dtype=np.float32)
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
            self._apply_processing()
        else:
            # Gain disabled: update last params so _compose_display_data knows
            # pre_agc is no longer baked into _proc_data.
            if hasattr(self, "_last_pipeline_params"):
                self._last_pipeline_params["pre_agc_gain"] = False
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

    def _channel_trace_vector(self, values, n_t: int) -> np.ndarray:
        target = max(0, int(n_t))
        if target <= 0:
            return np.zeros(0, dtype=np.float64)
        arr = np.asarray(values if values is not None else [], dtype=np.float64)
        idx = self._trace_source_indices
        if (
            isinstance(idx, np.ndarray)
            and idx.ndim == 1
            and idx.size == target
            and arr.size > 0
        ):
            try:
                imin = int(np.nanmin(idx))
                imax = int(np.nanmax(idx))
            except Exception:
                imin, imax = 0, -1
            if imin >= 0 and imax < arr.size:
                try:
                    return np.asarray(arr[idx.astype(np.int64)], dtype=np.float64)
                except Exception:
                    pass
        if arr.size == target:
            return arr.astype(np.float64, copy=False)
        if arr.size <= 0:
            return np.full(target, np.nan, dtype=np.float64)
        if arr.size == 1:
            return np.full(target, float(arr[0]), dtype=np.float64)
        src = np.linspace(0.0, 1.0, arr.size, dtype=np.float64)
        dst = np.linspace(0.0, 1.0, target, dtype=np.float64)
        return np.interp(dst, src, arr).astype(np.float64)

    def _display_distance_axis(self, prof, n_traces: int) -> np.ndarray:
        n_t = max(1, int(n_traces))
        try:
            ch = prof.channel(self._ch_idx)
            distances = self._channel_trace_vector(getattr(ch, "distances", []), n_t)
        except Exception:
            distances = np.zeros(0, dtype=np.float64)
        if distances.size != n_t:
            if distances.size > 1:
                src = np.linspace(0.0, 1.0, distances.size, dtype=np.float64)
                dst = np.linspace(0.0, 1.0, n_t, dtype=np.float64)
                distances = np.interp(dst, src, distances).astype(np.float64)
            else:
                step = float(getattr(prof, "sampling_step_m", 0.1) or 0.1)
                step = max(step, 1e-6)
                distances = (np.arange(n_t, dtype=np.float64) * step).astype(np.float64)
        finite = np.isfinite(distances)
        if not finite.any():
            return np.linspace(0.0, float(max(1, n_t - 1)), n_t, dtype=np.float64)
        if not finite.all():
            idx = np.arange(distances.size, dtype=np.float64)
            ok = np.where(finite)[0].astype(np.float64)
            distances = np.interp(idx, ok, distances[finite]).astype(np.float64)
        d0 = float(distances[0]) if np.isfinite(distances[0]) else 0.0
        distances = distances - d0
        if distances.size > 1:
            # Enforce monotonic axis for noisy/raw cumulative distance vectors.
            distances = np.maximum.accumulate(distances)
        span = float(distances[-1] - distances[0]) if distances.size > 1 else 0.0
        if (not np.isfinite(span)) or span <= 1e-9:
            step = float(getattr(prof, "sampling_step_m", 0.1) or 0.1)
            step = max(step, 1e-6)
            distances = (np.arange(n_t, dtype=np.float64) * step).astype(np.float64)
        return distances

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
        return {"version": 1, "params": params, "gain_hyper": gain_hyper}

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

    def _apply_full_processing_state(self, state: dict) -> None:
        """Ripristina l'intero stato di processing salvato nel progetto e
        riprocessa una sola volta."""
        if not isinstance(state, dict):
            return
        params = state.get("params") if isinstance(state.get("params"), dict) else {}
        gain_hyper = state.get("gain_hyper") if isinstance(state.get("gain_hyper"), dict) else {}
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

    def _draw_hyperbola_overlay_on_axis(self, ax, prof, dist_max: float, depth_max: float):
        if ax is None:
            return
        if not bool(getattr(self, "_chk_hyperbola", None) and self._chk_hyperbola.isChecked()):
            return
        if self._hyper_apex_x is None or self._hyper_apex_depth is None:
            return
        x0 = float(self._hyper_apex_x)
        z0_disp = float(self._hyper_apex_depth)
        if not (np.isfinite(x0) and np.isfinite(z0_disp) and np.isfinite(dist_max) and np.isfinite(depth_max)):
            return
        if z0_disp < 0 or z0_disp > depth_max:
            return

        v_ref = self._profile_reference_velocity(prof)
        rdp = float(self._spin_hyperbola_rdp.value())
        v_fit = self._rdp_to_velocity_m_s(rdp)
        s = self._hyperbola_txrx_offset()
        # Apex sample time from current displayed depth scale.
        t0 = (2.0 * z0_disp) / max(v_ref, 1e-9)
        x = np.linspace(0.0, float(dist_max), 600, dtype=np.float64)
        dx = x - x0
        # Bistatic model (s=0 -> monostatico), poi rimappato in profondita' display.
        z_disp = self._hyperbola_curve_depth(dx, t0, v_fit, v_ref, s)
        finite = np.isfinite(z_disp) & (z_disp >= 0.0) & (z_disp <= (depth_max * 1.02))
        if not finite.any():
            return
        ax.plot(x[finite], z_disp[finite], color="#ffd000", lw=1.5, alpha=0.95, zorder=7)
        ax.scatter([x0], [z0_disp], s=28, color="#ff8c00", edgecolor="#1f1f1f", linewidths=0.5, zorder=8)
        ax.text(
            x0,
            min(depth_max, z0_disp + 0.03 * max(depth_max, 1e-6)),
            f"RDP {rdp:.2f}",
            color="#ffd000",
            fontsize=8,
            ha="left",
            va="bottom",
            zorder=8,
        )

    def _draw_hyperbola_overlay(self, prof, dist_max: float, depth_max: float):
        self._draw_hyperbola_overlay_on_axis(self._ax, prof, dist_max, depth_max)

    def _build_depth_range_gain(self, n_samples: int) -> np.ndarray:
        n = max(1, int(n_samples))
        if not bool(getattr(self, "_chk_range_gain", None) and self._chk_range_gain.isChecked()):
            return np.ones((n, 1), dtype=np.float32)
        t = np.linspace(0.0, 1.0, n, dtype=np.float64)
        pts = self._sanitize_range_gain_breakpoints(self._range_gain_breakpoints)
        x = np.asarray(pts[:, 0], dtype=np.float64)
        y = np.asarray(pts[:, 1], dtype=np.float64)
        g = np.interp(t, x, y)
        g = np.clip(g, 1e-6, 1e6).astype(np.float32)
        return g.reshape(-1, 1)

    def _compose_display_data(self) -> np.ndarray | None:
        if self._proc_data is None:
            return None
        proc = np.asarray(self._proc_data, dtype=np.float32)
        if proc.ndim != 2 or proc.size <= 0:
            return None

        proc_disp = proc.astype(np.float32, copy=True)
        # The range-gain is applied inside the pipeline (pre_agc_gain step) only
        # when the checkbox is currently checked AND the pipeline was actually run
        # with that parameter. We detect this by checking both the current UI state
        # AND the last pipeline params — they must agree to avoid double-application
        # or missed application (e.g. when the checkbox is toggled after processing).
        chk_rg_on = bool(getattr(self, "_chk_range_gain", None) and self._chk_range_gain.isChecked())
        last_pre_agc = bool(getattr(self, "_last_pipeline_params", {}).get("pre_agc_gain", False))
        pre_agc_in_proc = chk_rg_on and last_pre_agc
        if not pre_agc_in_proc:
            # Apply depth gain as display-only transform BEFORE normalize_display.
            depth_gain = self._build_depth_range_gain(int(proc_disp.shape[0]))
            if depth_gain is not None:
                proc_disp = (proc_disp * depth_gain).astype(np.float32, copy=False)

        # Normalize to ±1 range first, then apply display gains on top.
        # The scalar gain and manual_gain are applied AFTER normalization so that
        # normalize_display does not cancel them.  Values that exceed ±1 after
        # the gains are intentionally saturated by the final clip — this is the
        # correct visual behaviour (deep samples clamp to full black/white).
        #
        # The burst zone is now attenuated by pre_agc_gain (TVG) before AGC,
        # so the amplitude is already uniform across the full profile here.
        # No special zeroing of the burst zone is needed.
        base = normalize_display(proc_disp).astype(np.float32, copy=False)

        # Scalar display gain (brightness knob).
        gain = float(self._spin_gain.value()) if hasattr(self, "_spin_gain") else 1.0
        disp = (base * gain).astype(np.float32, copy=False)

        # Manual depth-varying gain — applied last so it is not undone by
        # normalize_display.  The pipeline no longer runs manual_gain; the
        # function is called directly here as a display-only transform.
        chk_mg = getattr(self, "_chk_manual_gain", None)
        if chk_mg is not None and bool(chk_mg.isChecked()):
            try:
                mg_type = str(
                    self._cb_manual_gain_type.currentData() or "linear"
                ) if getattr(self, "_cb_manual_gain_type", None) else "linear"
                mg_start = float(
                    self._spin_manual_gain_start.value()
                ) if getattr(self, "_spin_manual_gain_start", None) else 1.0
                mg_end = float(
                    self._spin_manual_gain_end.value()
                ) if getattr(self, "_spin_manual_gain_end", None) else 10.0
                mg_s0 = int(
                    self._spin_manual_gain_sample_start.value()
                ) if getattr(self, "_spin_manual_gain_sample_start", None) else 0
                mg_s1 = int(
                    self._spin_manual_gain_sample_end.value()
                ) if getattr(self, "_spin_manual_gain_sample_end", None) else 0
                disp = manual_gain(
                    disp,
                    gain_type=mg_type,
                    gain_start=mg_start,
                    gain_end=mg_end,
                    sample_start=mg_s0,
                    sample_end=mg_s1,
                ).astype(np.float32, copy=False)
            except Exception:
                pass

        return np.clip(disp, -1.0, 1.0).astype(np.float32, copy=False)

    def _apply_gain_only(self):
        if self._proc_data is None:
            self._apply_processing()
            return
        self._disp_data = self._compose_display_data()
        if self._disp_data is None:
            return
        gain = float(self._spin_gain.value())
        disp_arr = np.asarray(self._disp_data, dtype=np.float64)
        finite = np.isfinite(disp_arr)
        if finite.any():
            disp_min = float(np.nanmin(disp_arr))
            disp_max = float(np.nanmax(disp_arr))
        else:
            disp_min = float("nan")
            disp_max = float("nan")
        self._safe_set_text(
            self._lbl_status,
            f"proc: min={self._proc_data.min():.3f}  "
            f"max={self._proc_data.max():.3f}  "
            f"disp: min={disp_min:.3f} max={disp_max:.3f}  "
            f"gain={gain:.1f}x  "
            f"range_gain={'on' if self._chk_range_gain.isChecked() else 'off'}"
        )
        self._redraw()

    @staticmethod
    def _clamp_axis_limits(
        lim0: float, lim1: float, axis_min: float, axis_max: float, min_span_ratio: float = 1e-3
    ) -> tuple[float, float]:
        if not (np.isfinite(lim0) and np.isfinite(lim1)):
            return axis_min, axis_max
        span_total = max(axis_max - axis_min, 1e-12)
        forward = lim1 >= lim0
        lo = min(lim0, lim1)
        hi = max(lim0, lim1)
        min_span = span_total * min_span_ratio
        if (hi - lo) < min_span:
            c = 0.5 * (lo + hi)
            lo = c - 0.5 * min_span
            hi = c + 0.5 * min_span
        if lo < axis_min:
            shift = axis_min - lo
            lo += shift
            hi += shift
        if hi > axis_max:
            shift = hi - axis_max
            lo -= shift
            hi -= shift
        lo = max(axis_min, lo)
        hi = min(axis_max, hi)
        return (lo, hi) if forward else (hi, lo)

    def _reset_zoom(self):
        self._view_xlim = None
        self._view_ylim = None
        # Explicit reset should restore full extent, not the initial short-window view.
        self._profile_view_initialized = True
        self._slice_view_xlim = None
        self._slice_view_ylim = None
        self._redraw()

    def _axis_pixel_size(self) -> tuple[float, float]:
        fig_w_px = fig_h_px = None
        try:
            if self._canvas_mpl is not None:
                fig_w_px, fig_h_px = self._canvas_mpl.get_width_height()
        except Exception:
            fig_w_px = fig_h_px = None
        if not (fig_w_px and fig_h_px):
            fig_w_px = float(self._fig.get_figwidth() * 100.0)
            fig_h_px = float(self._fig.get_figheight() * 100.0)
        bbox = self._ax.get_position()
        ax_w_px = max(1.0, float(fig_w_px) * float(bbox.width))
        ax_h_px = max(1.0, float(fig_h_px) * float(bbox.height))
        return float(ax_w_px), float(ax_h_px)

    def _compute_initial_xview(
        self,
        dist_max: float,
        depth_max: float,
        aspect_factor: float,
    ) -> tuple[float, float]:
        if not (np.isfinite(dist_max) and np.isfinite(depth_max)):
            return (0.0, 1.0)
        if dist_max <= 0.0 or depth_max <= 0.0:
            return (0.0, float(max(1.0, dist_max)))
        ax_w_px, ax_h_px = self._axis_pixel_size()
        if ax_h_px <= 1e-6:
            return (0.0, float(dist_max))
        af = float(np.clip(aspect_factor, 1e-6, 1e6))
        ideal_dist_window = float((ax_w_px * depth_max) / (ax_h_px * af))
        ideal_dist_window = float(np.clip(ideal_dist_window, 1.0, dist_max))
        # Keep full extent for short profiles; constrain only when profile is much longer.
        if dist_max <= (ideal_dist_window * 1.05):
            return (0.0, float(dist_max))
        return (0.0, float(ideal_dist_window))

    def _full_x_limits(self) -> tuple[float, float]:
        if self._disp_data is None or not self._profiles:
            return 0.0, 1.0
        prof = self._profiles[self._prof_idx]
        dist_arr = self._display_distance_axis(prof, int(self._disp_data.shape[1]))
        if dist_arr.size and np.isfinite(dist_arr).any():
            x_full_min = float(np.nanmin(dist_arr))
            x_full_max = float(np.nanmax(dist_arr))
        else:
            x_full_min = 0.0
            x_full_max = float(max(1.0, prof.sampling_step_m * max(self._disp_data.shape[1] - 1, 1)))
        if not (np.isfinite(x_full_min) and np.isfinite(x_full_max)) or x_full_max <= x_full_min:
            x_full_min, x_full_max = 0.0, 1.0
        return x_full_min, x_full_max

    def _sync_xpan_slider(self):
        if not hasattr(self, "_xpan_slider") or self._xpan_slider is None:
            return
        x_full_min, x_full_max = self._full_x_limits()
        full_w = float(max(x_full_max - x_full_min, 1e-9))
        x_view = self._view_xlim if self._view_xlim is not None else (x_full_min, x_full_max)
        x_view = self._clamp_axis_limits(x_view[0], x_view[1], x_full_min, x_full_max)
        view_w = float(max(abs(x_view[1] - x_view[0]), 1e-9))

        can_pan = view_w < (full_w - 1e-6)
        self._updating_xpan = True
        try:
            self._xpan_slider.setEnabled(can_pan)
            if not can_pan:
                self._xpan_slider.setValue(0)
                if self._lbl_xpan is not None:
                    self._safe_set_text(self._lbl_xpan, "Full")
                return
            movable = full_w - view_w
            ratio = (float(min(x_view)) - x_full_min) / max(movable, 1e-9)
            ratio = float(np.clip(ratio, 0.0, 1.0))
            self._xpan_slider.setValue(int(round(ratio * 1000.0)))
            if self._lbl_xpan is not None:
                self._safe_set_text(self._lbl_xpan, f"{min(x_view):.1f}-{max(x_view):.1f} m")
        finally:
            self._updating_xpan = False

    def _on_xpan_slider_changed(self, value: int):
        if bool(getattr(self, "_updating_xpan", False)):
            return
        if self._disp_data is None or self._ax is None:
            return
        x_full_min, x_full_max = self._full_x_limits()
        full_w = float(max(x_full_max - x_full_min, 1e-9))
        x_view = self._view_xlim if self._view_xlim is not None else (x_full_min, x_full_max)
        x_view = self._clamp_axis_limits(x_view[0], x_view[1], x_full_min, x_full_max)
        view_w = float(max(abs(x_view[1] - x_view[0]), 1e-9))
        movable = full_w - view_w
        if movable <= 1e-9:
            self._sync_xpan_slider()
            return
        ratio = float(np.clip(float(value) / 1000.0, 0.0, 1.0))
        lo = x_full_min + ratio * movable
        hi = lo + view_w
        self._view_xlim = (lo, hi)
        self._ax.set_xlim(*self._view_xlim)
        if self._canvas_mpl is not None:
            self._safe_draw_idle(self._canvas_mpl)
        self._sync_xpan_slider()

    def _export_radargram_image(self):
        if not HAS_MPL or self._disp_data is None or not self._profiles:
            QMessageBox.information(self, "Export Radargram", "Nessun radargramma da esportare.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Radargram",
            "radargram_export.png",
            "PNG (*.png);;TIFF (*.tif *.tiff)",
        )
        if not path:
            return
        dpi, ok = QInputDialog.getInt(
            self,
            "Export Radargram",
            "DPI:",
            300,
            72,
            2400,
            10,
        )
        if not ok:
            return
        prof = self._profiles[self._prof_idx]
        arr = np.asarray(self._disp_data, dtype=np.float64)
        if arr.ndim != 2 or arr.size <= 0:
            QMessageBox.information(self, "Export Radargram", "Dati radargramma non validi.")
            return
        n_s, n_t = arr.shape
        dist_arr = self._display_distance_axis(prof, n_t)
        if dist_arr.size > 0 and np.isfinite(dist_arr).any():
            dist_max = float(np.nanmax(dist_arr))
        else:
            dist_max = float(max(1.0, getattr(prof, "sampling_step_m", 0.1) * max(n_t - 1, 1)))
        depth_max = self._depth_max_display(prof)
        if not np.isfinite(depth_max) or depth_max <= 0:
            depth_max = float(max(1, n_s))

        finite = arr[np.isfinite(arr)]
        if finite.size:
            vmin = float(np.percentile(finite, 1.0))
            vmax = float(np.percentile(finite, 99.0))
            if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax - vmin < 1e-6):
                vmin = float(np.nanmin(finite))
                vmax = float(np.nanmax(finite))
            if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax - vmin < 1e-6):
                vmin, vmax = -1.0, 1.0
        else:
            vmin, vmax = -1.0, 1.0
        cmap = self._cb_cmap.currentText()
        interpolation_mode = str(self._cb_interp.currentData() or "bilinear")
        real_aspect = bool(self._chk_real_aspect.isChecked())
        vertical_exag = 1.0
        if hasattr(self, "_spin_vertical_exag") and self._spin_vertical_exag is not None:
            try:
                vertical_exag = float(self._spin_vertical_exag.value())
            except Exception:
                vertical_exag = 1.0
        vertical_exag = float(np.clip(vertical_exag, 0.5, 20.0))

        fig = Figure(figsize=(12.0, 5.5), tight_layout=True)
        ax = fig.add_subplot(111)
        aspect_export = "auto"
        if real_aspect and dist_max > 1e-9 and depth_max > 1e-9:
            # VE=1.0 preserves metric scale; VE>1 increases vertical readability.
            aspect_export = float(vertical_exag)
        disp_export = self._resample_disp_to_uniform_grid(self._disp_data, dist_arr)
        ax.imshow(
            disp_export,
            aspect=aspect_export,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=[0, dist_max, depth_max, 0],
            interpolation=interpolation_mode,
            resample=True,
        )
        ax.set_xlim(0.0, dist_max)
        ax.set_ylim(depth_max, 0.0)
        ax.set_xlabel("Distanza (m)")
        ax.set_ylabel("Profondita' (m)")
        ax.set_title(
            f"{os.path.basename(prof.path)}  |  Ch {self._ch_idx}  |  {float(getattr(prof, 'frequency_mhz', 0.0)):.0f} MHz"
        )
        self._draw_hyperbola_overlay_on_axis(ax, prof, dist_max, depth_max)

        ext = os.path.splitext(path)[1].strip().lower()
        if ext not in {".png", ".tif", ".tiff"}:
            path = f"{path}.png"
        try:
            fig.savefig(path, dpi=int(dpi), bbox_inches="tight", facecolor="white")
            self._safe_set_text(self._lbl_status, f"Radargram export: {path}  dpi={int(dpi)}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Radargram", f"Errore export:\n{exc}")

    # ------------------------------------------------------------------
    # Disegno
    # ------------------------------------------------------------------

    @staticmethod
    def _resample_disp_to_uniform_grid(
        disp: np.ndarray,
        dist_arr: np.ndarray,
    ) -> np.ndarray:
        """Resample disp (n_samples, n_traces) onto a uniform distance grid.

        Each column of *disp* corresponds to a trace at the cumulative distance
        dist_arr[i].  Because surveys are driven at variable speed the trace
        spacing is usually non-uniform.  imshow maps every column to an equal-
        width pixel, so non-uniform spacing produces a geometrically distorted
        radargram.  This function corrects that by interpolating each sample row
        onto a regular grid before rendering.
        """
        n_s, n_t = disp.shape
        if n_t < 2 or dist_arr.size < 2:
            return disp
        d = np.asarray(dist_arr, dtype=np.float64)
        if d.size != n_t:
            return disp
        d_min = float(d[0])
        d_max = float(d[-1])
        if not (np.isfinite(d_min) and np.isfinite(d_max)) or d_max <= d_min:
            return disp
        # Check whether the grid is already sufficiently uniform (saves CPU).
        steps = np.diff(d)
        step_mean = float(np.mean(steps))
        if step_mean > 0.0 and float(np.std(steps)) / step_mean < 0.01:
            return disp  # < 1 % variation — no resampling needed
        uniform = np.linspace(d_min, d_max, n_t, dtype=np.float64)
        out = np.empty((n_s, n_t), dtype=np.float32)
        for i in range(n_s):
            out[i] = np.interp(uniform, d, disp[i]).astype(np.float32)
        return out

    def _redraw(self):
        if not HAS_MPL or self._disp_data is None:
            return
        prof = self._profiles[self._prof_idx]
        ch   = prof.channel(self._ch_idx)
        cmap = self._cb_cmap.currentText()

        dist_arr = self._display_distance_axis(prof, int(self._disp_data.shape[1]))
        if dist_arr.size and np.isfinite(dist_arr).any():
            dist_max = float(np.nanmax(dist_arr))
        else:
            dist_max = float(max(1.0, prof.sampling_step_m * max(self._disp_data.shape[1] - 1, 1)))
        if not np.isfinite(dist_max) or dist_max <= 0:
            dist_max = float(max(1.0, prof.sampling_step_m * max(self._disp_data.shape[1] - 1, 1)))

        n_out     = self._disp_data.shape[0]
        n_orig    = prof.n_samples
        depth_base = float(prof.depth_max_m) if np.isfinite(prof.depth_max_m) else 0.0
        depth_max = depth_base * (n_out / n_orig) if n_orig > 0 else depth_base
        if not np.isfinite(depth_max) or depth_max <= 0:
            depth_max = max(1.0, float(n_out))

        # Resample to uniform distance grid so imshow is not geometrically distorted
        # by variable survey speed (non-uniform trace spacing).
        disp_uniform = self._resample_disp_to_uniform_grid(self._disp_data, dist_arr)

        self._ax.clear()
        # Keep fixed display limits so gain controls affect what the user sees.
        vmin, vmax = -1.0, 1.0
        real_aspect = bool(
            getattr(self, "_chk_real_aspect", None)
            and self._chk_real_aspect.isChecked()
        )
        interpolation_mode = "bilinear"
        if hasattr(self, "_cb_interp") and self._cb_interp is not None:
            interpolation_mode = str(self._cb_interp.currentData() or "bilinear")
        ve = 1.0
        if hasattr(self, "_spin_vertical_exag") and self._spin_vertical_exag is not None:
            try:
                ve = float(self._spin_vertical_exag.value())
            except Exception:
                ve = 1.0
        ve = float(np.clip(ve, 0.5, 20.0))
        self._im = self._ax.imshow(
            disp_uniform,
            # Mantieni il riempimento del pannello; la scala reale e' gestita sotto con set_aspect+datalim.
            aspect="auto",
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            extent=[0, dist_max, depth_max, 0],
            interpolation=interpolation_mode,
            resample=True,
        )
        x_full = (0.0, dist_max)
        y_full = (depth_max, 0.0)
        if self._view_xlim is None:
            if not bool(getattr(self, "_profile_view_initialized", True)):
                init_af = ve if real_aspect else 1.0
                self._view_xlim = self._compute_initial_xview(dist_max, depth_max, init_af)
            else:
                self._view_xlim = x_full
        x_view = self._view_xlim
        y_view = self._view_ylim if self._view_ylim is not None else y_full
        x_view = self._clamp_axis_limits(x_view[0], x_view[1], x_full[0], x_full[1])
        y_view = self._clamp_axis_limits(y_view[0], y_view[1], 0.0, depth_max)
        self._view_xlim = x_view
        self._view_ylim = y_view
        self._ax.set_xlim(*x_view)
        self._ax.set_ylim(*y_view)
        if real_aspect and dist_max > 1e-9 and depth_max > 1e-9:
            # VE=1.0 keeps metric proportions, VE>1.0 increases vertical emphasis.
            self._ax.set_aspect(float(np.clip(ve, 0.5, 20.0)), adjustable="box")
        else:
            # In auto mode avoid excessive horizontal stretch on short profiles.
            ax_w_px, ax_h_px = self._axis_pixel_size()
            x_range = max(1e-9, float(abs(x_view[1] - x_view[0])))
            y_range = max(1e-9, float(abs(y_view[0] - y_view[1])))
            px_per_m_x = ax_w_px / x_range
            px_per_m_y = ax_h_px / y_range
            max_ratio = 4.0
            if px_per_m_x > (px_per_m_y * max_ratio):
                ratio = float((y_range / x_range) / max_ratio)
                self._ax.set_aspect(float(np.clip(ratio, 1e-4, 1e4)), adjustable="box")
            else:
                self._ax.set_aspect("auto", adjustable="box")
        self._ax.set_xlabel("Distanza (m)")
        self._ax.set_ylabel("Profondit\u00e0 (m)")
        self._ax.set_title(
            f"{os.path.basename(prof.path)}  |  "
            f"Ch {self._ch_idx}  |  "
            f"{prof.frequency_mhz:.0f} MHz"
        )
        self._draw_hyperbola_overlay(prof, dist_max, depth_max)
        self._vline = self._ax.axvline(x=0, color="yellow", lw=1, visible=False)
        self._hline = self._ax.axhline(y=0, color="cyan",   lw=1, linestyle="--", visible=False)
        self._apply_wiggle_visibility_layout()
        self._sync_xpan_slider()
        self._update_wiggle_plot(draw=False, force=True)
        self._profile_view_initialized = True
        self._safe_draw_idle(self._canvas_mpl)
        self._redraw_slice_view()

    def _depth_max_display(self, prof) -> float:
        if self._disp_data is not None:
            n_out = int(self._disp_data.shape[0])
        elif self._proc_data is not None:
            n_out = int(self._proc_data.shape[0])
        elif self._raw_data is not None:
            n_out = int(self._raw_data.shape[0])
        else:
            n_out = int(getattr(prof, "n_samples", 0) or 0)
        n_orig = int(getattr(prof, "n_samples", 0) or 0)
        depth_base = float(getattr(prof, "depth_max_m", 0.0) or 0.0)
        depth_max = depth_base * (n_out / n_orig) if n_orig > 0 else depth_base
        if not np.isfinite(depth_max) or depth_max <= 0:
            depth_max = max(1.0, float(n_out))
        return float(depth_max)

    def _cursor_metrics(self):
        if (
            not self._profiles
            or self._disp_data is None
            or self._cursor_x is None
            or self._cursor_z is None
        ):
            return None
        try:
            prof = self._profiles[self._prof_idx]
            ch = prof.channel(self._ch_idx)
        except Exception:
            return None

        n_tr = int(self._disp_data.shape[1])
        distances = self._channel_trace_vector(getattr(ch, "distances", []), n_tr)
        if distances.size != n_tr:
            step = float(getattr(prof, "sampling_step_m", 0.1) or 0.1)
            distances = (np.arange(n_tr, dtype=np.float64) * max(step, 1e-6)).astype(np.float64)
        if distances.size <= 0:
            return None

        # Normalize distances to start at 0, matching the display axis extent
        # built by _display_distance_axis (which subtracts d0).
        if distances.size > 0 and np.isfinite(distances[0]):
            distances = distances - distances[0]
        finite_d = np.isfinite(distances)
        if not finite_d.any():
            return None
        if not finite_d.all():
            idx_arr = np.arange(distances.size, dtype=np.float64)
            ok = np.where(finite_d)[0].astype(np.float64)
            distances = np.interp(idx_arr, ok, distances[finite_d]).astype(np.float64)
        if distances.size > 1:
            distances = np.maximum.accumulate(distances)

        depth_max = self._depth_max_display(prof)
        idx_t = int(np.searchsorted(distances, float(self._cursor_x)))
        idx_t = int(np.clip(idx_t, 0, max(0, n_tr - 1)))
        n_s = int(self._disp_data.shape[0])
        idx_s = int(np.rint((float(self._cursor_z) / max(depth_max, 1e-9)) * max(0, n_s - 1)))
        idx_s = int(np.clip(idx_s, 0, max(0, n_s - 1)))

        amp = None
        try:
            a = float(self._disp_data[idx_s, idx_t])
            amp = a if np.isfinite(a) else None
        except Exception:
            amp = None

        east = north = None
        try:
            e_vec = self._channel_trace_vector(getattr(ch, "easting", []), n_tr)
            n_vec = self._channel_trace_vector(getattr(ch, "northing", []), n_tr)
            if idx_t < e_vec.size:
                east = float(e_vec[idx_t])
            if idx_t < n_vec.size:
                north = float(n_vec[idx_t])
            if east is not None and not np.isfinite(east):
                east = None
            if north is not None and not np.isfinite(north):
                north = None
        except Exception:
            east = north = None

        return {
            "idx_trace": int(idx_t),
            "idx_sample": int(idx_s),
            "distance_m": float(distances[idx_t]),
            "depth_m": float(self._cursor_z),
            "time_ns": float(idx_s * float(getattr(prof, "dt_ns", 0.0) or 0.0)),
            "amplitude": amp,
            "east": east,
            "north": north,
            "depth_max_m": depth_max,
        }

    def _update_wiggle_plot(self, draw: bool = True, force: bool = False):
        if not HAS_MPL or not hasattr(self, "_ax_wiggle"):
            return
        axw = self._ax_wiggle
        if not bool(getattr(self, "_show_wiggle", True)):
            try:
                axw.set_visible(False)
            except Exception:
                pass
            return
        else:
            try:
                axw.set_visible(True)
            except Exception:
                pass
        if self._disp_data is None or not self._profiles:
            axw.clear()
            axw.set_xticks([])
            axw.set_yticks([])
            self._wiggle_trace_idx = None
            self._wiggle_depth_line = None
            if draw:
                self._safe_draw_idle(self._canvas_mpl)
            return

        metrics = self._cursor_metrics()
        if metrics is None:
            idx_t = int(np.clip(self._disp_data.shape[1] // 2, 0, max(0, self._disp_data.shape[1] - 1)))
            idx_s = 0
            depth_max = self._depth_max_display(self._profiles[self._prof_idx])
        else:
            idx_t = int(metrics["idx_trace"])
            idx_s = int(metrics["idx_sample"])
            depth_max = float(metrics["depth_max_m"])

        filt_src = self._proc_data if self._proc_data is not None else self._disp_data
        raw_src = self._raw_data if self._raw_data is not None else filt_src
        if filt_src is None or raw_src is None:
            axw.clear()
            axw.set_xticks([])
            axw.set_yticks([])
            self._wiggle_trace_idx = None
            self._wiggle_depth_line = None
            if draw:
                self._safe_draw_idle(self._canvas_mpl)
            return

        idx_t_f = int(np.clip(idx_t, 0, max(0, int(filt_src.shape[1]) - 1)))
        idx_t_r = int(np.clip(idx_t, 0, max(0, int(raw_src.shape[1]) - 1)))
        trace_f = np.asarray(filt_src[:, idx_t_f], dtype=np.float64)
        trace_r = np.asarray(raw_src[:, idx_t_r], dtype=np.float64)
        if trace_f.size <= 0 or trace_r.size <= 0:
            axw.clear()
            axw.set_xticks([])
            axw.set_yticks([])
            self._wiggle_trace_idx = None
            self._wiggle_depth_line = None
            if draw:
                self._safe_draw_idle(self._canvas_mpl)
            return

        # Align raw to filtered length without distorting waveform:
        # if processing shortens the trace (e.g. time-zero), drop the leading
        # raw samples instead of interpolating the whole raw trace.
        n = int(trace_f.size)
        n_raw = int(trace_r.size)
        if n_raw >= n:
            off = max(0, n_raw - n)
            trace_r_plot = trace_r[off:off + n]
        else:
            trace_r_plot = np.full((n,), np.nan, dtype=np.float64)
            trace_r_plot[:n_raw] = trace_r
        trace_f_plot = trace_f[:n]
        depth_axis = np.linspace(0.0, float(depth_max), n, dtype=np.float64)
        can_incremental = (
            (not bool(force))
            and self._wiggle_trace_idx is not None
            and int(self._wiggle_trace_idx) == int(idx_t)
            and self._wiggle_depth_line is not None
        )

        if can_incremental:
            if 0 <= idx_s < depth_axis.size:
                y = float(depth_axis[idx_s])
                try:
                    self._wiggle_depth_line.set_ydata([y, y])
                except Exception:
                    can_incremental = False
            if can_incremental and draw:
                self._safe_draw_idle(self._canvas_mpl)
            if can_incremental:
                return

        # Full redraw only when trace index or data changes.
        axw.clear()
        finite_fil = np.isfinite(trace_f_plot)
        finite_raw = np.isfinite(trace_r_plot)
        scale_fil = 1.0
        if finite_fil.any():
            try:
                scale_fil = float(np.nanpercentile(np.abs(trace_f_plot[finite_fil]), 99.0))
            except Exception:
                scale_fil = float(np.nanmax(np.abs(trace_f_plot[finite_fil])))
        if (not np.isfinite(scale_fil)) or scale_fil <= 1e-9:
            scale_fil = 1.0

        scale_raw = 1.0
        if finite_raw.any():
            try:
                scale_raw = float(np.nanpercentile(np.abs(trace_r_plot[finite_raw]), 99.0))
            except Exception:
                scale_raw = float(np.nanmax(np.abs(trace_r_plot[finite_raw])))
        if (not np.isfinite(scale_raw)) or scale_raw <= 1e-9:
            scale_raw = 1.0

        # Separate normalization keeps RAW visible/stable across filter toggles.
        fil_norm = np.nan_to_num(trace_f_plot / scale_fil, nan=0.0, posinf=0.0, neginf=0.0)
        raw_norm = np.nan_to_num(trace_r_plot / scale_raw, nan=0.0, posinf=0.0, neginf=0.0)

        axw.plot(raw_norm, depth_axis, color="#dc5050", lw=0.9, alpha=0.45, antialiased=True)
        axw.plot(fil_norm, depth_axis, color="#50dc78", lw=0.9, alpha=0.95, antialiased=True)
        self._wiggle_depth_line = None
        if 0 <= idx_s < depth_axis.size:
            self._wiggle_depth_line = axw.axhline(
                float(depth_axis[idx_s]),
                color="goldenrod",
                lw=0.8,
                ls="--",
            )
        axw.set_ylim(float(depth_max), 0.0)
        xlim = float(np.nanmax(np.abs(np.r_[raw_norm, fil_norm]))) if raw_norm.size > 0 else 1.0
        if (not np.isfinite(xlim)) or xlim <= 0.1:
            xlim = 1.0
        xlim = float(np.clip(xlim * 1.05, 0.5, 5.0))
        axw.set_xlim(-xlim, xlim)
        axw.set_title(f"A-scan T{idx_t}", fontsize=7, pad=2)
        axw.set_xlabel("")
        axw.set_ylabel("")
        axw.set_xticks([])
        axw.set_yticks([])
        axw.grid(False)
        axw.axvline(0.0, color="#555555", lw=0.5)
        axw.set_facecolor("#1a1a1a")
        for spine in axw.spines.values():
            spine.set_edgecolor("#333333")
            spine.set_linewidth(0.5)
        self._wiggle_trace_idx = int(idx_t)
        if draw:
            self._safe_draw_idle(self._canvas_mpl)

    # ------------------------------------------------------------------
    # Cursore
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Pick-mode: click sul profilo → imposta un campo sample
    # ------------------------------------------------------------------

    def _set_pick_mode(self, target: str | None, source_btn=None):
        """Attiva/disattiva la modalità pick-from-profile.

        target: nome logico del campo da popolare ('agc_sample_start', ...).
                None = disattiva.
        source_btn: il QPushButton che ha scatenato l'azione (per sincronizzare
                    lo stato checked se la disattivazione arriva dall'esterno).
        """
        self._pick_mode_target = target
        # Deseleziona tutti i bottoni pick tranne quello sorgente
        for btn_name in ("_btn_pick_agc_s0",):
            btn = getattr(self, btn_name, None)
            if btn is not None and self._is_qt_alive(btn) and btn is not source_btn:
                btn.blockSignals(True)
                btn.setChecked(False)
                btn.blockSignals(False)
        # Cambia il cursore del canvas per dare feedback visivo
        if HAS_MPL and getattr(self, "_canvas_mpl", None) and self._is_qt_alive(self._canvas_mpl):
            if target is not None:
                self._canvas_mpl.setCursor(Qt.CrossCursor)
            else:
                self._canvas_mpl.unsetCursor()

    def _sample_from_ydata(self, ydata: float) -> int | None:
        """Converte la coordinata Y del canvas (profondita') in indice sample."""
        if self._disp_data is None:
            return None
        n_s = int(self._disp_data.shape[0])
        if n_s <= 0:
            return None
        prof = self._profiles[self._prof_idx] if self._profiles else None
        if prof is None:
            return None
        try:
            depth_max = float(getattr(prof, "depth_m", None) or 0.0)
            if depth_max <= 0.0:
                return None
            # ydata è in metri; convertiamo in indice sample
            frac = float(ydata) / depth_max
            idx = int(round(frac * (n_s - 1)))
            return max(0, min(n_s - 1, idx))
        except Exception:
            return None

    def _on_mouse_move(self, event):
        if event.inaxes != self._ax or self._disp_data is None:
            return
        self._cursor_x = event.xdata
        self._cursor_z = event.ydata
        self._update_cursor_lines()
        self._update_wiggle_plot(draw=False)
        self._update_status()
        self._canvas_timer.start()

    def _on_mouse_press(self, event):
        if event.inaxes == self._ax and event.button == 1:
            self._cursor_x = event.xdata
            self._cursor_z = event.ydata

            # Pick-mode: popola il campo target e disattiva la modalità
            if self._pick_mode_target is not None and event.ydata is not None:
                sample_idx = self._sample_from_ydata(float(event.ydata))
                if sample_idx is not None:
                    if self._pick_mode_target == "agc_sample_start":
                        if getattr(self, "_spin_agc_sample_start", None) and self._is_qt_alive(self._spin_agc_sample_start):
                            self._spin_agc_sample_start.setValue(sample_idx)
                    # Disattiva pick-mode dopo la selezione
                    self._set_pick_mode(None, None)
                return

            self._update_wiggle_plot(draw=False)
            if bool(getattr(event, "dblclick", False)) and bool(self._chk_hyperbola.isChecked()):
                self._set_hyperbola_apex_from_cursor()
            self._update_dial()
            self._emit_cursor_moved()

    def _on_scroll_zoom(self, event):
        if event.inaxes != self._ax or self._im is None:
            return
        if event.xdata is None or event.ydata is None:
            return

        step = float(getattr(event, "step", 0.0) or 0.0)
        btn = str(getattr(event, "button", "") or "").lower()
        if step > 0 or btn == "up":
            scale = 1.0 / 1.2
        elif step < 0 or btn == "down":
            scale = 1.2
        else:
            return

        x0, x1 = self._ax.get_xlim()
        y0, y1 = self._ax.get_ylim()
        x = float(event.xdata)
        y = float(event.ydata)

        nx0 = x - (x - x0) * scale
        nx1 = x + (x1 - x) * scale
        ny0 = y - (y - y0) * scale
        ny1 = y + (y1 - y) * scale

        ex0, ex1, ey0, ey1 = self._im.get_extent()
        x_min, x_max = float(min(ex0, ex1)), float(max(ex0, ex1))
        y_min, y_max = float(min(ey0, ey1)), float(max(ey0, ey1))

        self._view_xlim = self._clamp_axis_limits(nx0, nx1, x_min, x_max)
        self._view_ylim = self._clamp_axis_limits(ny0, ny1, y_min, y_max)

        self._ax.set_xlim(*self._view_xlim)
        self._ax.set_ylim(*self._view_ylim)
        self._sync_xpan_slider()
        self._safe_draw_idle(self._canvas_mpl)

    def _on_axes_leave(self, event):
        if self._vline: self._vline.set_visible(False)
        if self._hline: self._hline.set_visible(False)
        self._safe_draw_idle(self._canvas_mpl)

    def _update_cursor_lines(self):
        if self._vline and self._cursor_x is not None:
            self._vline.set_xdata([self._cursor_x])
            self._vline.set_visible(True)
        if self._hline and self._cursor_z is not None:
            self._hline.set_ydata([self._cursor_z])
            self._hline.set_visible(True)
        self._safe_draw_idle(self._canvas_mpl)

    def _update_status(self):
        if not self._profiles:
            return
        m = self._cursor_metrics()
        if not m:
            z_str = f"{self._cursor_z:.3f} m" if self._cursor_z is not None else "\u2014"
            x_str = f"{self._cursor_x:.2f} m" if self._cursor_x is not None else "\u2014"
            self._safe_set_text(
                self._lbl_status,
                f"Dist: {x_str}  |  Profondita': {z_str}"
            )
            return
        amp = m.get("amplitude")
        amp_str = f"{float(amp):.6g}" if amp is not None else "n/a"
        east = m.get("east")
        north = m.get("north")
        east_str = f"{float(east):.2f}" if east is not None else "n/a"
        north_str = f"{float(north):.2f}" if north is not None else "n/a"
        self._safe_set_text(
            self._lbl_status,
            f"Trace={int(m['idx_trace'])}  Sample={int(m['idx_sample'])}  "
            f"Amp={amp_str}  Time={float(m['time_ns']):.3f} ns  "
            f"Depth={float(m['depth_m']):.3f} m  Dist={float(m['distance_m']):.3f} m  "
            f"E {east_str}  N {north_str}"
        )

    def _cursor_world_position(self):
        m = self._cursor_metrics()
        if not m:
            return None, None, None
        east = m.get("east")
        north = m.get("north")
        depth = float(m.get("depth_m", np.nan))
        if east is None or north is None:
            return None, None, None
        if not (np.isfinite(float(east)) and np.isfinite(float(north)) and np.isfinite(depth)):
            return None, None, None
        return float(east), float(north), float(depth)

    def _emit_cursor_moved(self):
        east, north, depth = self._cursor_world_position()
        if east is None or north is None or depth is None:
            return
        try:
            self.cursor_moved.emit(float(east), float(north), float(depth))
        except Exception:
            pass
        if not self._suppress_3d_depth_sync and self._gpr_3d_viewer is not None:
            try:
                self._gpr_3d_viewer.set_depth_from_external(float(depth))
            except Exception:
                pass

    def _on_3d_depth_changed(self, depth: float):
        if not np.isfinite(depth):
            return
        if self._profiles:
            try:
                prof = self._profiles[self._prof_idx]
                n_out = int(self._disp_data.shape[0]) if self._disp_data is not None else int(prof.n_samples)
                n_orig = int(prof.n_samples) if int(prof.n_samples) > 0 else max(1, n_out)
                depth_base = float(prof.depth_max_m) if np.isfinite(prof.depth_max_m) else 0.0
                depth_max = depth_base * (n_out / n_orig) if n_orig > 0 else depth_base
                if not np.isfinite(depth_max) or depth_max <= 0:
                    depth_max = max(1.0, float(n_out))
                depth = float(np.clip(float(depth), 0.0, depth_max))
            except Exception:
                depth = float(depth)
        else:
            depth = float(depth)

        self._suppress_3d_depth_sync = True
        try:
            self._cursor_z = depth
            self._update_cursor_lines()
            self._update_wiggle_plot(draw=False)
            self._update_status()
            self._update_dial()
            self._emit_cursor_moved()
            self._canvas_timer.start()
        finally:
            self._suppress_3d_depth_sync = False

    # ------------------------------------------------------------------
    # Bridge QGIS
    # ------------------------------------------------------------------

    def _flush_canvas_update(self):
        if not self._is_qt_alive(self):
            return
        try:
            self._update_rubber_band()
            self._update_dial()
            self._update_timeslice_visibility_in_canvas()
            self._update_timeslice_crosshair()
            self._emit_cursor_moved()
            if bool(getattr(self, "_preview_mode_active", False)):
                self._request_preview_update(immediate=False)
            if self._is_qt_alive(getattr(self, "_canvas_slice", None)):
                self._redraw_slice_view()
        except RuntimeError:
            pass

    def _to_canvas_point(self, prof, east: float, north: float) -> QgsPointXY:
        pt = QgsPointXY(float(east), float(north))
        if self.iface is None:
            return pt
        canvas = self.iface.mapCanvas()
        if canvas is None:
            return pt
        try:
            epsg = int(getattr(prof, "epsg", 0) or 0)
        except Exception:
            epsg = 0
        if epsg <= 0:
            # Fallback: prova CRS progetto QGIS quando l'EPSG profilo manca.
            try:
                src = QgsProject.instance().crs()
                dst = canvas.mapSettings().destinationCrs()
                if src.isValid() and dst.isValid() and src != dst:
                    tr = QgsCoordinateTransform(src, dst, QgsProject.instance())
                    return tr.transform(pt)
            except Exception:
                pass
            return pt
        try:
            src = QgsCoordinateReferenceSystem.fromEpsgId(epsg)
            dst = canvas.mapSettings().destinationCrs()
            if not src.isValid() or not dst.isValid() or src == dst:
                return pt
            tr = QgsCoordinateTransform(src, dst, QgsProject.instance())
            return tr.transform(pt)
        except Exception:
            return pt

    def _update_rubber_band(self):
        if self._cursor_x is None or not self._profiles or self.iface is None:
            return
        m = self._cursor_metrics()
        if not m:
            return
        east = m.get("east")
        north = m.get("north")
        if east is None or north is None:
            return
        prof = self._profiles[self._prof_idx]
        canvas = self.iface.mapCanvas()
        if self._rb_point is None:
            self._rb_point = QgsRubberBand(canvas, QgsWkbTypes.PointGeometry)
            self._rb_point.setColor(QColor(255, 220, 0))
        self._rb_point.setIconSize(12)
        self._rb_point.setWidth(3)
        self._rb_point.setVisible(True)
        self._rb_point.reset(QgsWkbTypes.PointGeometry)
        pt_canvas = self._to_canvas_point(prof, float(east), float(north))
        self._rb_point.addPoint(
            pt_canvas, True
        )

    def _draw_profile_line_on_canvas(self):
        if not self._profiles or self.iface is None:
            return
        prof   = self._profiles[self._prof_idx]
        ch     = prof.channel(self._ch_idx)
        canvas = self.iface.mapCanvas()
        if self._rb_line is None:
            self._rb_line = QgsRubberBand(canvas, QgsWkbTypes.LineGeometry)
            self._rb_line.setColor(QColor(255, 165, 0))
            self._rb_line.setWidth(2)
            self._rb_line.setVisible(True)
        else:
            self._rb_line.reset(QgsWkbTypes.LineGeometry)
        if self._disp_data is not None and np.asarray(self._disp_data).ndim == 2:
            n_t = int(self._disp_data.shape[1])
        else:
            try:
                n_t = int(np.asarray(ch.data).shape[1])
            except Exception:
                n_t = 0
        e_vec = self._channel_trace_vector(getattr(ch, "easting", []), n_t)
        n_vec = self._channel_trace_vector(getattr(ch, "northing", []), n_t)
        finite = np.isfinite(e_vec) & np.isfinite(n_vec)
        if int(np.count_nonzero(finite)) < 2:
            return
        e_use = e_vec[finite]
        n_use = n_vec[finite]
        for i in range(len(e_use)):
            pt_canvas = self._to_canvas_point(prof, e_use[i], n_use[i])
            self._rb_line.addPoint(
                pt_canvas,
                i == len(e_use) - 1
            )

    def _clear_timeslice_crosshair(self):
        for rb in (self._rb_crosshair_h, self._rb_crosshair_v):
            if rb is not None:
                try:
                    rb.setVisible(False)
                except Exception:
                    pass
                try:
                    rb.reset(QgsWkbTypes.LineGeometry)
                except Exception:
                    pass

    def _get_cached_catalog(self, project_root: str):
        if not project_root:
            return None
        cat_path = os.path.join(project_root, "metadata", "project_catalog.json")
        try:
            mtime = float(os.path.getmtime(cat_path))
        except Exception:
            mtime = None
        needs_reload = (
            self._cached_catalog is None
            or self._cached_catalog_pr != project_root
            or self._cached_catalog_mtime != mtime
        )
        if needs_reload:
            from .project_catalog import load_catalog

            self._cached_catalog = load_catalog(project_root)
            self._cached_catalog_pr = project_root
            self._cached_catalog_mtime = mtime
        return self._cached_catalog

    def _active_group_timeslices_context(self):
        if self.plugin is None:
            return None, None
        settings = getattr(self.plugin, "settings", None)
        settings_key = getattr(self.plugin, "settings_key_active_project", None)
        if settings is None or not settings_key:
            return None, None
        try:
            pr = (settings.value(settings_key, "", type=str) or "").strip()
        except Exception:
            pr = ""
        if not pr:
            return None, None
        catalog = self._get_cached_catalog(pr)
        if not isinstance(catalog, dict):
            return None, None

        groups = [g for g in catalog.get("raster_groups", []) if isinstance(g, dict)]
        if not groups:
            return None, None

        active_group = None
        active_group_id = str(getattr(self.plugin, "_active_group_id", "") or "").strip()
        if active_group_id:
            active_group = next((g for g in groups if str(g.get("id") or "").strip() == active_group_id), None)

        if active_group is None:
            dlg = getattr(self.plugin, "dlg", None)
            glw = getattr(dlg, "groupListWidget", None)
            current_item = None
            try:
                current_item = glw.currentItem() if glw is not None else None
                if current_item is None and glw is not None:
                    selected_items = list(glw.selectedItems() or [])
                    current_item = selected_items[0] if selected_items else None
            except Exception:
                current_item = None
            if current_item is not None:
                gid = str(current_item.data(Qt.UserRole) or "").strip()
                gname = str(current_item.text() or "").strip()
                if gid:
                    active_group = next((g for g in groups if str(g.get("id") or "").strip() == gid), None)
                if active_group is None and gname:
                    active_group = next(
                        (g for g in groups if str(g.get("name") or "").strip().lower() == gname.lower()),
                        None,
                    )

        if active_group is None and hasattr(self.plugin, "_selected_group_names"):
            try:
                selected_names = list(getattr(self.plugin, "_selected_group_names")() or [])
            except Exception:
                selected_names = []
            for name in selected_names:
                key = str(name or "").strip().lower()
                if not key:
                    continue
                active_group = next(
                    (g for g in groups if str(g.get("name") or "").strip().lower() == key),
                    None,
                )
                if active_group is not None:
                    break

        if active_group is None:
            active_group = next((g for g in groups if g.get("timeslice_ids")), None)
        if active_group is None:
            return None, None

        tids = [str(tid).strip() for tid in (active_group.get("timeslice_ids") or []) if str(tid).strip()]
        if not tids:
            return active_group, []
        ts_by_id = {
            str(t.get("id") or "").strip(): t
            for t in catalog.get("timeslices", [])
            if isinstance(t, dict) and str(t.get("id") or "").strip()
        }
        slices = [ts_by_id[tid] for tid in tids if tid in ts_by_id]
        if not slices:
            tids_set = set(tids)
            slices = [
                t for t in catalog.get("timeslices", [])
                if isinstance(t, dict) and str(t.get("id") or "").strip() in tids_set
            ]
        return active_group, slices

    def _nearest_timeslice_index(self, slices, depth_m: float):
        if not slices:
            return None
        centers = []
        for t in slices:
            try:
                d0 = float(t.get("depth_from", 0.0) or 0.0)
                d1 = float(t.get("depth_to", d0) or d0)
            except Exception:
                d0 = 0.0
                d1 = 0.0
            centers.append((d0 + d1) * 0.5)
        depth_arr = np.asarray(centers, dtype=np.float64)
        if depth_arr.size <= 0 or not np.isfinite(depth_arr).any():
            return None
        try:
            target = float(depth_m)
        except Exception:
            return None
        if not np.isfinite(target):
            return None
        return int(np.nanargmin(np.abs(depth_arr - target)))

    def _active_project_root(self) -> str:
        root = ""
        if self.plugin is not None:
            settings = getattr(self.plugin, "settings", None)
            settings_key = getattr(self.plugin, "settings_key_active_project", None)
            if settings is not None and settings_key:
                try:
                    root = (settings.value(settings_key, "", type=str) or "").strip()
                except Exception:
                    root = ""
        if root:
            return root
        # Fallback: directory of current QGIS project file.
        try:
            qgs_file = str(QgsProject.instance().fileName() or "").strip()
        except Exception:
            qgs_file = ""
        if qgs_file:
            try:
                return os.path.dirname(os.path.normpath(qgs_file))
            except Exception:
                return ""
        return ""

    def _set_active_workspace(self, root: str) -> None:
        """Imposta la cartella-progetto attiva (dove vive GPR_Processing).

        Aggiorna sia l'override locale del viewer sia il setting del plugin,
        cosi' viewer e plugin (slicer/catalogo) puntano allo stesso progetto.
        Il valore persiste tra sessioni tramite le impostazioni del plugin.
        """
        root = os.path.normpath(str(root or "").strip())
        if not root:
            return
        self._gpr_workspace_override = root
        try:
            settings = getattr(self.plugin, "settings", None)
            key = getattr(self.plugin, "settings_key_active_project", None)
            if settings is not None and key:
                settings.setValue(key, root)
        except Exception:
            pass

    def _gpr_processing_paths(self) -> dict:
        # Workspace scelto dall'utente (Nuovo/Apri/import): priorita' sul setting.
        root = str(getattr(self, "_gpr_workspace_override", "") or "").strip()
        if not root:
            root = self._active_project_root()
        if not root:
            return {}
        base = os.path.normpath(os.path.join(root, "GPR_Processing"))
        return {
            "base": base,
            "raw": os.path.join(base, "raw"),
            "processed": os.path.join(base, "processed"),
            "timeslices": os.path.join(base, "timeslices"),
            "project_json": os.path.join(base, "project.json"),
        }

    def _ensure_gpr_processing_layout(self) -> dict:
        paths = self._gpr_processing_paths()
        if not paths:
            return {}
        try:
            os.makedirs(paths["raw"], exist_ok=True)
            os.makedirs(paths["processed"], exist_ok=True)
            os.makedirs(paths["timeslices"], exist_ok=True)
            os.makedirs(paths["base"], exist_ok=True)
        except Exception:
            return {}
        return paths

    def _path_to_gpr_processing_store(self, path: str) -> str:
        p = os.path.normpath(str(path or "").strip())
        if not p:
            return ""
        paths = self._gpr_processing_paths()
        base = paths.get("base", "")
        if not base:
            return p
        try:
            cp = os.path.commonpath([os.path.abspath(p), os.path.abspath(base)])
            if cp == os.path.abspath(base):
                rel = os.path.relpath(os.path.abspath(p), os.path.abspath(base))
                return rel.replace("\\", "/")
        except Exception:
            pass
        return p

    def _path_from_gpr_processing_store(self, stored_path: str) -> str:
        raw = str(stored_path or "").strip()
        if not raw:
            return ""
        paths = self._gpr_processing_paths()
        base = paths.get("base", "")
        if os.path.isabs(raw) or not base:
            return os.path.normpath(raw)
        return os.path.normpath(os.path.join(base, raw))

    def _load_gpr_processing_project_meta(self) -> dict:
        paths = self._gpr_processing_paths()
        pj = paths.get("project_json", "")
        if not pj or (not os.path.exists(pj)):
            return {}
        try:
            with open(pj, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_gpr_processing_project_meta(
        self,
        last_processing: Optional[str] = None,
        last_timeslice: Optional[str] = None,
    ) -> bool:
        paths = self._ensure_gpr_processing_layout()
        if not paths:
            return False
        now_iso = datetime.now().isoformat(timespec="seconds")
        meta = self._load_gpr_processing_project_meta()
        created = str(meta.get("created") or now_iso)
        profiles_payload = []
        for prof in list(getattr(self, "_profiles", []) or []):
            try:
                p_abs = os.path.normpath(str(getattr(prof, "path", "") or ""))
                if not p_abs:
                    continue
                profiles_payload.append({
                    "filename": os.path.basename(p_abs),
                    "path": self._path_to_gpr_processing_store(p_abs),
                    "channels": list(range(int(getattr(prof, "n_channels", 0) or 0))),
                    "loaded": True,
                })
            except Exception:
                continue
        ui_state = {
            "colormap": (
                str(self._cb_cmap.currentText())
                if self._is_qt_alive(getattr(self, "_cb_cmap", None))
                else ""
            ),
            "ve": (
                float(self._spin_vertical_exag.value())
                if self._is_qt_alive(getattr(self, "_spin_vertical_exag", None))
                else 1.0
            ),
            "render": (
                str(self._cb_interp.currentText())
                if self._is_qt_alive(getattr(self, "_cb_interp", None))
                else "Smooth"
            ),
            "show_wiggle": bool(
                self._chk_show_wiggle.isChecked()
                if self._is_qt_alive(getattr(self, "_chk_show_wiggle", None))
                else self._show_wiggle
            ),
        }
        # Fase D: persist the SliceSpec form verbatim.
        _panel = getattr(self, "_slice_spec_panel", None)
        if _panel is not None and self._is_qt_alive(_panel):
            try:
                ui_state["slice_spec"] = _panel.get_spec().to_dict()
            except Exception:
                pass
        if last_timeslice is None:
            try:
                outdir = str(self._le_slice_outdir.text() or "").strip()
            except Exception:
                outdir = ""
            if outdir:
                last_timeslice = self._path_to_gpr_processing_store(os.path.normpath(outdir))
        payload = {
            "version": "1.0",
            "created": created,
            "updated": now_iso,
            "profiles": profiles_payload,
            "last_processing": (
                last_processing
                if last_processing is not None
                else meta.get("last_processing")
            ),
            "last_timeslice": (
                last_timeslice
                if last_timeslice is not None
                else meta.get("last_timeslice")
            ),
            "ui_state": ui_state,
            # Preserve manual grid positioning params across sessions
            "grid_positioning": getattr(self, "_grid_positioning_params", None) or meta.get("grid_positioning"),
            # Ricetta completa della pipeline: evita di riprocessare da zero a
            # ogni riapertura del viewer.
            "processing_state": (
                self._collect_full_processing_state()
                if self._is_qt_alive(getattr(self, "_filter_chain_widget", None))
                else meta.get("processing_state")
            ),
        }
        try:
            with open(paths["project_json"], "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False

    def _stage_profile_in_project_raw(self, src_path: str) -> str:
        p_src = os.path.normpath(str(src_path or "").strip())
        if (not p_src) or (not os.path.exists(p_src)):
            return p_src
        paths = self._ensure_gpr_processing_layout()
        raw_dir = paths.get("raw", "")
        if not raw_dir:
            return p_src
        try:
            cp = os.path.commonpath([os.path.abspath(p_src), os.path.abspath(raw_dir)])
            if cp == os.path.abspath(raw_dir):
                return p_src
        except Exception:
            pass

        name = os.path.basename(p_src)
        base, ext = os.path.splitext(name)
        candidate = os.path.join(raw_dir, name)
        i = 1
        while os.path.exists(candidate):
            try:
                if os.path.getsize(candidate) == os.path.getsize(p_src):
                    return candidate
            except Exception:
                pass
            candidate = os.path.join(raw_dir, f"{base}_{i}{ext}")
            i += 1
            if i > 999:
                break

        # Prefer lightweight link, fallback to real copy.
        staged = None
        try:
            rel_src = os.path.relpath(p_src, raw_dir)
            os.symlink(rel_src, candidate)
            staged = candidate
        except Exception:
            pass
        if staged is None:
            try:
                shutil.copy2(p_src, candidate)
                staged = candidate
            except Exception:
                return p_src

        # Alcuni formati (es. .DT) hanno file sidecar (.gec/.gps/.gps.sav)
        # indispensabili per il posizionamento GNSS: li porta con se' nella
        # stessa cartella raw, accanto al file principale.
        try:
            for sc_src in sidecar_paths(p_src):
                sc_name = os.path.basename(sc_src)
                sc_dst = os.path.join(raw_dir, sc_name)
                if os.path.exists(sc_dst):
                    continue
                try:
                    rel_sc = os.path.relpath(sc_src, raw_dir)
                    os.symlink(rel_sc, sc_dst)
                except Exception:
                    try:
                        shutil.copy2(sc_src, sc_dst)
                    except Exception:
                        pass
        except Exception:
            pass
        return staged

    def _apply_gpr_processing_ui_state(self, ui_state: dict) -> None:
        if not isinstance(ui_state, dict):
            return
        try:
            cmap = str(ui_state.get("colormap") or "").strip()
            if cmap and self._is_qt_alive(getattr(self, "_cb_cmap", None)):
                self._cb_cmap.setCurrentText(cmap)
        except Exception:
            pass
        try:
            if self._is_qt_alive(getattr(self, "_spin_vertical_exag", None)):
                self._spin_vertical_exag.setValue(float(ui_state.get("ve", self._spin_vertical_exag.value())))
        except Exception:
            pass
        try:
            render = str(ui_state.get("render") or "").strip()
            if render and self._is_qt_alive(getattr(self, "_cb_interp", None)):
                self._cb_interp.setCurrentText(render)
        except Exception:
            pass
        try:
            if self._is_qt_alive(getattr(self, "_chk_show_wiggle", None)):
                self._chk_show_wiggle.setChecked(bool(ui_state.get("show_wiggle", self._chk_show_wiggle.isChecked())))
        except Exception:
            pass
        try:
            spec_d = ui_state.get("slice_spec")
            panel = getattr(self, "_slice_spec_panel", None)
            if isinstance(spec_d, dict) and panel is not None and self._is_qt_alive(panel):
                from .slice_spec import SliceSpec
                panel.set_spec(SliceSpec.from_dict(spec_d))
        except Exception as exc:
            print(f"[GPR viewer][WARN] ripristino SliceSpec fallito: {exc}")

    def _try_import_saved_gpr_processing_project(self, force: bool = False) -> bool:
        if self._profiles and (not bool(force)):
            return False
        if self._profiles and bool(force):
            ans = QMessageBox.question(
                self,
                "Import progetto GPR",
                "Sostituire i profili correnti con quelli del progetto GPR salvato?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if ans != QMessageBox.Yes:
                return False
        meta = self._load_gpr_processing_project_meta()
        entries = list(meta.get("profiles") or [])
        if not entries:
            return False
        paths = []
        for e in entries:
            if not isinstance(e, dict):
                continue
            p = self._path_from_gpr_processing_store(str(e.get("path") or ""))
            if p and os.path.exists(p):
                paths.append(p)
        if not paths:
            return False

        self._reset_session_state()
        errors = []
        for p in paths:
            try:
                self._profiles.append(detect_and_read(p, verify_md5=True))
            except Exception:
                try:
                    self._profiles.append(detect_and_read(p, verify_md5=False))
                except Exception as exc:
                    errors.append(f"{os.path.basename(p)}: {exc}")
        if not self._profiles:
            if errors:
                self._safe_set_text(self._lbl_status, f"Import progetto: {errors[0]}")
            return False
        self._prof_idx = 0
        self._load_current_profile()
        self._apply_gpr_processing_ui_state(meta.get("ui_state") or {})
        try:
            ps = meta.get("processing_state")
            if isinstance(ps, dict) and ps:
                self._apply_full_processing_state(ps)
        except Exception as exc:
            print(f"[GPR viewer][WARN] ripristino processing_state fallito: {exc}")
        self._update_slice_profile_info()
        try:
            ts_saved = self._path_from_gpr_processing_store(str(meta.get("last_timeslice") or ""))
            if ts_saved and self._is_qt_alive(getattr(self, "_le_slice_outdir", None)):
                self._le_slice_outdir.setText(ts_saved)
        except Exception:
            pass
        self._safe_set_text(self._lbl_status, f"Progetto GPR importato: {len(self._profiles)} profili.")
        return True

    def _resolve_timeslice_raster_path(self, ts: dict) -> str:
        if not isinstance(ts, dict):
            return ""
        project_root = self._active_project_root()
        candidates = []
        for key in ("raster_path", "project_path", "path", "file_path", "source_path"):
            raw = str(ts.get(key) or "").strip()
            if raw:
                candidates.append(raw)
        for raw in candidates:
            p = raw
            if not os.path.isabs(p) and project_root:
                p = os.path.normpath(os.path.join(project_root, p))
            p = os.path.normpath(p)
            if os.path.exists(p):
                return p
        if candidates:
            fallback = candidates[0]
            if not os.path.isabs(fallback) and project_root:
                fallback = os.path.normpath(os.path.join(project_root, fallback))
            return os.path.normpath(fallback)
        return ""

    def _load_slice_raster_cached(self, raster_path: str):
        p = str(raster_path or "").strip()
        if not p:
            return None, None
        try:
            mtime = float(os.path.getmtime(p))
        except Exception:
            mtime = None

        cache_hit = (
            self._slice_cache_path == p
            and self._slice_cache_arr is not None
            and self._slice_cache_extent is not None
            and self._slice_cache_mtime == mtime
        )
        if cache_hit:
            return self._slice_cache_arr, self._slice_cache_extent

        from osgeo import gdal

        ds = gdal.Open(p)
        if ds is None:
            return None, None
        band = ds.GetRasterBand(1)
        if band is None:
            ds = None
            return None, None
        arr = band.ReadAsArray()
        if arr is None:
            ds = None
            return None, None
        arr = np.asarray(arr, dtype=np.float32)
        nodata = band.GetNoDataValue()
        if nodata is not None and np.isfinite(float(nodata)):
            arr[arr == float(nodata)] = np.nan
        arr[~np.isfinite(arr)] = np.nan

        gt = ds.GetGeoTransform(can_return_null=True)
        if gt:
            xmin = float(gt[0])
            xmax = float(gt[0] + gt[1] * ds.RasterXSize)
            ymax = float(gt[3])
            ymin = float(gt[3] + gt[5] * ds.RasterYSize)
        else:
            xmin, xmax = 0.0, float(arr.shape[1])
            ymin, ymax = 0.0, float(arr.shape[0])
        ds = None

        extent = (xmin, xmax, ymin, ymax)
        self._slice_cache_path = p
        self._slice_cache_mtime = mtime
        self._slice_cache_arr = arr
        self._slice_cache_extent = extent
        return arr, extent

    def _slice_world_to_pixel(self, east: float, north: float) -> tuple[float, float] | tuple[None, None]:
        if self._slice_current_extent is None:
            return None, None
        xmin, xmax, ymin, ymax = [float(v) for v in self._slice_current_extent]
        if not (np.isfinite(xmin) and np.isfinite(xmax) and np.isfinite(ymin) and np.isfinite(ymax)):
            return None, None
        # Ora la slice e' visualizzata direttamente in coordinate geografiche (extent su imshow),
        # quindi il crosshair puo' usare world-coordinates senza conversione in pixel.
        if abs(float(xmax - xmin)) < 1e-12 or abs(float(ymax - ymin)) < 1e-12:
            return None, None
        ex = float(east)
        ny = float(north)
        if not (np.isfinite(ex) and np.isfinite(ny)):
            return None, None
        return ex, ny

    def _update_slice_crosshair_overlay(self, draw: bool = True) -> bool:
        if self._ax_slice is None or self._slice_im is None:
            return False
        m = self._cursor_metrics()
        if not m:
            return False
        east = m.get("east")
        north = m.get("north")
        try:
            east_f = float(east) if east is not None else np.nan
            north_f = float(north) if north is not None else np.nan
        except Exception:
            return False
        if not (np.isfinite(east_f) and np.isfinite(north_f)):
            return False
        xw, yw = self._slice_world_to_pixel(east_f, north_f)
        if xw is None or yw is None:
            return False
        if self._slice_vline is None:
            self._slice_vline = self._ax_slice.axvline(xw, color="yellow", lw=1.0, alpha=0.85)
        else:
            self._slice_vline.set_xdata([xw, xw])
        if self._slice_hline is None:
            self._slice_hline = self._ax_slice.axhline(yw, color="cyan", lw=1.0, ls="--", alpha=0.8)
        else:
            self._slice_hline.set_ydata([yw, yw])
        if draw and self._canvas_slice is not None:
            self._safe_draw_idle(self._canvas_slice)
        return True

    def _on_slice_clip_changed(self, _value):
        if not hasattr(self, "_spin_slice_vmin_pct") or not hasattr(self, "_spin_slice_vmax_pct"):
            return
        lo = float(self._spin_slice_vmin_pct.value())
        hi = float(self._spin_slice_vmax_pct.value())
        if hi <= lo + 0.1:
            sender = self.sender()
            if sender is self._spin_slice_vmin_pct:
                hi = min(100.0, lo + 0.1)
                self._spin_slice_vmax_pct.blockSignals(True)
                try:
                    self._spin_slice_vmax_pct.setValue(hi)
                finally:
                    self._spin_slice_vmax_pct.blockSignals(False)
            else:
                lo = max(0.0, hi - 0.1)
                self._spin_slice_vmin_pct.blockSignals(True)
                try:
                    self._spin_slice_vmin_pct.setValue(lo)
                finally:
                    self._spin_slice_vmin_pct.blockSignals(False)
        self._redraw_slice_view(force=True)
        self._request_preview_update(immediate=False)

    def _preview_resolution_m(self) -> float:
        quality = "medium"
        try:
            quality = str(self._cb_preview_quality.currentData() or "medium").strip().lower()
        except Exception:
            quality = "medium"
        mapping = {
            "low": 0.20,
            "medium": 0.10,
            "high": 0.05,
        }
        return float(mapping.get(quality, 0.10))

    def _preview_depth_window(self) -> tuple[float, float]:
        try:
            step = float(self._spin_slice_thickness.value())
        except Exception:
            step = 0.10
        if (not np.isfinite(step)) or step <= 0.0:
            step = 0.10
        z0 = 0.0
        if self._cursor_z is not None:
            try:
                z0 = float(self._cursor_z)
            except Exception:
                z0 = 0.0
        if not np.isfinite(z0):
            z0 = 0.0
        z0 = float(max(0.0, z0))
        z1 = float(z0 + step)
        return z0, z1

    def _on_toggle_preview_mode(self, checked: bool):
        self._preview_mode_active = bool(checked)
        if not self._preview_mode_active:
            try:
                self._preview_timer.stop()
            except Exception:
                pass
            self._preview_request_id += 1  # scarta un eventuale risultato ancora in volo
            self._set_preview_busy(False)
            self._preview_slice_grid = None
            self._preview_slice_extent = None
            self._preview_slice_z_from = None
            self._preview_slice_z_to = None
            self._preview_signature = None
            self._redraw_slice_view(force=True)
            return
        if not self._profiles:
            if self._is_qt_alive(getattr(self, "_btn_preview_slice", None)):
                self._btn_preview_slice.blockSignals(True)
                try:
                    self._btn_preview_slice.setChecked(False)
                finally:
                    self._btn_preview_slice.blockSignals(False)
            self._preview_mode_active = False
            QMessageBox.information(self, "Preview", "Importa almeno un profilo .ogpr.")
            return
        self._request_preview_update(immediate=True)

    def _request_preview_update(self, immediate: bool = False):
        if not bool(getattr(self, "_preview_mode_active", False)):
            return
        if not self._profiles:
            return
        if not self._is_qt_alive(getattr(self, "_preview_timer", None)):
            return
        try:
            self._preview_timer.start(1 if immediate else 300)
        except Exception:
            pass

    def _stop_preview_worker_blocking(self, timeout_ms: int = 3000):
        """Attende la fine di un eventuale thread di preview in corso.

        Chiamato solo alla chiusura reale della finestra, per evitare di
        distruggere il QThread mentre e' ancora attivo.
        """
        self._preview_request_id += 1  # scarta comunque il risultato
        thread = getattr(self, "_preview_thread", None)
        if self._is_qt_alive(thread) and thread.isRunning():
            try:
                thread.quit()
                thread.wait(int(timeout_ms))
            except Exception:
                pass

    def _set_preview_busy(self, busy: bool, message: str = ""):
        """Mostra/nasconde l'indicatore di avanzamento del preview.

        Non esiste una % reale (compute_timeslice_preview non riporta stati
        intermedi), quindi la QProgressBar resta in modalita' indeterminata
        (range 0,0): comunica "sta lavorando" senza promettere una % finta.
        """
        self._preview_busy = bool(busy)
        if self._is_qt_alive(getattr(self, "_progress_preview", None)):
            self._progress_preview.setVisible(bool(busy))
        if self._is_qt_alive(getattr(self, "_lbl_preview_status", None)):
            if busy:
                self._lbl_preview_status.setText(message or "Preview: elaborazione in corso...")
                self._lbl_preview_status.setVisible(True)
            elif message:
                self._lbl_preview_status.setText(message)
                self._lbl_preview_status.setVisible(True)
            else:
                self._lbl_preview_status.setVisible(False)

    @staticmethod
    def _stable_preview_signature(value):
        """Converte ricorsivamente un valore (dict/list/float inclusi) in una
        tupla hashabile e stabile, usata per capire se i parametri di preview
        sono davvero cambiati rispetto all'ultimo calcolo."""
        if isinstance(value, float):
            return round(value, 6)
        if isinstance(value, dict):
            return tuple(sorted(
                (k, GprProfileViewer._stable_preview_signature(v)) for k, v in value.items()
            ))
        if isinstance(value, (list, tuple)):
            return tuple(GprProfileViewer._stable_preview_signature(v) for v in value)
        return value

    def _compute_and_show_preview(self):
        if not bool(getattr(self, "_preview_mode_active", False)):
            return
        if not self._profiles:
            return
        try:
            from .gpr_ogpr_slicer import compute_timeslice_preview  # noqa: F401 (verifica disponibilita')
        except Exception as exc:
            self._set_preview_busy(False, f"Preview non disponibile: {exc}")
            return

        z_from, z_to = self._preview_depth_window()
        res = self._preview_resolution_m()
        slice_params = dict(self.get_slice_params() or {})
        chain_order = list(self._current_chain_order() or [])

        # Fase D: spec-native preview when the SliceSpec panel is active.
        preview_spec = None
        _panel = getattr(self, "_slice_spec_panel", None)
        if _panel is not None and self._is_qt_alive(_panel):
            try:
                preview_spec = _panel.get_spec().resolve(self._profiles)
            except Exception as exc:
                self._set_preview_busy(False, f"Parametri slice non validi: {exc}")
                return
            # Never render the preview finer than the final output; the
            # "Qualita' preview" combo only makes it coarser.
            res = max(float(res), float(preview_spec.resolution_m))

        kwargs = {
            "channel": int(self._ch_idx),
            "combine_method": "mean",
            "pipeline_params": dict(slice_params.get("pipeline_params") or {}),
            "normalize_channels": bool(slice_params.get("normalize_channels", False)),
            "extraction_mode": str(slice_params.get("extraction_mode", "las_like") or "las_like"),
            "use_hilbert": bool(slice_params.get("use_hilbert", True)),
            "use_processing": bool(slice_params.get("use_processing", False)),
            "amplitude_sigma": slice_params.get("amplitude_sigma"),
            "use_anisotropic_idw": bool(slice_params.get("use_anisotropic_idw", False)),
            "auto_radius": bool(slice_params.get("auto_radius", True)),
            "idw_mode": str(slice_params.get("idw_mode", "quality") or "quality"),
            "radius": slice_params.get("search_radius"),
            "idw_power": float(slice_params.get("idw_power", 2) or 2),
            "min_points": int(slice_params.get("min_points", 1) or 1),
            "fill_nodata": bool(slice_params.get("fill_nodata", True)),
            "fill_nodata_max_distance": 0.0,
            "blanking_distance": float(slice_params.get("blanking_distance", 0.0) or 0.0),
            "smooth_sigma": float(slice_params.get("smooth_sigma", 0.0) or 0.0),
            "depth_radius_factor": float(slice_params.get("depth_radius_factor", 0.6) or 0.0),
            "balance_profiles": bool(slice_params.get("balance_profiles", True)),
            "pre_slice_bg_removal": bool(slice_params.get("pre_slice_bg_removal", False)),
            "pre_slice_bg_mode": str(slice_params.get("pre_slice_bg_mode", "line_by_line") or "line_by_line"),
            "pre_slice_bg_window": int(slice_params.get("pre_slice_bg_window", 0) or 0),
            "pre_slice_bg_sample_start": int(slice_params.get("pre_slice_bg_sample_start", 0) or 0),
            "pre_slice_bg_sample_end": int(slice_params.get("pre_slice_bg_sample_end", 0) or 0),
            "stack_n": int(slice_params.get("stack_n", 1) or 1),
            "stack_kernel": str(slice_params.get("stack_kernel", "boxcar") or "boxcar"),
            "flip_traces_mode": str(slice_params.get("flip_traces_mode", "none") or "none"),
            "topographic_correction": bool(slice_params.get("topographic_correction", False)),
            "topo_reference_mode": str(slice_params.get("topo_reference_mode", "median") or "median"),
            "topo_reference_elevation": slice_params.get("topo_reference_elevation"),
            "parallel_profiles": bool(slice_params.get("parallel_profiles", False)),
            "profile_workers": int(slice_params.get("profile_workers", 0) or 0),
        }

        # La firma di cache e' derivata direttamente da kwargs (invece di una
        # lista di campi mantenuta a mano): cosi' un parametro come Blanking
        # o Search radius che cambia non puo' piu' passare inosservato e
        # lasciare il preview "congelato" sul risultato precedente.
        sig = (
            round(float(z_from), 4),
            round(float(z_to), 4),
            round(float(res), 4),
            tuple(chain_order),
            (self._stable_preview_signature(preview_spec.to_dict())
             if preview_spec is not None
             else self._stable_preview_signature(kwargs)),
        )
        if sig == self._preview_signature:
            return

        # Calcolo su thread separato: il thread GUI resta reattivo e Windows
        # non mostra piu' il cursore "in caricamento" per l'intera durata.
        self._preview_request_id += 1
        request_id = self._preview_request_id
        self._preview_signature = sig
        self._preview_pending_zwindow = (float(z_from), float(z_to))
        self._set_preview_busy(True)

        old_thread = getattr(self, "_preview_thread", None)
        if self._is_qt_alive(old_thread):
            try:
                old_thread.finished.disconnect()
            except Exception:
                pass

        thread = QThread(self)
        worker = _PreviewSliceWorker(
            request_id, self._profiles, float(z_from), float(z_to), float(res), chain_order, kwargs,
            spec=preview_spec,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_preview_worker_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._preview_thread = thread
        self._preview_worker = worker
        thread.start()

    def _on_preview_worker_finished(self, request_id: int, grid, extent, error: str):
        if int(request_id) != int(self._preview_request_id):
            return  # superato da una richiesta di preview piu' recente
        self._set_preview_busy(False)
        if error:
            self._safe_set_text(self._lbl_status, f"Preview errore: {error}")
            return
        if grid is None or extent is None:
            self._safe_set_text(self._lbl_status, "Preview: nessun dato nella finestra selezionata.")
            return

        z_from, z_to = getattr(self, "_preview_pending_zwindow", (0.0, 0.0))
        self._preview_slice_grid = np.asarray(grid, dtype=np.float32)
        self._preview_slice_extent = tuple(float(v) for v in extent)
        self._preview_slice_z_from = float(z_from)
        self._preview_slice_z_to = float(z_to)
        self._redraw_slice_view(force=True)

    def _redraw_slice_view(self, force: bool = False):
        if not HAS_MPL or self._ax_slice is None or self._canvas_slice is None:
            return

        if bool(getattr(self, "_preview_mode_active", False)):
            arr = np.asarray(getattr(self, "_preview_slice_grid", None)) if getattr(self, "_preview_slice_grid", None) is not None else None
            extent = getattr(self, "_preview_slice_extent", None)
            if arr is None or extent is None or arr.ndim != 2 or arr.size <= 0:
                self._ax_slice.clear()
                self._slice_im = None
                self._slice_vline = None
                self._slice_hline = None
                self._ax_slice.text(
                    0.5,
                    0.5,
                    "Preview in attesa...\nMuovi il cursore nel radargramma.",
                    transform=self._ax_slice.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="#666666",
                )
                self._ax_slice.set_title("Timeslice preview")
                self._ax_slice.set_xticks([])
                self._ax_slice.set_yticks([])
                self._safe_draw_idle(self._canvas_slice)
                return

            finite = arr[np.isfinite(arr)]
            p_lo = 2.0
            p_hi = 98.0
            if hasattr(self, "_spin_slice_vmin_pct") and self._spin_slice_vmin_pct is not None:
                try:
                    p_lo = float(self._spin_slice_vmin_pct.value())
                except Exception:
                    p_lo = 2.0
            if hasattr(self, "_spin_slice_vmax_pct") and self._spin_slice_vmax_pct is not None:
                try:
                    p_hi = float(self._spin_slice_vmax_pct.value())
                except Exception:
                    p_hi = 98.0
            if p_hi <= p_lo + 0.1:
                p_hi = min(100.0, p_lo + 0.1)
            if finite.size > 0:
                vmin = float(np.nanpercentile(finite, p_lo))
                vmax = float(np.nanpercentile(finite, p_hi))
                if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax <= vmin):
                    vmin = float(np.nanmin(finite))
                    vmax = float(np.nanmax(finite))
            else:
                vmin, vmax = 0.0, 1.0
            if not np.isfinite(vmin):
                vmin = 0.0
            if not np.isfinite(vmax) or vmax <= vmin:
                vmax = vmin + 1e-6

            cmap = (
                self._cb_slice_cmap.currentText()
                if hasattr(self, "_cb_slice_cmap") and self._cb_slice_cmap is not None
                else DEFAULT_CMAP
            )
            arr_masked = np.ma.masked_invalid(arr)
            slice_cmap = cmap
            try:
                from matplotlib import colormaps

                cm_obj = colormaps.get_cmap(str(cmap)).copy()
                cm_obj.set_bad(color="#e6e6e6", alpha=1.0)
                slice_cmap = cm_obj
            except Exception:
                slice_cmap = cmap
            self._ax_slice.clear()
            self._slice_im = self._ax_slice.imshow(
                arr_masked,
                cmap=slice_cmap,
                vmin=vmin,
                vmax=vmax,
                extent=[float(extent[0]), float(extent[1]), float(extent[2]), float(extent[3])],
                origin="upper",
                aspect="equal",
                interpolation="bilinear",
            )
            xmin, xmax, ymin, ymax = [float(v) for v in extent]
            x_full = (min(xmin, xmax), max(xmin, xmax))
            y_full = (min(ymin, ymax), max(ymin, ymax))
            if self._slice_view_xlim is None:
                self._slice_view_xlim = x_full
            if self._slice_view_ylim is None:
                self._slice_view_ylim = y_full
            self._slice_view_xlim = self._clamp_axis_limits(
                self._slice_view_xlim[0], self._slice_view_xlim[1], x_full[0], x_full[1]
            )
            self._slice_view_ylim = self._clamp_axis_limits(
                self._slice_view_ylim[0], self._slice_view_ylim[1], y_full[0], y_full[1]
            )
            self._ax_slice.set_xlim(*self._slice_view_xlim)
            self._ax_slice.set_ylim(*self._slice_view_ylim)
            self._ax_slice.set_aspect("equal", adjustable="datalim")
            self._ax_slice.grid(False)
            self._ax_slice.set_xticks([])
            self._ax_slice.set_yticks([])
            self._ax_slice.set_xlabel("")
            self._ax_slice.set_ylabel("")
            self._ax_slice.set_facecolor("#e6e6e6")
            for spine in self._ax_slice.spines.values():
                spine.set_visible(False)
            z0 = float(getattr(self, "_preview_slice_z_from", 0.0) or 0.0)
            z1 = float(getattr(self, "_preview_slice_z_to", z0) or z0)
            self._ax_slice.set_title(f"Timeslice preview {z0:.2f}-{z1:.2f} m")
            self._slice_vline = None
            self._slice_hline = None
            self._update_slice_crosshair_overlay(draw=False)
            self._safe_draw_idle(self._canvas_slice)
            return

        local_catalog = list(getattr(self, "_slice_catalog", []) or [])
        using_local_catalog = len(local_catalog) > 0
        slices = local_catalog if using_local_catalog else []

        if not slices:
            self._ax_slice.clear()
            self._slice_im = None
            self._slice_vline = None
            self._slice_hline = None
            self._slice_current_idx = None
            self._slice_current_path = None
            self._slice_current_extent = None
            self._slice_current_shape = None
            self._slice_current_cmap = ""
            self._ax_slice.text(
                0.5,
                0.5,
                "Nessuna timeslice calcolata.\nUsa 'Crea Timeslice...' nel pannello Timeslice.",
                transform=self._ax_slice.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color="#666666",
            )
            self._ax_slice.set_title("Timeslice locale")
            self._ax_slice.set_xticks([])
            self._ax_slice.set_yticks([])
            self._safe_draw_idle(self._canvas_slice)
            return

        idx = 0 if self._slice_current_idx is None else int(self._slice_current_idx)
        idx = int(np.clip(idx, 0, len(slices) - 1))
        ts = dict(slices[idx] or {})
        raster_path = str(ts.get("path") or "").strip()
        cmap = (
            self._cb_slice_cmap.currentText()
            if hasattr(self, "_cb_slice_cmap") and self._cb_slice_cmap is not None
            else (self._cb_cmap.currentText() if hasattr(self, "_cb_cmap") else DEFAULT_CMAP)
        )

        if not raster_path:
            self._ax_slice.clear()
            self._slice_im = None
            self._ax_slice.set_title(f"Slice {idx} - percorso raster non disponibile")
            self._ax_slice.set_xticks([])
            self._ax_slice.set_yticks([])
            self._safe_draw_idle(self._canvas_slice)
            return
        if not os.path.exists(raster_path):
            self._ax_slice.clear()
            self._slice_im = None
            self._ax_slice.set_title(f"Slice {idx} - file non trovato")
            self._ax_slice.set_xticks([])
            self._ax_slice.set_yticks([])
            self._safe_draw_idle(self._canvas_slice)
            return

        needs_full_redraw = bool(
            force
            or self._slice_im is None
            or self._slice_current_idx != idx
            or str(self._slice_current_path or "") != str(raster_path)
        )

        if needs_full_redraw:
            try:
                arr, extent = self._load_slice_raster_cached(raster_path)
            except Exception as exc:
                self._ax_slice.clear()
                self._slice_im = None
                self._ax_slice.set_title(f"Errore lettura raster: {exc}")
                self._ax_slice.set_xticks([])
                self._ax_slice.set_yticks([])
                self._safe_draw_idle(self._canvas_slice)
                return
            if arr is None or extent is None:
                self._ax_slice.clear()
                self._slice_im = None
                self._ax_slice.set_title("Errore lettura raster")
                self._ax_slice.set_xticks([])
                self._ax_slice.set_yticks([])
                self._safe_draw_idle(self._canvas_slice)
                return

            finite = arr[np.isfinite(arr)]
            p_lo = 2.0
            p_hi = 98.0
            if hasattr(self, "_spin_slice_vmin_pct") and self._spin_slice_vmin_pct is not None:
                try:
                    p_lo = float(self._spin_slice_vmin_pct.value())
                except Exception:
                    p_lo = 2.0
            if hasattr(self, "_spin_slice_vmax_pct") and self._spin_slice_vmax_pct is not None:
                try:
                    p_hi = float(self._spin_slice_vmax_pct.value())
                except Exception:
                    p_hi = 98.0
            if p_hi <= p_lo + 0.1:
                p_hi = min(100.0, p_lo + 0.1)
            if finite.size > 0:
                vmin = float(np.nanpercentile(finite, p_lo))
                vmax = float(np.nanpercentile(finite, p_hi))
                if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax <= vmin):
                    vmin = float(np.nanmin(finite))
                    vmax = float(np.nanmax(finite))
            else:
                vmin, vmax = 0.0, 1.0
            if not np.isfinite(vmin):
                vmin = 0.0
            if not np.isfinite(vmax) or vmax <= vmin:
                vmax = vmin + 1e-6

            self._ax_slice.clear()
            self._slice_vline = None
            self._slice_hline = None
            arr_masked = np.ma.masked_invalid(arr)
            slice_cmap = cmap
            try:
                from matplotlib import colormaps

                cm_obj = colormaps.get_cmap(str(cmap)).copy()
                cm_obj.set_bad(color="#e6e6e6", alpha=1.0)
                slice_cmap = cm_obj
            except Exception:
                slice_cmap = cmap
            self._slice_im = self._ax_slice.imshow(
                arr_masked,
                cmap=slice_cmap,
                vmin=vmin,
                vmax=vmax,
                extent=[float(extent[0]), float(extent[1]), float(extent[2]), float(extent[3])],
                origin="upper",
                aspect="equal",
                interpolation="bilinear",
            )

            n_rows, n_cols = int(arr.shape[0]), int(arr.shape[1])
            xmin, xmax, ymin, ymax = [float(v) for v in extent]
            x_full = (min(xmin, xmax), max(xmin, xmax))
            y_full = (min(ymin, ymax), max(ymin, ymax))
            if self._slice_view_xlim is None:
                self._slice_view_xlim = x_full
            if self._slice_view_ylim is None:
                self._slice_view_ylim = y_full
            self._slice_view_xlim = self._clamp_axis_limits(
                self._slice_view_xlim[0], self._slice_view_xlim[1], x_full[0], x_full[1]
            )
            self._slice_view_ylim = self._clamp_axis_limits(
                self._slice_view_ylim[0], self._slice_view_ylim[1], y_full[0], y_full[1]
            )
            self._ax_slice.set_xlim(*self._slice_view_xlim)
            self._ax_slice.set_ylim(*self._slice_view_ylim)
            self._ax_slice.set_aspect("equal", adjustable="datalim")

            self._ax_slice.grid(False)
            self._ax_slice.set_xticks([])
            self._ax_slice.set_yticks([])
            self._ax_slice.set_xlabel("")
            self._ax_slice.set_ylabel("")
            self._ax_slice.set_facecolor("#e6e6e6")
            for spine in self._ax_slice.spines.values():
                spine.set_visible(False)

            try:
                d0 = float(ts.get("z_top", 0.0) or 0.0)
                d1 = float(ts.get("z_bot", d0) or d0)
            except Exception:
                d0 = 0.0
                d1 = 0.0
            source_name = os.path.basename(str(self._slice_catalog_source_dir or "")) or "local"
            title_left = f"{source_name} | "
            base = os.path.basename(raster_path)
            self._ax_slice.set_title(f"{title_left}Timeslice {d0:.2f}-{d1:.2f} m  |  {base}")

            self._slice_current_idx = idx
            self._slice_current_path = raster_path
            self._slice_current_extent = tuple(float(v) for v in extent)
            self._slice_current_shape = (n_rows, n_cols)
            self._slice_current_cmap = str(cmap)
        else:
            if self._slice_im is not None and str(self._slice_current_cmap) != str(cmap):
                cm_to_set = cmap
                try:
                    from matplotlib import colormaps

                    cm_obj = colormaps.get_cmap(str(cmap)).copy()
                    cm_obj.set_bad(color="#e6e6e6", alpha=1.0)
                    cm_to_set = cm_obj
                except Exception:
                    cm_to_set = cmap
                self._slice_im.set_cmap(cm_to_set)
                self._slice_current_cmap = str(cmap)

        changed = self._update_slice_crosshair_overlay(draw=False)
        if changed or needs_full_redraw:
            self._safe_draw_idle(self._canvas_slice)

    def _on_slice_press(self, event):
        if event is None or event.inaxes != self._ax_slice:
            return
        if int(getattr(event, "button", 0) or 0) != 1:
            return
        if self._slice_im is None or event.xdata is None or event.ydata is None:
            return
        try:
            xlim0 = tuple(self._ax_slice.get_xlim())
            ylim0 = tuple(self._ax_slice.get_ylim())
            self._slice_pan_anchor = (
                float(event.xdata),
                float(event.ydata),
                (float(xlim0[0]), float(xlim0[1])),
                (float(ylim0[0]), float(ylim0[1])),
            )
        except Exception:
            self._slice_pan_anchor = None

    def _on_slice_release(self, _event):
        self._slice_pan_anchor = None

    def _on_slice_mouse_move(self, event):
        if event is None or event.inaxes != self._ax_slice:
            return
        if event.xdata is None or event.ydata is None:
            return
        if self._slice_pan_anchor is not None and self._slice_im is not None:
            try:
                px, py, xlim0, ylim0 = self._slice_pan_anchor
                dx = float(px) - float(event.xdata)
                dy = float(py) - float(event.ydata)
                nx0 = float(xlim0[0]) + dx
                nx1 = float(xlim0[1]) + dx
                ny0 = float(ylim0[0]) + dy
                ny1 = float(ylim0[1]) + dy
                ex0, ex1, ey0, ey1 = self._slice_im.get_extent()
                x_min, x_max = float(min(ex0, ex1)), float(max(ex0, ex1))
                y_min, y_max = float(min(ey0, ey1)), float(max(ey0, ey1))
                self._slice_view_xlim = self._clamp_axis_limits(nx0, nx1, x_min, x_max)
                self._slice_view_ylim = self._clamp_axis_limits(ny0, ny1, y_min, y_max)
                self._ax_slice.set_xlim(*self._slice_view_xlim)
                self._ax_slice.set_ylim(*self._slice_view_ylim)
                self._safe_draw_idle(self._canvas_slice)
            except Exception:
                pass
            return
        try:
            self._safe_set_text(
                self._lbl_status,
                f"Slice cursor: x {float(event.xdata):.1f}  y {float(event.ydata):.1f}"
            )
        except Exception:
            pass

    def _on_slice_scroll_zoom(self, event):
        if event is None or event.inaxes != self._ax_slice or self._slice_im is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        step = float(getattr(event, "step", 0.0) or 0.0)
        btn = str(getattr(event, "button", "") or "").lower()
        if step > 0 or btn == "up":
            scale = 1.0 / 1.2
        elif step < 0 or btn == "down":
            scale = 1.2
        else:
            return

        x0, x1 = self._ax_slice.get_xlim()
        y0, y1 = self._ax_slice.get_ylim()
        x = float(event.xdata)
        y = float(event.ydata)

        nx0 = x - (x - x0) * scale
        nx1 = x + (x1 - x) * scale
        ny0 = y - (y - y0) * scale
        ny1 = y + (y1 - y) * scale

        ex0, ex1, ey0, ey1 = self._slice_im.get_extent()
        x_min, x_max = float(min(ex0, ex1)), float(max(ex0, ex1))
        y_min, y_max = float(min(ey0, ey1)), float(max(ey0, ey1))

        self._slice_view_xlim = self._clamp_axis_limits(nx0, nx1, x_min, x_max)
        self._slice_view_ylim = self._clamp_axis_limits(ny0, ny1, y_min, y_max)
        self._ax_slice.set_xlim(*self._slice_view_xlim)
        self._ax_slice.set_ylim(*self._slice_view_ylim)
        self._safe_draw_idle(self._canvas_slice)

    def _update_dial(self):
        if self._cursor_z is None or self.plugin is None:
            return
        settings = getattr(self.plugin, "settings", None)
        settings_key = getattr(self.plugin, "settings_key_active_project", None)
        if settings is None or not settings_key:
            return
        dlg = getattr(self.plugin, "dlg", None)
        if dlg is None:
            return
        dial = getattr(dlg, "Dial", None) or getattr(dlg, "dial", None)
        if dial is None:
            return
        try:
            slices = list(getattr(self, "_slice_catalog", []) or [])
            if not slices:
                return
            centers = []
            for i, s in enumerate(slices):
                try:
                    zc = s.get("z_center", None)
                    if zc is None:
                        z0 = float(s.get("z_top", 0.0) or 0.0)
                        z1 = float(s.get("z_bot", z0) or z0)
                        zc = 0.5 * (z0 + z1)
                    centers.append((i, float(zc)))
                except Exception:
                    continue
            if not centers:
                idx = int(self._slice_current_idx if self._slice_current_idx is not None else 0)
                idx = int(np.clip(idx, 0, len(slices) - 1))
            else:
                target = float(self._cursor_z)
                idx = min(centers, key=lambda p: abs(p[1] - target))[0]
            dial_val = int(np.clip(idx, 0, int(dial.maximum())))
            if dial.value() != dial_val:
                dial.setValue(dial_val)
        except Exception:
            pass

    def _update_timeslice_visibility_in_canvas(self):
        if not bool(getattr(self, "_chk_timeslice_sync", None) and self._chk_timeslice_sync.isChecked()):
            return
        if self._cursor_z is None or self.plugin is None:
            return

        slices = list(getattr(self, "_slice_catalog", []) or [])
        if not slices:
            return

        centers = []
        for i, s in enumerate(slices):
            try:
                zc = s.get("z_center", None)
                if zc is None:
                    z0 = float(s.get("z_top", 0.0) or 0.0)
                    z1 = float(s.get("z_bot", z0) or z0)
                    zc = 0.5 * (z0 + z1)
                centers.append((i, float(zc)))
            except Exception:
                continue
        if not centers:
            return

        target = float(self._cursor_z)
        idx = min(centers, key=lambda p: abs(p[1] - target))[0]

        try:
            root = QgsProject.instance().layerTreeRoot()
            qgis_group = root.findGroup("GPR Timeslices")
            if qgis_group is None:
                return

            raster_nodes = []
            for child in qgis_group.children():
                if not hasattr(child, "setItemVisibilityChecked"):
                    continue
                layer = child.layer() if hasattr(child, "layer") else None
                if layer is None:
                    continue
                if not hasattr(layer, "bandCount"):
                    continue
                raster_nodes.append(child)
            if not raster_nodes:
                return
            index = int(np.clip(idx, 0, len(raster_nodes) - 1))
            for i, node in enumerate(raster_nodes):
                node.setItemVisibilityChecked(i == index)
        except Exception:
            pass

    def _update_timeslice_crosshair(self):
        if not bool(getattr(self, "_chk_timeslice_sync", None) and self._chk_timeslice_sync.isChecked()):
            self._clear_timeslice_crosshair()
            return
        if self.iface is None or not self._profiles:
            self._clear_timeslice_crosshair()
            return
        m = self._cursor_metrics()
        if not m:
            self._clear_timeslice_crosshair()
            return
        east = m.get("east")
        north = m.get("north")
        if east is None or north is None:
            self._clear_timeslice_crosshair()
            return
        try:
            east_f = float(east)
            north_f = float(north)
        except Exception:
            self._clear_timeslice_crosshair()
            return
        if not (np.isfinite(east_f) and np.isfinite(north_f)):
            self._clear_timeslice_crosshair()
            return
        prof = self._profiles[self._prof_idx]
        canvas = self.iface.mapCanvas()
        if canvas is None:
            self._clear_timeslice_crosshair()
            return
        try:
            size = float(getattr(self, "_spin_crosshair_size").value())
        except Exception:
            size = float(getattr(self, "_crosshair_size_m", 5.0) or 5.0)
        size = float(np.clip(size, 0.5, 1000.0))

        if self._rb_crosshair_h is None:
            self._rb_crosshair_h = QgsRubberBand(canvas, QgsWkbTypes.LineGeometry)
            self._rb_crosshair_h.setColor(QColor(255, 50, 50))
            self._rb_crosshair_h.setWidth(2)
        if self._rb_crosshair_v is None:
            self._rb_crosshair_v = QgsRubberBand(canvas, QgsWkbTypes.LineGeometry)
            self._rb_crosshair_v.setColor(QColor(255, 50, 50))
            self._rb_crosshair_v.setWidth(2)

        try:
            self._rb_crosshair_h.reset(QgsWkbTypes.LineGeometry)
            p_left = self._to_canvas_point(prof, east_f - size, north_f)
            p_right = self._to_canvas_point(prof, east_f + size, north_f)
            self._rb_crosshair_h.addPoint(p_left, False)
            self._rb_crosshair_h.addPoint(p_right, True)
            self._rb_crosshair_h.setVisible(True)
        except Exception:
            pass
        try:
            self._rb_crosshair_v.reset(QgsWkbTypes.LineGeometry)
            p_bottom = self._to_canvas_point(prof, east_f, north_f - size)
            p_top = self._to_canvas_point(prof, east_f, north_f + size)
            self._rb_crosshair_v.addPoint(p_bottom, False)
            self._rb_crosshair_v.addPoint(p_top, True)
            self._rb_crosshair_v.setVisible(True)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Timeslice
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_slice_depth_label(label: str) -> Optional[float]:
        tok = str(label or "").strip().lower()
        if not tok:
            return None
        sign = -1.0 if tok.startswith("m") else 1.0
        if tok.startswith("m"):
            tok = tok[1:]
        if "_" in tok:
            tok = tok.replace("_", ".", 1)
        try:
            return sign * float(tok)
        except Exception:
            return None

    def _scan_slice_catalog(self, outdir: str, dz: float) -> list[dict]:
        base = str(outdir or "").strip()
        if not base or not os.path.isdir(base):
            return []

        # Manifesto autorevole: se c'e' un sliceset.json valido leggi da li'
        # profondita'/ordine/nomi invece di dedurli dai nomi file.
        try:
            from . import slice_manifest as _sm
            if _sm.has_sliceset(base):
                rows = _sm.slice_catalog_from_sliceset(base)
                if rows:
                    rows.sort(key=lambda it: (float(it.get("z_center", 0.0)),
                                              int(it.get("index", 0))))
                    self._slice_catalog_from_manifest = True
                    return rows
        except Exception:
            pass
        self._slice_catalog_from_manifest = False

        paths = sorted(glob.glob(os.path.join(base, "slice_*.tif")))
        if not paths:
            paths = sorted(glob.glob(os.path.join(base, "slice_*.tiff")))
        if not paths:
            return []

        catalog = []
        centers = []
        for fallback_idx, path in enumerate(paths):
            name = os.path.basename(path)
            idx = fallback_idx
            z_center = None
            m = re.search(r"^slice_(\d+)_z([A-Za-z0-9_]+)\.tiff?$", name, flags=re.IGNORECASE)
            if m:
                try:
                    idx = int(m.group(1))
                except Exception:
                    idx = fallback_idx
                z_center = self._parse_slice_depth_label(m.group(2))
            if z_center is not None and np.isfinite(z_center):
                centers.append(float(z_center))
            catalog.append(
                {
                    "path": os.path.normpath(path),
                    "index": int(idx),
                    "z_center": float(z_center) if z_center is not None else None,
                    "name": os.path.splitext(name)[0],
                }
            )

        catalog.sort(key=lambda item: (int(item.get("index", 0)), str(item.get("path", ""))))

        dz_use = float(dz) if np.isfinite(dz) and float(dz) > 0.0 else 0.0
        if dz_use <= 0.0:
            z_vals = [
                float(item["z_center"])
                for item in catalog
                if item.get("z_center") is not None and np.isfinite(float(item.get("z_center")))
            ]
            if len(z_vals) >= 2:
                z_vals = sorted(z_vals)
                diffs = np.diff(np.asarray(z_vals, dtype=np.float64))
                diffs = diffs[np.isfinite(diffs) & (diffs > 1e-9)]
                if diffs.size > 0:
                    dz_use = float(np.nanmedian(diffs))
        if dz_use <= 0.0:
            dz_use = 0.10

        for i, item in enumerate(catalog):
            zc = item.get("z_center")
            if zc is None or not np.isfinite(float(zc)):
                z_top = float(i) * dz_use
                z_center = z_top + 0.5 * dz_use
            else:
                z_center = float(zc)
                z_top = z_center - 0.5 * dz_use
            item["z_top"] = float(z_top)
            item["z_bot"] = float(z_top + dz_use)
            item["z_center"] = float(z_center)
        return catalog

    def _refresh_slice_catalog(
        self,
        outdir: str,
        dz: float,
        reset_index: bool = False,
        redraw: bool = True,
    ):
        base = os.path.normpath(str(outdir or "").strip()) if str(outdir or "").strip() else ""
        catalog = self._scan_slice_catalog(base, dz)
        self._slice_catalog = list(catalog or [])
        self._slice_catalog_source_dir = base
        try:
            self._slice_catalog_dz = float(dz) if np.isfinite(float(dz)) else 0.0
        except Exception:
            self._slice_catalog_dz = 0.0
        if not self._slice_catalog:
            self._slice_current_idx = None
        else:
            if reset_index or self._slice_current_idx is None:
                self._slice_current_idx = 0
            else:
                self._slice_current_idx = int(
                    np.clip(int(self._slice_current_idx), 0, len(self._slice_catalog) - 1)
                )
        self._update_slice_navigator()
        if redraw:
            self._slice_view_xlim = None
            self._slice_view_ylim = None
            self._redraw_slice_view(force=True)

    def _on_slice_depth_slider(self, value: int):
        if self._updating_slice_nav:
            return
        if not self._slice_catalog:
            return
        idx = int(np.clip(int(value), 0, len(self._slice_catalog) - 1))
        if self._slice_current_idx == idx:
            return
        self._slice_current_idx = idx
        try:
            center = float(self._slice_catalog[idx].get("z_center"))
            if np.isfinite(center):
                self._cursor_z = center
        except Exception:
            pass
        self._update_slice_navigator()
        self._redraw_slice_view(force=True)
        self._update_timeslice_visibility_in_canvas()

    def _step_slice(self, delta: int):
        if not self._slice_catalog:
            return
        curr = int(self._slice_current_idx if self._slice_current_idx is not None else 0)
        nxt = int(np.clip(curr + int(delta), 0, len(self._slice_catalog) - 1))
        if self._slider_slice_depth is not None:
            self._slider_slice_depth.setValue(nxt)
        else:
            self._on_slice_depth_slider(nxt)

    def _update_slice_navigator(self):
        has_slider = hasattr(self, "_slider_slice_depth") and self._slider_slice_depth is not None
        if (not has_slider) or (not hasattr(self, "_lbl_slice_nav")):
            return
        if not self._slice_catalog:
            self._updating_slice_nav = True
            try:
                self._slider_slice_depth.setRange(0, 0)
                self._slider_slice_depth.setValue(0)
                self._slider_slice_depth.setEnabled(False)
            finally:
                self._updating_slice_nav = False
            if self._is_qt_alive(getattr(self, "_btn_slice_up", None)):
                self._btn_slice_up.setEnabled(False)
            if self._is_qt_alive(getattr(self, "_btn_slice_down", None)):
                self._btn_slice_down.setEnabled(False)
            self._safe_set_text(self._lbl_slice_nav, "\u2014 nessuna slice \u2014")
            if hasattr(self, "_lbl_slice_depth_top"):
                self._safe_set_text(self._lbl_slice_depth_top, "0.00 m")
            if hasattr(self, "_lbl_slice_depth_bottom"):
                self._safe_set_text(self._lbl_slice_depth_bottom, "\u2014")
            return

        idx = int(self._slice_current_idx if self._slice_current_idx is not None else 0)
        idx = int(np.clip(idx, 0, len(self._slice_catalog) - 1))
        self._slice_current_idx = idx
        n = int(len(self._slice_catalog))
        entry = self._slice_catalog[idx]
        z_top = float(entry.get("z_top", 0.0) or 0.0)
        z_bot = float(entry.get("z_bot", z_top) or z_top)

        self._updating_slice_nav = True
        try:
            self._slider_slice_depth.setRange(0, n - 1)
            self._slider_slice_depth.setValue(idx)
            self._slider_slice_depth.setEnabled(True)
        finally:
            self._updating_slice_nav = False
        if self._is_qt_alive(getattr(self, "_btn_slice_up", None)):
            self._btn_slice_up.setEnabled(idx > 0)
        if self._is_qt_alive(getattr(self, "_btn_slice_down", None)):
            self._btn_slice_down.setEnabled(idx < (n - 1))

        self._safe_set_text(self._lbl_slice_nav, f"Slice {idx + 1}/{n}  |  {z_top:.2f}-{z_bot:.2f} m")
        if hasattr(self, "_lbl_slice_depth_top"):
            self._safe_set_text(self._lbl_slice_depth_top, "0.00 m")
        if hasattr(self, "_lbl_slice_depth_bottom"):
            z_max = float(self._slice_catalog[-1].get("z_bot", z_bot) or z_bot)
            self._safe_set_text(self._lbl_slice_depth_bottom, f"{z_max:.2f} m")

    def _on_timeslice_build_finished(self, ok: bool, payload: dict | None = None):
        if not bool(ok):
            return
        data = dict(payload or {})
        outdir = str(data.get("output_dir") or "").strip()
        if not outdir:
            outdir = str(getattr(self, "_le_slice_outdir", None).text() if hasattr(self, "_le_slice_outdir") else "").strip()
        if outdir and hasattr(self, "_le_slice_outdir") and self._le_slice_outdir is not None:
            try:
                self._le_slice_outdir.setText(os.path.normpath(outdir))
            except Exception:
                pass
        dz = data.get("z_step")
        if dz is None:
            try:
                dz = float(self._spin_slice_thickness.value())
            except Exception:
                dz = 0.10
        try:
            dz_f = float(dz)
        except Exception:
            dz_f = 0.10
        if hasattr(self, "_chk_slice_thickness_locked") and self._chk_slice_thickness_locked is not None:
            self._chk_slice_thickness_locked.setChecked(True)
        self._refresh_slice_catalog(outdir, dz=dz_f, reset_index=True, redraw=True)
        self._save_gpr_processing_project_meta(
            last_timeslice=self._path_to_gpr_processing_store(outdir),
        )

    def _default_slice_output_dir(self) -> str:
        paths = self._ensure_gpr_processing_layout()
        if paths:
            return os.path.normpath(paths.get("timeslices", ""))
        project_root = self._active_project_root()
        if project_root and os.path.isdir(project_root):
            return os.path.normpath(os.path.join(project_root, "timeslices_2d"))
        return ""

    def _choose_slice_outdir(self):
        start = str(getattr(self, "_le_slice_outdir", None).text() if hasattr(self, "_le_slice_outdir") else "").strip()
        if not start:
            start = self._default_slice_output_dir() or os.path.expanduser("~")
        folder = QFileDialog.getExistingDirectory(self, "Output folder Timeslice", start)
        if folder:
            try:
                norm = os.path.normpath(folder)
                self._le_slice_outdir.setText(norm)
                try:
                    dz = float(self._spin_slice_thickness.value())
                except Exception:
                    dz = 0.10
                self._refresh_slice_catalog(norm, dz=dz, reset_index=True, redraw=True)
                self._save_gpr_processing_project_meta(
                    last_timeslice=self._path_to_gpr_processing_store(norm),
                )
            except Exception:
                pass

    def _update_slice_profile_info(self):
        if not hasattr(self, "_lbl_slice_profiles") or self._lbl_slice_profiles is None:
            return
        n_prof = int(len(self._profiles or []))
        if n_prof <= 0:
            self._safe_set_text(self._lbl_slice_profiles, "Nessun profilo caricato")
            return
        total_traces = 0
        for prof in self._profiles:
            try:
                if int(getattr(prof, "n_channels", 0) or 0) <= 0:
                    continue
                total_traces += int(prof.channel(0).data.shape[1])
            except Exception:
                continue
        self._safe_set_text(self._lbl_slice_profiles, f"{n_prof} profili  |  ~{total_traces} tracce")

    def _import_slices_to_canvas(self):
        outdir = str(getattr(self, "_le_slice_outdir", None).text() if hasattr(self, "_le_slice_outdir") else "").strip()
        if not outdir:
            QMessageBox.warning(self, "Import Timeslice", "Seleziona prima la cartella output.")
            return
        if not os.path.isdir(outdir):
            QMessageBox.warning(self, "Import Timeslice", "Cartella output non valida.")
            return

        tifs = sorted(glob.glob(os.path.join(outdir, "*.tif"))) + sorted(glob.glob(os.path.join(outdir, "*.tiff")))
        if not tifs:
            QMessageBox.information(self, "Import Timeslice", "Nessun GeoTIFF trovato nella cartella output.")
            return

        root = QgsProject.instance().layerTreeRoot()
        existing = root.findGroup("GPR Timeslices")
        if existing is not None:
            try:
                root.removeChildNode(existing)
            except Exception:
                pass
        group = root.addGroup("GPR Timeslices")

        added = 0
        for tif in tifs:
            name = os.path.splitext(os.path.basename(tif))[0]
            rl = QgsRasterLayer(tif, name)
            if not rl.isValid():
                continue
            QgsProject.instance().addMapLayer(rl, False)
            group.addLayer(rl)
            added += 1

        for i, child in enumerate(group.children()):
            try:
                child.setItemVisibilityChecked(i == 0)
            except Exception:
                pass

        if self.iface is not None and self.iface.mapCanvas() is not None:
            try:
                self.iface.mapCanvas().refresh()
            except Exception:
                pass
        self._safe_set_text(
            self._lbl_status,
            f"Import completato: {added} layer nel gruppo 'GPR Timeslices'."
        )
        if added <= 0:
            QMessageBox.warning(self, "Import Timeslice", "Nessun layer valido importato.")
        else:
            try:
                dz = float(self._spin_slice_thickness.value())
            except Exception:
                dz = 0.10
            self._refresh_slice_catalog(outdir, dz=dz, reset_index=False, redraw=True)
            self._save_gpr_processing_project_meta(
                last_timeslice=self._path_to_gpr_processing_store(outdir),
            )

    @staticmethod
    def _set_combo_to_data(combo: QComboBox, value) -> None:
        if combo is None:
            return
        idx = combo.findData(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _save_processed_variant(self):
        """Applica la pipeline corrente a TUTTI i profili/canali dal grezzo e
        salva i radargrammi processati come .npz in processed/<nome variante>.

        A differenza di _apply_processing (solo RAM/anteprima), questo scrive
        davvero i dati filtrati su disco, cosi' da poterli riusare per le
        time-slice e confrontare varianti diverse.
        """
        if not self._profiles:
            QMessageBox.information(
                self, "Salva variante", "Importa almeno un profilo .ogpr prima."
            )
            return
        paths = self._ensure_gpr_processing_layout()
        if not paths:
            QMessageBox.warning(
                self, "Salva variante",
                "Nessun workspace disponibile. Reimporta scegliendo una cartella di lavoro.",
            )
            return

        name, ok = QInputDialog.getText(
            self, "Salva variante processata",
            "Nome variante (es. bp_bg, solo_bg):",
        )
        if not ok:
            return
        variant = "".join(
            c if c.isalnum() or c in "_-" else "_" for c in str(name).strip()
        ).strip("_")
        if not variant:
            QMessageBox.warning(self, "Salva variante", "Nome variante non valido.")
            return

        params = self._current_processing_params()
        chain = params.get("chain_order")
        variant_dir = os.path.join(paths["processed"], variant)
        if os.path.isdir(variant_dir) and any(
            f.lower().endswith(".npz") for f in os.listdir(variant_dir)
        ):
            ans = QMessageBox.question(
                self, "Variante esistente",
                f"La variante '{variant}' esiste gia'. Sovrascrivere i file processati?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if ans != QMessageBox.Yes:
                return
        os.makedirs(variant_dir, exist_ok=True)

        from PyQt5.QtWidgets import QProgressDialog
        from PyQt5.QtCore import Qt as _Qt

        n = len(self._profiles)
        prog = QProgressDialog("Salvataggio variante in corso...", "Annulla", 0, n, self)
        prog.setWindowModality(_Qt.WindowModal)
        prog.setMinimumDuration(0)
        prog.setValue(0)

        files = []
        errors = []
        for i, prof in enumerate(self._profiles):
            if prog.wasCanceled():
                break
            try:
                proc_by_ch = []
                for ci in range(int(getattr(prof, "n_channels", 0) or 0)):
                    raw = np.asarray(prof.channel(ci).data, dtype=np.float32)
                    p_ch = params
                    if params.get("kirchhoff") and params.get("kirchhoff_topo"):
                        p_ch = dict(params)
                        p_ch["kirchhoff_elev_m"] = self._profile_channel_elevation(prof, ci)
                    out = apply_pipeline(
                        raw, p_ch, dt_ns=prof.dt_ns,
                        normalize_output=False, chain_order=chain,
                    )
                    proc_by_ch.append(out)
                if proc_by_ch:
                    fp = save_processed_profile_npz(
                        variant_dir, prof, proc_by_ch, params, chain
                    )
                    files.append(fp)
            except Exception as e:
                errors.append(f"{os.path.basename(getattr(prof, 'path', '?'))}: {e}")
            prog.setValue(i + 1)
        prog.setValue(n)

        if files:
            try:
                write_variant_manifest(variant_dir, variant, params, chain, files)
            except Exception:
                pass
            try:
                from . import variant_provenance as vp
                parent = getattr(self, "_variant_parent_ctx", None)
                if parent == variant:
                    parent = None
                vp.set_entry(
                    paths["processed"], variant,
                    parent=parent,
                    chain=chain or params.get("chain_order"),
                    step=self._last_enabled_step(params, chain),
                    n_profiles=len(files),
                )
                # la prossima variante salvata discende da questa
                self._variant_parent_ctx = variant
            except Exception:
                pass
            self._refresh_slice_variant_choices()
        if errors:
            QMessageBox.warning(
                self, "Salva variante",
                f"Salvati {len(files)}/{n} profili.\nErrori:\n" + "\n".join(errors[:8]),
            )
        elif files:
            QMessageBox.information(
                self, "Salva variante",
                f"Variante '{variant}' salvata: {len(files)} profili in\n{variant_dir}",
            )
        else:
            QMessageBox.warning(self, "Salva variante", "Nessun profilo salvato.")
        try:
            self._save_gpr_processing_project_meta(
                last_processing=self._path_to_gpr_processing_store(variant_dir)
            )
        except Exception:
            pass

    def _blend_multifreq_to_variant(self):
        """Fonde antenne dual/triple frequenza in un unico radargramma
        (somma pesata + allineamento tempo-zero) e salva il risultato come
        variante a 1 canale.

        Due sorgenti:
          * i canali del profilo corrente (dati multicanale in un solo file);
          * piu' profili caricati (antenna dual-freq esportata come file
            separati, es. L1_LF + L1_HF): li si seleziona e vengono fusi in
            un unico profilo di uscita.
        La frequenza nominale dell'antenna NON serve per la fusione: conta solo
        che le sorgenti siano lo stesso tracciato.
        """
        if not self._profiles:
            QMessageBox.information(self, "Combina multi-frequenza",
                                   "Importa almeno un profilo prima.")
            return
        n_ch = int(getattr(self._profiles[self._prof_idx], "n_channels", 0) or 0)
        n_prof = len(self._profiles)
        if n_ch < 2 and n_prof < 2:
            QMessageBox.information(
                self, "Combina multi-frequenza",
                "Serve un profilo con >=2 canali, oppure >=2 profili caricati\n"
                "(un file per frequenza dello stesso tracciato).",
            )
            return
        paths = self._ensure_gpr_processing_layout()
        if not paths:
            QMessageBox.warning(self, "Combina multi-frequenza",
                                "Nessun workspace disponibile.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Combina multi-frequenza")
        outer = QVBoxLayout(dlg)
        cb_source = QComboBox()
        if n_ch >= 2:
            cb_source.addItem(f"Canali del profilo corrente ({n_ch})", "channels")
        if n_prof >= 2:
            cb_source.addItem(f"Profili caricati ({n_prof})", "profiles")
        form_top = QFormLayout()
        form_top.addRow("Sorgente:", cb_source)
        outer.addLayout(form_top)
        outer.addWidget(QLabel(
            "Seleziona SOLO sorgenti dello stesso tracciato.\n"
            "Peso e shift tempo-zero (campioni) per sorgente:"
        ))

        rows_box = QWidget()
        rows_lay = QVBoxLayout(rows_box)
        rows_lay.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(rows_box)
        cb_target = QComboBox()
        le_name = QLineEdit("blend")
        form_bot = QFormLayout()
        form_bot.addRow("Sorgente target (asse tempi/geometria):", cb_target)
        form_bot.addRow("Nome variante:", le_name)
        outer.addLayout(form_bot)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        outer.addWidget(bb)

        state = {"rows": []}

        def _rebuild_rows():
            for i in reversed(range(rows_lay.count())):
                it = rows_lay.takeAt(i)
                if it.widget():
                    it.widget().setParent(None)
            state["rows"] = []
            cb_target.clear()
            src = str(cb_source.currentData() or "channels")
            if src == "channels":
                labels = [f"Canale {ci}" for ci in range(n_ch)]
            else:
                labels = [
                    os.path.basename(getattr(p, "path", f"prof {i}") or f"prof {i}")
                    for i, p in enumerate(self._profiles)
                ]
            for k, lab in enumerate(labels):
                chk = QCheckBox(lab)
                chk.setChecked(src == "channels" or k < 2)
                ws = QDoubleSpinBox(); ws.setRange(0.0, 10.0); ws.setSingleStep(0.05)
                ws.setDecimals(3); ws.setValue(round(1.0 / max(2, len(labels)), 3))
                ss = QSpinBox(); ss.setRange(-8192, 8192); ss.setValue(0)
                rw = QWidget(); rl = QHBoxLayout(rw); rl.setContentsMargins(0, 0, 0, 0)
                rl.addWidget(chk, 1)
                rl.addWidget(QLabel("peso")); rl.addWidget(ws)
                rl.addWidget(QLabel("shift t0")); rl.addWidget(ss)
                rows_lay.addWidget(rw)
                state["rows"].append((k, lab, chk, ws, ss))
                cb_target.addItem(lab, k)

        cb_source.currentIndexChanged.connect(lambda _i: _rebuild_rows())
        _rebuild_rows()
        if dlg.exec_() != QDialog.Accepted:
            return

        src = str(cb_source.currentData() or "channels")
        chosen = [(k, w.value(), s.value()) for (k, _l, c, w, s) in state["rows"] if c.isChecked()]
        if len(chosen) < 2:
            QMessageBox.warning(self, "Combina multi-frequenza",
                                "Seleziona almeno 2 sorgenti.")
            return
        idxs = [k for (k, _w, _s) in chosen]
        weights = [float(w) for (_k, w, _s) in chosen]
        shifts = [int(s) for (_k, _w, s) in chosen]
        target_k = int(cb_target.currentData() or idxs[0])
        try:
            target_pos = idxs.index(target_k)
        except ValueError:
            target_pos = 0
        variant = "".join(
            c if c.isalnum() or c in "_-" else "_" for c in str(le_name.text()).strip()
        ).strip("_") or "blend"
        variant_dir = os.path.join(paths["processed"], variant)
        os.makedirs(variant_dir, exist_ok=True)
        meta = {"blend": {"source": src, "indices": idxs, "weights": weights,
                          "t0_shift_samples": shifts, "target": target_k}}

        files, errors = [], []
        try:
            if src == "channels":
                prof = self._profiles[self._prof_idx]
                arrs = [np.asarray(prof.channel(ci).data, dtype=np.float32) for ci in idxs]
                dts = [float(getattr(prof, "dt_ns", 0.1) or 0.1)] * len(arrs)
                blend = multifreq_blend(arrs, weights, dts, shifts, target=target_pos)
                # scrivo un profilo mono-canale usando la geometria del canale target
                fp = save_processed_profile_npz(variant_dir, prof, [blend], meta, [])
                files.append(fp)
            else:
                sel_profs = [self._profiles[k] for k in idxs]
                arrs = [np.asarray(p.channel(0).data, dtype=np.float32) for p in sel_profs]
                dts = [float(getattr(p, "dt_ns", 0.1) or 0.1) for p in sel_profs]
                blend = multifreq_blend(arrs, weights, dts, shifts, target=target_pos)
                fp = save_processed_profile_npz(
                    variant_dir, sel_profs[target_pos], [blend], meta, []
                )
                files.append(fp)
        except Exception as e:
            errors.append(str(e))

        if files:
            try:
                write_variant_manifest(variant_dir, variant, meta, [], files)
            except Exception:
                pass
            QMessageBox.information(
                self, "Combina multi-frequenza",
                f"Variante '{variant}' salvata ({len(files)} radargramma/i).\n"
                "Aprila dalla scheda 'Varianti' per generare le time-slice.",
            )
            try:
                self._refresh_variants_list()
            except Exception:
                pass
        else:
            QMessageBox.warning(self, "Combina multi-frequenza",
                                "Fusione non riuscita:\n" + "\n".join(errors[:5]))

    def _filter_timeslices_grid(self):
        """Applica un filtro nel dominio griglia a tutte le GeoTIFF di un set di
        time-slice, scrivendo un nuovo set filtrato accanto."""
        start = str(self._le_slice_outdir.text() or "").strip() if self._is_qt_alive(
            getattr(self, "_le_slice_outdir", None)) else ""
        if not start or not os.path.isdir(start):
            paths = self._gpr_processing_paths()
            start = (paths.get("timeslices", "") if paths else "") or os.path.expanduser("~")
        folder = QFileDialog.getExistingDirectory(
            self, "Cartella time-slice (GeoTIFF) da filtrare", start
        )
        if not folder:
            return
        tifs = sorted(
            glob.glob(os.path.join(folder, "*.tif"))
            + glob.glob(os.path.join(folder, "*.tiff"))
        )
        if not tifs:
            QMessageBox.warning(self, "Filtra time-slice",
                                "Nessuna GeoTIFF nella cartella selezionata.")
            return
        try:
            from .raster_io import (
                read_tif_singleband, write_tif_singleband, write_qml_singleband,
            )
            from . import grid_filters as gf
        except Exception as exc:
            QMessageBox.critical(self, "Filtra time-slice",
                                 f"Modulo non disponibile:\n{exc}")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Filtra time-slice (dominio griglia)")
        fl = QFormLayout(dlg)
        cb = QComboBox()
        cb.addItem("Passa-basso 3x3", ("lowpass", 3))
        cb.addItem("Passa-basso 5x5", ("lowpass", 5))
        cb.addItem("Passa-basso 7x7", ("lowpass", 7))
        cb.addItem("Passa-alto 3x3", ("hipass", 3))
        cb.addItem("Laplaciano (enfasi bordi)", ("laplacian", 0))
        cb.addItem("Background di linea — X (righe)", ("bg_x", 0))
        cb.addItem("Background di linea — Y (colonne)", ("bg_y", 0))
        cb.addItem("Notch FFT direzionale", ("fftnotch", 0))
        fl.addRow("Filtro:", cb)
        sp_len = QSpinBox(); sp_len.setRange(0, 99999); sp_len.setValue(0)
        sp_len.setToolTip("Lunghezza del filtro di linea in celle (0 = intera linea).")
        sp_clip = QDoubleSpinBox(); sp_clip.setRange(50.0, 100.0); sp_clip.setValue(100.0)
        sp_clip.setToolTip("Background di linea: esclude dal calcolo i valori sopra\n"
                           "questo percentile della linea (protegge i target forti).")
        sp_ang = QDoubleSpinBox(); sp_ang.setRange(0.0, 180.0); sp_ang.setValue(0.0)
        sp_ang.setSuffix(" °")
        sp_hw = QDoubleSpinBox(); sp_hw.setRange(1.0, 45.0); sp_hw.setValue(10.0)
        sp_hw.setSuffix(" °")
        le_suffix = QLineEdit("filt")
        fl.addRow("Lunghezza filtro linea:", sp_len)
        fl.addRow("Clip percentile (bg linea):", sp_clip)
        fl.addRow("Angolo (notch FFT):", sp_ang)
        fl.addRow("Semi-larghezza (notch FFT):", sp_hw)
        fl.addRow("Suffisso cartella output:", le_suffix)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        fl.addRow(bb)
        if dlg.exec_() != QDialog.Accepted:
            return

        kind, param = cb.currentData()
        suffix = "".join(c if c.isalnum() or c in "_-" else "_"
                         for c in str(le_suffix.text()).strip()).strip("_") or "filt"
        out_dir = os.path.normpath(folder).rstrip("\\/") + "_" + suffix
        os.makedirs(out_dir, exist_ok=True)

        from PyQt5.QtWidgets import QProgressDialog
        from PyQt5.QtCore import Qt as _Qt
        prog = QProgressDialog("Filtraggio time-slice...", "Annulla", 0, len(tifs), self)
        prog.setWindowModality(_Qt.WindowModal)
        prog.setMinimumDuration(0)

        n_ok, errors = 0, []
        for i, tif in enumerate(tifs):
            if prog.wasCanceled():
                break
            try:
                grid, x_min, y_min, y_max, res, epsg = read_tif_singleband(tif)
                if kind == "lowpass":
                    out = gf.lowpass(grid, int(param))
                elif kind == "hipass":
                    out = gf.hipass(grid, int(param))
                elif kind == "laplacian":
                    out = gf.laplacian(grid)
                elif kind == "bg_x":
                    out = gf.line_background(grid, "x", int(sp_len.value()),
                                            0.0, float(sp_clip.value()))
                elif kind == "bg_y":
                    out = gf.line_background(grid, "y", int(sp_len.value()),
                                            0.0, float(sp_clip.value()))
                elif kind == "fftnotch":
                    out = gf.fft2d_directional_notch(
                        grid, float(sp_ang.value()), float(sp_hw.value()), "reject"
                    )
                else:
                    out = grid
                out_path = os.path.join(out_dir, os.path.basename(tif))
                write_tif_singleband(out, out_path, x_min, y_min, y_max, res, epsg or 0)
                try:
                    write_qml_singleband(out_path, out)
                except Exception:
                    pass
                n_ok += 1
            except Exception as e:
                errors.append(f"{os.path.basename(tif)}: {e}")
            prog.setValue(i + 1)
        prog.setValue(len(tifs))

        # porta con se' il manifesto (i nomi file combaciano)
        src_manifest = os.path.join(folder, "sliceset.json")
        if os.path.isfile(src_manifest):
            try:
                shutil.copy2(src_manifest, os.path.join(out_dir, "sliceset.json"))
            except Exception:
                pass

        if n_ok:
            msg = f"{n_ok}/{len(tifs)} time-slice filtrate in\n{out_dir}"
            if errors:
                msg += "\n\nErrori:\n" + "\n".join(errors[:5])
            QMessageBox.information(self, "Filtra time-slice", msg)
            if self._is_qt_alive(getattr(self, "_le_slice_outdir", None)):
                self._le_slice_outdir.setText(out_dir)
                try:
                    dz0 = float(self._spin_slice_thickness.value())
                except Exception:
                    dz0 = 0.10
                try:
                    self._refresh_slice_catalog(out_dir, dz=dz0, reset_index=True, redraw=True)
                except Exception:
                    pass
        else:
            QMessageBox.warning(self, "Filtra time-slice",
                                "Nessuna slice filtrata.\n" + "\n".join(errors[:5]))

    # ------------------------------------------------------------------
    # grid_geometry in UI — unione / rototraslazione di set di time-slice
    # ------------------------------------------------------------------
    @staticmethod
    def _gg_list_tifs(folder):
        return sorted(
            glob.glob(os.path.join(folder, "*.tif"))
            + glob.glob(os.path.join(folder, "*.tiff"))
        )

    @staticmethod
    def _gg_read_manifest(folder):
        """Raw sliceset.json (dict) o None; niente rehydrate della spec."""
        path = os.path.join(folder, "sliceset.json")
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def _gg_slice_start_dir(self):
        start = str(self._le_slice_outdir.text() or "").strip() if self._is_qt_alive(
            getattr(self, "_le_slice_outdir", None)) else ""
        if not start or not os.path.isdir(start):
            paths = self._gpr_processing_paths()
            start = (paths.get("timeslices", "") if paths else "") or os.path.expanduser("~")
        return start

    def _gg_repoint_slice_view(self, out_dir):
        if not self._is_qt_alive(getattr(self, "_le_slice_outdir", None)):
            return
        self._le_slice_outdir.setText(out_dir)
        try:
            dz0 = float(self._spin_slice_thickness.value())
        except Exception:
            dz0 = 0.10
        try:
            self._refresh_slice_catalog(out_dir, dz=dz0, reset_index=True, redraw=True)
        except Exception:
            pass

    def _merge_timeslice_sets(self):
        """Unisce 2+ cartelle di GeoTIFF (stessa cella, stessi z) in un set unico
        via ``grid_geometry.append_grids`` (edge-match opzionale sulla giunzione)."""
        try:
            from .raster_io import read_tif_singleband, write_tif_singleband, write_qml_singleband
            from . import grid_geometry as gg
            from . import slice_manifest as sm
        except Exception as exc:
            QMessageBox.critical(self, "Unisci set di time-slice",
                                 f"Modulo non disponibile:\n{exc}")
            return

        start = self._gg_slice_start_dir()
        folders = []
        while True:
            title = ("Cartella time-slice #%d (Annulla per terminare)"
                     % (len(folders) + 1)) if folders else "Prima cartella time-slice"
            d = QFileDialog.getExistingDirectory(self, title, start)
            if not d:
                break
            d = os.path.normpath(d)
            if d in folders:
                continue
            if not self._gg_list_tifs(d):
                QMessageBox.warning(self, "Unisci set di time-slice",
                                    "Nessuna GeoTIFF in:\n" + d)
                continue
            folders.append(d)
            start = os.path.dirname(d)
            if len(folders) >= 2:
                if QMessageBox.question(
                    self, "Unisci set di time-slice",
                    "Aggiungere un'altra cartella?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No
                ) != QMessageBox.Yes:
                    break
        if len(folders) < 2:
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Unisci set di time-slice")
        fl = QFormLayout(dlg)
        chk_em = QCheckBox("Edge-match sulla giunzione (pareggia la media)")
        chk_em.setChecked(True)
        cb_axis = QComboBox()
        cb_axis.addItem("Affiancate lungo X (bordo verticale)", "x")
        cb_axis.addItem("Impilate lungo Y (bordo orizzontale)", "y")
        sp_ov = QSpinBox(); sp_ov.setRange(1, 999); sp_ov.setValue(3)
        cb_overlap = QComboBox()
        for lbl, val in (("Media", "mean"), ("Prima griglia", "first"), ("Ultima griglia", "last")):
            cb_overlap.addItem(lbl, val)
        le_suffix = QLineEdit("merged")
        fl.addRow(chk_em)
        fl.addRow("Asse giunzione (edge-match):", cb_axis)
        fl.addRow("Celle di sovrapposizione:", sp_ov)
        fl.addRow("Celle coperte da piu' griglie:", cb_overlap)
        fl.addRow("Suffisso cartella output:", le_suffix)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        fl.addRow(bb)
        if dlg.exec_() != QDialog.Accepted:
            return

        do_em = bool(chk_em.isChecked())
        em_axis = cb_axis.currentData()
        em_ov = int(sp_ov.value())
        overlap = cb_overlap.currentData()
        suffix = "".join(c if c.isalnum() or c in "_-" else "_"
                         for c in str(le_suffix.text()).strip()).strip("_") or "merged"
        out_dir = os.path.normpath(folders[0]).rstrip("\\/") + "_" + suffix
        os.makedirs(out_dir, exist_ok=True)

        # indicizza ogni cartella per chiave z (manifesto se c'e', altrimenti
        # posizione ordinata dei file)
        manifests = [self._gg_read_manifest(f) for f in folders]
        per_folder = []           # list[dict[zkey] -> (tif_path, z_from, z_to, z_center)]
        for f, man in zip(folders, manifests):
            tifs = self._gg_list_tifs(f)
            entries = {}
            if man and isinstance(man.get("slices"), list) and len(man["slices"]) == len(tifs):
                for s in man["slices"]:
                    gp = os.path.join(f, str(s.get("grid", "")))
                    if not os.path.isfile(gp):
                        entries = {}
                        break
                    zc = round(float(s.get("z_center", 0.0)), 4)
                    entries[zc] = (gp, float(s.get("z_from", zc)),
                                   float(s.get("z_to", zc)), float(s.get("z_center", zc)))
            if not entries:
                for i, gp in enumerate(tifs):
                    entries[float(i)] = (gp, float(i), float(i), float(i))
            per_folder.append(entries)

        all_keys = sorted(set().union(*[set(d.keys()) for d in per_folder]))
        if not all_keys:
            QMessageBox.warning(self, "Unisci set di time-slice", "Nessuna slice da unire.")
            return

        from PyQt5.QtWidgets import QProgressDialog
        from PyQt5.QtCore import Qt as _Qt
        prog = QProgressDialog("Unione time-slice...", "Annulla", 0, len(all_keys), self)
        prog.setWindowModality(_Qt.WindowModal)
        prog.setMinimumDuration(0)

        epsg_out = 0
        for man in manifests:
            if man and man.get("crs_epsg"):
                epsg_out = int(man["crs_epsg"]); break

        out_slices = []
        n_ok, errors = 0, []
        ext_xmin = ext_ymin = ext_xmax = ext_ymax = None
        for i, zk in enumerate(all_keys):
            if prog.wasCanceled():
                break
            try:
                grids, meta = [], None
                for entries in per_folder:
                    if zk not in entries:
                        continue
                    gp, zf, zt, zc = entries[zk]
                    arr, x_min, y_min, y_max, res, epsg = read_tif_singleband(gp)
                    grids.append({"array": arr, "x0": x_min, "y0": y_min, "cell": res})
                    if meta is None:
                        meta = (zf, zt, zc)
                    if not epsg_out and epsg:
                        epsg_out = int(epsg)
                if not grids:
                    continue
                if do_em and len(grids) >= 2:
                    base = grids[0]
                    for k in range(1, len(grids)):
                        grids[k] = gg.edge_match(base, grids[k], axis=em_axis,
                                                 overlap_cells=em_ov)
                merged = grids[0] if len(grids) == 1 else gg.append_grids(grids, overlap=overlap)
                arr = merged["array"]
                x0, y0, cell = float(merged["x0"]), float(merged["y0"]), float(merged["cell"])
                n_y, n_x = arr.shape
                y_top = y0 + (n_y - 1) * cell
                x_right = x0 + (n_x - 1) * cell
                name = "slice_%04d.tif" % i
                out_path = os.path.join(out_dir, name)
                write_tif_singleband(arr, out_path, x0, y0, y_top, cell, epsg_out or 0)
                try:
                    write_qml_singleband(out_path, arr)
                except Exception:
                    pass
                zf, zt, zc = meta if meta else (float(i), float(i), float(i))
                out_slices.append({
                    "index": i, "z_from": zf, "z_to": zt, "z_center": zc,
                    "grid": name, "stats": sm.compute_slice_stats(arr),
                })
                hw = cell / 2.0
                ext_xmin = (x0 - hw) if ext_xmin is None else min(ext_xmin, x0 - hw)
                ext_ymin = (y0 - hw) if ext_ymin is None else min(ext_ymin, y0 - hw)
                ext_xmax = (x_right + hw) if ext_xmax is None else max(ext_xmax, x_right + hw)
                ext_ymax = (y_top + hw) if ext_ymax is None else max(ext_ymax, y_top + hw)
                n_ok += 1
            except Exception as e:
                errors.append("z=%s: %s" % (zk, e))
            prog.setValue(i + 1)
        prog.setValue(len(all_keys))

        # manifesto unito (se almeno una sorgente ne aveva uno)
        base_man = next((m for m in manifests if m), None)
        if base_man is not None and out_slices:
            merged_man = dict(base_man)
            merged_man["slices"] = out_slices
            merged_man["extent"] = [ext_xmin, ext_ymin, ext_xmax, ext_ymax]
            if epsg_out:
                merged_man["crs_epsg"] = int(epsg_out)
            src = dict(merged_man.get("source") or {})
            src["merged_from"] = [os.path.basename(f) for f in folders]
            merged_man["source"] = src
            try:
                with open(os.path.join(out_dir, "sliceset.json"), "w", encoding="utf-8") as f:
                    json.dump(merged_man, f, indent=2, ensure_ascii=True, sort_keys=True)
            except Exception:
                pass

        if n_ok:
            msg = "%d/%d slice unite in\n%s" % (n_ok, len(all_keys), out_dir)
            if errors:
                msg += "\n\nErrori:\n" + "\n".join(errors[:5])
            QMessageBox.information(self, "Unisci set di time-slice", msg)
            self._gg_repoint_slice_view(out_dir)
        else:
            QMessageBox.warning(self, "Unisci set di time-slice",
                                "Nessuna slice unita.\n" + "\n".join(errors[:5]))

    def _rotate_translate_timeslice_set(self):
        """Ruota/trasla ogni GeoTIFF di un set via
        ``grid_geometry.rotate_translate`` e scrive un nuovo set ``_rt``."""
        try:
            from .raster_io import read_tif_singleband, write_tif_singleband, write_qml_singleband
            from . import grid_geometry as gg
            from . import slice_manifest as sm
        except Exception as exc:
            QMessageBox.critical(self, "Ruota/trasla set", f"Modulo non disponibile:\n{exc}")
            return

        folder = QFileDialog.getExistingDirectory(
            self, "Cartella time-slice (GeoTIFF) da ruotare/traslare",
            self._gg_slice_start_dir()
        )
        if not folder:
            return
        tifs = self._gg_list_tifs(folder)
        if not tifs:
            QMessageBox.warning(self, "Ruota/trasla set",
                                "Nessuna GeoTIFF nella cartella selezionata.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Ruota/trasla set di time-slice")
        fl = QFormLayout(dlg)
        sp_ang = QDoubleSpinBox(); sp_ang.setRange(-360.0, 360.0); sp_ang.setDecimals(4)
        sp_ang.setValue(0.0); sp_ang.setSuffix(" ° CCW")
        sp_e = QDoubleSpinBox(); sp_e.setRange(-1e9, 1e9); sp_e.setDecimals(3); sp_e.setValue(0.0)
        sp_n = QDoubleSpinBox(); sp_n.setRange(-1e9, 1e9); sp_n.setDecimals(3); sp_n.setValue(0.0)
        cb_mode = QComboBox()
        cb_mode.addItem("Angolo SW sorgente -> origine (fix_origin)", "fix_origin")
        cb_mode.addItem("Angolo SW inviluppo ruotato -> origine (translate_origin)",
                        "translate_origin")
        le_suffix = QLineEdit("rt")
        fl.addRow("Angolo:", sp_ang)
        fl.addRow("Origine E:", sp_e)
        fl.addRow("Origine N:", sp_n)
        fl.addRow("Modo:", cb_mode)
        fl.addRow("Suffisso cartella output:", le_suffix)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        fl.addRow(bb)
        if dlg.exec_() != QDialog.Accepted:
            return

        angle = float(sp_ang.value())
        origin = (float(sp_e.value()), float(sp_n.value()))
        mode = cb_mode.currentData()
        suffix = "".join(c if c.isalnum() or c in "_-" else "_"
                         for c in str(le_suffix.text()).strip()).strip("_") or "rt"
        out_dir = os.path.normpath(folder).rstrip("\\/") + "_" + suffix
        os.makedirs(out_dir, exist_ok=True)

        man = self._gg_read_manifest(folder)
        stats_by_name = {}

        from PyQt5.QtWidgets import QProgressDialog
        from PyQt5.QtCore import Qt as _Qt
        prog = QProgressDialog("Rototraslazione time-slice...", "Annulla", 0, len(tifs), self)
        prog.setWindowModality(_Qt.WindowModal)
        prog.setMinimumDuration(0)

        n_ok, errors = 0, []
        ext_xmin = ext_ymin = ext_xmax = ext_ymax = None
        for i, tif in enumerate(tifs):
            if prog.wasCanceled():
                break
            try:
                arr, x_min, y_min, y_max, res, epsg = read_tif_singleband(tif)
                res_g = gg.rotate_translate(
                    {"array": arr, "x0": x_min, "y0": y_min, "cell": res},
                    angle, origin, mode=mode,
                )
                oarr = res_g["array"]
                x0, y0, cell = float(res_g["x0"]), float(res_g["y0"]), float(res_g["cell"])
                n_y, n_x = oarr.shape
                y_top = y0 + (n_y - 1) * cell
                x_right = x0 + (n_x - 1) * cell
                out_path = os.path.join(out_dir, os.path.basename(tif))
                write_tif_singleband(oarr, out_path, x0, y0, y_top, cell, epsg or 0)
                try:
                    write_qml_singleband(out_path, oarr)
                except Exception:
                    pass
                stats_by_name[os.path.basename(tif)] = sm.compute_slice_stats(oarr)
                hw = cell / 2.0
                ext_xmin = (x0 - hw) if ext_xmin is None else min(ext_xmin, x0 - hw)
                ext_ymin = (y0 - hw) if ext_ymin is None else min(ext_ymin, y0 - hw)
                ext_xmax = (x_right + hw) if ext_xmax is None else max(ext_xmax, x_right + hw)
                ext_ymax = (y_top + hw) if ext_ymax is None else max(ext_ymax, y_top + hw)
                n_ok += 1
            except Exception as e:
                errors.append("%s: %s" % (os.path.basename(tif), e))
            prog.setValue(i + 1)
        prog.setValue(len(tifs))

        if man is not None and n_ok:
            new_man = dict(man)
            slices = []
            for s in (man.get("slices") or []):
                s2 = dict(s)
                nm = str(s.get("grid", ""))
                if nm in stats_by_name:
                    s2["stats"] = stats_by_name[nm]
                slices.append(s2)
            if slices:
                new_man["slices"] = slices
            new_man["extent"] = [ext_xmin, ext_ymin, ext_xmax, ext_ymax]
            src = dict(new_man.get("source") or {})
            src["rotate_translate"] = {"angle_deg": angle, "origin": list(origin), "mode": mode}
            new_man["source"] = src
            try:
                with open(os.path.join(out_dir, "sliceset.json"), "w", encoding="utf-8") as f:
                    json.dump(new_man, f, indent=2, ensure_ascii=True, sort_keys=True)
            except Exception:
                pass

        if n_ok:
            msg = "%d/%d slice rototraslate in\n%s" % (n_ok, len(tifs), out_dir)
            if errors:
                msg += "\n\nErrori:\n" + "\n".join(errors[:5])
            QMessageBox.information(self, "Ruota/trasla set", msg)
            self._gg_repoint_slice_view(out_dir)
        else:
            QMessageBox.warning(self, "Ruota/trasla set",
                                "Nessuna slice scritta.\n" + "\n".join(errors[:5]))

    # ------------------------------------------------------------------
    # Pannello "Varianti" — elenco delle varianti processate su disco
    # ------------------------------------------------------------------
    def _build_variants_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)

        lay.addWidget(QLabel(
            "Varianti di radargrammi processati in GPR_Processing/processed/.\n"
            "L'albero segue la provenienza (variant_graph.json): ogni variante\n"
            "e' figlia di quella da cui e' stata derivata. Tasto destro per le azioni."
        ))
        self._tree_variants = QTreeWidget()
        self._tree_variants.setHeaderLabels(["Variante", "Profili", "Ultimo step"])
        self._tree_variants.setAlternatingRowColors(True)
        self._tree_variants.setContextMenuPolicy(Qt.CustomContextMenu)
        self._tree_variants.customContextMenuRequested.connect(self._on_variants_context_menu)
        self._tree_variants.itemSelectionChanged.connect(self._on_variant_selection_changed)
        self._tree_variants.itemDoubleClicked.connect(
            lambda _it, _c: self._slice_selected_variant()
        )
        lay.addWidget(self._tree_variants, 1)

        row = QHBoxLayout()
        self._btn_variants_refresh = QPushButton("Aggiorna")
        self._btn_variants_refresh.clicked.connect(self._refresh_variants_list)
        self._btn_variants_newfrom = QPushButton("Nuova variante da qui")
        self._btn_variants_newfrom.setEnabled(False)
        self._btn_variants_newfrom.setToolTip(
            "Carica la pipeline di questa variante nei controlli di processing\n"
            "e la registra come genitore della prossima variante salvata."
        )
        self._btn_variants_newfrom.clicked.connect(self._new_variant_from_selected)
        self._btn_variants_slice = QPushButton("Genera time-slice")
        self._btn_variants_slice.setEnabled(False)
        self._btn_variants_slice.setToolTip(
            "Genera le time-slice dai .npz della variante selezionata,\n"
            "senza riapplicare la pipeline (i dati sono gia' filtrati)."
        )
        self._btn_variants_slice.clicked.connect(self._slice_selected_variant)
        self._btn_variants_open = QPushButton("Apri cartella")
        self._btn_variants_open.setEnabled(False)
        self._btn_variants_open.clicked.connect(self._open_selected_variant_folder)
        row.addWidget(self._btn_variants_refresh)
        row.addWidget(self._btn_variants_newfrom)
        row.addWidget(self._btn_variants_slice)
        row.addWidget(self._btn_variants_open)
        lay.addLayout(row)

        self._lbl_variant_detail = QLabel("—")
        self._lbl_variant_detail.setWordWrap(True)
        self._lbl_variant_detail.setStyleSheet("color: palette(mid);")
        lay.addWidget(self._lbl_variant_detail)
        return w

    def _refresh_variants_list(self):
        tree = getattr(self, "_tree_variants", None)
        if not self._is_qt_alive(tree):
            return
        tree.clear()
        if self._is_qt_alive(getattr(self, "_lbl_variant_detail", None)):
            self._lbl_variant_detail.setText("—")
        paths = self._gpr_processing_paths()
        processed = paths.get("processed", "") if paths else ""
        if not processed or not os.path.isdir(processed):
            tree.addTopLevelItem(QTreeWidgetItem(["(nessun progetto GPR attivo)"]))
            return
        try:
            dirs = list_variants(processed)
        except Exception:
            dirs = []
        if not dirs:
            tree.addTopLevelItem(QTreeWidgetItem(["(nessuna variante salvata)"]))
            return

        try:
            from . import variant_provenance as vp
        except Exception:
            vp = None
        entries = vp.load(processed) if vp is not None else {}

        by_name = {}
        for d in dirs:
            man = read_variant_manifest(d) or {}
            nm = str(man.get("name") or os.path.basename(os.path.normpath(d)))
            chain = [s for s in (man.get("chain_order") or []) if s]
            n_prof = int(man.get("n_profiles") or len(man.get("files") or []))
            by_name[nm] = {
                "dir": os.path.normpath(d),
                "chain": chain,
                "n_prof": n_prof,
                "created": man.get("created", ""),
            }

        present = list(by_name.keys())
        if vp is not None:
            forest = vp.build_forest(entries, present)
        else:
            forest = [{"name": n, "meta": {}, "children": []} for n in present]

        def _add(node, parent_item):
            nm = node["name"]
            info = by_name.get(nm, {})
            chain = info.get("chain") or []
            last_step = (entries.get(nm, {}) or {}).get("step") or (chain[-1] if chain else "")
            it = QTreeWidgetItem([
                nm, str(info.get("n_prof", "")), str(last_step or ""),
            ])
            it.setData(0, Qt.UserRole, info.get("dir", ""))
            it.setData(0, Qt.UserRole + 1, {"chain": chain, "created": info.get("created", "")})
            it.setToolTip(0, f"{info.get('dir','')}\ncatena: "
                             f"{', '.join(chain) if chain else '(non registrata)'}")
            if parent_item is None:
                self._tree_variants.addTopLevelItem(it)
            else:
                parent_item.addChild(it)
            for ch in node.get("children") or []:
                _add(ch, it)

        for root in forest:
            _add(root, None)
        tree.expandAll()
        self._refresh_slice_variant_choices()

    def _selected_variant_dir(self) -> str:
        tree = getattr(self, "_tree_variants", None)
        if not self._is_qt_alive(tree):
            return ""
        it = tree.currentItem()
        if it is None:
            return ""
        d = it.data(0, Qt.UserRole)
        return str(d) if d else ""

    def _selected_variant_name(self) -> str:
        tree = getattr(self, "_tree_variants", None)
        if not self._is_qt_alive(tree):
            return ""
        it = tree.currentItem()
        return str(it.text(0)) if it is not None else ""

    def _on_variant_selection_changed(self):
        d = self._selected_variant_dir()
        has = bool(d)
        for b in ("_btn_variants_slice", "_btn_variants_open", "_btn_variants_newfrom"):
            w = getattr(self, b, None)
            if self._is_qt_alive(w):
                w.setEnabled(has)
        tree = getattr(self, "_tree_variants", None)
        it = tree.currentItem() if (has and self._is_qt_alive(tree)) else None
        meta = it.data(0, Qt.UserRole + 1) if it is not None else None
        if isinstance(meta, dict):
            chain = ", ".join(meta.get("chain") or []) or "(catena non registrata)"
            self._lbl_variant_detail.setText(f"Catena: {chain}\n{d}")
        else:
            self._lbl_variant_detail.setText("—")

    def _on_variants_context_menu(self, pos):
        tree = getattr(self, "_tree_variants", None)
        if not self._is_qt_alive(tree):
            return
        it = tree.itemAt(pos)
        if it is None or not it.data(0, Qt.UserRole):
            return
        tree.setCurrentItem(it)
        menu = QMenu(tree)
        act_new = menu.addAction("Nuova variante da qui")
        act_slice = menu.addAction("Genera time-slice")
        act_open = menu.addAction("Apri cartella")
        menu.addSeparator()
        act_del = menu.addAction("Elimina variante…")
        chosen = menu.exec_(tree.viewport().mapToGlobal(pos))
        if chosen == act_new:
            self._new_variant_from_selected()
        elif chosen == act_slice:
            self._slice_selected_variant()
        elif chosen == act_open:
            self._open_selected_variant_folder()
        elif chosen == act_del:
            self._delete_selected_variant()

    def _new_variant_from_selected(self):
        d = self._selected_variant_dir()
        name = self._selected_variant_name()
        if not d or not os.path.isdir(d):
            return
        man = read_variant_manifest(d) or {}
        params = man.get("pipeline_params") if isinstance(man.get("pipeline_params"), dict) else {}
        if not params:
            QMessageBox.warning(self, "Nuova variante da qui",
                                "La variante non registra i parametri della pipeline "
                                "(variant.json privo di 'pipeline_params').")
            return
        chain = [s for s in (man.get("chain_order") or params.get("chain_order") or []) if s]
        try:
            self._apply_pipeline_params_dict(dict(params), chain_order=chain)
        except Exception as exc:
            QMessageBox.critical(self, "Nuova variante da qui",
                                 f"Impossibile caricare la pipeline:\n{exc}")
            return
        self._variant_parent_ctx = str(name)
        self._safe_set_text(
            self._lbl_status,
            f"Pipeline della variante '{name}' caricata. La prossima variante "
            f"salvata sara' registrata come sua figlia.",
        )

    def _delete_selected_variant(self):
        d = self._selected_variant_dir()
        name = self._selected_variant_name()
        if not d or not os.path.isdir(d):
            return
        ans = QMessageBox.question(
            self, "Elimina variante",
            f"Eliminare la cartella della variante '{name}'?\n{d}\n\n"
            "I .npz e il manifesto verranno rimossi. Operazione non annullabile.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if ans != QMessageBox.Yes:
            return
        try:
            shutil.rmtree(d)
        except Exception as exc:
            QMessageBox.critical(self, "Elimina variante", f"Rimozione fallita:\n{exc}")
            return
        try:
            paths = self._gpr_processing_paths()
            processed = paths.get("processed", "") if paths else ""
            if processed:
                from . import variant_provenance as vp
                vp.remove_entry(processed, name)
        except Exception:
            pass
        self._refresh_variants_list()

    def _slice_selected_variant(self):
        d = self._selected_variant_dir()
        if d:
            self._slice_from_variant_folder(d)

    def _open_selected_variant_folder(self):
        d = self._selected_variant_dir()
        if not d:
            return
        try:
            from qgis.PyQt.QtGui import QDesktopServices
            from qgis.PyQt.QtCore import QUrl
            QDesktopServices.openUrl(QUrl.fromLocalFile(d))
        except Exception:
            QMessageBox.information(self, "Varianti", d)

    def _build_slices_from_variant(self):
        """Genera time-slice a partire da una variante di radargrammi gia'
        processati (.npz), senza riapplicare la pipeline. Sceglie la cartella
        con un dialog; per la selezione dal pannello Varianti vedi
        `_slice_from_variant_folder`."""
        paths = self._gpr_processing_paths()
        start = paths.get("processed", "") if paths else ""
        if not start or not os.path.isdir(start):
            start = (paths.get("base", "") if paths else "") or os.path.expanduser("~")
        folder = QFileDialog.getExistingDirectory(
            self, "Seleziona cartella variante processata (.npz)", start
        )
        if not folder:
            return
        self._slice_from_variant_folder(folder)

    def _variant_names_for_picker(self) -> list[str]:
        """Nomi delle varianti salvate (cartelle in processed/ con .npz)."""
        paths = self._gpr_processing_paths()
        processed = paths.get("processed", "") if paths else ""
        if not processed or not os.path.isdir(processed):
            return []
        out = []
        try:
            for d in list_variants(processed):
                man = read_variant_manifest(d) or {}
                nm = str(man.get("name") or os.path.basename(os.path.normpath(d))).strip()
                # solo varianti con radargrammi affettabili
                if nm and glob.glob(os.path.join(d, "*.npz")):
                    out.append(nm)
        except Exception:
            return []
        return sorted(set(out))

    def _refresh_slice_variant_choices(self) -> None:
        panel = getattr(self, "_slice_spec_panel", None)
        if panel is not None and self._is_qt_alive(panel) and hasattr(panel, "set_variant_choices"):
            try:
                panel.set_variant_choices(self._variant_names_for_picker())
            except Exception:
                pass

    def _variant_dir_for_name(self, name: str) -> str:
        name = str(name or "").strip()
        if not name or name.lower() == "raw":
            return ""
        paths = self._gpr_processing_paths()
        processed = paths.get("processed", "") if paths else ""
        if not processed:
            return ""
        d = os.path.join(processed, name)
        return d if os.path.isdir(d) and glob.glob(os.path.join(d, "*.npz")) else ""

    def _profiles_for_slicing(self):
        """(profiles, variant_name) da usare per la generazione time-slice.

        Se il pannello ha una 'Variante sorgente' diversa da 'raw' e la cartella
        esiste con .npz, ritorna i radargrammi gia' filtrati caricati da disco
        (il chiamante deve disabilitare la pipeline). Altrimenti i profili
        correnti e ``""``.
        """
        panel = getattr(self, "_slice_spec_panel", None)
        name = ""
        if panel is not None and self._is_qt_alive(panel):
            try:
                name = str(panel.get_spec().source_variant or "").strip()
            except Exception:
                name = ""
        d = self._variant_dir_for_name(name)
        if not d:
            return list(self._profiles or []), ""
        npzs = sorted(glob.glob(os.path.join(d, "*.npz")))
        try:
            profs = [load_processed_profile_npz(p) for p in npzs]
        except Exception as exc:
            QMessageBox.warning(
                self, "Variante sorgente",
                f"Impossibile leggere i .npz della variante '{name}':\n{exc}\n\n"
                "Uso i profili correnti.",
            )
            return list(self._profiles or []), ""
        return profs, name

    def _slice_from_variant_folder(self, folder: str):
        """Carica i .npz di una cartella-variante e lancia la generazione slice."""
        paths = self._gpr_processing_paths()
        folder = os.path.normpath(str(folder or ""))
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "Time-slice da variante", "Cartella non valida.")
            return
        npzs = sorted(glob.glob(os.path.join(folder, "*.npz")))
        if not npzs:
            QMessageBox.warning(
                self, "Time-slice da variante",
                "Nessun file .npz nella cartella selezionata.",
            )
            return
        try:
            profiles = [load_processed_profile_npz(p) for p in npzs]
        except Exception as e:
            QMessageBox.critical(
                self, "Time-slice da variante", f"Errore lettura .npz: {e}"
            )
            return
        if self.plugin is None or not hasattr(self.plugin, "import_ogpr_as_slices"):
            QMessageBox.information(
                self, "Time-slice da variante",
                "Funzione disponibile dal plugin principale.",
            )
            return
        try:
            dz = float(self._spin_slice_thickness.value())
        except Exception:
            dz = 0.10
        slice_params = self.get_slice_params()
        # I dati sono gia' filtrati su disco: nessun ri-processing pipeline.
        slice_params["use_processing"] = False
        slice_params["pipeline_params"] = {}
        slice_params["pre_slice_bg_removal"] = False
        slice_params["z_step_override"] = float(dz)
        slice_params["thickness_m"] = float(dz)
        variant_name = os.path.basename(os.path.normpath(folder)) or "variante"
        if paths.get("timeslices"):
            slice_params["output_dir"] = os.path.join(paths["timeslices"], variant_name)
        try:
            self.plugin.import_ogpr_as_slices(
                profiles, slice_params=slice_params,
                completion_callback=self._on_timeslice_build_finished,
            )
        except TypeError:
            self.plugin.import_ogpr_as_slices(profiles, slice_params=slice_params)

    def _slice_pipeline_enabled_map(self) -> dict:
        enabled = dict(self._pipeline_enabled_map_from_controls() or {})
        if self._is_qt_alive(self._filter_chain_widget):
            try:
                from_widget = dict(self._filter_chain_widget.get_enabled_map() or {})
                for sid, on in from_widget.items():
                    enabled[str(sid)] = bool(on)
            except Exception:
                pass
        return enabled

    def _sync_slice_source_mode(self) -> None:
        if not self._is_qt_alive(getattr(self, "_cb_slice_pipeline_source", None)):
            return
        mode = str(self._cb_slice_pipeline_source.currentData() or "auto").strip().lower()
        manual = (mode == "manual")
        for w in (
            getattr(self, "_cb_slice_extraction", None),
            getattr(self, "_chk_slice_use_hilbert", None),
            getattr(self, "_chk_slice_use_processing", None),
        ):
            if self._is_qt_alive(w):
                try:
                    w.setEnabled(manual)
                except Exception:
                    pass

    def _resolve_slice_source(
        self,
        proc_params: dict,
        extraction_mode_manual: str,
        use_hilbert_manual: bool,
        use_processing_manual: bool,
    ) -> tuple[dict, bool, str, bool, str]:
        params = dict(proc_params or {})
        chain_order = [str(s) for s in list(params.get("chain_order") or self._current_chain_order() or DEFAULT_CHAIN_ORDER)]
        enabled_map = self._slice_pipeline_enabled_map()

        def _is_enabled(step_id: str) -> bool:
            if step_id in enabled_map:
                return bool(enabled_map.get(step_id))
            return bool(params.get(step_id, False))

        mode = "auto"
        if self._is_qt_alive(getattr(self, "_cb_slice_pipeline_source", None)):
            mode = str(self._cb_slice_pipeline_source.currentData() or "auto").strip().lower()
        if mode == "manual":
            return params, bool(use_processing_manual), str(extraction_mode_manual), bool(use_hilbert_manual), "manual"

        # Auto mode: prefer Hilbert/envelope when active, otherwise use last active step.
        if mode == "auto":
            if ("envelope" in chain_order) and _is_enabled("envelope"):
                target = "envelope"
            else:
                target = next((sid for sid in reversed(chain_order) if _is_enabled(sid)), "raw")
        elif mode == "raw":
            target = "raw"
        else:
            target = str(mode)
            if (target not in chain_order) or (not _is_enabled(target)):
                target = next((sid for sid in reversed(chain_order) if _is_enabled(sid)), "raw")

        if target == "raw":
            return params, False, "las_like", False, "raw"

        if target in chain_order:
            truncated = chain_order[: chain_order.index(target) + 1]
        else:
            truncated = [sid for sid in chain_order if _is_enabled(sid)]
        if not truncated:
            return params, False, "las_like", False, "raw"

        resolved = dict(params)
        for sid in DEFAULT_CHAIN_ORDER:
            resolved[sid] = bool(_is_enabled(sid) and (sid in truncated))
        resolved["chain_order"] = list(truncated)

        if target == "envelope":
            extraction_mode = "envelope"
            use_hilbert = True
        else:
            extraction_mode = "las_like"
            use_hilbert = False
        return resolved, True, extraction_mode, use_hilbert, target

    def _apply_selected_timeslice_preset(self):
        key = str(self._cb_slice_preset.currentData() or "base").strip().lower()
        self._apply_timeslice_preset(key)

    def _apply_timeslice_preset(self, preset_key: str):
        key = str(preset_key or "base").strip().lower()
        if key not in {"base", "quality", "aggressive"}:
            key = "base"

        if key == "quality":
            cfg = {
                "extraction_mode": "envelope",
                "use_hilbert": True,
                "use_processing": True,
                "pre_slice_bg_removal": True,
                "pre_slice_bg_mode": "line_by_line",
                "pre_slice_bg_auto": True,
                "pre_slice_bg_window": 200,
                "pre_slice_bg_sample_start": 10,
                "pre_slice_bg_sample_end": 0,
                "stack_n": 3,
                "stack_kernel": "triangular",
                "flip_traces_mode": "none",
                "topographic_correction": False,
                "topo_reference_mode": "median",
                "topo_reference_elevation": 0.0,
                "normalize_channels": False,
                "amplitude_filter": True,
                "amplitude_sigma": 3.0,
                "idw_mode": "quality",
                "slice_source": "auto",
                "search_radius": 0.0,
                "idw_power": 2,
                "overlap_fraction": 0.5,
                "blanking_distance": 0.30,
                "use_anisotropic_idw": True,
                "auto_radius": True,
                "balance_profiles": True,
                "depth_radius_factor": 0.8,
                "min_points": 2,
                "parallel_profiles": True,
                "profile_workers": 0,
                "fill_nodata": True,
                "smooth": True,
                "smooth_sigma": 1.0,
            }
            lbl = "Preset timeslice: Qualita' applicato."
        elif key == "aggressive":
            cfg = {
                "extraction_mode": "envelope",
                "use_hilbert": True,
                "use_processing": True,
                "pre_slice_bg_removal": True,
                "pre_slice_bg_mode": "grid_by_grid",
                "pre_slice_bg_auto": False,
                "pre_slice_bg_window": 300,
                "pre_slice_bg_sample_start": 10,
                "pre_slice_bg_sample_end": 0,
                "stack_n": 5,
                "stack_kernel": "triangular",
                "flip_traces_mode": "none",
                "topographic_correction": False,
                "topo_reference_mode": "median",
                "topo_reference_elevation": 0.0,
                "normalize_channels": False,
                "amplitude_filter": True,
                "amplitude_sigma": 2.5,
                "idw_mode": "quality",
                "slice_source": "auto",
                "search_radius": 0.0,
                "idw_power": 3,
                "overlap_fraction": 0.5,
                "blanking_distance": 0.30,
                "use_anisotropic_idw": True,
                "auto_radius": True,
                "balance_profiles": True,
                "depth_radius_factor": 1.0,
                "min_points": 3,
                "parallel_profiles": True,
                "profile_workers": 0,
                "fill_nodata": True,
                "smooth": True,
                "smooth_sigma": 1.5,
            }
            lbl = "Preset timeslice: Aggressivo applicato."
        else:
            cfg = {
                "extraction_mode": "las_like",
                "use_hilbert": True,
                "use_processing": True,
                "pre_slice_bg_removal": True,
                "pre_slice_bg_mode": "line_by_line",
                "pre_slice_bg_auto": True,
                "pre_slice_bg_window": 200,
                "pre_slice_bg_sample_start": 10,
                "pre_slice_bg_sample_end": 0,
                "stack_n": 1,
                "stack_kernel": "boxcar",
                "flip_traces_mode": "none",
                "topographic_correction": False,
                "topo_reference_mode": "median",
                "topo_reference_elevation": 0.0,
                "normalize_channels": False,
                "amplitude_filter": False,
                "amplitude_sigma": 3.0,
                "idw_mode": "quality",
                "slice_source": "auto",
                "search_radius": 0.0,
                "idw_power": 2,
                "overlap_fraction": 0.5,
                "blanking_distance": 0.30,
                "use_anisotropic_idw": False,
                "auto_radius": True,
                "balance_profiles": True,
                "depth_radius_factor": 0.6,
                "min_points": 1,
                "parallel_profiles": True,
                "profile_workers": 0,
                "fill_nodata": True,
                "smooth": True,
                "smooth_sigma": 0.8,
            }
            lbl = "Preset timeslice: Base applicato."

        self._set_combo_to_data(self._cb_slice_extraction, cfg["extraction_mode"])
        self._chk_slice_use_hilbert.setChecked(bool(cfg.get("use_hilbert", True)))
        self._chk_slice_use_processing.setChecked(bool(cfg["use_processing"]))
        self._chk_slice_bg.setChecked(bool(cfg["pre_slice_bg_removal"]))
        self._set_combo_to_data(self._cb_slice_bg_mode, cfg["pre_slice_bg_mode"])
        self._chk_slice_bg_auto.setChecked(bool(cfg["pre_slice_bg_auto"]))
        self._spin_slice_bg_window.setValue(int(cfg["pre_slice_bg_window"]))
        self._spin_slice_bg_sample_start.setValue(int(cfg["pre_slice_bg_sample_start"]))
        self._spin_slice_bg_sample_end.setValue(int(cfg["pre_slice_bg_sample_end"]))
        self._spin_slice_stack_n.setValue(int(cfg["stack_n"]))
        self._set_combo_to_data(self._cb_slice_stack_kernel, cfg["stack_kernel"])
        self._set_combo_to_data(self._cb_slice_flip_mode, cfg["flip_traces_mode"])
        self._chk_slice_topographic.setChecked(bool(cfg["topographic_correction"]))
        self._set_combo_to_data(self._cb_slice_topo_ref_mode, cfg["topo_reference_mode"])
        self._spin_slice_topo_ref_custom.setValue(float(cfg["topo_reference_elevation"]))
        self._chk_normalize_ch.setChecked(bool(cfg["normalize_channels"]))
        self._chk_amplitude_filter.setChecked(bool(cfg["amplitude_filter"]))
        self._spin_amplitude_sigma.setValue(float(cfg["amplitude_sigma"]))
        self._set_combo_to_data(self._cb_slice_idw_mode, cfg["idw_mode"])
        self._spin_slice_search_radius.setValue(float(cfg.get("search_radius", 0.0) or 0.0))
        self._set_combo_to_data(self._cb_slice_pipeline_source, str(cfg.get("slice_source", "auto") or "auto"))
        self._spin_slice_idw_power.setValue(int(cfg.get("idw_power", 2)))
        self._spin_slice_overlap_pct.setValue(int(round(float(cfg.get("overlap_fraction", 0.5)) * 100.0)))
        self._spin_slice_blanking_m.setValue(float(cfg.get("blanking_distance", 0.30)))
        self._chk_anisotropic_idw.setChecked(bool(cfg["use_anisotropic_idw"]))
        self._chk_auto_radius.setChecked(bool(cfg["auto_radius"]))
        self._chk_slice_balance_profiles.setChecked(bool(cfg["balance_profiles"]))
        self._spin_slice_depth_radius_factor.setValue(float(cfg["depth_radius_factor"]))
        self._spin_slice_min_points.setValue(int(cfg["min_points"]))
        self._chk_slice_parallel_profiles.setChecked(bool(cfg["parallel_profiles"]))
        self._spin_slice_profile_workers.setValue(int(cfg["profile_workers"]))
        self._chk_fill_nodata.setChecked(bool(cfg["fill_nodata"]))
        self._chk_smooth.setChecked(bool(cfg["smooth"]))
        self._spin_smooth_sigma.setValue(float(cfg["smooth_sigma"]))

        # Sync dependent controls if state has not emitted toggles.
        self._spin_slice_bg_window.setEnabled(
            bool(self._chk_slice_bg.isChecked()) and (not bool(self._chk_slice_bg_auto.isChecked()))
        )
        self._cb_slice_bg_mode.setEnabled(bool(self._chk_slice_bg.isChecked()))
        self._chk_slice_bg_auto.setEnabled(bool(self._chk_slice_bg.isChecked()))
        self._spin_slice_bg_sample_start.setEnabled(bool(self._chk_slice_bg.isChecked()))
        self._spin_slice_bg_sample_end.setEnabled(bool(self._chk_slice_bg.isChecked()))
        topo_on = bool(self._chk_slice_topographic.isChecked())
        self._cb_slice_topo_ref_mode.setEnabled(topo_on)
        self._spin_slice_topo_ref_custom.setEnabled(
            topo_on and (str(self._cb_slice_topo_ref_mode.currentData() or "") == "custom")
        )
        self._spin_amplitude_sigma.setEnabled(bool(self._chk_amplitude_filter.isChecked()))
        self._spin_smooth_sigma.setEnabled(bool(self._chk_smooth.isChecked()))
        self._spin_slice_profile_workers.setEnabled(bool(self._chk_slice_parallel_profiles.isChecked()))
        self._sync_slice_source_mode()

        # I preset "Qualita'"/"Aggressivo" impostano parametri della sezione
        # Esperto: senza abilitarla esplicitamente quei valori verrebbero
        # ignorati da get_slice_params(). "Base" li riporta ai default e
        # ricollassa la sezione per tornare alla vista semplice.
        if self._is_qt_alive(getattr(self, "_btn_slice_expert", None)):
            self._btn_slice_expert.setChecked(key != "base")

        self._safe_set_text(self._lbl_status, lbl)

    def get_slice_params(self) -> dict:
        # Fase D: quando il pannello SliceSpec e' attivo e' l'unica sorgente
        # dei parametri di generazione. Il dict restituito ha la stessa forma
        # legacy (piu' la chiave "_spec") cosi' il resto della pipeline non
        # cambia; il ramo Esperto sotto resta solo come fallback.
        panel = getattr(self, "_slice_spec_panel", None)
        if panel is not None and self._is_qt_alive(panel):
            try:
                from .slice_params_compat import spec_to_legacy_params

                spec = panel.get_spec()
                if self._profiles:
                    spec = spec.resolve(self._profiles)
                elif spec.z_max_m is None:
                    from dataclasses import replace as _replace
                    spec = _replace(spec, z_max_m=5.0)  # placeholder finche' non ci sono profili
                open_3d = bool(
                    self._is_qt_alive(getattr(self, "_chk_slice_open_3d_on_complete", None))
                    and self._chk_slice_open_3d_on_complete.isChecked()
                )
                out = spec_to_legacy_params(
                    spec, channel=-1, combine_method="mean",
                    open_3d_on_complete=open_3d,
                )
                return out
            except Exception as exc:
                print(f"[GPR viewer][WARN] get_slice_params via SliceSpec fallito: {exc}")

        # "Esperto" e' un flag: finche' non e' esplicitamente abilitato, questi
        # campi vengono ignorati e si usa il comportamento di default, anche
        # se i widget (nascosti) contengono un valore diverso.
        expert_on = bool(
            self._is_qt_alive(getattr(self, "_btn_slice_expert", None))
            and self._btn_slice_expert.isChecked()
        )
        slice_bg_auto = bool(self._chk_slice_bg_auto.isChecked()) if expert_on else True
        topo_mode = str(self._cb_slice_topo_ref_mode.currentData() or "median") if expert_on else "median"
        topo_on = bool(self._chk_slice_topographic.isChecked()) if expert_on else False
        topo_custom = None
        if expert_on and topo_mode == "custom":
            topo_custom = float(self._spin_slice_topo_ref_custom.value())
        proc_params = dict(self._current_processing_params() or {})
        proc_params, use_processing, extraction_mode, use_hilbert, source_step = self._resolve_slice_source(
            proc_params,
            extraction_mode_manual=(
                str(self._cb_slice_extraction.currentData() or "las_like") if expert_on else "las_like"
            ),
            use_hilbert_manual=(bool(self._chk_slice_use_hilbert.isChecked()) if expert_on else True),
            use_processing_manual=(bool(self._chk_slice_use_processing.isChecked()) if expert_on else True),
        )
        # For timeslice pre-processing, avoid surface over-saturation with conservative defaults
        # when the panel is still using broad generic values.
        if bool(use_processing):
            try:
                if int(proc_params.get("bg_sample_start", 0) or 0) <= 0:
                    proc_params["bg_sample_start"] = 10
            except Exception:
                proc_params["bg_sample_start"] = 10
            try:
                agc_on = bool(proc_params.get("agc", False))
                agc_win = int(proc_params.get("agc_win", 0) or 0)
                if agc_on and agc_win >= 96:
                    proc_params["agc_win"] = 32
            except Exception:
                pass
        try:
            search_radius = float(self._spin_slice_search_radius.value())
        except Exception:
            search_radius = 0.0
        radius_override = None
        if np.isfinite(search_radius) and search_radius > 0.0:
            radius_override = float(search_radius)
        auto_radius = radius_override is None
        return {
            "normalize_channels":  (self._chk_normalize_ch.isChecked() if expert_on else False),
            "slice_source":        str(source_step or "auto"),
            "extraction_mode":     str(extraction_mode or "las_like"),
            "use_hilbert":         bool(use_hilbert),
            "use_processing":      bool(use_processing),
            "pre_slice_bg_removal": self._chk_slice_bg.isChecked(),
            "pre_slice_bg_mode": (
                str(self._cb_slice_bg_mode.currentData() or "line_by_line") if expert_on else "line_by_line"
            ),
            "pre_slice_bg_window": (0 if slice_bg_auto else int(self._spin_slice_bg_window.value())),
            "pre_slice_bg_sample_start": (int(self._spin_slice_bg_sample_start.value()) if expert_on else 0),
            "pre_slice_bg_sample_end": (int(self._spin_slice_bg_sample_end.value()) if expert_on else 0),
            "stack_n": (int(self._spin_slice_stack_n.value()) if expert_on else 1),
            "stack_kernel": (str(self._cb_slice_stack_kernel.currentData() or "boxcar") if expert_on else "boxcar"),
            "flip_traces_mode": (str(self._cb_slice_flip_mode.currentData() or "none") if expert_on else "none"),
            "pipeline_params": proc_params,
            "topographic_correction": topo_on,
            "topo_reference_mode": topo_mode,
            "topo_reference_elevation": topo_custom,
            "amplitude_sigma":     (
                float(self._spin_amplitude_sigma.value())
                if (expert_on and self._chk_amplitude_filter.isChecked()) else None
            ),
            "idw_mode":            str(self._cb_slice_idw_mode.currentData() or "quality"),
            "radius":              radius_override,
            "search_radius":       radius_override,
            "idw_power":           int(self._spin_slice_idw_power.value()),
            "overlap_fraction":    float(self._spin_slice_overlap_pct.value()) / 100.0,
            "blanking_distance":   float(self._spin_slice_blanking_m.value()),
            "use_anisotropic_idw": (self._chk_anisotropic_idw.isChecked() if expert_on else False),
            "auto_radius":         bool(auto_radius),
            "balance_profiles":    self._chk_slice_balance_profiles.isChecked(),
            "depth_radius_factor": (float(self._spin_slice_depth_radius_factor.value()) if expert_on else 0.6),
            "min_points":          (int(self._spin_slice_min_points.value()) if expert_on else 1),
            "parallel_profiles":   (self._chk_slice_parallel_profiles.isChecked() if expert_on else True),
            "profile_workers":     (int(self._spin_slice_profile_workers.value()) if expert_on else 0),
            "fill_nodata":         self._chk_fill_nodata.isChecked(),
            "smooth_sigma":        (
                float(self._spin_smooth_sigma.value())
                if self._chk_smooth.isChecked() else 0.0
            ),
            "open_3d_on_complete": bool(
                self._is_qt_alive(self._chk_slice_open_3d_on_complete)
                and self._chk_slice_open_3d_on_complete.isChecked()
            ),
        }

    def _open_slice_dialog(self):
        if not self._profiles:
            QMessageBox.information(self, "Nessun profilo",
                                    "Importa almeno un file .ogpr prima.")
            return
        slice_params = self.get_slice_params()
        slice_profiles, variant_src = self._profiles_for_slicing()
        if not slice_profiles:
            QMessageBox.warning(self, "Timeslice", "Nessun radargramma da affettare.")
            return
        if variant_src:
            # dati gia' filtrati su disco: niente ri-processing della pipeline
            slice_params["use_processing"] = False
            slice_params["pipeline_params"] = {}
            slice_params["pre_slice_bg_removal"] = False
        panel = getattr(self, "_slice_spec_panel", None)
        if panel is not None and self._is_qt_alive(panel):
            try:
                dz = float(panel.get_spec().thickness_m)
            except Exception:
                dz = float(slice_params.get("z_step", 0.10) or 0.10)
        else:
            try:
                dz = float(self._spin_slice_thickness.value())
            except Exception:
                dz = 0.10
        if (not np.isfinite(dz)) or dz <= 0.0:
            QMessageBox.warning(self, "Timeslice", "Spessore slice non valido.")
            return
        slice_params["z_step_override"] = float(dz)
        slice_params["thickness_m"] = float(dz)
        outdir = str(getattr(self, "_le_slice_outdir", None).text() if hasattr(self, "_le_slice_outdir") else "").strip()
        if not outdir and variant_src:
            paths = self._gpr_processing_paths()
            if paths.get("timeslices"):
                outdir = os.path.join(paths["timeslices"], variant_src)
        if outdir:
            slice_params["output_dir"] = outdir
        if self.plugin and hasattr(self.plugin, "import_ogpr_as_slices"):
            try:
                self.plugin.import_ogpr_as_slices(
                    slice_profiles,
                    slice_params=slice_params,
                    completion_callback=self._on_timeslice_build_finished,
                )
            except TypeError:
                self.plugin.import_ogpr_as_slices(slice_profiles, slice_params=slice_params)
        else:
            QMessageBox.information(
                self, "Timeslice",
                "Funzione disponibile dal plugin principale."
            )

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


