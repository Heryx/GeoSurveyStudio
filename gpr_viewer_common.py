# -*- coding: utf-8 -*-
"""Elementi condivisi del GPR Profile Viewer: import, costanti, classi e funzioni ausiliarie.

Estratto dalla testa di gpr_profile_viewer.py senza modifiche; serve alla finestra principale e ai mixin."""

from __future__ import annotations

import json
import os
import glob
import re
import shutil
from datetime import datetime
from typing import Optional

import numpy as np

from qgis.PyQt.QtCore import Qt, QTimer, pyqtSignal, QSettings, QThread, QObject, QSize
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import (
    QApplication, QDialog, QMainWindow, QVBoxLayout, QHBoxLayout, QToolBar,
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

from .gpr_ogpr_reader import read_ogpr_cached, clear_ogpr_cache, OgprProfile, OgprChannel
from . import gpr_debug_log as _dbg
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
    RANGE_GAIN_MAX,
    auto_tune_params,
    multifreq_blend,
    log_exception,
    _log_gpr,
    LOG_FILE_PATH,
)
from .gpr_npz_io import (
    save_processed_profile_npz,
    load_processed_profile_npz,
    apply_geometry_from_profile,
    write_variant_manifest,
    read_variant_manifest,
    list_variants,
)

try:
    from .filter_chain_widget import FilterChainWidget, FilterBlockWidget
except Exception:
    FilterChainWidget = None
    FilterBlockWidget = None

try:
    from . import gpr_palettes as _gpr_palettes
except Exception:  # pragma: no cover
    _gpr_palettes = None


GPR_BUILTIN_CMAPS = [
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

# Lista completa (native + palette esterne registrate in matplotlib).
GPR_CMAPS = list(GPR_BUILTIN_CMAPS)


def refresh_gpr_cmaps():
    """Ri-scansiona le palette esterne e aggiorna GPR_CMAPS. Idempotente."""
    global GPR_CMAPS
    extra = []
    if _gpr_palettes is not None:
        try:
            extra = _gpr_palettes.register_all()
        except Exception:
            extra = []
    GPR_CMAPS = GPR_BUILTIN_CMAPS + [c for c in extra if c not in GPR_BUILTIN_CMAPS]
    return GPR_CMAPS


def _populate_cmap_combo(cb, *, current=None):
    """Riempie una QComboBox con i nomi colormap + icona anteprima, preservando
    la selezione corrente quando possibile."""
    if cb is None:
        return
    keep = current if current is not None else cb.currentText()
    try:
        cb.blockSignals(True)
        cb.clear()
        for name in GPR_CMAPS:
            icon = None
            if _gpr_palettes is not None:
                try:
                    icon = _gpr_palettes.make_cmap_icon(name, 64, 14)
                except Exception:
                    icon = None
            if icon is not None:
                cb.addItem(icon, name)
            else:
                cb.addItem(name)
        if keep and keep in GPR_CMAPS:
            cb.setCurrentText(keep)
        elif DEFAULT_CMAP in GPR_CMAPS:
            cb.setCurrentText(DEFAULT_CMAP)
    finally:
        cb.blockSignals(False)


refresh_gpr_cmaps()


def _nice_ceil(value: float) -> float:
    """Arrotonda per eccesso a 1-2-5 x 10^n (per un asse "tondo")."""
    v = float(value)
    if not np.isfinite(v) or v <= 0.0:
        return 1.0
    base = 10.0 ** np.floor(np.log10(v))
    for m in (1.0, 2.0, 5.0, 10.0):
        if v <= m * base * (1.0 + 1e-9):
            return float(m * base)
    return float(10.0 * base)


class _RangeGainCurveDialog(QDialog):
    """Interactive breakpoint editor for range gain curve.

    Il gain dei punti arriva fino a ``RANGE_GAIN_MAX``. L'asse X si adatta da solo
    alla curva ("Auto") e passa a scala log quando il gain copre piu' decadi:
    l'attenuazione GPR e' esponenziale, la compensazione richiede decine di dB.
    """

    _AUTO_HEADROOM = 2.0     # l'asse auto e' ~2x il gain massimo dei punti
    _LOG_FROM_MAX = 30.0     # sopra questo gain massimo la scala log parte da sola

    def __init__(self, points, max_points=16, gain_max=None, on_curve_changed=None, parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("Range Gain Curve")
        self.resize(620, 460)
        self._max_points = int(max(2, min(32, max_points)))
        self._on_curve_changed = on_curve_changed
        self._drag_idx = None
        self._selected_idx = None
        self._did_drag = False
        self._log_user_set = False
        self._view_max = 10.0   # asse in uso; lo fissa _redraw
        self._points = self._sanitize(points)
        # gain_max esplicito = asse manuale; None = automatico.
        self._manual_max = (
            float(np.clip(float(gain_max), 1.0, RANGE_GAIN_MAX)) if gain_max is not None else 10.0
        )

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
        self._spin_gain_max.setRange(1.0, RANGE_GAIN_MAX)
        self._spin_gain_max.setSingleStep(1.0)
        self._spin_gain_max.setDecimals(1)
        self._spin_gain_max.setValue(self._manual_max)
        self._spin_gain_max.setToolTip("Estremo dell'asse del gain (solo scala del grafico, non limita i punti).")
        self._spin_gain_max.valueChanged.connect(self._on_gain_max_changed)
        gain_row.addWidget(self._spin_gain_max)
        self._chk_auto_axis = QCheckBox("Auto")
        self._chk_auto_axis.setToolTip("Adatta l'asse del gain alla curva.")
        self._chk_auto_axis.setChecked(gain_max is None)
        self._chk_auto_axis.toggled.connect(self._on_auto_axis_toggled)
        gain_row.addWidget(self._chk_auto_axis)
        self._chk_log = QCheckBox("Scala log")
        self._chk_log.setToolTip(
            "Asse gain logaritmico (lineare sotto 1): utile quando la curva passa da <1 a centinaia."
        )
        self._chk_log.setChecked(self._axis_max() >= self._LOG_FROM_MAX)
        self._chk_log.toggled.connect(self._on_log_toggled)
        gain_row.addWidget(self._chk_log)
        gain_row.addStretch(1)
        root.addLayout(gain_row)
        self._spin_gain_max.setEnabled(not self._chk_auto_axis.isChecked())

        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Step:"))
        self._spin_step = QDoubleSpinBox()
        self._spin_step.setRange(0.01, RANGE_GAIN_MAX)
        self._spin_step.setSingleStep(0.01)
        self._spin_step.setValue(0.05 if self._axis_max() <= 10.0 else _nice_ceil(self._axis_max() / 100.0))
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
                        float(np.clip(y, 0.0, RANGE_GAIN_MAX)),
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

    def _axis_max_auto(self) -> float:
        top = float(np.max(self._points[:, 1])) if self._points.size else 1.0
        return float(min(RANGE_GAIN_MAX, _nice_ceil(max(10.0, self._AUTO_HEADROOM * top))))

    def _axis_max(self) -> float:
        """Estremo dell'asse gain: mai sotto i punti, cosi' restano tutti visibili."""
        if self._chk_auto_axis.isChecked():
            return self._axis_max_auto()
        top = float(np.max(self._points[:, 1])) if self._points.size else 1.0
        return float(min(RANGE_GAIN_MAX, max(self._manual_max, _nice_ceil(1.05 * top))))

    def gain_max(self) -> float:
        return float(self._axis_max())

    def _emit_changed(self):
        if callable(self._on_curve_changed):
            try:
                self._on_curve_changed(self.points(), float(self._axis_max()))
            except Exception:
                pass

    def _apply_auto_log(self):
        """Attiva la scala log da sola quando il gain esce dalla zona "lineare"."""
        if self._log_user_set or not hasattr(self, "_chk_log"):
            return
        want = float(np.max(self._points[:, 1])) >= self._LOG_FROM_MAX
        if want != self._chk_log.isChecked():
            self._chk_log.blockSignals(True)
            try:
                self._chk_log.setChecked(want)
            finally:
                self._chk_log.blockSignals(False)

    def _redraw(self, rescale: bool = True):
        if not HAS_MPL or self._ax is None or self._canvas is None:
            return
        if rescale:
            self._apply_auto_log()
            self._view_max = self._axis_max()
        xmax = self._view_max
        self._ax.clear()
        p = self._points
        # X=gain, Y=depth fraction (0 top, 1 bottom).
        if self._chk_log.isChecked():
            self._ax.set_xscale("symlog", linthresh=1.0)
        self._ax.plot(p[:, 1], p[:, 0], color="#2d6ba3", lw=1.4)
        self._ax.scatter(p[:, 1], p[:, 0], s=40, color="#c0392b", zorder=3)
        self._ax.set_xlim(0.0, xmax)
        self._ax.set_ylim(1.0, 0.0)
        self._ax.grid(True, ls=":", lw=0.5, alpha=0.7)
        self._ax.set_xlabel("Gain (log)" if self._chk_log.isChecked() else "Gain")
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
        """Punto piu' vicino a (gain=x, depth=y), con distanza in frazione d'asse.

        La distanza si misura sullo schermo, cosi' vale anche con l'asse log.
        """
        p = self._points
        if p.shape[0] <= 0 or self._ax is None:
            return None, 1e9
        to_axes = self._ax.transAxes.inverted()
        pts = to_axes.transform(self._ax.transData.transform(np.column_stack([p[:, 1], p[:, 0]])))
        q = to_axes.transform(self._ax.transData.transform([[float(x), float(y)]]))[0]
        d = np.hypot(pts[:, 0] - q[0], pts[:, 1] - q[1])
        idx = int(np.argmin(d))
        return idx, float(d[idx])

    def _on_press(self, event):
        if event is None or event.inaxes != self._ax or event.xdata is None or event.ydata is None:
            return
        gain = float(np.clip(event.xdata, 0.0, self._view_max))
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
        gain = float(np.clip(event.xdata, 0.0, self._view_max))
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
        # Durante il drag l'asse resta fermo: se si adattasse a ogni movimento il
        # punto sfuggirebbe al mouse (asse auto = ~2x il massimo dei punti).
        self._redraw(rescale=False)

    def _on_release(self, _event):
        was_dragging = self._drag_idx is not None
        self._did_drag = False
        self._drag_idx = None
        if was_dragging:
            self._redraw()

    def _move_selected(self, delta: float):
        if self._selected_idx is None:
            return
        i = int(self._selected_idx)
        p = self._points.copy()
        if i < 0 or i >= len(p):
            return
        p[i, 1] = float(np.clip(p[i, 1] + float(delta), 0.0, RANGE_GAIN_MAX))
        self._points = self._sanitize(p)
        self._emit_changed()
        self._redraw()

    def _on_gain_max_changed(self, value: float):
        # Solo la scala del grafico: i punti non vengono toccati.
        self._manual_max = float(np.clip(float(value), 1.0, RANGE_GAIN_MAX))
        self._redraw()

    def _on_auto_axis_toggled(self, checked: bool):
        self._spin_gain_max.setEnabled(not checked)
        if not checked:
            # Riparte dall'asse che si sta vedendo, non da un valore scollegato.
            self._manual_max = float(np.clip(self._axis_max_auto(), 1.0, RANGE_GAIN_MAX))
            self._spin_gain_max.blockSignals(True)
            try:
                self._spin_gain_max.setValue(self._manual_max)
            finally:
                self._spin_gain_max.blockSignals(False)
        self._redraw()

    def _on_log_toggled(self, _checked: bool):
        self._log_user_set = True
        self._redraw()

    def _on_reset(self):
        self._points = np.asarray([(0.0, 1.0), (1.0, 6.0)], dtype=np.float64)
        self._selected_idx = None
        self._drag_idx = None
        self._emit_changed()
        self._redraw()


class _BlitManager:
    """Ridisegna solo gli artisti "animati" (cursore, A-scan) senza rifare l'immagine.

    Il ricampionamento del radargramma costa ~100+ ms: farlo a ogni movimento
    del mouse rende il cursore a scatti.  Dopo ogni draw completo salviamo lo
    sfondo; il movimento del mouse ripristina lo sfondo e ridisegna solo gli
    artisti registrati (matplotlib: "blitting").
    """

    def __init__(self, canvas):
        self.canvas = canvas
        self._bg = None
        self._artists: list = []
        self._cid = canvas.mpl_connect("draw_event", self._on_draw)
        # Dopo un resize lo sfondo salvato ha la dimensione sbagliata.
        canvas.mpl_connect("resize_event", lambda _e: self.invalidate())

    def set_artists(self, artists) -> None:
        self._artists = []
        for art in artists:
            if art is None:
                continue
            art.set_animated(True)
            self._artists.append(art)

    def _draw_animated(self) -> None:
        fig = self.canvas.figure
        for art in self._artists:
            ax = getattr(art, "axes", None)
            if ax is not None and not ax.get_visible():
                continue
            fig.draw_artist(art)

    def _on_draw(self, event) -> None:
        if event is not None and getattr(event, "canvas", self.canvas) is not self.canvas:
            return
        self._bg = self.canvas.copy_from_bbox(self.canvas.figure.bbox)
        self._draw_animated()

    def update(self) -> None:
        """Aggiorna gli artisti animati; ripiega su un draw completo se serve."""
        if self._bg is None:
            self.canvas.draw_idle()
            return
        self.canvas.restore_region(self._bg)
        self._draw_animated()
        self.canvas.blit(self.canvas.figure.bbox)

    def invalidate(self) -> None:
        """Lo sfondo non e' piu' valido (resize, nuovo draw in arrivo)."""
        self._bg = None


class _PreviewSliceWorker(QObject):
    """Esegue compute_timeslice_preview() su un thread separato dalla GUI.

    Evita il blocco del thread principale (che produce il cursore "in
    caricamento" permanente di Windows/Qt) durante il preview live delle
    timeslice.
    """

    # request_id, grid, extent, error, load_result. ``load_result`` e' None se i
    # profili erano gia' in memoria; altrimenti ("ok", profili) / ("fail", None)
    # dopo il caricamento della variante da disco fatto in questo thread.
    finished = pyqtSignal(int, object, object, str, object)

    def __init__(self, request_id: int, profiles, z_from, z_to, resolution, chain_order, kwargs,
                 spec=None, loader=None):
        super().__init__()
        self._request_id = int(request_id)
        self._profiles = profiles
        self._loader = loader
        self._z_from = z_from
        self._z_to = z_to
        self._resolution = resolution
        self._chain_order = chain_order
        self._kwargs = kwargs
        self._spec = spec

    def run(self):
        load_result = None
        try:
            if self._profiles is None and self._loader is not None:
                # Lettura dei .npz della variante (~1 GB): qui, non nel thread GUI.
                try:
                    self._profiles = self._loader()
                except Exception:
                    load_result = ("fail", None)
                    raise
                load_result = ("ok", self._profiles)
            if self._spec is not None:
                # Spec-native preview (single window), same code path as the
                # final generation. For interp="bin" (default) the engine is
                # already fast; the idw_mode/fill_nodata knobs only cheapen the
                # legacy interp="idw" path. No diagnostics either way.
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
            self.finished.emit(self._request_id, grid, extent, "", load_result)
        except Exception as exc:
            self.finished.emit(self._request_id, None, None, str(exc), load_result)


def _run_pipeline_job(job: dict) -> np.ndarray:
    """apply_pipeline() sui dati di un job (usabile da qualunque thread)."""
    return apply_pipeline(
        job["raw"],
        job["params"],
        dt_ns=job["dt_ns"],
        bg_reference_trace=job["bg_reference_trace"],
        normalize_output=False,
        chain_order=job["chain_order"],
        stage_cache=job["stage_cache"],
    )


class _PipelineWorker(QObject):
    """Esegue apply_pipeline() fuori dal thread della GUI."""

    finished = pyqtSignal(int, object, str)  # seq, risultato, errore

    def __init__(self, job: dict):
        super().__init__()
        self._job = job

    def run(self):
        seq = int(self._job["seq"])
        try:
            result = _run_pipeline_job(self._job)
        except Exception as exc:
            try:
                log_exception("apply_pipeline", exc)
            except Exception:
                pass
            self.finished.emit(seq, None, f"{exc}")
            return
        self.finished.emit(seq, result, "")
