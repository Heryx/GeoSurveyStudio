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
    "jet",
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


class RangeGainPanel(QWidget):
    """Controlli per la curva di range gain (TVG), sotto il radargramma.

    Il grafico dei breakpoint e' un canvas a parte (``curve_canvas()``) che il
    viewer mette a sinistra del radargramma, allineato in profondita'. La curva
    si legge e si trascina in dB (20*log10 del guadagno): 1 dB sposta il segnale
    della stessa frazione a qualunque livello. Le ampiezze prima/dopo le disegna
    il viewer al posto dell'A-scan. "Prima" e' il dato che arriva al passo di
    guadagno, cioe' dopo i passi attivi che lo precedono nella catena. Il gruppo
    "Automatico" genera la curva dal profilo con inizio, compensazione e tetto
    regolabili.

    Segnali: ``finished_with`` ("one": applica a questo profilo, "all": copia la
    curva su tutti i profili, "cancel") e ``view_changed`` (cambia la scala delle
    ampiezze).
    """

    finished_with = pyqtSignal(str)
    view_changed = pyqtSignal()
    _DB_FLOOR = 1e-3         # sotto, il guadagno e' praticamente nullo (-60 dB)

    def __init__(self, points, max_points=16, gain_max=None, on_curve_changed=None,
                 parent=None, envelope_db=None, interp="log", auto_settings=None,
                 on_auto_all=None, trace_data=None):
        super().__init__(parent)
        self._disposed = False
        self._max_points = int(max(2, min(32, max_points)))
        self._on_curve_changed = on_curve_changed
        self._on_auto_all = on_auto_all
        self._data = None
        if trace_data is not None:
            arr = np.asarray(trace_data)
            if arr.ndim == 2 and arr.shape[0] >= 8 and arr.shape[1] >= 1:
                self._data = np.nan_to_num(arr.astype(np.float64))
        if envelope_db is None and self._data is not None:
            from .gpr_profile_gain import envelope_db as _env_db
            envelope_db = _env_db(self._data)
        self._env = np.asarray(envelope_db if envelope_db is not None else [], dtype=np.float64)
        self._drag_idx = None
        self._selected_idx = None
        self._did_drag = False
        self._view_lo = -3.0
        self._view_max = 20.0   # estremi dell'asse dB in uso; li fissa _redraw
        self._depth = 1.0       # profondita' (m) corrispondente alla finestra intera
        self._ylim_fn = None
        self._points = self._sanitize(points)
        self._manual_max = float(gain_max) if gain_max is not None else 40.0
        from .gpr_profile_gain import AUTO_DEFAULTS
        self._auto = {**AUTO_DEFAULTS, **(auto_settings or {})}

        if HAS_MPL:
            # Stessi margini verticali della figura del radargramma: i due canvas
            # hanno la stessa altezza e gli assi della profondita' combaciano.
            self._fig = Figure(figsize=(2.4, 3.6))
            self._fig.subplots_adjust(left=0.30, right=0.95, top=0.94, bottom=0.11)
            self._ax = self._fig.add_subplot(111)
            self._canvas = FigureCanvasQTAgg(self._fig)
            self._canvas.setFixedWidth(240)
            self._canvas.mpl_connect("button_press_event", self._on_press)
            self._canvas.mpl_connect("button_release_event", self._on_release)
            self._canvas.mpl_connect("motion_notify_event", self._on_move)
        else:
            self._fig = None
            self._ax = None
            self._canvas = None

        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        form = QFormLayout()
        self._cb_interp = QComboBox()
        self._cb_interp.addItem("Esponenziale (dB)", "log")
        self._cb_interp.addItem("Lineare", "linear")
        self._cb_interp.setCurrentIndex(max(0, self._cb_interp.findData(str(interp))))
        self._cb_interp.setToolTip(
            "Come si passa da un punto al successivo. Esponenziale: il guadagno\n"
            "cresce in modo regolare in dB, senza gradini. Lineare: retta sul\n"
            "guadagno (comportamento delle curve salvate in precedenza)."
        )
        self._cb_interp.currentIndexChanged.connect(self._on_interp_changed)
        form.addRow("Interpolazione:", self._cb_interp)
        step_row = QHBoxLayout()
        self._spin_step = QDoubleSpinBox()
        self._spin_step.setRange(0.1, 20.0)
        self._spin_step.setSingleStep(0.5)
        self._spin_step.setValue(1.0)
        self._spin_step.setSuffix(" dB")
        step_row.addWidget(self._spin_step)
        self._btn_step_up = QPushButton("▲")
        self._btn_step_down = QPushButton("▼")
        self._btn_step_up.setToolTip("Alza il guadagno del punto selezionato.")
        self._btn_step_down.setToolTip("Abbassa il guadagno del punto selezionato.")
        self._btn_step_up.clicked.connect(lambda: self._move_selected(+float(self._spin_step.value())))
        self._btn_step_down.clicked.connect(lambda: self._move_selected(-float(self._spin_step.value())))
        step_row.addWidget(self._btn_step_up)
        step_row.addWidget(self._btn_step_down)
        self._btn_all_shift = QPushButton("Tutta la curva")
        self._btn_all_shift.setCheckable(True)
        self._btn_all_shift.setToolTip(
            "Attivo: i pulsanti su/giu' spostano tutti i punti (guadagno complessivo)."
        )
        step_row.addWidget(self._btn_all_shift)
        form.addRow("Passo:", step_row)
        axis_row = QHBoxLayout()
        self._spin_gain_max = QDoubleSpinBox()
        self._spin_gain_max.setRange(6.0, 120.0)
        self._spin_gain_max.setDecimals(0)
        self._spin_gain_max.setSuffix(" dB")
        self._spin_gain_max.setValue(self._manual_max)
        self._spin_gain_max.valueChanged.connect(self._on_gain_max_changed)
        axis_row.addWidget(self._spin_gain_max)
        self._chk_auto_axis = QCheckBox("Auto")
        self._chk_auto_axis.setChecked(gain_max is None)
        self._chk_auto_axis.toggled.connect(self._on_auto_axis_toggled)
        axis_row.addWidget(self._chk_auto_axis)
        self._spin_gain_max.setEnabled(not self._chk_auto_axis.isChecked())
        form.addRow("Asse max:", axis_row)
        self._chk_same_scale = QCheckBox("Ampiezze reali (stessa scala)")
        self._chk_same_scale.setChecked(True)
        self._chk_same_scale.setToolTip(
            "Attivo: prima e dopo hanno lo stesso fondo scala (il massimo del segnale\n"
            "prima del guadagno): dove la traccia esce dal grafico, il segnale satura.\n"
            "Disattivo: ognuna e' normalizzata sul proprio massimo."
        )
        self._chk_same_scale.toggled.connect(lambda _c: self.view_changed.emit())
        form.addRow("", self._chk_same_scale)
        self._lbl_hint = QLabel(
            "Click: seleziona/trascina | Doppio click: aggiungi | Click destro: elimina"
        )
        self._lbl_hint.setStyleSheet("color: #666;")
        self._lbl_hint.setWordWrap(True)
        form.addRow(self._lbl_hint)
        root.addLayout(form, 1)

        box = QGroupBox("Automatico dal profilo")
        fl = QFormLayout(box)
        self._chk_start_auto = QCheckBox("onda diretta rilevata")
        self._chk_start_auto.setChecked(self._auto["start"] is None)
        self._spin_start = QDoubleSpinBox()
        self._spin_start.setRange(0.0, 90.0)
        self._spin_start.setDecimals(1)
        self._spin_start.setSuffix(" %")
        self._spin_start.setValue(100.0 * float(self._auto["start"] or 0.0))
        self._spin_start.setToolTip(
            "Il guadagno e' 0 dB sopra questo punto (% della finestra): l'onda diretta non viene amplificata."
        )
        self._spin_start.setEnabled(not self._chk_start_auto.isChecked())
        self._chk_start_auto.toggled.connect(lambda c: self._spin_start.setEnabled(not c))
        start_row = QHBoxLayout()
        start_row.addWidget(self._chk_start_auto)
        start_row.addWidget(self._spin_start)
        fl.addRow("Inizio:", start_row)
        self._spin_strength = QDoubleSpinBox()
        self._spin_strength.setRange(0.0, 150.0)
        self._spin_strength.setDecimals(0)
        self._spin_strength.setSuffix(" %")
        self._spin_strength.setValue(100.0 * float(self._auto["strength"]))
        self._spin_strength.setToolTip(
            "Quanta parte dell'attenuazione misurata viene compensata.\n"
            "Il guadagno cresce in dB in modo lineare (esponenziale sul guadagno)\n"
            "con questa quota della pendenza di attenuazione del segnale."
        )
        fl.addRow("Compensazione:", self._spin_strength)
        self._spin_max_db = QDoubleSpinBox()
        self._spin_max_db.setRange(0.0, 120.0)
        self._spin_max_db.setDecimals(0)
        self._spin_max_db.setSuffix(" dB")
        self._spin_max_db.setValue(float(self._auto["max_db"]))
        self._spin_max_db.setToolTip("Guadagno massimo: limita l'amplificazione del rumore in fondo.")
        fl.addRow("Tetto:", self._spin_max_db)
        self._spin_npts = QSpinBox()
        self._spin_npts.setRange(5, self._max_points)
        self._spin_npts.setValue(int(min(self._auto["n_points"], self._max_points)))
        fl.addRow("Punti:", self._spin_npts)
        self._btn_auto = QPushButton("Calcola per questo profilo")
        self._btn_auto.clicked.connect(self._on_auto_clicked)
        fl.addRow(self._btn_auto)
        self._btn_auto_all = QPushButton("Calcola per tutti i profili")
        self._btn_auto_all.setToolTip(
            "Stesse impostazioni; ogni profilo e' analizzato dopo i passi attivi che\n"
            "precedono il guadagno, in parallelo e annullabile."
        )
        self._btn_auto_all.setEnabled(callable(on_auto_all))
        self._btn_auto_all.clicked.connect(self._on_auto_all_clicked)
        fl.addRow(self._btn_auto_all)
        root.addWidget(box, 1)

        btns = QVBoxLayout()
        self._btn_apply_one = QPushButton("Applica a questo profilo")
        self._btn_apply_one.setToolTip("Tiene la curva solo per il profilo mostrato e chiude.")
        self._btn_apply_one.clicked.connect(lambda: self.finished_with.emit("one"))
        btns.addWidget(self._btn_apply_one)
        self._btn_apply_all = QPushButton("Applica a tutti i profili")
        self._btn_apply_all.setToolTip("Copia questa curva su tutti i profili e chiude.")
        self._btn_apply_all.clicked.connect(lambda: self.finished_with.emit("all"))
        btns.addWidget(self._btn_apply_all)
        self._btn_cancel = QPushButton("Annulla")
        self._btn_cancel.clicked.connect(lambda: self.finished_with.emit("cancel"))
        btns.addWidget(self._btn_cancel)
        btns.addStretch(1)
        self._btn_reset = QPushButton("Reset")
        self._btn_reset.clicked.connect(self._on_reset)
        btns.addWidget(self._btn_reset)
        root.addLayout(btns, 0)

        self._redraw()

    # --- integrazione col viewer --------------------------------------------
    def curve_canvas(self):
        """Canvas del grafico dei breakpoint (da mettere a sinistra del radargramma)."""
        return self._canvas

    def trace_data(self):
        return self._data

    def same_scale(self) -> bool:
        return bool(self._chk_same_scale.isChecked())

    def amplitude_pair(self, k: int):
        """(prima, dopo) della traccia `k` del dato mostrato."""
        if self._data is None:
            return None
        k = int(np.clip(k, 0, self._data.shape[1] - 1))
        before = self._data[:, k] - np.median(self._data[:, k])
        return before, before * self.gain_vector(before.size)

    def set_depth(self, depth_m: float, ylim_fn=None):
        """Profondita' (m) della finestra intera e funzione che da' i limiti y
        del radargramma (per tenere gli assi allineati)."""
        self._depth = float(depth_m) if depth_m and depth_m > 0 else 1.0
        self._ylim_fn = ylim_fn
        self._redraw(rescale=False)

    def sync_ylim(self):
        """Allinea l'asse della profondita' a quello del radargramma (zoom/scorrimento)."""
        if self._ax is None or self._disposed or self._ylim_fn is None:
            return
        try:
            self._ax.set_ylim(*self._ylim_fn())
            if self.isVisible() or (self._canvas is not None and self._canvas.isVisible()):
                self._canvas.draw_idle()
        except Exception:
            pass

    # --- dati --------------------------------------------------------------
    @classmethod
    def _to_db(cls, gain):
        return 20.0 * np.log10(np.maximum(np.asarray(gain, dtype=np.float64), cls._DB_FLOOR))

    @staticmethod
    def _from_db(db):
        return np.clip(10.0 ** (np.asarray(db, dtype=np.float64) / 20.0), 0.0, RANGE_GAIN_MAX)

    def _sanitize(self, points):
        arr = np.asarray(points if points is not None else [], dtype=np.float64).reshape(-1, 2)
        clean = []
        for x, y in arr:
            if np.isfinite(x) and np.isfinite(y):
                clean.append((float(np.clip(x, 0.0, 1.0)), float(np.clip(y, 0.0, RANGE_GAIN_MAX))))
        if not clean:
            clean = [(0.0, 1.0), (1.0, 6.0)]
        clean.sort(key=lambda p: p[0])
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

    def dispose(self):
        """Smette di disegnare: da chiamare prima di eliminare il pannello."""
        self._disposed = True
        self._on_curve_changed = None
        self._on_auto_all = None
        if self._canvas is not None:
            try:
                self._canvas.close()
            except Exception:
                pass

    def interp(self) -> str:
        return str(self._cb_interp.currentData() or "log")

    def auto_settings(self) -> dict:
        return {
            "start": None if self._chk_start_auto.isChecked() else self._spin_start.value() / 100.0,
            "strength": self._spin_strength.value() / 100.0,
            "max_db": float(self._spin_max_db.value()),
            "n_points": int(self._spin_npts.value()),
        }

    def gain_vector(self, n: int) -> np.ndarray:
        """Guadagno (lineare) della curva attuale su n campioni, come lo applica il motore."""
        from .gpr_processing import _build_pre_agc_gain
        return _build_pre_agc_gain(int(n), {
            "pre_agc_breakpoints": self._points.tolist(), "pre_agc_interp": self.interp(),
        })[:, 0].astype(np.float64)

    def _curve_db(self, n: int) -> np.ndarray:
        return self._to_db(self.gain_vector(n))

    def _axis_bounds(self):
        db = self._to_db(self._points[:, 1]) if self._points.size else np.zeros(1)
        lo = float(min(-3.0, np.floor(db.min() / 5.0) * 5.0 - 3.0))
        if self._chk_auto_axis.isChecked():
            hi = float(max(12.0, np.ceil((db.max() + 6.0) / 5.0) * 5.0))
        else:
            hi = float(max(self._manual_max, np.ceil(db.max() + 1.0)))
        return lo, hi

    def _axis_max(self) -> float:
        return self._axis_bounds()[1]

    def gain_max(self) -> float:
        return float(self._axis_max())

    def _emit_changed(self):
        if callable(self._on_curve_changed):
            try:
                self._on_curve_changed(self.points(), float(self._axis_max()))
            except Exception:
                pass

    # --- disegno -----------------------------------------------------------
    def _redraw(self, rescale: bool = True):
        if not HAS_MPL or self._ax is None or self._canvas is None or self._disposed:
            return
        if rescale:
            self._view_lo, self._view_max = self._axis_bounds()
        p = self._points
        db = self._to_db(p[:, 1])
        ax = self._ax
        ax.clear()
        n = 400
        xs = np.linspace(0.0, 1.0, n)
        ax.plot(self._curve_db(n), xs * self._depth, color="#2d6ba3", lw=1.4)
        ax.scatter(db, p[:, 0] * self._depth, s=36, color="#c0392b", zorder=3)
        ax.axvline(0.0, color="#888888", lw=0.8, ls="--")
        ax.set_xlim(self._view_lo, self._view_max)
        try:
            ax.set_ylim(*(self._ylim_fn() if self._ylim_fn is not None else (self._depth, 0.0)))
        except Exception:
            ax.set_ylim(self._depth, 0.0)
        ax.grid(True, ls=":", lw=0.5, alpha=0.7)
        ax.set_xlabel("Guadagno (dB)")
        ax.set_title(f"Breakpoint {len(p)}/{self._max_points}", fontsize=9)
        if self._selected_idx is not None and 0 <= int(self._selected_idx) < len(p):
            i = int(self._selected_idx)
            ax.scatter([float(db[i])], [float(p[i, 0] * self._depth)], s=70, color="#ffd000",
                       edgecolor="#1f1f1f", linewidths=1.0, zorder=5)
        if self._canvas.isVisible():
            self._canvas.draw_idle()

    def _nearest_idx(self, db, depth_frac):
        """Punto piu' vicino a (guadagno in dB, profondita' come frazione),
        con distanza in frazione d'asse."""
        p = self._points
        if p.shape[0] <= 0 or self._ax is None:
            return None, 1e9
        to_axes = self._ax.transAxes.inverted()
        pts = to_axes.transform(self._ax.transData.transform(
            np.column_stack([self._to_db(p[:, 1]), p[:, 0] * self._depth])))
        q = to_axes.transform(self._ax.transData.transform(
            [[float(db), float(depth_frac) * self._depth]]))[0]
        d = np.hypot(pts[:, 0] - q[0], pts[:, 1] - q[1])
        idx = int(np.argmin(d))
        return idx, float(d[idx])

    # --- mouse -------------------------------------------------------------
    def _event_values(self, event):
        db = float(np.clip(event.xdata, self._view_lo, self._view_max))
        frac = float(np.clip(event.ydata / self._depth, 0.0, 1.0))
        return db, frac

    def _on_press(self, event):
        if event is None or event.inaxes != self._ax or event.xdata is None or event.ydata is None:
            return
        db, depth = self._event_values(event)
        idx, dist = self._nearest_idx(db, depth)
        button = int(getattr(event, "button", 0) or 0)

        if button == 3:
            if idx is not None and dist < 0.05 and idx not in (0, len(self._points) - 1):
                self._points = self._sanitize(np.delete(self._points, idx, axis=0))
                self._selected_idx = None
                self._emit_changed()
                self._redraw()
            return

        if bool(getattr(event, "dblclick", False)) and button == 1:
            if len(self._points) < self._max_points and (idx is None or dist >= 0.02):
                self._points = self._sanitize(np.vstack(
                    [self._points, [[depth, float(self._from_db(db))]]]))
                self._selected_idx, _ = self._nearest_idx(db, depth)
                self._emit_changed()
                self._redraw()
            return

        if button == 1 and idx is not None and dist < 0.08:
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
        db, depth = self._event_values(event)
        p = self._points.copy()
        if i == 0:
            depth = 0.0
        elif i == (len(p) - 1):
            depth = 1.0
        else:
            lo = float(p[i - 1, 0] + 1e-4)
            hi = float(p[i + 1, 0] - 1e-4)
            depth = float(p[i, 0]) if hi <= lo else float(np.clip(depth, lo, hi))
        p[i, 0] = depth
        p[i, 1] = float(self._from_db(db))
        self._points = self._sanitize(p)
        self._did_drag = True
        self._emit_changed()
        # Durante il drag l'asse resta fermo: il punto non deve sfuggire al mouse.
        self._redraw(rescale=False)

    def _on_release(self, _event):
        was_dragging = self._drag_idx is not None
        self._did_drag = False
        self._drag_idx = None
        if was_dragging:
            self._redraw()

    # --- comandi -----------------------------------------------------------
    def _move_selected(self, delta_db: float):
        p = self._points.copy()
        if self._btn_all_shift.isChecked():
            p[:, 1] = self._from_db(self._to_db(p[:, 1]) + float(delta_db))
        else:
            if self._selected_idx is None:
                return
            i = int(self._selected_idx)
            if i < 0 or i >= len(p):
                return
            p[i, 1] = float(self._from_db(self._to_db(p[i, 1]) + float(delta_db)))
        self._points = self._sanitize(p)
        self._emit_changed()
        self._redraw()

    def _on_interp_changed(self, _i):
        self._emit_changed()
        self._redraw()

    def _on_gain_max_changed(self, value: float):
        self._manual_max = float(value)
        self._redraw()

    def _on_auto_axis_toggled(self, checked: bool):
        self._spin_gain_max.setEnabled(not checked)
        if not checked:
            self._manual_max = float(self._axis_bounds()[1])
            self._spin_gain_max.blockSignals(True)
            try:
                self._spin_gain_max.setValue(self._manual_max)
            finally:
                self._spin_gain_max.blockSignals(False)
        self._redraw()

    def _set_interp_exponential(self):
        i = self._cb_interp.findData("log")
        if i >= 0 and self._cb_interp.currentIndex() != i:
            self._cb_interp.blockSignals(True)
            self._cb_interp.setCurrentIndex(i)
            self._cb_interp.blockSignals(False)

    def _on_auto_clicked(self):
        from .gpr_profile_gain import auto_gain_points
        if self._env.size < 16:
            return
        self._points = self._sanitize(auto_gain_points(self._env, **self.auto_settings()))
        self._selected_idx = None
        self._set_interp_exponential()
        self._lbl_hint.setText(
            "Nessuna attenuazione misurabile dopo l'onda diretta: curva piatta (0 dB). "
            "Prova un Inizio manuale piu' basso o regola i punti a mano."
            if np.all(self._points[:, 1] <= 1.0 + 1e-9) else
            "Click: seleziona/trascina | Doppio click: aggiungi | Click destro: elimina"
        )
        self._emit_changed()
        self._redraw()

    def _on_auto_all_clicked(self):
        if not callable(self._on_auto_all):
            return
        mine = self._on_auto_all(self.auto_settings())
        if mine:
            self._points = self._sanitize(mine)
            self._selected_idx = None
            self._set_interp_exponential()
            self._emit_changed()
            self._redraw()

    def _on_reset(self):
        self._points = np.asarray([(0.0, 1.0), (1.0, 6.0)], dtype=np.float64)
        self._selected_idx = None
        self._drag_idx = None
        self._emit_changed()
        self._redraw()


# Nome storico (usato da test e import esistenti).
_RangeGainCurveDialog = RangeGainPanel


class AgcPanel(QWidget):
    """Controlli dell'AGC, sotto il radargramma (stessa impostazione del pannello TVG).

    L'AGC non ha una curva da disegnare: si adatta al segnale in finestre mobili.
    A sinistra del radargramma il viewer mostra il guadagno effettivo medio che
    produce sul profilo (dB in funzione della profondita'); le ampiezze
    prima/dopo sostituiscono l'A-scan. "Prima" e' il dato che arriva al passo AGC,
    cioe' dopo i passi attivi che lo precedono (TVG incluso).

    Segnali: ``finished_with`` ("one" / "all" / "cancel") e ``view_changed``.
    """

    finished_with = pyqtSignal(str)
    view_changed = pyqtSignal()
    _MAX_COLS = 600

    def __init__(self, params, trace_data=None, on_changed=None, parent=None,
                 auto_settings=None, on_auto_all=None):
        super().__init__(parent)
        self.resize(1000, 200)
        self._disposed = False
        self._on_changed = on_changed
        self._on_auto_all = on_auto_all
        self._depth = 1.0
        self._ylim_fn = None
        self._data = None
        self._sub = None
        self._after = None
        self._eff_db = None
        self._step = 1
        if trace_data is not None:
            arr = np.asarray(trace_data)
            if arr.ndim == 2 and arr.shape[0] >= 8 and arr.shape[1] >= 1:
                self._data = np.nan_to_num(arr.astype(np.float64))
                self._step = max(1, int(np.ceil(self._data.shape[1] / self._MAX_COLS)))
                self._sub = self._data[:, ::self._step]
        from .gpr_profile_gain import AGC_AUTO_DEFAULTS
        self._auto = {**AGC_AUTO_DEFAULTS, **(auto_settings or {})}
        p = dict(params or {})

        if HAS_MPL:
            self._fig = Figure(figsize=(2.4, 3.6))
            self._fig.subplots_adjust(left=0.30, right=0.95, top=0.94, bottom=0.11)
            self._ax = self._fig.add_subplot(111)
            self._canvas = FigureCanvasQTAgg(self._fig)
            self._canvas.setFixedWidth(240)
        else:
            self._fig = None
            self._ax = None
            self._canvas = None

        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        form = QFormLayout()
        self._spin_win = QSpinBox()
        self._spin_win.setRange(8, 512)
        self._spin_win.setValue(int(p.get("agc_win", 128)))
        self._spin_win.setSuffix(" campioni")
        self._spin_win.setToolTip(
            "Ampiezza della finestra mobile. Finestre corte esaltano ogni riflessione,\n"
            "finestre lunghe seguono solo il decadimento medio."
        )
        form.addRow("Finestra:", self._spin_win)
        self._spin_max = QDoubleSpinBox()
        self._spin_max.setRange(0.0, 1000.0)
        self._spin_max.setSingleStep(5.0)
        self._spin_max.setValue(float(p.get("agc_max_gain", 0.0)))
        self._spin_max.setToolTip(
            "Guadagno massimo rispetto al livello tipico (0 = nessun limite).\n"
            "Impedisce di portare il rumore profondo al livello del segnale."
        )
        form.addRow("Max gain (0=nessuno):", self._spin_max)
        start_row = QHBoxLayout()
        self._spin_start = QSpinBox()
        self._spin_start.setRange(0, 9999)
        self._spin_start.setValue(int(p.get("agc_sample_start", 0)))
        self._spin_start.setToolTip(
            "Primo campione su cui agisce l'AGC: sopra, l'onda diretta resta com'e'."
        )
        start_row.addWidget(self._spin_start)
        self._btn_start_gw = QPushButton("Onda diretta")
        self._btn_start_gw.setToolTip("Imposta l'inizio alla fine dell'onda diretta rilevata.")
        self._btn_start_gw.clicked.connect(self._on_start_from_ground_wave)
        start_row.addWidget(self._btn_start_gw)
        form.addRow("Da campione:", start_row)
        self._spin_lat = QSpinBox()
        self._spin_lat.setRange(0, 999)
        self._spin_lat.setValue(int(p.get("agc_lateral", 0)))
        self._spin_lat.setToolTip(
            "0 = guadagno per singola traccia (appiattisce le anomalie).\n"
            "N = energia mediata su N tracce: conserva le differenze tra tracce."
        )
        form.addRow("Tracce laterali:", self._spin_lat)
        self._chk_same_scale = QCheckBox("Ampiezze reali (stessa scala)")
        self._chk_same_scale.setChecked(False)
        self._chk_same_scale.setToolTip(
            "L'AGC riporta il segnale a scala ~1: di norma conviene confrontare le\n"
            "forme (disattivo). Attivo: stessa scala di prima del guadagno."
        )
        self._chk_same_scale.toggled.connect(lambda _c: self.view_changed.emit())
        form.addRow("", self._chk_same_scale)
        self._lbl_hint = QLabel(
            "L'AGC si adatta al segnale: a sinistra il guadagno effettivo medio, "
            "al posto dell'A-scan le ampiezze prima/dopo."
        )
        self._lbl_hint.setStyleSheet("color: #666;")
        self._lbl_hint.setWordWrap(True)
        form.addRow(self._lbl_hint)
        root.addLayout(form, 1)

        box = QGroupBox("Automatico dal profilo")
        fl = QFormLayout(box)
        self._spin_periods = QDoubleSpinBox()
        self._spin_periods.setRange(1.0, 20.0)
        self._spin_periods.setSingleStep(0.5)
        self._spin_periods.setValue(float(self._auto["periods"]))
        self._spin_periods.setToolTip("Finestra = tanti periodi del segnale dominante.")
        fl.addRow("Finestra (periodi):", self._spin_periods)
        self._chk_limit = QCheckBox("limita il max gain al rumore")
        self._chk_limit.setChecked(bool(self._auto["limit_noise"]))
        fl.addRow("", self._chk_limit)
        self._btn_auto = QPushButton("Calcola per questo profilo")
        self._btn_auto.clicked.connect(self._on_auto_clicked)
        fl.addRow(self._btn_auto)
        self._btn_auto_all = QPushButton("Calcola per tutti i profili")
        self._btn_auto_all.setToolTip(
            "Stesse impostazioni; ogni profilo e' analizzato dopo i passi attivi che\n"
            "precedono l'AGC, in parallelo e annullabile."
        )
        self._btn_auto_all.setEnabled(callable(on_auto_all))
        self._btn_auto_all.clicked.connect(self._on_auto_all_clicked)
        fl.addRow(self._btn_auto_all)
        root.addWidget(box, 1)

        btns = QVBoxLayout()
        b_one = QPushButton("Applica a questo profilo")
        b_one.setToolTip("Tiene questi valori solo per il profilo mostrato e chiude.")
        b_one.clicked.connect(lambda: self.finished_with.emit("one"))
        btns.addWidget(b_one)
        b_all = QPushButton("Applica a tutti i profili")
        b_all.setToolTip("Copia questi valori su tutti i profili e chiude.")
        b_all.clicked.connect(lambda: self.finished_with.emit("all"))
        btns.addWidget(b_all)
        b_cancel = QPushButton("Annulla")
        b_cancel.clicked.connect(lambda: self.finished_with.emit("cancel"))
        btns.addWidget(b_cancel)
        btns.addStretch(1)
        root.addLayout(btns, 0)

        for w in (self._spin_win, self._spin_max, self._spin_start, self._spin_lat):
            w.valueChanged.connect(lambda _v: self._changed())
        self._recompute()
        self._redraw()

    # --- integrazione col viewer --------------------------------------------
    def curve_canvas(self):
        return self._canvas

    def trace_data(self):
        return self._data

    def same_scale(self) -> bool:
        return bool(self._chk_same_scale.isChecked())

    def set_depth(self, depth_m: float, ylim_fn=None):
        self._depth = float(depth_m) if depth_m and depth_m > 0 else 1.0
        self._ylim_fn = ylim_fn
        self._redraw()

    def sync_ylim(self):
        if self._ax is None or self._disposed or self._ylim_fn is None:
            return
        try:
            self._ax.set_ylim(*self._ylim_fn())
            if self._canvas is not None and self._canvas.isVisible():
                self._canvas.draw_idle()
        except Exception:
            pass

    def dispose(self):
        self._disposed = True
        self._on_changed = None
        self._on_auto_all = None
        if self._canvas is not None:
            try:
                self._canvas.close()
            except Exception:
                pass

    def params(self) -> dict:
        return {
            "agc_win": int(self._spin_win.value()),
            "agc_max_gain": float(self._spin_max.value()),
            "agc_sample_start": int(self._spin_start.value()),
            "agc_lateral": int(self._spin_lat.value()),
        }

    def set_params(self, p: dict):
        for spin, key in ((self._spin_win, "agc_win"), (self._spin_max, "agc_max_gain"),
                          (self._spin_start, "agc_sample_start"), (self._spin_lat, "agc_lateral")):
            if key in p:
                spin.blockSignals(True)
                try:
                    spin.setValue(type(spin.value())(p[key]))
                finally:
                    spin.blockSignals(False)
        self._changed()

    def auto_settings(self) -> dict:
        return {"periods": float(self._spin_periods.value()),
                "limit_noise": bool(self._chk_limit.isChecked())}

    def amplitude_pair(self, k: int):
        """(prima, dopo) della traccia `k` del dato mostrato."""
        if self._sub is None or self._after is None:
            return None
        kk = int(np.clip(round(k / self._step), 0, self._sub.shape[1] - 1))
        before = self._sub[:, kk] - np.median(self._sub[:, kk])
        return before, np.asarray(self._after[:, kk], dtype=np.float64)

    # --- calcolo e disegno ----------------------------------------------------
    def _recompute(self):
        if self._sub is None:
            return
        from .gpr_processing import agc_gain
        p = self.params()
        lat = p["agc_lateral"]
        lat_eff = 0 if lat <= 1 else max(3, int(round(lat / self._step)))
        try:
            self._after = agc_gain(
                self._sub, window=p["agc_win"], clip_percentile=99.5,
                sample_start=p["agc_sample_start"],
                max_gain_factor=(p["agc_max_gain"] if p["agc_max_gain"] > 0 else None),
                lateral=lat_eff).astype(np.float64)
        except Exception:
            self._after = self._sub.copy()
        rb = np.sqrt(np.mean(self._sub ** 2, axis=1)) + 1e-12
        ra = np.sqrt(np.mean(self._after ** 2, axis=1)) + 1e-12
        s0 = min(p["agc_sample_start"], rb.size - 1)
        eff = 20.0 * np.log10(ra / rb)
        eff = eff - (eff[s0] if s0 > 0 else 0.0)      # relativo all'inizio dell'AGC
        eff[:s0] = 0.0
        self._eff_db = eff

    def _changed(self):
        if self._disposed:
            return
        self._recompute()
        self._redraw()
        if callable(self._on_changed):
            try:
                self._on_changed(self.params())
            except Exception:
                pass

    def _redraw(self):
        if not HAS_MPL or self._ax is None or self._canvas is None or self._disposed:
            return
        ax = self._ax
        ax.clear()
        if self._eff_db is not None:
            n = self._eff_db.size
            ax.plot(self._eff_db, np.linspace(0.0, self._depth, n), color="#2d6ba3", lw=1.4)
            s0 = self._spin_start.value()
            if s0 > 0:
                ax.axhline(self._depth * s0 / n, color="#e0a040", lw=1.0, ls="--")
            lo = float(min(-2.0, np.floor(self._eff_db.min()) - 2.0))
            hi = float(max(6.0, np.ceil(self._eff_db.max()) + 2.0))
            ax.set_xlim(lo, hi)
        ax.axvline(0.0, color="#888888", lw=0.8, ls="--")
        try:
            ax.set_ylim(*(self._ylim_fn() if self._ylim_fn is not None else (self._depth, 0.0)))
        except Exception:
            ax.set_ylim(self._depth, 0.0)
        ax.grid(True, ls=":", lw=0.5, alpha=0.7)
        ax.set_xlabel("Guadagno AGC (dB)")
        ax.set_title("AGC effettivo", fontsize=9)
        if self._canvas.isVisible():
            self._canvas.draw_idle()

    # --- comandi ---------------------------------------------------------------
    def _on_start_from_ground_wave(self):
        if self._data is None:
            return
        from .gpr_profile_gain import envelope_db, ground_wave_end
        s0 = int(round(ground_wave_end(envelope_db(self._data)) * self._data.shape[0]))
        self._spin_start.setValue(s0)

    def _on_auto_clicked(self):
        from .gpr_profile_gain import auto_agc_params
        if self._data is None:
            return
        self.set_params(auto_agc_params(self._data, **self.auto_settings()))

    def _on_auto_all_clicked(self):
        if not callable(self._on_auto_all):
            return
        mine = self._on_auto_all(self.auto_settings())
        if mine:
            self.set_params(mine)


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
                 spec=None, loader=None, points_cache=None):
        super().__init__()
        self._points_cache = points_cache
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
                    points_cache=self._points_cache,
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
