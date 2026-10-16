# -*- coding: utf-8 -*-
"""Vista della sezione ERT sotto il radargramma, con l'asse delle distanze condiviso.

Il pannello non possiede dati: il dock gli passa la sezione ERT, il suo allineamento al
canale GPR mostrato e la griglia gia' pronta. Qui si disegna soltanto, nelle coordinate del
radargramma (distanza lungo il profilo GPR, profondita' in m), cosi' un oggetto sta alla
stessa x in entrambe le viste.

Modalita' verticale: sempre "piana" (profondita' dal piano di campagna). Il radargramma del
pannello non porta ancora una quota per traccia; una sezione ERT con z in quota si mostra
comunque come profondita' dalla sua quota piu' alta e il pannello lo dichiara.
"""

from __future__ import annotations

import numpy as np
from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

try:
    from matplotlib.ticker import FuncFormatter

    from .gpr_viewer_common import FigureCanvasQTAgg, Figure, HAS_MPL, _BlitManager
except Exception:                                    # matplotlib mancante
    HAS_MPL = False

# Stessi margini (in punti) di RadargramDock._layout_axes: gli assi x restano allineati.
_LEFT_PT, _RIGHT_PT, _TOP_PT, _BOTTOM_PT = 38, 8, 5, 27
_CMAP = "viridis"
_CLIM_PERCENTILES = (2.0, 98.0)


def _fmt_ohm(log_value: float) -> str:
    v = 10.0 ** float(log_value)
    return f"{v:.0f}" if v >= 100 else (f"{v:.1f}" if v >= 10 else f"{v:.2f}")


class ErtSectionPanel(QWidget):
    """Barra (sezione, importa, elimina, profondita') e canvas della sezione ERT."""

    import_requested = pyqtSignal()
    delete_requested = pyqtSignal(str)
    section_changed = pyqtSignal(str)      # nome scelto nel combo ("" = nessuna)
    depth_follow_changed = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._updating = False
        self._grid = None            # (x_gpr, depth, data) mostrati, per il suggerimento sotto il cursore
        self._fig = self._ax = self._canvas = self._blit = self._vline = None
        self._build()

    # --- interfaccia ---------------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)
        bar = QHBoxLayout()
        bar.setContentsMargins(4, 0, 4, 0)
        bar.setSpacing(6)
        bar.addWidget(QLabel("ERT"))
        self._cb = QComboBox()
        self._cb.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self._cb.setMinimumContentsLength(10)
        self._cb.setToolTip("Sezioni ERT importate nel progetto (cartella ERT).")
        self._cb.currentIndexChanged.connect(self._on_combo)
        bar.addWidget(self._cb)
        self._btn_import = QToolButton()
        self._btn_import.setText("+")
        self._btn_import.setAutoRaise(True)
        self._btn_import.setToolTip("Importa una sezione ERT invertita (CSV di ResIPY) e georiferiscila.")
        self._btn_import.clicked.connect(self.import_requested)
        bar.addWidget(self._btn_import)
        self._btn_delete = QToolButton()
        self._btn_delete.setText("\U0001F5D1")
        self._btn_delete.setAutoRaise(True)
        self._btn_delete.setToolTip("Toglie dal progetto la sezione ERT scelta.")
        self._btn_delete.clicked.connect(lambda: self.delete_requested.emit(self.current_name()))
        bar.addWidget(self._btn_delete)
        self._chk_depth = QCheckBox("Stessa profondità del radargramma")
        self._chk_depth.setChecked(True)
        self._chk_depth.setToolTip("Limita la sezione ERT alla profondità mostrata dal radargramma, "
                                   "per confrontare le due viste alla stessa scala verticale.")
        self._chk_depth.toggled.connect(self.depth_follow_changed)
        bar.addWidget(self._chk_depth)
        self._lbl = QLabel("")
        self._lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self._lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        bar.addWidget(self._lbl, 1)
        self._lbl_hover = QLabel("")
        self._lbl_hover.setMinimumWidth(190)
        bar.addWidget(self._lbl_hover)
        root.addLayout(bar)
        if not HAS_MPL:
            root.addWidget(QLabel("matplotlib non trovato: impossibile mostrare la sezione ERT."))
            return
        self._fig = Figure(figsize=(8, 2.0))
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setMinimumHeight(90)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._blit = _BlitManager(self._canvas)
        self._canvas.mpl_connect("resize_event", self._layout_axes)
        self._canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._canvas.mpl_connect("axes_leave_event", lambda _e: self._hover_text(None))
        root.addWidget(self._canvas, 1)
        self._draw_empty("Nessuna sezione ERT: usa + per importarne una.")

    # --- elenco ----------------------------------------------------------------------------
    def set_names(self, names, current: str = "") -> None:
        self._updating = True
        try:
            self._cb.clear()
            self._cb.addItem("(nessuna)", "")
            for n in names:
                self._cb.addItem(n, n)
            i = self._cb.findData(current)
            self._cb.setCurrentIndex(i if i >= 0 else 0)
        finally:
            self._updating = False
        self._btn_delete.setEnabled(bool(self.current_name()))

    def current_name(self) -> str:
        return str(self._cb.currentData() or "")

    def follows_depth(self) -> bool:
        return self._chk_depth.isChecked()

    def _on_combo(self, _i: int) -> None:
        self._btn_delete.setEnabled(bool(self.current_name()))
        if not self._updating:
            self.section_changed.emit(self.current_name())

    # --- disegno -----------------------------------------------------------------------------
    def _layout_axes(self, _event=None) -> None:
        if self._ax is None:
            return
        w, h = float(self._fig.bbox.width), float(self._fig.bbox.height)
        if w < 60 or h < 40:
            return
        k = self._fig.dpi / 72.0
        left, right, top, bottom = _LEFT_PT * k, _RIGHT_PT * k, _TOP_PT * k, _BOTTOM_PT * k
        self._ax.set_position([left / w, bottom / h, max(0.1, 1.0 - (left + right) / w),
                               max(0.1, 1.0 - (top + bottom) / h)])

    def _draw_empty(self, text: str) -> None:
        if not HAS_MPL:
            return
        self._fig.clf()
        self._ax = self._fig.add_axes([0.05, 0.1, 0.9, 0.8])
        self._ax.set_axis_off()
        self._ax.text(0.5, 0.5, text, ha="center", va="center", transform=self._ax.transAxes,
                      fontsize=9, color="gray")
        self._grid = None
        self._vline = None
        self._blit.set_artists([])
        self._blit.invalidate()
        self._canvas.draw_idle()

    def clear(self, text: str = "Nessuna sezione ERT: usa + per importarne una.") -> None:
        self._lbl.setText("")
        self._draw_empty(text)

    def show_section(self, name: str, grid, alignment, z_mode: str, x_view=None, depth_max=None) -> None:
        """Disegna ``grid`` (``ErtGrid``) con le colonne spostate sulla distanza GPR.

        ``alignment`` e' un ``ErtAlignment``. Le colonne fuori dal profilo GPR non si disegnano.
        """
        if not HAS_MPL:
            return
        x_gpr = alignment.ert_to_gpr(grid.x_axis)
        keep = np.isfinite(x_gpr)
        if keep.sum() < 2:
            self.clear(f"La sezione ERT «{name}» non cade sul profilo GPR mostrato.")
            self._lbl.setText(f"scostamento laterale mediano {alignment.median_offset_m:.1f} m")
            return
        order = np.argsort(x_gpr[keep])
        xs = x_gpr[keep][order]
        data = grid.data[:, keep][:, order]
        depth = grid.depth_axis
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            self.clear(f"La sezione ERT «{name}» non ha valori nella parte mostrata.")
            return
        lo, hi = np.percentile(finite, _CLIM_PERCENTILES)
        self._fig.clf()
        ax = self._ax = self._fig.add_axes([0.05, 0.15, 0.9, 0.8])
        mesh = ax.pcolormesh(xs, depth, np.ma.masked_invalid(data), cmap=_CMAP, vmin=lo, vmax=hi,
                             shading="auto", rasterized=True)
        ax.set_ylim(depth[-1] if depth_max is None else float(depth_max), 0.0)
        if x_view is not None:
            ax.set_xlim(*x_view)
        else:
            ax.set_xlim(float(xs[0]), float(xs[-1]))
        ax.set_xlabel("Distanza (m)", fontsize=8, labelpad=1)
        ax.set_ylabel("Profondità (m)", fontsize=8, labelpad=1)
        ax.tick_params(labelsize=8, pad=1)
        cax = ax.inset_axes([0.70, 0.88, 0.28, 0.07])
        cb = self._fig.colorbar(mesh, cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=7, pad=1)
        cb.ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: _fmt_ohm(v)))
        cb.set_label("Resistività (Ω·m)", fontsize=7, labelpad=1)
        self._vline = ax.axvline(0.0, color="yellow", lw=1, visible=False)
        self._blit.set_artists([self._vline])
        self._grid = (xs, depth, data)
        self._layout_axes()
        self._blit.invalidate()
        self._canvas.draw_idle()
        mode = ("piana (profondità dal piano di campagna)" if z_mode == "depth"
                else "piana (z in quota: usata come profondità dal punto più alto, quota GPR non disponibile)")
        self._lbl.setText(f"{name} · scostamento laterale {alignment.median_offset_m:.1f} m "
                          f"· sovrapposta per {alignment.overlap_m:.0f} m · modalità {mode}")
        self._lbl.setToolTip(self._lbl.text())

    # --- sincronizzazione con il radargramma ---------------------------------------------------
    def set_xlim(self, x0: float, x1: float) -> None:
        if self._ax is not None and self._grid is not None:
            self._ax.set_xlim(x0, x1)
            self._blit.invalidate()
            self._canvas.draw_idle()

    def set_ylim(self, depth_max: float) -> None:
        if self._ax is not None and self._grid is not None:
            self._ax.set_ylim(float(depth_max), 0.0)
            self._blit.invalidate()
            self._canvas.draw_idle()

    def set_cursor_x(self, x) -> None:
        if self._vline is None:
            return
        if x is None:
            self._vline.set_visible(False)
        else:
            self._vline.set_xdata([x, x])
            self._vline.set_visible(True)
        self._blit.update()

    # --- passaggio del mouse --------------------------------------------------------------------
    def value_at(self, x: float, depth: float):
        """log10 della resistivita' alla distanza ``x`` e profondita' ``depth``, o None."""
        if self._grid is None:
            return None
        xs, ds, data = self._grid
        if not (xs[0] <= x <= xs[-1]) or not (ds[0] <= depth <= ds[-1]):
            return None
        ix = int(np.clip(np.searchsorted(xs, x), 1, len(xs) - 1))
        ix -= int(abs(x - xs[ix - 1]) < abs(x - xs[ix]))
        iz = int(np.argmin(np.abs(ds - depth)))
        v = data[iz, ix]
        return float(v) if np.isfinite(v) else None

    def _on_motion(self, event) -> None:
        if event.inaxes is not self._ax or event.xdata is None:
            return
        self._hover_text((float(event.xdata), float(event.ydata)))

    def _hover_text(self, xy) -> None:
        if xy is None:
            return
        v = self.value_at(*xy)
        self._lbl_hover.setText(
            f"x {xy[0]:.1f} m · prof. {xy[1]:.2f} m · "
            + ("fuori copertura" if v is None else f"ρ {_fmt_ohm(v)} Ω·m"))
