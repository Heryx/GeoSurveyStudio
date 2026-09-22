# -*- coding: utf-8 -*-
"""Istogramma dei valori della time-slice con limiti colore trascinabili.

Mostra la distribuzione delle ampiezze della slice visibile (dopo tutta la
rifinitura) e, sotto, la barra colore compresa fra i due marcatori: e' il modo
piu' diretto per scegliere a occhio quali valori vanno a fondo scala. Trascinare
un marcatore emette ``limits_changed(vmin, vmax)``; chi ascolta decide se passare
ai limiti fissi.

Le altezze delle barre sono in radice del conteggio, cosi' le code (dove stanno i
bersagli) restano visibili accanto al picco del fondo.
"""

from __future__ import annotations

import numpy as np
from qgis.PyQt.QtCore import QPointF, QRectF, QSize, Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor, QLinearGradient, QPainter, QPen, QPolygonF
from qgis.PyQt.QtWidgets import QSizePolicy, QWidget

from .amplitude_stats import N_BINS, finite_values, histogram_counts, robust_range

_PAD_X = 6.0
_PAD_TOP = 4.0
_STRIP_H = 9.0
_GAP = 3.0
_HANDLE = 4.0          # half-width of the triangular grab handle
_GRAB_PX = 9.0         # how close (px) a click must be to take a marker
_MIN_GAP_FRAC = 0.005  # markers never closer than this fraction of the range


def _event_x(event) -> float:
    try:
        return float(event.position().x())        # Qt6
    except AttributeError:
        return float(event.pos().x())             # Qt5


class AmplitudeHistogram(QWidget):
    """Histogram + colour strip; ``limits_changed(vmin, vmax)`` on drag."""

    limits_changed = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._counts = None
        self._lo, self._hi = 0.0, 1.0
        self._vmin, self._vmax = 0.0, 1.0
        self._colors: list[str] = []
        self._drag: str | None = None
        self.setMouseTracking(True)
        self.setMinimumHeight(64)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(84)
        self.setToolTip(
            "Distribuzione dei valori della slice (altezza in radice del conteggio).\n"
            "Trascina i due marcatori per fissare i limiti della scala colore: i valori\n"
            "fuori saturano al colore estremo."
        )

    # ------------------------------------------------------------------ data
    def sizeHint(self):
        return QSize(240, 84)

    def set_data(self, values, vmin: float, vmax: float) -> None:
        """New values (any shape, NaN ignored) and the current colour limits."""
        if self._drag is not None:      # never re-scale under the user's hand
            return
        v = finite_values(values)
        self._vmin, self._vmax = float(vmin), float(vmax)
        if v.size < 2:
            self._counts = None
        else:
            self._lo, self._hi = robust_range(v, vmin, vmax)
            self._counts = histogram_counts(v, self._lo, self._hi, N_BINS)
        self.update()

    def set_limits(self, vmin: float, vmax: float) -> None:
        """Move the markers without touching the histogram (no signal)."""
        if self._drag is not None:
            return
        self._vmin, self._vmax = float(vmin), float(vmax)
        self._lo = min(self._lo, self._vmin)
        self._hi = max(self._hi, self._vmax)
        self.update()

    def set_colors(self, colors) -> None:
        """``#rrggbb`` list of the colormap, drawn between the two markers."""
        self._colors = [str(c) for c in (colors or [])]
        self.update()

    # -------------------------------------------------------------- geometry
    def _plot_rect(self) -> QRectF:
        h = float(self.height())
        return QRectF(
            _PAD_X, _PAD_TOP,
            max(1.0, float(self.width()) - 2.0 * _PAD_X),
            max(8.0, h - _PAD_TOP - _STRIP_H - _GAP - 3.0),
        )

    def x_of(self, value: float) -> float:
        """Widget x of a data value."""
        r = self._plot_rect()
        span = max(self._hi - self._lo, 1e-30)
        return r.left() + (float(value) - self._lo) / span * r.width()

    def value_of(self, x: float) -> float:
        """Data value under the widget x."""
        r = self._plot_rect()
        span = max(self._hi - self._lo, 1e-30)
        return self._lo + (float(x) - r.left()) / max(r.width(), 1.0) * span

    # ----------------------------------------------------------------- paint
    def paintEvent(self, _event):
        p = QPainter(self)
        try:
            pal = self.palette()
            p.fillRect(self.rect(), pal.base())
            r = self._plot_rect()
            strip = QRectF(r.left(), r.bottom() + _GAP, r.width(), _STRIP_H)
            grey = QColor(pal.mid().color())

            if self._counts is None or int(self._counts.max()) <= 0:
                p.setPen(pal.text().color())
                p.drawText(self.rect(), Qt.AlignCenter, "nessun dato")
                return

            n = len(self._counts)
            peak = float(self._counts.max())
            bw = r.width() / n
            span = self._hi - self._lo
            hi_col = QColor(pal.highlight().color())
            out_col = QColor(hi_col)
            out_col.setAlpha(70)
            for i, c in enumerate(self._counts):
                if c <= 0:
                    continue
                centre = self._lo + (i + 0.5) * span / n
                col = hi_col if self._vmin <= centre <= self._vmax else out_col
                hbar = r.height() * float(np.sqrt(c / peak))
                p.fillRect(QRectF(r.left() + i * bw, r.bottom() - hbar, max(bw - 0.6, 1.0), hbar), col)

            x0 = min(max(self.x_of(self._vmin), r.left()), r.right())
            x1 = min(max(self.x_of(self._vmax), r.left()), r.right())
            p.fillRect(strip, grey)
            if x1 > x0 + 0.5:
                if self._colors:
                    grad = QLinearGradient(x0, 0.0, x1, 0.0)
                    m = len(self._colors)
                    for k, c in enumerate(self._colors):
                        grad.setColorAt(k / max(m - 1, 1), QColor(c))
                    p.fillRect(QRectF(x0, strip.top(), x1 - x0, strip.height()), grad)
                else:
                    grad = QLinearGradient(x0, 0.0, x1, 0.0)
                    grad.setColorAt(0.0, QColor("#000000"))
                    grad.setColorAt(1.0, QColor("#ffffff"))
                    p.fillRect(QRectF(x0, strip.top(), x1 - x0, strip.height()), grad)

            pen = QPen(pal.text().color())
            pen.setWidthF(1.4)
            p.setPen(pen)
            for x in (x0, x1):
                p.drawLine(QPointF(x, r.top()), QPointF(x, strip.bottom()))
                p.setBrush(pal.text().color())
                p.drawPolygon(QPolygonF([
                    QPointF(x - _HANDLE, r.top()), QPointF(x + _HANDLE, r.top()),
                    QPointF(x, r.top() + _HANDLE * 1.6),
                ]))
        finally:
            p.end()

    # ----------------------------------------------------------------- mouse
    def _pick(self, x: float) -> str | None:
        dmin = abs(x - self.x_of(self._vmin))
        dmax = abs(x - self.x_of(self._vmax))
        if min(dmin, dmax) > _GRAB_PX:
            return None
        return "min" if dmin <= dmax else "max"

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._counts is not None:
            self._drag = self._pick(_event_x(event))
            if self._drag is not None:
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        x = _event_x(event)
        if self._drag is None:
            self.setCursor(Qt.SplitHCursor if self._pick(x) else Qt.ArrowCursor)
            super().mouseMoveEvent(event)
            return
        gap = _MIN_GAP_FRAC * max(self._hi - self._lo, 1e-30)
        v = min(max(self.value_of(x), self._lo), self._hi)
        if self._drag == "min":
            self._vmin = min(v, self._vmax - gap)
        else:
            self._vmax = max(v, self._vmin + gap)
        self.update()
        self.limits_changed.emit(float(self._vmin), float(self._vmax))
        event.accept()

    def mouseReleaseEvent(self, event):
        if self._drag is not None:
            self._drag = None
            self.update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    @property
    def limits(self) -> tuple[float, float]:
        return float(self._vmin), float(self._vmax)


__all__ = ["AmplitudeHistogram"]
