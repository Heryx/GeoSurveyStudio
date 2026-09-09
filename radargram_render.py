# -*- coding: utf-8 -*-
"""Disegno di un radargramma su un ``matplotlib.axes.Axes`` — foglia condivisa
fra il canvas principale del Profile Viewer e i mini-canvas dei dialog
(es. l'anteprima marker del Posizionamento Manuale).

Nessuno stato: prende un ``Axes`` e un array 2D ``(n_samples, n_traces)`` e
disegna. Puro rispetto a Qt/QGIS; l'unica dipendenza è numpy + matplotlib.
"""

from __future__ import annotations

import numpy as np

__all__ = ["downsample_radargram", "robust_vlim", "draw_radargram"]


def downsample_radargram(data: np.ndarray, max_rows: int = 700, max_cols: int = 1400) -> np.ndarray:
    """Sottocampiona (stride) un radargramma grande per un disegno rapido.
    Non interpola: prende una traccia/campione ogni ``k``."""
    a = np.asarray(data, dtype=np.float32)
    if a.ndim != 2 or a.size == 0:
        return a
    n_r, n_c = a.shape
    r_step = max(1, int(np.ceil(n_r / float(max_rows))))
    c_step = max(1, int(np.ceil(n_c / float(max_cols))))
    return a[::r_step, ::c_step]


def robust_vlim(data: np.ndarray, pct: float = 98.0) -> tuple[float, float]:
    """Limiti simmetrici (-v, +v) dal percentile ``pct`` dei valori finiti."""
    a = np.asarray(data, dtype=np.float64)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return -1.0, 1.0
    v = float(np.percentile(np.abs(finite), float(np.clip(pct, 50.0, 100.0))))
    if not np.isfinite(v) or v <= 0.0:
        v = float(np.max(np.abs(finite))) or 1.0
    return -v, v


def draw_radargram(
    ax,
    data: np.ndarray,
    *,
    dist_max: float | None = None,
    depth_max: float | None = None,
    cmap: str = "gray",
    vmin: float | None = None,
    vmax: float | None = None,
    interpolation: str = "nearest",
    aspect: str = "auto",
    marker_traces=None,
    n_traces_full: int | None = None,
    marker_kwargs: dict | None = None,
    clear: bool = True,
):
    """Disegna ``data`` su ``ax`` come immagine e, se dati, i marker come linee
    verticali.

    - ``dist_max`` / ``depth_max`` : estensione in unità reali (m). Se None,
      l'asse X è in tracce e l'asse Y in campioni.
    - ``marker_traces`` : indici di traccia (nella griglia PIENA, non
      sottocampionata) su cui tracciare una linea verticale.
    - ``n_traces_full`` : numero di tracce della griglia piena, necessario per
      mappare ``marker_traces`` sull'asse X quando ``data`` è sottocampionato o
      quando l'asse X è in metri.

    Ritorna ``(image, vlines)``.
    """
    a = np.asarray(data, dtype=np.float32)
    if clear:
        ax.clear()
    if a.ndim != 2 or a.size == 0:
        return None, []

    n_r, n_c = a.shape
    n_full = int(n_traces_full) if n_traces_full else n_c
    if vmin is None or vmax is None:
        lo, hi = robust_vlim(a)
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax

    x_hi = float(dist_max) if (dist_max is not None and dist_max > 0) else float(n_full)
    y_hi = float(depth_max) if (depth_max is not None and depth_max > 0) else float(n_r)
    image = ax.imshow(
        a, aspect=aspect, cmap=cmap, vmin=vmin, vmax=vmax,
        extent=[0.0, x_hi, y_hi, 0.0], interpolation=interpolation, resample=True,
    )

    vlines = []
    if marker_traces is not None and n_full > 1:
        mk = dict(color="#ff3b30", lw=1.0, alpha=0.9)
        mk.update(marker_kwargs or {})
        scale = x_hi / float(max(n_full - 1, 1))
        for t in np.asarray(list(marker_traces), dtype=np.float64).ravel():
            if np.isfinite(t):
                vlines.append(ax.axvline(float(t) * scale, **mk))

    ax.set_xlim(0.0, x_hi)
    ax.set_ylim(y_hi, 0.0)
    return image, vlines
