# -*- coding: utf-8 -*-
"""Statistiche dei valori di una time-slice (senza dipendenze Qt).

Servono all'istogramma del pannello Slicing (`amplitude_histogram.py`) e ai test:
tenerle qui permette di provarle con qualunque Python.
"""

from __future__ import annotations

import numpy as np

N_BINS = 64
# fraction of the values cut off each end when choosing the histogram range, so a
# handful of spikes does not squeeze the whole distribution into one bin
_TAIL_PCT = 0.2


def finite_values(grid) -> np.ndarray:
    """1-D float64 array of the finite values of ``grid`` (any shape / None)."""
    if grid is None:
        return np.empty(0, dtype=np.float64)
    a = np.asarray(grid, dtype=np.float64).ravel()
    return a[np.isfinite(a)]


def value_stats(values) -> dict | None:
    """``n, min, p2, p50, p98, max, mean, std`` of the finite values, or None."""
    v = finite_values(values)
    if v.size == 0:
        return None
    p2, p50, p98 = np.percentile(v, [2.0, 50.0, 98.0])
    return {
        "n": int(v.size),
        "min": float(v.min()),
        "p2": float(p2),
        "p50": float(p50),
        "p98": float(p98),
        "max": float(v.max()),
        "mean": float(v.mean()),
        "std": float(v.std()),
    }


def _fmt(x: float) -> str:
    if x == 0.0:
        return "0"
    a = abs(x)
    if a >= 1e5 or a < 1e-3:
        return f"{x:.3g}"
    return f"{x:.4g}"


def format_stats(stats: dict | None) -> str:
    """One compact line for the label under the histogram."""
    if not stats:
        return "nessun dato"
    return (
        f"n {stats['n']:,} · min {_fmt(stats['min'])} · p2 {_fmt(stats['p2'])} · "
        f"mediana {_fmt(stats['p50'])} · p98 {_fmt(stats['p98'])} · "
        f"max {_fmt(stats['max'])} · σ {_fmt(stats['std'])}"
    ).replace(",", " ")


def robust_range(values, vmin=None, vmax=None) -> tuple[float, float]:
    """Histogram x-range: [p0.2, p99.8] of the data, widened to contain the
    colour limits so their markers are always visible. Never empty."""
    v = finite_values(values)
    if v.size == 0:
        lo, hi = 0.0, 1.0
    else:
        lo, hi = (float(x) for x in np.percentile(v, [_TAIL_PCT, 100.0 - _TAIL_PCT]))
    for lim in (vmin, vmax):
        if lim is not None and np.isfinite(lim):
            lo = min(lo, float(lim))
            hi = max(hi, float(lim))
    if not hi > lo:
        hi = lo + 1.0
    return lo, hi


def histogram_counts(values, lo: float, hi: float, n_bins: int = N_BINS) -> np.ndarray:
    """Counts of the finite values in ``n_bins`` equal bins over ``[lo, hi]``
    (values outside the range fall in the first / last bin: nothing is lost)."""
    v = np.clip(finite_values(values), lo, hi)
    counts, _ = np.histogram(v, bins=int(n_bins), range=(float(lo), float(hi)))
    return counts.astype(np.int64)


__all__ = [
    "N_BINS", "finite_values", "value_stats", "format_stats", "robust_range",
    "histogram_counts",
]
