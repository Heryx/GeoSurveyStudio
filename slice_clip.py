# -*- coding: utf-8 -*-
"""Limite della griglia delle time-slice: poligoni di ritaglio.

Quando i profili di andata e ritorno non finiscono tutti alla stessa distanza,
ai bordi della griglia restano "tendoni": strisce di dati che sporgono oltre i
profili vicini. Lo spec porta uno o piu' poligoni (coordinate mondo, stesso CRS
dei profili); le celle fuori da *tutti* i poligoni diventano NoData.

Niente matplotlib/QGIS qui: il test punto-in-poligono e' una scanline numpy
(regola pari-dispari), cosi' funziona identico nel thread di preview, nel
salvataggio a processi e nei test.
"""

from __future__ import annotations

import numpy as np

MIN_VERTICES = 3


def normalize_rings(rings) -> list[list[list[float]]]:
    """Valida e normalizza i poligoni in ``[[ [x, y], ... ], ...]`` (float).

    Toglie i vertici non finiti e l'eventuale punto finale uguale al primo;
    scarta gli anelli con meno di 3 vertici distinti. Accetta ``None``.
    """
    out: list[list[list[float]]] = []
    for ring in rings or []:
        pts: list[list[float]] = []
        for p in ring or []:
            try:
                x, y = float(p[0]), float(p[1])
            except (TypeError, ValueError, IndexError):
                continue
            if np.isfinite(x) and np.isfinite(y):
                pts.append([x, y])
        if len(pts) > 1 and pts[0] == pts[-1]:
            pts.pop()
        if len(pts) >= MIN_VERTICES:
            out.append(pts)
    return out


def ring_mask(ring, x_min: float, y_min: float, n_x: int, n_y: int, res: float) -> np.ndarray:
    """Maschera booleana ``(n_y, n_x)`` delle celle il cui CENTRO sta nell'anello.

    Convenzione dello slicer: il centro della cella (riga r, colonna c) e'
    ``(x_min + c*res, y_min + r*res)``, riga 0 = ``y_min``.
    """
    mask = np.zeros((int(n_y), int(n_x)), dtype=bool)
    pts = np.asarray(ring, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < MIN_VERTICES:
        return mask
    # lavoro in unita' di cella: gli indici delle righe sono interi
    gx = (pts[:, 0] - float(x_min)) / float(res)
    gy = (pts[:, 1] - float(y_min)) / float(res)
    gx2 = np.roll(gx, -1)
    gy2 = np.roll(gy, -1)

    rows = np.arange(int(n_y), dtype=np.float64)
    crossings: list[list[float]] = [[] for _ in range(int(n_y))]
    for x1, y1, x2, y2 in zip(gx, gy, gx2, gy2):
        if y1 == y2:
            continue  # lato orizzontale: nessun incrocio con le scanline
        lo, hi = (y1, y2) if y1 < y2 else (y2, y1)
        # semi-aperto [lo, hi): un vertice sulla scanline non conta due volte
        r0 = max(0, int(np.ceil(lo)))
        r1 = min(int(n_y) - 1, int(np.ceil(hi)) - 1)
        if r1 < r0:
            continue
        rr = rows[r0:r1 + 1]
        xs = x1 + (rr - y1) * (x2 - x1) / (y2 - y1)
        for r, xv in zip(range(r0, r1 + 1), xs):
            crossings[r].append(float(xv))

    for r, xs in enumerate(crossings):
        if len(xs) < 2:
            continue
        xs.sort()
        for a, b in zip(xs[0::2], xs[1::2]):
            c0 = max(0, int(np.ceil(a)))
            c1 = min(int(n_x) - 1, int(np.floor(b)))
            if c1 >= c0:
                mask[r, c0:c1 + 1] = True
    return mask


def polygons_mask(rings, x_min: float, y_min: float, n_x: int, n_y: int, res: float) -> np.ndarray:
    """Unione delle maschere di tutti i poligoni."""
    mask = np.zeros((int(n_y), int(n_x)), dtype=bool)
    for ring in normalize_rings(rings):
        mask |= ring_mask(ring, x_min, y_min, n_x, n_y, res)
    return mask


def apply_clip(grid: np.ndarray, rings, x_min: float, y_min: float, res: float) -> np.ndarray:
    """Mette a NaN le celle fuori dai poligoni. Senza poligoni restituisce ``grid``
    invariata (stesso oggetto)."""
    if not normalize_rings(rings):
        return grid
    g = np.asarray(grid)
    mask = polygons_mask(rings, x_min, y_min, g.shape[1], g.shape[0], res)
    return np.where(mask, g, np.nan).astype(np.float32, copy=False)


def apply_clip_to_grids(grids: list, meta: dict, rings) -> None:
    """Applica il ritaglio in-place a una lista di ``{"grid": ...}`` dello slicer
    e riallinea le statistiche di riempimento."""
    if not grids or not normalize_rings(rings):
        return
    x_min = float(meta["x_min"])
    y_min = float(meta["y_min"])
    res = float(meta["resolution"])
    first = np.asarray(grids[0]["grid"])
    mask = polygons_mask(rings, x_min, y_min, first.shape[1], first.shape[0], res)
    for item in grids:
        g = np.asarray(item["grid"])
        item["grid"] = np.where(mask, g, np.nan).astype(np.float32, copy=False)
        n_valid = int(np.count_nonzero(np.isfinite(item["grid"])))
        item["n_valid_cells"] = n_valid
        item["fill_pct"] = 100.0 * n_valid / max(int(item["grid"].size), 1)
    fills = np.asarray([it["fill_pct"] for it in grids], dtype=np.float64)
    meta["fill_pct_mean"] = float(np.mean(fills))
    meta["fill_pct_min"] = float(np.min(fills))
    meta["clip_applied"] = True


__all__ = ["normalize_rings", "ring_mask", "polygons_mask", "apply_clip",
           "apply_clip_to_grids"]
