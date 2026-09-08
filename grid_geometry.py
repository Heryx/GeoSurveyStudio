# -*- coding: utf-8 -*-
"""Geometria di griglie / gridset — Fase 5 §5.4 del blueprint processing.

Operazioni sui raster time-slice come *insiemi georiferiti*, non filtri:
- ``rotate_translate`` : ruota/trasla una griglia locale (ricampionamento
  bilineare) per georeferenziarla o allinearla a un'origine comune;
- ``append_grids``      : fonde più gridset (stessa cella) in un unico raster
  che ne contiene l'inviluppo;
- ``edge_match``        : rende continua la giunzione fra due griglie affiancate
  pareggiando la media nella striscia di sovrapposizione.

Puro numpy, nessuna dipendenza da Qt/QGIS/scipy/GDAL. Una "griglia" qui è
``{"array": 2D float (NaN=nodata), "x0": float, "y0": float, "cell": float}``
dove ``x0/y0`` è la coordinata del centro del pixel ``[0, 0]`` (riga 0 = Y più
piccolo; ``array`` è ``(n_y, n_x)``).
"""

from __future__ import annotations

import numpy as np

__all__ = ["rotate_translate", "append_grids", "edge_match"]


def _as_grid(g) -> tuple[np.ndarray, float, float, float]:
    if isinstance(g, dict):
        a = np.asarray(g["array"], dtype=np.float64)
        return a, float(g.get("x0", 0.0)), float(g.get("y0", 0.0)), float(g.get("cell", 1.0))
    a = np.asarray(g, dtype=np.float64)
    return a, 0.0, 0.0, 1.0


def _bilinear(src: np.ndarray, fx: np.ndarray, fy: np.ndarray) -> np.ndarray:
    """Campionamento bilineare NaN-aware a indici frazionari (fx=colonna, fy=riga)."""
    n_y, n_x = src.shape
    x0 = np.floor(fx).astype(np.int64)
    y0 = np.floor(fy).astype(np.int64)
    wx = fx - x0
    wy = fy - y0
    out = np.full(fx.shape, np.nan, dtype=np.float64)

    def _g(iy, ix):
        m = (ix >= 0) & (ix < n_x) & (iy >= 0) & (iy < n_y)
        v = np.full(fx.shape, np.nan)
        if np.any(m):
            v[m] = src[np.clip(iy[m], 0, n_y - 1), np.clip(ix[m], 0, n_x - 1)]
        return v

    v00 = _g(y0, x0)
    v01 = _g(y0, x0 + 1)
    v10 = _g(y0 + 1, x0)
    v11 = _g(y0 + 1, x0 + 1)
    w00 = (1 - wx) * (1 - wy)
    w01 = wx * (1 - wy)
    w10 = (1 - wx) * wy
    w11 = wx * wy
    stack_v = np.stack([v00, v01, v10, v11])
    stack_w = np.stack([w00, w01, w10, w11])
    ok = np.isfinite(stack_v)
    stack_v = np.where(ok, stack_v, 0.0)
    stack_w = np.where(ok, stack_w, 0.0)
    wsum = stack_w.sum(axis=0)
    res = np.divide((stack_v * stack_w).sum(axis=0), wsum, out=out, where=wsum > 1e-9)
    return res


def rotate_translate(
    grid,
    angle_deg: float,
    out_origin: tuple[float, float] = (0.0, 0.0),
    mode: str = "fix_origin",
    cell_out: float | None = None,
):
    """Ruota di ``angle_deg`` (CCW) e trasla una griglia locale.

    ``mode='fix_origin'``      : l'angolo SW della griglia sorgente
        (pixel [0,0]) viene portato esattamente su ``out_origin`` dopo la
        rotazione — utile per georeferenziare un grid locale di cui si conosce
        solo l'angolo SW in UTM.
    ``mode='translate_origin'``: l'angolo SW del nuovo raster (minimo
        dell'inviluppo ruotato) viene portato su ``out_origin`` — utile per
        riferire più grid a un'origine comune così da poterli appendere.

    Ritorna ``{"array", "x0", "y0", "cell"}``.
    """
    src, sx0, sy0, cell = _as_grid(grid)
    if src.ndim != 2 or src.size == 0:
        return {"array": src.copy(), "x0": sx0, "y0": sy0, "cell": cell}
    c = float(cell_out) if cell_out and cell_out > 0 else float(cell)
    n_y, n_x = src.shape
    th = np.radians(float(angle_deg))
    ct, st = np.cos(th), np.sin(th)

    # pivot = centro pixel [0,0] della sorgente
    px, py = sx0, sy0
    # coordinate mondo dei 4 vertici sorgente
    cx = np.array([sx0, sx0 + (n_x - 1) * cell, sx0, sx0 + (n_x - 1) * cell])
    cy = np.array([sy0, sy0, sy0 + (n_y - 1) * cell, sy0 + (n_y - 1) * cell])
    rx = px + (cx - px) * ct - (cy - py) * st
    ry = py + (cx - px) * st + (cy - py) * ct

    if str(mode) == "fix_origin":
        tx = float(out_origin[0]) - rx[0]
        ty = float(out_origin[1]) - ry[0]
    else:  # translate_origin
        tx = float(out_origin[0]) - rx.min()
        ty = float(out_origin[1]) - ry.min()

    rx += tx
    ry += ty
    new_x0 = float(rx.min())
    new_y0 = float(ry.min())
    out_nx = int(np.ceil((rx.max() - new_x0) / c)) + 1
    out_ny = int(np.ceil((ry.max() - new_y0) / c)) + 1
    out_nx = max(1, out_nx)
    out_ny = max(1, out_ny)

    ox = new_x0 + np.arange(out_nx) * c
    oy = new_y0 + np.arange(out_ny) * c
    OX, OY = np.meshgrid(ox, oy)                       # (out_ny, out_nx)

    # inversa: togli traslazione, ruota di -th attorno al pivot, → indici src
    ux = OX - tx - px
    uy = OY - ty - py
    sxw = px + ux * ct + uy * st
    syw = py - ux * st + uy * ct
    fx = (sxw - sx0) / cell
    fy = (syw - sy0) / cell
    out = _bilinear(src, fx, fy)
    return {"array": out.astype(np.float32), "x0": new_x0, "y0": new_y0, "cell": c}


def append_grids(grids, overlap: str = "mean"):
    """Fonde una lista di griglie (stessa ``cell``) in un unico raster che ne
    contiene l'inviluppo. ``overlap`` = ``'mean'`` | ``'first'`` | ``'last'``
    per le celle coperte da più griglie.

    Ritorna ``{"array", "x0", "y0", "cell"}``.
    """
    items = [_as_grid(g) for g in (grids or [])]
    items = [it for it in items if it[0].ndim == 2 and it[0].size > 0]
    if not items:
        return {"array": np.zeros((0, 0), dtype=np.float32), "x0": 0.0, "y0": 0.0, "cell": 1.0}
    cell = items[0][3]
    for _, _, _, c in items:
        if abs(c - cell) > 1e-9:
            raise ValueError("append_grids: tutte le griglie devono avere la stessa cell size")

    xmins = [x0 for _, x0, _, _ in items]
    ymins = [y0 for _, _, y0, _ in items]
    xmaxs = [x0 + (a.shape[1] - 1) * cell for a, x0, _, _ in items]
    ymaxs = [y0 + (a.shape[0] - 1) * cell for a, _, y0, _ in items]
    X0, Y0 = min(xmins), min(ymins)
    n_x = int(round((max(xmaxs) - X0) / cell)) + 1
    n_y = int(round((max(ymaxs) - Y0) / cell)) + 1

    acc = np.zeros((n_y, n_x), dtype=np.float64)
    cnt = np.zeros((n_y, n_x), dtype=np.float64)
    out = np.full((n_y, n_x), np.nan, dtype=np.float64)
    for a, x0, y0, _ in items:
        j0 = int(round((x0 - X0) / cell))
        i0 = int(round((y0 - Y0) / cell))
        sub_out = out[i0:i0 + a.shape[0], j0:j0 + a.shape[1]]
        m = np.isfinite(a)
        if overlap == "first":
            take = m & ~np.isfinite(sub_out)
            sub_out[take] = a[take]
        elif overlap == "last":
            sub_out[m] = a[m]
        else:  # mean
            acc[i0:i0 + a.shape[0], j0:j0 + a.shape[1]][m] += a[m]
            cnt[i0:i0 + a.shape[0], j0:j0 + a.shape[1]][m] += 1.0
    if overlap == "mean":
        out = np.divide(acc, cnt, out=np.full_like(acc, np.nan), where=cnt > 0.0)
    return {"array": out.astype(np.float32), "x0": float(X0), "y0": float(Y0), "cell": float(cell)}


def edge_match(grid_a, grid_b, axis: str = "x", overlap_cells: int = 3):
    """Pareggia la media di ``grid_b`` a quella di ``grid_a`` nella striscia di
    sovrapposizione (``overlap_cells`` celle sul bordo condiviso), così la
    giunzione non mostra un salto. ``axis='x'`` = griglie affiancate lungo X
    (bordo verticale), ``axis='y'`` = impilate lungo Y.

    Ritorna una copia di ``grid_b`` corretta (stesso dict di ingresso).
    """
    a, ax0, ay0, ac = _as_grid(grid_a)
    b, bx0, by0, bc = _as_grid(grid_b)
    k = max(1, int(overlap_cells))
    if str(axis).lower().startswith("y"):
        strip_a = a[:k, :]
        strip_b = b[-k:, :]
    else:
        strip_a = a[:, :k]
        strip_b = b[:, -k:]
    ma = np.nanmean(strip_a) if np.isfinite(strip_a).any() else 0.0
    mb = np.nanmean(strip_b) if np.isfinite(strip_b).any() else 0.0
    shift = float(ma - mb)
    out = b + shift
    return {"array": out.astype(np.float32), "x0": bx0, "y0": by0, "cell": bc, "shift": shift}
