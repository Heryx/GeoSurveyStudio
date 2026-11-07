# -*- coding: utf-8 -*-
"""Geometria delle "tende" verticali per la vista 3D dei profili (senza Qt ne' pyvista).

Un radargramma o una sezione ERT diventa una striscia verticale lungo la sua linea sulla mappa:
due file di vertici (in alto a z=0, in basso a z=-profondita'), una texture RGBA con i colori
e le coordinate di texture ``u`` = distanza lungo la linea, ``v`` = profondita'. La striscia ha
vertici solo dove la linea cambia direzione (non uno per traccia): la texture porta il dettaglio.

Le coordinate si traslano di ``origin`` (est, nord) perche' in float32 la GPU non regge
numeri come 4049823.4 senza tremolii.
"""

from __future__ import annotations

import numpy as np

MAX_TEXTURE_COLS = 2048
MAX_TEXTURE_ROWS = 768


def _uniform_indices(n: int, max_pts: int) -> np.ndarray:
    if n <= max_pts:
        return np.arange(n)
    idx = np.unique(np.round(np.linspace(0, n - 1, max_pts)).astype(int))
    return idx


def track_vertices(east, north, dist, max_pts: int = 300):
    """Vertici ``(xy (k,2), d (k,))`` di un profilo GPR: tracce con coordinate valide, decimate.

    ``d`` e' la distanza lungo il profilo (la stessa dell'asse x del radargramma), crescente.
    """
    east = np.asarray(east, dtype=np.float64)
    north = np.asarray(north, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64)
    ok = np.isfinite(east) & np.isfinite(north) & np.isfinite(dist)
    if ok.sum() < 2:
        raise ValueError("il profilo ha meno di 2 tracce con coordinate valide")
    e, n, d = east[ok], north[ok], dist[ok]
    order = np.argsort(d, kind="stable")
    e, n, d = e[order], n[order], d[order]
    keep = np.concatenate([[True], np.diff(d) > 1e-9])           # distanze strettamente crescenti
    e, n, d = e[keep], n[keep], d[keep]
    idx = _uniform_indices(len(d), max_pts)
    return np.column_stack([e[idx], n[idx]]), d[idx]


def polyline_vertices(georef, d0: float, d1: float, step_m: float = 0.5, max_pts: int = 400):
    """Vertici ``(xy, d)`` di una linea georiferita (ERT) tra due distanze.

    Include sempre i punti di controllo compresi nell'intervallo, cosi' la linea spezzata non si smussa.
    """
    d0, d1 = float(d0), float(d1)
    n = max(int(np.ceil((d1 - d0) / max(step_m, 1e-6))), 1) + 1
    d = np.linspace(d0, d1, min(n, max_pts))
    ctrl = np.array([p.distance for p in georef.points if d0 < p.distance < d1], dtype=np.float64)
    d = np.unique(np.concatenate([d, ctrl]))
    x, y = georef.locate(d)
    return np.column_stack([x, y]), d


def curtain(xy, d, tex_span, depth_top: float, depth_bottom: float, origin=(0.0, 0.0)) -> dict:
    """Striscia verticale lungo ``xy``: ``points (2k,3)``, ``faces`` (formato VTK) e ``tcoords (2k,2)``.

    ``tex_span`` = (d_inizio, d_fine) coperti dalla texture: ``u`` vale 0 e 1 ai due estremi.
    Riga alta: z = -depth_top; riga bassa: z = -depth_bottom (profondita' positive verso il basso).
    ``v`` = 1 in alto e 0 in basso (convenzione delle texture VTK).
    """
    xy = np.asarray(xy, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    k = len(d)
    if k < 2 or xy.shape != (k, 2):
        raise ValueError("servono almeno 2 vertici")
    t0, t1 = float(tex_span[0]), float(tex_span[1])
    if t1 - t0 <= 0:
        raise ValueError("intervallo di texture vuoto")
    x = xy[:, 0] - origin[0]
    y = xy[:, 1] - origin[1]
    top = np.column_stack([x, y, np.full(k, -float(depth_top))])
    bot = np.column_stack([x, y, np.full(k, -float(depth_bottom))])
    u = np.clip((d - t0) / (t1 - t0), 0.0, 1.0)
    tc = np.vstack([np.column_stack([u, np.ones(k)]), np.column_stack([u, np.zeros(k)])])
    i = np.arange(k - 1)
    faces = np.column_stack([np.full(k - 1, 4), i, i + 1, k + i + 1, k + i]).ravel()
    return {"points": np.vstack([top, bot]), "faces": faces.astype(np.int64), "tcoords": tc}


def plane(corners_xy, depth: float, origin=(0.0, 0.0)) -> dict:
    """Quadrilatero orizzontale a profondita' ``depth``: ``points (4,3)``, ``faces``, ``tcoords (4,2)``.

    ``corners_xy`` = angoli in ordine: alto-sinistra, alto-destra, basso-destra, basso-sinistra
    (alto = nord nella texture, riga 0 in alto). Gli angoli possono non formare un rettangolo
    (raster in un CRS diverso da quello della scena).
    """
    c = np.asarray(corners_xy, dtype=np.float64)
    if c.shape != (4, 2):
        raise ValueError("servono 4 angoli")
    pts = np.column_stack([c[:, 0] - origin[0], c[:, 1] - origin[1], np.full(4, -float(depth))])
    tc = np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    return {"points": pts, "faces": np.array([4, 0, 1, 2, 3], dtype=np.int64), "tcoords": tc}


def downsample(data, max_rows: int = MAX_TEXTURE_ROWS, max_cols: int = MAX_TEXTURE_COLS):
    """Riduce a blocchi (media ignorando i NaN) fino a stare nei limiti della texture."""
    a = np.asarray(data, dtype=np.float32)
    fr = int(np.ceil(a.shape[0] / float(max_rows)))
    fc = int(np.ceil(a.shape[1] / float(max_cols)))
    if fr == 1 and fc == 1:
        return a
    h, w = (a.shape[0] // fr) * fr, (a.shape[1] // fc) * fc
    blocks = a[:h, :w].reshape(h // fr, fr, w // fc, fc)
    with np.errstate(all="ignore"):
        valid = np.isfinite(blocks)
        s = np.where(valid, blocks, 0.0).sum(axis=(1, 3))
        c = valid.sum(axis=(1, 3))
        out = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    return out.astype(np.float32)


def rgba_from_array(data, cmap_name, vmin: float, vmax: float):
    """Texture ``uint8 (h, w, 4)`` con la riga 0 in alto; i NaN sono trasparenti.

    ``cmap_name`` e' il nome di una palette matplotlib o una ``Colormap`` gia' costruita.
    """
    from matplotlib import colormaps

    a = np.asarray(data, dtype=np.float64)
    span = (vmax - vmin) if vmax > vmin else 1.0
    norm = np.clip((a - vmin) / span, 0.0, 1.0)
    cmap = colormaps[cmap_name] if isinstance(cmap_name, str) else cmap_name
    rgba = cmap(np.nan_to_num(norm, nan=0.0))
    rgba[..., 3] = np.where(np.isfinite(a), 1.0, 0.0)
    return (rgba * 255.0 + 0.5).astype(np.uint8)


def crossings(xy_a, d_a, xy_b, d_b) -> list:
    """Punti in cui due polilinee si incrociano: ``[(distanza_a, distanza_b, x, y), ...]``.

    Le distanze sono interpolate tra i vertici; incroci quasi coincidenti (stesso punto visto da due
    segmenti adiacenti) si fondono.
    """
    xa, da = np.asarray(xy_a, dtype=np.float64), np.asarray(d_a, dtype=np.float64)
    xb, db = np.asarray(xy_b, dtype=np.float64), np.asarray(d_b, dtype=np.float64)
    p, r = xa[:-1], xa[1:] - xa[:-1]                          # segmenti A: p + t r
    q, s = xb[:-1], xb[1:] - xb[:-1]                          # segmenti B: q + u s
    rxs = r[:, None, 0] * s[None, :, 1] - r[:, None, 1] * s[None, :, 0]
    qp = q[None, :, :] - p[:, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (qp[..., 0] * s[None, :, 1] - qp[..., 1] * s[None, :, 0]) / rxs
        u = (qp[..., 0] * r[:, None, 1] - qp[..., 1] * r[:, None, 0]) / rxs
    hit = np.isfinite(t) & (np.abs(rxs) > 1e-12) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    out: list = []
    for i, j in zip(*np.nonzero(hit)):
        ti, uj = float(t[i, j]), float(u[i, j])
        pt = p[i] + ti * r[i]
        row = (float(da[i] + ti * (da[i + 1] - da[i])), float(db[j] + uj * (db[j + 1] - db[j])),
               float(pt[0]), float(pt[1]))
        if not any(abs(row[0] - o[0]) < 1e-6 and abs(row[1] - o[1]) < 1e-6 for o in out):
            out.append(row)
    return out
