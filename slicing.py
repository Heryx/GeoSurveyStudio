# -*- coding: utf-8 -*-
"""Shared time-slice numeric core: amplitude extraction, depth windowing and
IDW gridding helpers.

Extracted verbatim from ``gpr_ogpr_slicer.py`` (Fase A of the time-slice
redesign) so every slice generator shares one implementation. Behaviour is
unchanged; only the module location moved.

SciPy is imported lazily inside the functions that need it.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Amplitude extraction / depth windowing
# ---------------------------------------------------------------------------


def _envelope(data: np.ndarray) -> np.ndarray:
    """Calculate Hilbert envelope along axis 0 (samples/depth).

    Falls back to absolute value if SciPy is not available.
    """
    try:
        from scipy.signal import hilbert  # type: ignore
        analytic = hilbert(data.astype(np.float64), axis=0)
        return np.abs(analytic).astype(np.float32)
    except Exception:
        return np.abs(data).astype(np.float32)


def _depth_to_sample_range(
    z_from: float,
    z_to: float,
    depth_max_m: float,
    n_samples: int,
) -> tuple[int, int]:
    if n_samples <= 0:
        return 0, 0
    if depth_max_m <= 0:
        return 0, n_samples

    z0 = float(np.clip(min(z_from, z_to), 0.0, depth_max_m))
    z1 = float(np.clip(max(z_from, z_to), 0.0, depth_max_m))
    scale = float(n_samples) / max(float(depth_max_m), 1e-12)
    s_lo = int(np.floor(z0 * scale))
    s_hi = int(np.ceil(z1 * scale))
    s_lo = int(np.clip(s_lo, 0, n_samples))
    s_hi = int(np.clip(s_hi, 0, n_samples))
    # Ensure at least one sample when a positive depth window collapses by rounding.
    if z1 > z0 and s_hi <= s_lo:
        s_hi = min(n_samples, s_lo + 1)
    return s_lo, s_hi


# ---------------------------------------------------------------------------
# Radius / grid post-processing
# ---------------------------------------------------------------------------


def _depth_adaptive_radius(
    base_radius: float,
    z_center: float,
    z_max: float,
    depth_factor: float = 0.6,
) -> float:
    radius = float(base_radius)
    if not np.isfinite(radius) or radius <= 0.0:
        return 1e-6
    try:
        df = float(depth_factor)
    except Exception:
        df = 0.0
    if not np.isfinite(df) or df <= 0.0:
        return radius
    try:
        zmax = float(z_max)
    except Exception:
        zmax = 0.0
    if not np.isfinite(zmax) or zmax <= 0.0:
        return radius
    try:
        zc = float(z_center)
    except Exception:
        zc = 0.0
    if not np.isfinite(zc):
        zc = 0.0
    zc = min(max(zc, 0.0), zmax)
    scale = 1.0 + df * (zc / max(zmax, 1e-9))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return radius * scale


def _fill_nodata_grid(grid: np.ndarray, max_distance: int = 5) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt
    if grid is None or grid.ndim != 2:
        return grid
    try:
        max_d = int(max_distance)
    except Exception:
        max_d = 0
    if max_d <= 0:
        return grid
    nan_mask = np.isnan(grid)
    if not nan_mask.any():
        return grid
    # Safety guard: EDT can allocate large temporary buffers on huge grids.
    MAX_SAFE_CELLS = 5_000_000
    if int(nan_mask.size) > MAX_SAFE_CELLS:
        print(
            f"[OGPR slicer][WARN] _fill_nodata_grid saltato: griglia {nan_mask.shape[0]}x{nan_mask.shape[1]} "
            f"({nan_mask.size:,} celle) supera il limite di sicurezza ({MAX_SAFE_CELLS:,}). "
            "I buchi non verranno colmati."
        )
        return grid
    mask_u8 = nan_mask.astype(np.uint8, copy=False)
    try:
        dist, (row_idx, col_idx) = distance_transform_edt(
            mask_u8, return_distances=True, return_indices=True,
        )
    except MemoryError:
        print("[OGPR slicer][WARN] _fill_nodata_grid: memoria insufficiente per distance_transform_edt. Fallback senza fill.")
        return grid
    except Exception as exc:
        print(f"[OGPR slicer][WARN] _fill_nodata_grid: errore imprevisto ({exc}). Fallback senza fill.")
        return grid
    filled = grid.copy()
    fw = nan_mask & np.isfinite(dist) & (dist <= float(max_d))
    filled[fw] = grid[row_idx[fw], col_idx[fw]]
    return filled


def _smooth_grid_gaussian(grid: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    if sigma <= 0:
        return grid
    nan_mask = np.isnan(grid)
    filled = np.where(nan_mask, 0.0, grid.astype(np.float64))
    weight = np.where(nan_mask, 0.0, 1.0)
    smooth_num = gaussian_filter(filled, sigma=sigma)
    smooth_den = gaussian_filter(weight, sigma=sigma)
    result = np.full_like(grid, np.nan, dtype=np.float32)
    valid = smooth_den > 1e-9
    result[valid] = (smooth_num[valid] / smooth_den[valid]).astype(np.float32)
    return result


def _blanking_mask_from_distance(
    e_pts: np.ndarray,
    n_pts: np.ndarray,
    x_min: float,
    y_min: float,
    n_x: int,
    n_y: int,
    resolution: float,
    blanking_distance: float,
) -> np.ndarray | None:
    """Return mask for cells farther than blanking_distance from real samples."""
    try:
        dmax = float(blanking_distance)
    except Exception:
        dmax = 0.0
    if (not np.isfinite(dmax)) or dmax <= 0.0:
        return None
    if len(e_pts) <= 0 or len(n_pts) <= 0:
        return None
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return None
    try:
        gx = float(x_min) + np.arange(int(n_x), dtype=np.float64) * float(resolution)
        gy = float(y_min) + np.arange(int(n_y), dtype=np.float64) * float(resolution)
        gxx, gyy = np.meshgrid(gx, gy)
        qpts = np.column_stack([gxx.ravel(), gyy.ravel()])
        tree = cKDTree(np.column_stack([e_pts.astype(np.float64), n_pts.astype(np.float64)]))
        dist, _ = tree.query(qpts, k=1, workers=-1)
        return np.asarray(dist, dtype=np.float64).reshape(int(n_y), int(n_x)) > dmax
    except Exception:
        return None


# ---------------------------------------------------------------------------
# IDW helpers
# ---------------------------------------------------------------------------


def _estimate_idw_knn_k(
    x_pts: np.ndarray,
    y_pts: np.ndarray,
    radius: float,
    min_points: int,
    max_k: int = 512,
) -> int:
    n_pts = int(len(x_pts))
    if n_pts <= 0:
        return 1
    r = float(radius)
    if not np.isfinite(r) or r <= 0.0:
        r = 1e-6

    try:
        area = float(np.ptp(x_pts)) * float(np.ptp(y_pts))
    except Exception:
        area = 0.0
    if not np.isfinite(area) or area <= 1e-12:
        area = max(float(n_pts), 1.0)

    density = float(n_pts) / area
    expected = np.pi * (r ** 2) * density
    try:
        k = int(np.ceil(expected * 1.8)) + 8
    except Exception:
        k = 32
    k = max(int(min_points), k, 8)
    k = min(k, int(max_k), n_pts)
    return max(1, int(k))


def _idw_chunk_size_from_k(k: int, target_elements: int = 500_000) -> int:
    kk = max(1, int(k))
    cs = int(target_elements // kk)
    return max(1_000, cs)


def _ckdtree_query_knn(tree, qpts: np.ndarray, k: int, distance_upper_bound: float):
    try:
        return tree.query(
            qpts,
            k=int(k),
            distance_upper_bound=float(distance_upper_bound),
            workers=-1,
        )
    except TypeError:
        return tree.query(
            qpts,
            k=int(k),
            distance_upper_bound=float(distance_upper_bound),
        )


def _normalize_idw_mode(mode: str | None) -> str:
    txt = str(mode or "fast").strip().lower()
    if txt in {"quality", "accurate", "radius", "ball"}:
        return "quality"
    return "fast"


def _idw_grid_guard(n_x: int, n_y: int, label: str, max_cells: int = 10_000_000) -> tuple[bool, int]:
    """Return (ok, n_cells) for safe IDW grid allocation."""
    try:
        nx = int(n_x)
        ny = int(n_y)
    except Exception:
        return False, 0
    if nx <= 0 or ny <= 0:
        return False, 0
    n_cells = int(nx) * int(ny)
    if n_cells > int(max_cells):
        print(
            f"[OGPR slicer][ERROR] {label} abortito: griglia {ny}x{nx}={n_cells:,} celle "
            f"supera il limite di sicurezza ({int(max_cells):,}). "
            "Aumenta la risoluzione o riduci il raggio IDW."
        )
        return False, n_cells
    return True, n_cells


def _bin_with_idw_ball(
    x_pts, y_pts, i_pts, x_min, y_min, n_x, n_y, resolution, radius, power=2.0, min_points=1,
    tree=None,
):
    """IDW quality-mode (ball query) — fully vectorised via ragged-list padding."""
    from scipy.spatial import cKDTree

    ok_grid, _ = _idw_grid_guard(n_x, n_y, "IDW quality")
    if not ok_grid:
        return np.full((n_y, n_x), np.nan, dtype=np.float32)
    gx = x_min + np.arange(n_x) * resolution
    gy = y_min + np.arange(n_y) * resolution
    gxx, gyy = np.meshgrid(gx, gy)
    gpts = np.column_stack([gxx.ravel(), gyy.ravel()])
    x_arr = np.asarray(x_pts, dtype=np.float64)
    y_arr = np.asarray(y_pts, dtype=np.float64)
    i64 = np.asarray(i_pts, dtype=np.float64)
    n_pts = int(x_arr.size)
    if tree is None:
        tree = cKDTree(np.column_stack([x_arr, y_arr]))
    try:
        results = tree.query_ball_point(gpts, r=radius, workers=-1)
    except TypeError:
        results = tree.query_ball_point(gpts, r=radius)
    pwr = float(power) if np.isfinite(power) and float(power) > 0 else 2.0
    min_pts = max(1, int(min_points))

    # Vectorised: pad ragged result lists to a fixed-width array.
    n_cells = len(results)
    max_k = max((len(r) for r in results), default=1)
    max_k = max(max_k, 1)
    idx_pad = np.full((n_cells, max_k), -1, dtype=np.int32)
    dist_pad = np.full((n_cells, max_k), np.inf, dtype=np.float64)
    for k, lst in enumerate(results):
        n = len(lst)
        if n == 0:
            continue
        idxs = np.asarray(lst, dtype=np.int32)
        idx_pad[k, :n] = idxs
        cx, cy = gpts[k]
        dx = x_arr[idxs] - cx
        dy = y_arr[idxs] - cy
        dist_pad[k, :n] = np.sqrt(dx * dx + dy * dy)
    valid = (idx_pad >= 0) & np.isfinite(dist_pad)
    cnt = valid.sum(axis=1)
    w = np.where(valid, 1.0 / (dist_pad ** pwr + 1e-9), 0.0)
    ws = w.sum(axis=1)
    idx_safe = np.clip(idx_pad, 0, n_pts - 1)
    vals = i64[idx_safe]
    ok = (cnt >= min_pts) & (ws > 1e-12)
    num = np.sum(w * np.where(valid, vals, 0.0), axis=1)
    grid = np.full(n_cells, np.nan, dtype=np.float32)
    grid[ok] = (num[ok] / ws[ok]).astype(np.float32)
    return grid.reshape(n_y, n_x)


def _bin_with_idw_anisotropic_ball(
    x_pts, y_pts, i_pts, x_min, y_min, n_x, n_y, resolution, radius, power=2.0, min_points=1,
    anisotropy_ratio=1.0, anisotropy_angle=0.0, max_points_per_cell=4096,
    tree=None,
):
    """IDW anisotropic quality-mode (ball query) — fully vectorised via ragged-list padding."""
    from scipy.spatial import cKDTree

    ok_grid, _ = _idw_grid_guard(n_x, n_y, "IDW quality anisotropo")
    if not ok_grid:
        return np.full((n_y, n_x), np.nan, dtype=np.float32)
    if anisotropy_ratio <= 0:
        anisotropy_ratio = 1.0
    ar = np.radians(anisotropy_angle)
    ca, sa = np.cos(ar), np.sin(ar)

    gx = x_min + np.arange(n_x) * resolution
    gy = y_min + np.arange(n_y) * resolution
    gxx, gyy = np.meshgrid(gx, gy)
    gpts = np.column_stack([gxx.ravel(), gyy.ravel()])
    x_arr = np.asarray(x_pts, dtype=np.float64)
    y_arr = np.asarray(y_pts, dtype=np.float64)
    i64 = np.asarray(i_pts, dtype=np.float64)
    n_pts = int(x_arr.size)
    if tree is None:
        tree = cKDTree(np.column_stack([x_arr, y_arr]))
    r_search = float(radius) * max(1.0, float(anisotropy_ratio))
    try:
        results = tree.query_ball_point(gpts, r=r_search, workers=-1)
    except TypeError:
        results = tree.query_ball_point(gpts, r=r_search)

    pwr = float(power) if np.isfinite(power) and float(power) > 0 else 2.0
    min_pts = max(1, int(min_points))
    cap_pts = int(max_points_per_cell or 0)

    # Vectorised: pad ragged result lists to a fixed-width array.
    n_cells = len(results)
    max_k = max((len(r) for r in results), default=1)
    max_k = max(max_k, 1)
    if cap_pts > 0:
        max_k = min(max_k, cap_pts)
    idx_pad = np.full((n_cells, max_k), -1, dtype=np.int32)
    for k, lst in enumerate(results):
        n = len(lst)
        if n == 0:
            continue
        arr = np.asarray(lst, dtype=np.int32)
        if cap_pts and n > cap_pts:
            # Keep the closest cap_pts points (cap by Euclidean pre-distance).
            cx, cy = gpts[k]
            d2 = (x_arr[arr] - cx) ** 2 + (y_arr[arr] - cy) ** 2
            arr = arr[np.argpartition(d2, min(cap_pts - 1, len(d2) - 1))[:cap_pts]]
            n = len(arr)
        idx_pad[k, :n] = arr

    # Compute anisotropic distances in bulk.
    valid_mask = idx_pad >= 0
    idx_safe = np.clip(idx_pad, 0, n_pts - 1)
    cx_col = gpts[:, 0:1]  # (n_cells, 1)
    cy_col = gpts[:, 1:2]
    dx = x_arr[idx_safe] - cx_col  # (n_cells, max_k)
    dy = y_arr[idx_safe] - cy_col
    # Anisotropic distance: stretch along principal axis, compress across it.
    d_aniso = np.sqrt((dx * ca + dy * sa) ** 2 + ((-dx * sa + dy * ca) / anisotropy_ratio) ** 2)
    in_radius = valid_mask & (d_aniso <= radius)
    cnt = in_radius.sum(axis=1)
    w = np.where(in_radius, 1.0 / (d_aniso ** pwr + 1e-9), 0.0)
    ws = w.sum(axis=1)
    vals = i64[idx_safe]
    ok = (cnt >= min_pts) & (ws > 1e-12)
    num = np.sum(w * np.where(in_radius, vals, 0.0), axis=1)
    grid = np.full(n_cells, np.nan, dtype=np.float32)
    grid[ok] = (num[ok] / ws[ok]).astype(np.float32)
    return grid.reshape(n_y, n_x)


def _bin_with_idw(
    x_pts, y_pts, i_pts, x_min, y_min, n_x, n_y, resolution, radius, power=2.0, min_points=1,
    tree=None,
):
    from scipy.spatial import cKDTree
    ok_grid, n_cells = _idw_grid_guard(n_x, n_y, "IDW")
    if not ok_grid:
        return np.full((n_y, n_x), np.nan, dtype=np.float32)
    gx = x_min + np.arange(n_x) * resolution
    gy = y_min + np.arange(n_y) * resolution
    gxx, gyy = np.meshgrid(gx, gy)
    gpts = np.column_stack([gxx.ravel(), gyy.ravel()])
    n_pts = int(np.asarray(x_pts).size)
    if n_cells <= 0 or n_pts <= 0:
        return np.full((n_y, n_x), np.nan, dtype=np.float32)

    r = float(radius)
    if not np.isfinite(r) or r <= 0.0:
        r = 1e-6
    pwr = float(power) if np.isfinite(power) and float(power) > 0.0 else 2.0
    min_pts = max(1, int(min_points))

    x_arr = np.asarray(x_pts, dtype=np.float64)
    y_arr = np.asarray(y_pts, dtype=np.float64)
    i64 = np.asarray(i_pts, dtype=np.float64)

    if tree is None:
        tree = cKDTree(np.column_stack([x_arr, y_arr]))
    k = _estimate_idw_knn_k(x_arr, y_arr, r, min_pts, max_k=32)
    chunk_size = _idw_chunk_size_from_k(k)

    grid = np.full(n_cells, np.nan, dtype=np.float32)
    for start in range(0, n_cells, chunk_size):
        end = min(n_cells, start + chunk_size)
        q = gpts[start:end]
        dists, idxs = _ckdtree_query_knn(tree, q, k=k, distance_upper_bound=r)
        dists = np.asarray(dists, dtype=np.float64)
        idxs = np.asarray(idxs, dtype=np.int64)
        if dists.ndim == 1:
            dists = dists[:, np.newaxis]
            idxs = idxs[:, np.newaxis]

        valid = np.isfinite(dists) & (idxs >= 0) & (idxs < n_pts)
        if not np.any(valid):
            continue

        idx_clip = np.clip(idxs, 0, max(0, n_pts - 1))
        vals = i64[idx_clip]
        d_safe = np.where(valid, dists, 1.0)
        w = np.where(valid, 1.0 / (np.power(d_safe, pwr) + 1e-9), 0.0)
        ws = np.sum(w, axis=1)
        cnt = np.sum(valid, axis=1)
        ok = (cnt >= min_pts) & (ws > 1e-12)
        if not np.any(ok):
            continue
        num = np.sum(w * vals, axis=1)
        out = np.full(q.shape[0], np.nan, dtype=np.float32)
        out[ok] = (num[ok] / ws[ok]).astype(np.float32)
        grid[start:end] = out

    return grid.reshape(n_y, n_x)


def _bin_with_idw_anisotropic(
    x_pts, y_pts, i_pts, x_min, y_min, n_x, n_y, resolution, radius, power=2.0, min_points=1,
    anisotropy_ratio=1.0, anisotropy_angle=0.0, max_points_per_cell=4096,
    tree=None,
):
    from scipy.spatial import cKDTree
    ok_grid, n_cells = _idw_grid_guard(n_x, n_y, "IDW anisotropo")
    if not ok_grid:
        return np.full((n_y, n_x), np.nan, dtype=np.float32)
    if anisotropy_ratio <= 0:
        anisotropy_ratio = 1.0
    ar = np.radians(anisotropy_angle)
    ca, sa = np.cos(ar), np.sin(ar)

    def _ad(px, py, cx, cy):
        dx, dy = px - cx, py - cy
        return np.sqrt((dx * ca + dy * sa) ** 2 + ((-dx * sa + dy * ca) / anisotropy_ratio) ** 2)

    gx = x_min + np.arange(n_x) * resolution
    gy = y_min + np.arange(n_y) * resolution
    gxx, gyy = np.meshgrid(gx, gy)
    gpts = np.column_stack([gxx.ravel(), gyy.ravel()])
    n_pts = int(np.asarray(x_pts).size)
    if n_cells <= 0 or n_pts <= 0:
        return np.full((n_y, n_x), np.nan, dtype=np.float32)

    r = float(radius)
    if not np.isfinite(r) or r <= 0.0:
        r = 1e-6
    r_search = r * max(1.0, float(anisotropy_ratio))
    pwr = float(power) if np.isfinite(power) and float(power) > 0.0 else 2.0
    min_pts = max(1, int(min_points))
    max_k = int(max_points_per_cell) if max_points_per_cell else 32
    max_k = max(min_pts, min(32, max(8, max_k)))

    x_arr = np.asarray(x_pts, dtype=np.float64)
    y_arr = np.asarray(y_pts, dtype=np.float64)
    i64 = np.asarray(i_pts, dtype=np.float64)

    if tree is None:
        tree = cKDTree(np.column_stack([x_arr, y_arr]))
    k = _estimate_idw_knn_k(x_arr, y_arr, r_search, min_pts, max_k=max_k)
    chunk_size = _idw_chunk_size_from_k(k)

    grid = np.full(n_cells, np.nan, dtype=np.float32)
    for start in range(0, n_cells, chunk_size):
        end = min(n_cells, start + chunk_size)
        q = gpts[start:end]
        dists, idxs = _ckdtree_query_knn(tree, q, k=k, distance_upper_bound=r_search)
        dists = np.asarray(dists, dtype=np.float64)
        idxs = np.asarray(idxs, dtype=np.int64)
        if dists.ndim == 1:
            dists = dists[:, np.newaxis]
            idxs = idxs[:, np.newaxis]

        valid0 = np.isfinite(dists) & (idxs >= 0) & (idxs < n_pts)
        if not np.any(valid0):
            continue

        idx_clip = np.clip(idxs, 0, max(0, n_pts - 1))
        px = x_arr[idx_clip]
        py = y_arr[idx_clip]
        cx = q[:, 0][:, np.newaxis]
        cy = q[:, 1][:, np.newaxis]

        d_aniso = _ad(px, py, cx, cy)
        valid = valid0 & np.isfinite(d_aniso) & (d_aniso <= r)
        if not np.any(valid):
            continue

        vals = i64[idx_clip]
        d_safe = np.where(valid, d_aniso, 1.0)
        w = np.where(valid, 1.0 / (np.power(d_safe, pwr) + 1e-9), 0.0)
        ws = np.sum(w, axis=1)
        cnt = np.sum(valid, axis=1)
        ok = (cnt >= min_pts) & (ws > 1e-12)
        if not np.any(ok):
            continue
        num = np.sum(w * vals, axis=1)
        out = np.full(q.shape[0], np.nan, dtype=np.float32)
        out[ok] = (num[ok] / ws[ok]).astype(np.float32)
        grid[start:end] = out

    return grid.reshape(n_y, n_x)
