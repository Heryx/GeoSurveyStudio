# -*- coding: utf-8 -*-
"""Shared time-slice numeric core: amplitude extraction, depth windowing and
gridding helpers.

The legacy helpers (``_envelope``, ``_depth_to_sample_range``, the ``_bin_with_idw*``
family, ...) were extracted verbatim from ``gpr_ogpr_slicer.py`` (Fase A of the
time-slice redesign) so every slice generator shares one implementation.

The "binned gridding" section (``window_amplitude``, ``window_sample_range``,
``grid_binned``, ...) is the numeric core of ``slice_engine`` - the default
engine that replaced IDW for array data. It is pure NumPy/SciPy and is covered
by ``test/test_slice_engine.py``.

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


# ---------------------------------------------------------------------------
# Binned gridding (replaces IDW as the default): splat + normalised convolution
# ---------------------------------------------------------------------------
#
# Why not IDW: with 1/d^p weights every grid cell that happens to sit next to a
# sample takes (almost) that single sample's value, while cells between samples
# are blurred averages. On a track-based survey this leaves a "string of beads"
# along every line (isolated one-cell dots), which is exactly what the array
# GPR time-slices showed. Here every sample is instead *splatted* onto the four
# surrounding nodes with bilinear weights (so density is uniform and no sample
# can dominate), the accumulated field is smoothed with a Gaussian normalised by
# the accumulated weights (normalised convolution), and holes between lines
# are filled from a wider Gaussian, blended in smoothly. No KD-tree, no Python
# loop over cells: a 90k-cell slice takes ~0.1 s.

AMPLITUDE_WINDOW_MODES = ("abs_mean", "rms", "env_peak", "env_mean", "signed_mean", "max_abs")
_HILBERT_PAD = 48  # samples of context around the window for the envelope


def depth_per_sample_m(dt_ns: float, velocity_m_s: float) -> float:
    """Two-way-travel depth increment of one sample: ``dt * v / 2`` (metres).

    Returns 0.0 when either input is not a positive finite number.
    """
    try:
        dt = float(dt_ns)
        v = float(velocity_m_s)
    except (TypeError, ValueError):
        return 0.0
    if not (np.isfinite(dt) and np.isfinite(v)) or dt <= 0.0 or v <= 0.0:
        return 0.0
    return dt * 1e-9 * v / 2.0


def window_sample_range(
    z_from: float, z_to: float, dz_sample: float, n_samples: int,
) -> tuple[int, int]:
    """Sample rows covering the depth window ``[z_from, z_to]``.

    Depth is ``row * dz_sample`` (row 0 = time zero). Unlike
    ``_depth_to_sample_range`` this does not assume the matrix spans the whole
    nominal depth range, so a matrix cropped by time-zero alignment or channel
    stacking still maps depth correctly. Returns ``(0, 0)`` for an empty window.
    """
    if n_samples <= 0 or not (dz_sample > 0.0):
        return 0, 0
    z0 = max(0.0, min(float(z_from), float(z_to)))
    z1 = max(0.0, max(float(z_from), float(z_to)))
    lo = int(np.floor(z0 / dz_sample + 1e-9))
    hi = int(np.ceil(z1 / dz_sample - 1e-9))
    if hi <= lo:
        hi = lo + 1
    if lo >= n_samples:
        return 0, 0
    return lo, min(hi, n_samples)


def window_amplitude(data: np.ndarray, s_lo: int, s_hi: int, mode: str) -> np.ndarray:
    """One amplitude value per trace over the sample rows ``[s_lo, s_hi)``.

    ``data`` is a ``(n_samples, n_traces)`` radargram. Modes:
    ``abs_mean`` mean(|a|), ``rms`` sqrt(mean(a^2)), ``max_abs`` max(|a|),
    ``signed_mean`` mean(a), ``env_peak`` max of the Hilbert envelope and
    ``env_mean`` its mean over the window (the envelope is computed on the window
    plus a few samples of context, so it is not distorted at the window edges and
    never touches the rest of the trace). ``env_mean`` is phase-independent like
    ``env_peak`` but averages instead of picking one sample, so it is far less
    sensitive to isolated spikes.
    """
    if mode not in AMPLITUDE_WINDOW_MODES:
        raise ValueError(f"amplitude mode {mode!r} not in {AMPLITUDE_WINDOW_MODES}")
    n_s = int(data.shape[0])
    s_lo = max(0, int(s_lo))
    s_hi = min(n_s, int(s_hi))
    if s_hi <= s_lo:
        return np.full(int(data.shape[1]), np.nan, dtype=np.float64)
    if mode in ("env_peak", "env_mean"):
        a = max(0, s_lo - _HILBERT_PAD)
        b = min(n_s, s_hi + _HILBERT_PAD)
        env = _envelope(data[a:b])[s_lo - a:s_hi - a]
        if mode == "env_mean":
            return env.mean(axis=0, dtype=np.float64)
        return env.max(axis=0).astype(np.float64)
    seg = data[s_lo:s_hi]
    if mode == "abs_mean":
        return np.abs(seg).mean(axis=0, dtype=np.float64)
    if mode == "rms":
        return np.sqrt(np.mean(np.square(seg, dtype=np.float64), axis=0))
    if mode == "max_abs":
        return np.abs(seg).max(axis=0).astype(np.float64)
    return seg.mean(axis=0, dtype=np.float64)  # signed_mean


# ---------------------------------------------------------------------------
# Amplitude tools: conditioning of the values before / after gridding
# ---------------------------------------------------------------------------
#
# Every function is pure NumPy/SciPy, NaN-aware and returns a new array; the
# ``slice_engine`` chains them according to the ``SliceSpec`` (so the live
# preview and the saved GeoTIFFs go through exactly the same steps).

AMPLITUDE_SCALES = ("linear", "sqrt", "db")
NORMALIZE_MODES = ("none", "median")
DESPECKLE_SIZES = (0, 3, 5)
_DB_FLOOR_REL = 1e-6      # dB floor: 1e-6 of the slice maximum (-120 dB), never log(0)
_MAX_SAFE_CELLS = 5_000_000


def clip_outliers(values, upper_pct: float, symmetric: bool = False) -> np.ndarray:
    """Cap the extreme values at the ``upper_pct`` percentile of ``values``.

    Tames isolated spikes (a single noisy trace) before gridding, so they do not
    turn into a bright blob after the smoothing. ``upper_pct >= 100`` is a no-op.
    With ``symmetric`` (zero-centred amplitudes such as ``signed_mean``) the low
    tail is capped at the ``100 - upper_pct`` percentile as well. NaN are ignored.
    """
    v = np.asarray(values, dtype=np.float64)
    p = float(upper_pct)
    if not (p < 100.0):
        return v
    fin = np.isfinite(v)
    if not fin.any():
        return v
    p = max(p, 50.0)
    hi = float(np.percentile(v[fin], p))
    lo = float(np.percentile(v[fin], 100.0 - p)) if symmetric else -np.inf
    return np.clip(v, lo, hi)


def scale_amplitude(grid, scale: str) -> np.ndarray:
    """Non-linear scale of a *positive* amplitude grid.

    ``sqrt`` compresses the dynamic range gently; ``db`` is ``20*log10`` (values
    are floored at 1e-6 of the slice maximum so zeros never give -inf). ``linear``
    (or anything else) returns the grid unchanged. NaN stay NaN.
    """
    g = np.asarray(grid, dtype=np.float64)
    if scale == "sqrt":
        return np.sqrt(np.maximum(g, 0.0))
    if scale == "db":
        fin = g[np.isfinite(g)]
        peak = float(fin.max()) if fin.size else 0.0
        floor = max(peak * _DB_FLOOR_REL, 1e-12)
        return 20.0 * np.log10(np.maximum(g, floor))
    return g


def normalize_grid(grid, mode: str) -> np.ndarray:
    """``median``: divide by the slice median so 1.0 is the typical background.

    Puts slices of different depths on one comparable scale (amplitude falls
    with depth) - the basis for a single fixed colour range across the whole set.
    A non-positive median leaves the grid untouched.
    """
    g = np.asarray(grid, dtype=np.float64)
    if mode != "median":
        return g
    fin = g[np.isfinite(g)]
    if fin.size == 0:
        return g
    med = float(np.median(fin))
    if not med > 1e-12:
        return g
    return g / med


def _nearest_fill(grid: np.ndarray):
    """``(filled, nan_mask)``: NaN cells take the value of the nearest finite cell,
    so an order filter does not see the survey outline as data."""
    nan = ~np.isfinite(grid)
    if not nan.any() or nan.all():
        return grid, nan
    if grid.size > _MAX_SAFE_CELLS:
        return np.where(nan, float(np.nanmedian(grid)), grid), nan
    from scipy.ndimage import distance_transform_edt

    idx = distance_transform_edt(nan, return_distances=False, return_indices=True)
    return grid[idx[0], idx[1]], nan


def median_despeckle(grid, size: int) -> np.ndarray:
    """``size`` x ``size`` (odd) median filter that respects the NaN outline.

    Removes isolated bright/dark cells without blurring edges the way a Gaussian
    does. ``size < 3`` is a no-op.
    """
    g = np.asarray(grid, dtype=np.float64)
    size = int(size)
    if size < 3 or not np.isfinite(g).any():
        return g
    from scipy.ndimage import median_filter

    filled, nan = _nearest_fill(g)
    out = median_filter(filled, size=size | 1, mode="nearest")
    out[nan] = np.nan
    return out


def highpass_background(grid, sigma_cells: float) -> np.ndarray:
    """``grid`` minus its NaN-aware Gaussian of ``sigma_cells``: drops the slow
    trend (geology, soil moisture) and leaves the small local anomalies."""
    g = np.asarray(grid, dtype=np.float64)
    s = float(sigma_cells)
    if not s > 0.0 or not np.isfinite(g).any():
        return g
    return g - _smooth_grid_gaussian(g, sigma=s).astype(np.float64)


def _splat_bilinear(x, y, v, x_min, y_min, n_x, n_y, resolution):
    """Accumulate ``sum(w*v)`` and ``sum(w)`` on the grid nodes (bilinear weights)."""
    fx = (x - x_min) / resolution
    fy = (y - y_min) / resolution
    ix = np.floor(fx).astype(np.int64)
    iy = np.floor(fy).astype(np.int64)
    tx = fx - ix
    ty = fy - iy
    n_cells = int(n_x) * int(n_y)
    s = np.zeros(n_cells, dtype=np.float64)
    w = np.zeros(n_cells, dtype=np.float64)
    for dx, wx in ((0, 1.0 - tx), (1, tx)):
        for dy, wy in ((0, 1.0 - ty), (1, ty)):
            jx = ix + dx
            jy = iy + dy
            ok = (jx >= 0) & (jx < n_x) & (jy >= 0) & (jy < n_y)
            if not ok.any():
                continue
            idx = jy[ok] * n_x + jx[ok]
            ww = (wx * wy)[ok]
            s += np.bincount(idx, weights=ww * v[ok], minlength=n_cells)
            w += np.bincount(idx, weights=ww, minlength=n_cells)
    return s.reshape(n_y, n_x), w.reshape(n_y, n_x)


def _normalized_convolution(s, w, sigma):
    """Gaussian-smoothed ``s / w`` (NaN where the kernel saw no weight)."""
    from scipy.ndimage import gaussian_filter

    num = gaussian_filter(s, sigma, mode="constant")
    den = gaussian_filter(w, sigma, mode="constant")
    out = np.full(s.shape, np.nan, dtype=np.float64)
    ok = den > 1e-9
    out[ok] = num[ok] / den[ok]
    return out


def grid_binned(
    x_pts,
    y_pts,
    v_pts,
    x_min: float,
    y_min: float,
    n_x: int,
    n_y: int,
    resolution: float,
    reach_m: float,
    sigma_cells: float = 1.0,
    blanking_m: float = 0.0,
    min_cover: float = 0.3,
) -> np.ndarray:
    """Grid scattered samples without IDW's point-dominated bull's-eyes.

    ``sigma_cells`` is the Gaussian sigma (in cells) of the base smoothing.
    ``reach_m`` is the largest gap between data that is bridged (gaps up to
    ``2*reach_m`` wide are closed; the outer edge of the survey only grows by
    ~1.5 cells). ``blanking_m`` > 0 additionally NaNs every cell farther than
    that from a sample. Returns ``(n_y, n_x)`` float32, row 0 = ``y_min``.
    """
    from scipy.ndimage import distance_transform_edt

    n_x = int(n_x)
    n_y = int(n_y)
    res = float(resolution)
    x = np.asarray(x_pts, dtype=np.float64).ravel()
    y = np.asarray(y_pts, dtype=np.float64).ravel()
    v = np.asarray(v_pts, dtype=np.float64).ravel()
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    if not ok.any():
        return np.full((n_y, n_x), np.nan, dtype=np.float32)
    x, y, v = x[ok], y[ok], v[ok]

    s, w = _splat_bilinear(x, y, v, float(x_min), float(y_min), n_x, n_y, res)
    cover = w >= float(min_cover)
    if not cover.any():
        return np.full((n_y, n_x), np.nan, dtype=np.float32)
    s = np.where(cover, s, 0.0)
    w = np.where(cover, w, 0.0)

    from scipy.ndimage import gaussian_filter

    sig = max(0.3, float(sigma_cells))
    fine = _normalized_convolution(s, w, sig)
    coverage = gaussian_filter(cover.astype(np.float64), sig, mode="constant")

    reach_cells = max(1.0, float(reach_m) / res)
    sig_wide = float(min(max(2.0 * sig, reach_cells / 2.0), 64.0))
    wide = _normalized_convolution(s, w, sig_wide)

    # Smooth hand-over from the fine field (where it has data all around) to the
    # wide field (holes / survey edge): no visible seam between the two scales.
    t = np.clip((coverage - 0.15) / 0.45, 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)
    fine_f = np.where(np.isfinite(fine), fine, 0.0)
    wide_f = np.where(np.isfinite(wide), wide, 0.0)
    out = np.where(np.isfinite(fine), t * fine_f + (1.0 - t) * wide_f, wide)

    # Which cells are allowed to carry a value.
    # Morphological closing (bridges gaps up to 2*reach). The dilation must be
    # computed on a PADDED array: if the survey touches the grid border, a
    # dilation cut at the array edge is eroded back by ``reach`` and the body
    # loses up to ``reach`` metres in front of every line end, leaving the line
    # tails sticking out as thin strips.
    pad = int(np.ceil(reach_cells)) + 2
    dist_p = distance_transform_edt(~np.pad(cover, pad))
    dist = dist_p[pad:-pad, pad:-pad]
    dilated = dist_p <= reach_cells
    closed = (distance_transform_edt(dilated) > reach_cells)[pad:-pad, pad:-pad]
    allowed = closed | (dist <= 1.5)
    if float(blanking_m) > 0.0:
        allowed &= dist <= float(blanking_m) / res
    out = np.where(allowed & np.isfinite(out), out, np.nan)
    return out.astype(np.float32)


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
