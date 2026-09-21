# -*- coding: utf-8 -*-
"""Time-slice engine "bin": every trace at its own position, binned gridding.

Replaces the profile-centre-line + IDW path of ``compute_ogpr_slice_grids`` for
a ``SliceSpec`` with ``interp == "bin"`` (the default). What changes:

* **Every channel is a set of points at its own coordinates.** The old path
  averaged the coordinates *and* the amplitudes of all the channels of a swath
  into one centre line; a 19-channel array (4 cm channel pitch, ~0.75 m swath)
  was thus reduced to one line per swath and everything between lines was
  interpolated, which is what produced the one-cell dots along the survey lines.
* **Depth is ``row * dt * v / 2``**, like the radargram axis, not
  ``row * depth_max / n_rows``: after time-zero alignment / channel stacking the
  matrices are shorter than the nominal record and the old mapping stretched
  depth (a slice labelled 4.08 m showed what the radargram has at ~3.6 m).
  ``SliceSpec.velocity_m_s`` (``None`` = the profile's own velocity) is honoured.
* **Channels are equalised** (``SliceSpec.balance``): array channels differ in
  coupling/gain, which shows up as stripes along the tracks once the lateral
  resolution is kept.
* **Gridding is ``slicing.grid_binned``** (bilinear splat + normalised Gaussian
  convolution): no bull's-eyes, ~0.1 s per slice, so the live preview is
  interactive.

The output is ``(grids, meta)`` with the same shape as
``gpr_ogpr_slicer.compute_ogpr_slice_grids``, so GeoTIFF writing, the manifest
and the 3D viewer are unchanged. Geometry always comes from the profiles handed
in (single geometry source); nothing is read from disk here.
"""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter

import numpy as np

from .slicing import (
    AMPLITUDE_WINDOW_MODES,
    _smooth_grid_gaussian,
    depth_per_sample_m,
    grid_binned,
    window_amplitude,
    window_sample_range,
)

MAX_CELLS_ABORT = 50_000_000
_EDGE_MARGIN_CELLS = 2.0
_MIN_FLOOR = 1e-12


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


class _ChannelGeom:
    __slots__ = ("prof_idx", "ch_idx", "x", "y")

    def __init__(self, prof_idx, ch_idx, x, y):
        self.prof_idx = prof_idx
        self.ch_idx = ch_idx
        self.x = x
        self.y = y


def _fill_nonfinite(v: np.ndarray) -> np.ndarray | None:
    """Linear-interpolate NaN traces (fix-less GPS epochs); None if all bad."""
    fin = np.isfinite(v)
    if not fin.any():
        return None
    if fin.all():
        return v
    idx = np.arange(v.size, dtype=np.float64)
    return np.interp(idx, idx[fin], v[fin])


def _channel_xy(prof, ci: int, prof_idx: int, synthetic: bool):
    """(x, y) of every trace of one channel, or None if unusable."""
    from .gpr_ogpr_slicer import (
        _has_plausible_geo_xy,
        _resample_vec_to_n,
        _synthetic_profile_xy,
    )

    ch = prof.channel(ci)
    n_t = int(np.asarray(ch.data).shape[1])
    if n_t <= 0:
        return None
    if synthetic:
        x, y = _synthetic_profile_xy(ch, prof, prof_idx)
        return _resample_vec_to_n(x, n_t), _resample_vec_to_n(y, n_t)
    x = _fill_nonfinite(np.asarray(ch.easting, dtype=np.float64))
    y = _fill_nonfinite(np.asarray(ch.northing, dtype=np.float64))
    if x is None or y is None:
        return None
    x = _resample_vec_to_n(x, n_t)
    y = _resample_vec_to_n(y, n_t)
    if not _has_plausible_geo_xy(x, y):
        return None
    return x, y


def collect_geometry(profiles: list, channel: int = -1):
    """Per-channel trace coordinates for the whole survey.

    Returns ``(channels, diag)``. ``channels`` is a list of ``_ChannelGeom``;
    ``diag`` carries the same georeferencing counters as the legacy path
    (``profiles_total`` / ``profiles_geo_valid`` / ``profiles_geo_skipped`` /
    ``using_synthetic_coords``).
    """
    from .gpr_ogpr_slicer import _has_plausible_geo_xy

    def _selected(prof):
        n_ch = int(getattr(prof, "n_channels", 0) or 0)
        if n_ch <= 0:
            return []
        return list(range(n_ch)) if int(channel) < 0 else [min(int(channel), n_ch - 1)]

    valid_flags = []
    for prof in profiles:
        ok = False
        for ci in _selected(prof):
            try:
                ch = prof.channel(ci)
                if _has_plausible_geo_xy(
                    np.asarray(ch.easting, dtype=np.float64),
                    np.asarray(ch.northing, dtype=np.float64),
                ):
                    ok = True
                    break
            except Exception:
                continue
        valid_flags.append(ok)
    synthetic = not any(valid_flags)

    out: list[_ChannelGeom] = []
    skipped = 0
    for pi, prof in enumerate(profiles):
        if (not synthetic) and (not valid_flags[pi]):
            skipped += 1
            continue
        for ci in _selected(prof):
            try:
                xy = _channel_xy(prof, ci, pi, synthetic)
            except Exception:
                xy = None
            if xy is not None:
                out.append(_ChannelGeom(pi, ci, xy[0], xy[1]))
    diag = {
        "profiles_total": int(len(profiles)),
        "profiles_geo_valid": int(sum(1 for v in valid_flags if v)),
        "profiles_geo_skipped": int(skipped),
        "using_synthetic_coords": bool(synthetic),
    }
    return out, diag


# ---------------------------------------------------------------------------
# windows
# ---------------------------------------------------------------------------


def build_windows(spec) -> list[tuple[float, float]]:
    """``[(z_from, z_to), ...]`` covering ``[z_min, z_max]`` for a resolved spec.

    Window ``i`` starts at ``z_min + i * thickness * (1 - overlap)`` and is
    ``thickness`` thick; only windows that fit entirely in ``[z_min, z_max]``
    are produced (a range thinner than one window yields one clipped window).
    """
    z0 = float(spec.z_min_m)
    z1 = float(spec.z_max_m)
    thick = float(spec.thickness_m)
    stride = float(spec.stride_m)
    if stride <= 0.0:
        stride = thick
    wins = []
    i = 0
    while True:
        a = z0 + i * stride
        if a + thick > z1 + 1e-9:
            break
        wins.append((a, a + thick))
        i += 1
        if i > 100_000:
            break
    if not wins:
        wins.append((z0, max(z1, z0 + 1e-6)))
    return wins


# ---------------------------------------------------------------------------
# points of one window
# ---------------------------------------------------------------------------


def _profile_dz(prof, velocity_override) -> float:
    v = velocity_override
    if v is None:
        v = getattr(prof, "velocity_m_s", None)
    dz = depth_per_sample_m(getattr(prof, "dt_ns", None), v)
    if dz > 0.0:
        return dz
    # last resort (profile without dt/velocity): nominal depth over nominal rows
    try:
        n = float(getattr(prof, "n_samples", 0) or 0)
        d = float(getattr(prof, "depth_max_m", 0.0) or 0.0)
        return d / n if n > 0 and d > 0 else 0.0
    except Exception:
        return 0.0


def _balance_factors(records, mode: str, amplitude_mode: str):
    """Per-record ``(scale, offset)`` so channels/profiles share one level.

    Amplitude-like values (abs_mean, rms, env_peak, max_abs) are equalised with a
    multiplicative gain; ``signed_mean`` (zero-centred) with an additive offset.
    The reference is the median over everything, and each record's own level is
    its median over the slice window, so isolated targets do not bias it.
    """
    n = len(records)
    scale = np.ones(n, dtype=np.float64)
    offset = np.zeros(n, dtype=np.float64)
    if n == 0 or mode == "none":
        return scale, offset
    med = np.array(
        [float(np.median(r[3][np.isfinite(r[3])])) if np.isfinite(r[3]).any() else np.nan
         for r in records]
    )
    if mode == "profile":
        pid = np.array([r[0] for r in records])
        lvl = np.full(n, np.nan)
        for p in np.unique(pid):
            m = pid == p
            fin = med[m][np.isfinite(med[m])]
            if fin.size:
                lvl[m] = np.median(fin)
    else:  # "channel"
        lvl = med
    fin = lvl[np.isfinite(lvl)]
    if fin.size == 0:
        return scale, offset
    if amplitude_mode == "signed_mean":
        offset = np.where(np.isfinite(lvl), -lvl, 0.0)
        return scale, offset
    ref = float(np.median(fin[fin > _MIN_FLOOR])) if (fin > _MIN_FLOOR).any() else 0.0
    if ref <= _MIN_FLOOR:
        return scale, offset
    ok = np.isfinite(lvl) & (lvl > _MIN_FLOOR)
    scale[ok] = ref / lvl[ok]
    return scale, offset


def window_points(profiles, geoms, spec, z_from, z_to):
    """``(x, y, v)`` of every trace of every selected channel for one window."""
    mode = str(spec.amplitude_mode)
    records = []  # (prof_idx, ch_idx, x, v)
    dz_cache: dict[int, float] = {}
    for g in geoms:
        prof = profiles[g.prof_idx]
        dz = dz_cache.get(g.prof_idx)
        if dz is None:
            dz = _profile_dz(prof, spec.velocity_m_s)
            dz_cache[g.prof_idx] = dz
        data = np.asarray(prof.channel(g.ch_idx).data)
        if data.ndim != 2 or data.size == 0:
            continue
        lo, hi = window_sample_range(z_from, z_to, dz, int(data.shape[0]))
        if hi <= lo:
            continue
        v = window_amplitude(data, lo, hi, mode)
        if v.shape[0] != g.x.shape[0]:
            continue
        records.append((g.prof_idx, g.ch_idx, g, v))
    if not records:
        return None
    scale, offset = _balance_factors(records, str(getattr(spec, "balance", "channel")), mode)
    xs = np.concatenate([r[2].x for r in records])
    ys = np.concatenate([r[2].y for r in records])
    vs = np.concatenate([r[3] * s + o for r, s, o in zip(records, scale, offset)])
    return xs, ys, vs


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------


def _grid_params(geoms, resolution: float):
    xs = np.concatenate([g.x for g in geoms])
    ys = np.concatenate([g.y for g in geoms])
    margin = _EDGE_MARGIN_CELLS * resolution
    x_min = float(xs.min() - margin)
    y_min = float(ys.min() - margin)
    n_x = max(2, int(np.ceil((float(xs.max()) + margin - x_min) / resolution)) + 1)
    n_y = max(2, int(np.ceil((float(ys.max()) + margin - y_min) / resolution)) + 1)
    return x_min, y_min, n_x, n_y, margin


def compute_grids(
    profiles: list,
    spec,
    *,
    channel: int = -1,
    single_window: tuple[float, float] | None = None,
    resolution_override: float | None = None,
    progress_callback=None,
    cancel_event=None,
    emit_diagnostics: bool = True,
) -> tuple[list[dict], dict]:
    """Compute the slice grids for a ``SliceSpec`` (``interp == "bin"``).

    ``single_window=(z0, z1)`` yields exactly one slice on that window (the live
    preview); ``resolution_override`` forces the cell size (coarser preview).
    Returns ``(grids, meta)`` shaped like ``compute_ogpr_slice_grids``.
    """
    from .slice_spec import SliceSpec

    if not isinstance(spec, SliceSpec):
        raise TypeError("spec deve essere una slice_spec.SliceSpec")
    t_start = perf_counter()
    if not profiles:
        return [], {}

    rspec = spec if spec.is_resolved() else spec.resolve(profiles)
    if resolution_override is not None and float(resolution_override) > 0.0:
        rspec = replace(rspec, resolution_m=float(resolution_override))
    if single_window is not None:
        z0, z1 = sorted(float(v) for v in single_window)
        z1 = max(z1, z0 + 1e-6)
        windows = [(z0, z1)]
    else:
        windows = build_windows(rspec)
    rspec.validate()

    geoms, diag = collect_geometry(profiles, channel)
    if not geoms:
        return [], {**diag, "timing_s": {"total": round(perf_counter() - t_start, 6)}}

    res = float(rspec.resolution_m)
    x_min, y_min, n_x, n_y, margin = _grid_params(geoms, res)
    n_cells = int(n_x) * int(n_y)
    if n_cells > MAX_CELLS_ABORT:
        raise MemoryError(
            f"Griglia troppo grande ({n_cells:,} celle, shape {n_y}x{n_x}). "
            f"Aumenta la risoluzione da {res:.4f}m a {res * 3.0:.2f}m o superiore."
        )
    x_max = x_min + (n_x - 1) * res
    y_max = y_min + (n_y - 1) * res
    reach = float(rspec.search_radius_m or (res * 5.0))
    sigma_cells = 1.0
    smooth_sigma = float(rspec.smooth_sigma)

    grids: list[dict] = []
    t_grid = 0.0
    total = len(windows)
    for iz, (z_from, z_to) in enumerate(windows):
        if cancel_event is not None and cancel_event.is_set():
            break
        pts = window_points(profiles, geoms, rspec, z_from, z_to)
        if pts is None:
            if progress_callback is not None:
                progress_callback(iz + 1, total)
            continue
        xs, ys, vs = pts
        t0 = perf_counter()
        grid = grid_binned(
            xs, ys, vs, x_min, y_min, n_x, n_y, res,
            reach_m=reach, sigma_cells=sigma_cells,
            blanking_m=float(rspec.blanking_radius_m),
        )
        keep = np.isfinite(grid)
        if rspec.grid_bg_spec:
            from .grid_filters import apply_grid_bg
            grid = apply_grid_bg(grid, rspec.grid_bg_spec).astype(np.float32, copy=False)
        if smooth_sigma > 0.0:
            grid = _smooth_grid_gaussian(grid, sigma=smooth_sigma)
        grid = np.where(keep, grid, np.nan).astype(np.float32, copy=False)
        t_grid += perf_counter() - t0

        n_valid = int(np.count_nonzero(keep))
        grids.append({
            "z_lev": float(0.5 * (z_from + z_to)),
            "z_from": round(float(z_from), 6),
            "z_to": round(float(z_to), 6),
            "index": iz,
            "grid": grid,
            "n_pts": int(xs.size),
            "n_valid_cells": n_valid,
            "n_total_cells": int(grid.size),
            "fill_pct": 100.0 * n_valid / max(int(grid.size), 1),
            "radius": reach,
            "effective_radius": reach,
            "amplitude_diag": {},
        })
        if progress_callback is not None:
            progress_callback(iz + 1, total)

    stride = float(rspec.stride_m) if single_window is None else float(z1 - z0)
    from .slice_spec import _AMPLITUDE_MAP
    legacy_mode, legacy_hilbert = _AMPLITUDE_MAP[str(rspec.amplitude_mode)]
    meta = dict(
        x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max,
        n_x=int(n_x), n_y=int(n_y), resolution=res,
        z_min=float(windows[0][0]), z_max=float(windows[-1][1]),
        z_step=float(windows[0][1] - windows[0][0]),
        channel=int(channel), combine_method="mean",
        extraction_mode=legacy_mode, use_hilbert=bool(legacy_hilbert),
        amplitude_mode=str(rspec.amplitude_mode),
        use_processing=False, pipeline_params={},
        radius=reach, effective_radius=reach, depth_radius_factor=0.0,
        overlap_fraction=float(rspec.overlap), slice_step=stride,
        idw_power=0.0, use_anisotropic_idw=False, idw_mode="bin", min_points=1,
        balance_profiles=bool(rspec.balance != "none"), per_slice_balance=False,
        fill_nodata_max_distance_m=reach, blanking_distance_m=float(rspec.blanking_radius_m),
        pre_slice_bg_removal=False, pre_slice_bg_mode="line_by_line", pre_slice_bg_window=0,
        stack_n=1, stack_kernel="boxcar", flip_traces_mode="none",
        topographic_correction=False, topo_reference_mode="median", topo_reference_elevation=None,
        parallel_profiles=False, profile_workers_used=1, bounds_margin_m=float(margin),
        slice_spec=rspec.to_dict(), engine="bin",
        **diag,
    )
    if grids:
        z_lev = [float(g["z_lev"]) for g in grids]
        fills = np.asarray([g["fill_pct"] for g in grids], dtype=np.float64)
        meta.update(
            n_z=len(grids), z_first=min(z_lev), z_last=max(z_lev), z_levels=z_lev,
            fill_pct_mean=float(np.mean(fills)), fill_pct_min=float(np.min(fills)),
            fill_pct_low_count=0, fill_pct_low_threshold=60.0,
        )
    t_total = perf_counter() - t_start
    meta["timing_s"] = {
        "total": round(t_total, 6),
        "preprocess": 0.0,
        "grid_setup": 0.0,
        "slice_loop": round(t_total, 6),
        "interpolation": round(t_grid, 6),
        "interpolation_avg_per_slice": round(t_grid / max(1, len(grids)), 6),
    }
    meta["timing_counts"] = {"slices_requested": int(total), "slices_computed": int(len(grids))}
    if emit_diagnostics:
        print(
            f"[OGPR slicer][bin] griglia {n_x}x{n_y} (res={res:.3f}m, raggio={reach:.3f}m) "
            f"canali={len(geoms)} slice={len(grids)}/{total} totale={t_total:.2f}s"
        )
    return grids, meta


__all__ = ["compute_grids", "build_windows", "collect_geometry", "window_points",
           "AMPLITUDE_WINDOW_MODES"]
