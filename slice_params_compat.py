# -*- coding: utf-8 -*-
"""Bridge ``SliceSpec`` -> the legacy slice-parameter dicts (Fase D).

During the migration the OGPR slice pipeline still speaks two dict shapes:

* ``params``           - the small dict ``OgprSliceBuildTask`` reads for
                         ``channel`` / ``combine_method`` / ``z_*`` / ``radius`` /
                         ``resolution`` / ``group_name``.
* ``extra_slice_params`` - the ~35-key "advanced" bag flattened into the
                         ``compute_ogpr_slice_grids(**...)`` call.

``SliceSpec`` models ~13 of those; the rest are the old "Esperto" knobs the
redesign deliberately drops. This module produces both dicts from a *resolved*
spec, with fixed, conservative defaults for the dropped knobs, so the rest of
the pipeline keeps working unchanged while the single form drives everything.
"""

from __future__ import annotations

from .slice_spec import SliceSpec

# Fixed values for parameters SliceSpec does not model. These match the old
# "Esperto flag OFF" behaviour of GprProfileViewer.get_slice_params().
_DROPPED_KNOB_DEFAULTS = {
    "normalize_channels": False,
    "use_processing": False,
    "pipeline_params": None,
    "amplitude_sigma": None,
    "use_anisotropic_idw": False,
    "idw_mode": "quality",
    "idw_power": 1.5,
    "min_points": 1,
    "depth_radius_factor": 0.6,
    "balance_profiles": True,
    "pre_slice_bg_removal": False,
    "pre_slice_bg_mode": "line_by_line",
    "pre_slice_bg_window": 0,
    "pre_slice_bg_sample_start": 0,
    "pre_slice_bg_sample_end": 0,
    "stack_n": 1,
    "stack_kernel": "boxcar",
    "flip_traces_mode": "none",
    "topographic_correction": False,
    "topo_reference_mode": "median",
    "topo_reference_elevation": None,
    "parallel_profiles": True,
    "profile_workers": 0,
    "fill_nodata": True,
}


def _require_resolved(spec: SliceSpec) -> SliceSpec:
    if not isinstance(spec, SliceSpec):
        raise TypeError("expected a slice_spec.SliceSpec")
    spec.validate()
    if spec.z_max_m is None:
        raise ValueError("spec must be resolved (z_max_m is None); call spec.resolve(profiles)")
    return spec


def spec_to_extra_slice_params(spec: SliceSpec) -> dict:
    """Build the flattened ``extra_slice_params`` bag from a resolved spec."""
    spec = _require_resolved(spec)
    ck = spec.to_compute_kwargs()  # extraction_mode/use_hilbert/overlap/blanking/smooth/(radius)/auto_radius
    out = dict(_DROPPED_KNOB_DEFAULTS)
    out.update({
        "slice_source": str(spec.source_variant or "raw"),
        "extraction_mode": ck["extraction_mode"],
        "use_hilbert": ck["use_hilbert"],
        "overlap_fraction": ck["overlap_fraction"],
        "blanking_distance": ck["blanking_distance"],
        "smooth_sigma": ck["smooth_sigma"],
        "auto_radius": ck["auto_radius"],
        "radius": ck.get("radius"),
        "search_radius": ck.get("radius"),
        "thickness_m": float(spec.thickness_m),
    })
    return out


def spec_to_build_params(
    spec: SliceSpec,
    *,
    channel: int = -1,
    combine_method: str = "mean",
    group_name: str = "gpr_slices",
) -> dict:
    """Build the small ``params`` dict ``OgprSliceBuildTask`` consumes.

    ``channel`` / ``combine_method`` / ``group_name`` stay viewer-level choices
    (SliceSpec has no opinion on them).
    """
    spec = _require_resolved(spec)
    radius = spec.search_radius_m
    return {
        "channel": int(channel),
        "combine_method": str(combine_method or "mean"),
        "z_min": float(spec.z_min_m),
        "z_max": float(spec.z_max_m),
        "z_step": float(spec.thickness_m),
        "resolution": float(spec.resolution_m),
        "radius": (float(radius) if radius is not None else 0.0),
        "group_name": str(group_name or "gpr_slices"),
        "advanced": {},
    }


def spec_to_legacy_params(
    spec: SliceSpec,
    *,
    channel: int = -1,
    combine_method: str = "mean",
    group_name: str = "gpr_slices",
    **viewer_extras,
) -> dict:
    """Full replacement for ``GprProfileViewer.get_slice_params()``.

    Returns the historical flat dict (build params + extra bag merged) plus the
    live ``_spec`` object, and echoes any ``viewer_extras`` (e.g.
    ``open_3d_on_complete``, ``output_dir``).
    """
    spec = _require_resolved(spec)
    merged = spec_to_extra_slice_params(spec)
    bp = spec_to_build_params(
        spec, channel=channel, combine_method=combine_method, group_name=group_name
    )
    merged.update({
        "channel": bp["channel"],
        "combine_method": bp["combine_method"],
        "z_min": bp["z_min"],
        "z_max": bp["z_max"],
        "z_step": bp["z_step"],
        "resolution": bp["resolution"],
        "radius": bp["radius"] or None,
    })
    merged["_spec"] = spec
    merged["crs_epsg"] = spec.crs_epsg
    for k, v in viewer_extras.items():
        merged[k] = v
    return merged
