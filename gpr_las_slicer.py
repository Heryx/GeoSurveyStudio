"""LAS/LAZ → GeoTIFF slice generator with IDW interpolation.

Supports:
  - single-band (any field)
  - 3-band RGB (value_field='rgb') → GeoTIFF + QML sidecar multibandcolor
    with user-chosen R, G, B fields

Interpolation methods:
  - 'idw' : Inverse Distance Weighting (isotropic, cKDTree gather approach)
"""

from __future__ import annotations

import json
import os

import numpy as np


# ---------------------------------------------------------------------------
# LAS field diagnosis
# ---------------------------------------------------------------------------

_SKIP_FIELDS = frozenset((
    "x", "y", "z", "X", "Y", "Z",
    "return_number", "number_of_returns",
    "scan_direction_flag", "edge_of_flight_line",
    "scan_angle_rank", "scan_angle",
    "gps_time",
))

_PRIORITY_FIELDS = ("intensity", "red", "green", "blue",
                    "amplitude", "reflectance", "deviation", "signal")


def diagnose_las_fields_detailed(
    las_path: str,
    n_preview: int = 6,
    n_sample: int = 30000,
) -> dict:
    import laspy
    result = {
        "rows":            [],
        "all_field_names": [],
        "suggested_r":     "red",
        "suggested_g":     "green",
        "suggested_b":     "blue",
        "suggested_single":"intensity",
        "point_format":    -1,
        "n_points":        0,
    }
    try:
        with laspy.LasReader(las_path) as reader:
            result["point_format"] = int(reader.header.point_format.id)
            result["n_points"]     = int(reader.header.point_count)
            fmt      = reader.header.point_format
            all_dims = list(fmt.dimension_names)
            try:
                for ed in fmt.extra_dims:
                    all_dims.append(ed.name)
            except Exception:
                pass
            value_dims = [d for d in all_dims if d not in _SKIP_FIELDS]
            result["all_field_names"] = value_dims
            sample = None
            try:
                for chunk in reader.chunk_iterator(n_sample):
                    sample = chunk
                    break
            except Exception:
                pass
            if sample is None:
                return result
            n_pts = min(n_preview, len(np.asarray(sample.x)))
            for name in value_dims:
                try:
                    arr = np.asarray(getattr(sample, name), dtype=np.float64)
                    if arr.size == 0:
                        continue
                    finite = arr[np.isfinite(arr)]
                    if finite.size == 0:
                        continue
                    mn       = float(finite.min())
                    mx       = float(finite.max())
                    has_data = (mx > 0) or (mx > mn)
                    preview  = [round(float(arr[i]), 3) for i in range(min(n_pts, arr.size))]
                    result["rows"].append({
                        "name":     name,
                        "values":   preview,
                        "min":      mn,
                        "max":      mx,
                        "has_data": has_data,
                    })
                except Exception:
                    pass
    except Exception:
        return result

    def _key(row):
        n = row["name"]
        p = _PRIORITY_FIELDS.index(n) if n in _PRIORITY_FIELDS else 99
        return (0 if row["has_data"] else 1, p, n)

    result["rows"].sort(key=_key)
    has_data_fields = [r["name"] for r in result["rows"] if r["has_data"]]

    def _first(candidates):
        for c in candidates:
            if c in has_data_fields:
                return c
        return candidates[0] if candidates else "intensity"

    result["suggested_r"]      = _first(["red",   "intensity"])
    result["suggested_g"]      = _first(["green", "intensity"])
    result["suggested_b"]      = _first(["blue",  "intensity"])
    result["suggested_single"] = _first(["intensity", "red", "green", "blue"])
    return result


# ---------------------------------------------------------------------------
# LAS reading
# ---------------------------------------------------------------------------

def _read_las_single(las_path: str, value_field: str):
    import laspy
    las    = laspy.read(las_path)
    x      = np.asarray(las.x, dtype=np.float64)
    y      = np.asarray(las.y, dtype=np.float64)
    z      = np.asarray(las.z, dtype=np.float64)
    try:
        raw = np.asarray(getattr(las, value_field), dtype=np.float32)
    except Exception as e:
        raise ValueError(
            f"Campo '{value_field}' non presente nel file LAS: {e}"
        )
    if raw.size != len(x):
        raise ValueError(
            f"Campo '{value_field}' ha dimensione incoerente "
            f"({raw.size} valori per {len(x)} punti)."
        )
    return x, y, z, raw


def _read_las_rgb(
    las_path: str,
    r_field: str = "red",
    g_field: str = "green",
    b_field: str = "blue",
):
    import laspy
    las = laspy.read(las_path)
    x   = np.asarray(las.x, dtype=np.float64)
    y   = np.asarray(las.y, dtype=np.float64)
    z   = np.asarray(las.z, dtype=np.float64)

    def _get(field):
        try:
            arr = np.asarray(getattr(las, field), dtype=np.float32)
        except Exception as e:
            raise ValueError(
                f"Campo RGB '{field}' non presente nel file LAS: {e}"
            )
        if arr.size != len(x):
            raise ValueError(
                f"Campo RGB '{field}' ha dimensione incoerente "
                f"({arr.size} valori per {len(x)} punti)."
            )
        return arr

    return x, y, z, _get(r_field), _get(g_field), _get(b_field)


# ---------------------------------------------------------------------------
# IDW binning — isotropic gather with cKDTree
# ---------------------------------------------------------------------------

def _idw_gather(
    x_pts: np.ndarray,
    y_pts: np.ndarray,
    x_min: float,
    y_min: float,
    n_x: int,
    n_y: int,
    resolution: float,
    radius: float,
    power: float = 2.0,
    min_points: int = 1,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Precompute IDW neighbours and normalized weights per grid cell.

    Only depends on point geometry (x, y), not on the interpolated value,
    so the result can be reused across several value channels (e.g. R/G/B)
    without rebuilding the KD-tree. Returns a list of
    (cell_index, point_indices, normalized_weights).
    """
    from scipy.spatial import cKDTree

    gx = x_min + np.arange(n_x) * resolution
    gy = y_min + np.arange(n_y) * resolution
    gxx, gyy  = np.meshgrid(gx, gy)
    grid_pts  = np.column_stack([gxx.ravel(), gyy.ravel()])

    tree    = cKDTree(np.column_stack([x_pts, y_pts]))
    results = tree.query_ball_point(grid_pts, r=radius, workers=-1)

    gather: list[tuple[int, np.ndarray, np.ndarray]] = []
    for k, idx_list in enumerate(results):
        if len(idx_list) < min_points:
            continue
        idx = np.asarray(idx_list, dtype=np.int64)
        cx  = grid_pts[k, 0]
        cy  = grid_pts[k, 1]
        d   = np.sqrt((x_pts[idx] - cx) ** 2 + (y_pts[idx] - cy) ** 2)
        w   = 1.0 / (d ** power + 1e-9)
        ws  = w.sum()
        if ws > 0:
            gather.append((k, idx, w / ws))
    return gather


def _apply_idw(
    gather: list[tuple[int, np.ndarray, np.ndarray]],
    values: np.ndarray,
    n_x: int,
    n_y: int,
) -> np.ndarray:
    """Apply precomputed IDW weights to one value array -> (n_y, n_x) grid."""
    v    = np.asarray(values, dtype=np.float64)
    grid = np.full(n_x * n_y, np.nan, dtype=np.float32)
    for k, idx, wn in gather:
        grid[k] = float(np.dot(wn, v[idx]))
    return grid.reshape(n_y, n_x)


def _bin_with_idw(
    x_pts: np.ndarray,
    y_pts: np.ndarray,
    i_pts: np.ndarray,
    x_min: float,
    y_min: float,
    n_x: int,
    n_y: int,
    resolution: float,
    radius: float,
    power: float = 2.0,
    min_points: int = 1,
) -> np.ndarray:
    """Isotropic single-channel IDW (thin wrapper over gather/apply)."""
    gather = _idw_gather(
        x_pts, y_pts, x_min, y_min, n_x, n_y,
        resolution, radius, power, min_points,
    )
    return _apply_idw(gather, i_pts, n_x, n_y)


# ---------------------------------------------------------------------------
# GeoTIFF / QML writers — single source of truth in raster_io.py (Fase A).
# Aliases keep the historical private names for callers that import them
# from this module (e.g. gpr_csv_slicer).
# ---------------------------------------------------------------------------

from .raster_io import (  # noqa: E402,F401
    make_geotransform as _make_geotransform,
    set_projection as _set_projection,
    write_qml_multiband as _write_qml_multiband,
    write_tif_rgb as _write_tif_rgb,
    write_tif_singleband as _write_tif_singleband,
)


# ---------------------------------------------------------------------------
# Sidecar params
# ---------------------------------------------------------------------------

SIDECAR_FILENAME = ".las_slicer_params.json"


def save_slicer_params(output_dir, params):
    path = os.path.join(output_dir, SIDECAR_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=True)
    return path


def load_slicer_params(output_dir):
    path = os.path.join(output_dir, SIDECAR_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def slice_las_to_tifs(
    las_path: str,
    output_dir: str,
    resolution: float = 0.10,
    z_step: float = 0.05,
    z_min: float | None = None,
    z_max: float | None = None,
    radius: float | None = None,
    epsg: int | None = None,
    value_field: str = "intensity",
    r_field: str = "red",
    g_field: str = "green",
    b_field: str = "blue",
    idw_power: float = 2.0,
    min_points: int = 1,
) -> list[dict]:
    """
    Slice LAS/LAZ data into GeoTIFF rasters (one per Z interval).

    Parameters
    ----------
    las_path, output_dir : percorsi I/O
    resolution           : passo griglia XY in metri
    z_step               : spessore finestra di profondita'
    z_min / z_max        : range Z (None = auto)
    radius               : raggio IDW in metri (None = resolution * sqrt(2))
    epsg                 : EPSG del CRS (None = non georiferito)
    value_field          : campo LAS da interpolare ('rgb' per RGB)
    r_field, g_field, b_field : campi RGB (usati se value_field='rgb')
    idw_power            : esponente IDW
    min_points           : punti minimi per stimare una cella

    Returns
    -------
    Lista di dict: path, z_from, z_to, z_center, index, name.
    """
    if radius is None:
        radius = resolution * (2 ** 0.5)

    os.makedirs(output_dir, exist_ok=True)
    use_rgb = (value_field == "rgb")

    if use_rgb:
        x, y, z, r, g, b = _read_las_rgb(las_path, r_field, g_field, b_field)
    else:
        x, y, z, values = _read_las_single(las_path, value_field)

    if z_min is None:
        z_min = float(z.min())
    if z_max is None:
        z_max = float(z.max())

    x_min = float(x.min())
    y_min = float(y.min())
    y_max = float(y.max())
    x_max = float(x.max())

    n_x = len(np.arange(x_min, x_max + resolution * 0.5, resolution))
    n_y = len(np.arange(y_min, y_max + resolution * 0.5, resolution))

    _kw = dict(x_min=x_min, y_min=y_min, n_x=n_x, n_y=n_y,
               resolution=resolution, radius=radius,
               power=idw_power, min_points=min_points)

    # Slice centers tile [z_min, z_max]: each window is [c - dz/2, c + dz/2),
    # so no slice reaches below z_min or above z_max (coherent with the PDAL
    # path in gpr_slice_generator.generate_slices_pdal).
    z_centers = np.arange(z_min + z_step / 2.0, z_max, z_step)
    if z_centers.size == 0:
        z_centers = np.array([(z_min + z_max) / 2.0])
    results  = []

    for iz, z_lev in enumerate(z_centers):
        z_from = float(z_lev - z_step / 2.0)
        z_to   = float(z_lev + z_step / 2.0)
        mask   = (z >= z_from) & (z < z_to)
        if not np.any(mask):
            continue

        z_label  = f"{z_lev:.4f}".replace(".", "_").replace("-", "m")
        tif_name = f"slice_{iz:04d}_z{z_label}.tif"
        tif_path = os.path.join(output_dir, tif_name)

        if use_rgb:
            # Build the KD-tree / weights once, reuse for the 3 channels.
            gather = _idw_gather(x[mask], y[mask], **_kw)
            r_grid = _apply_idw(gather, r[mask], n_x, n_y)
            g_grid = _apply_idw(gather, g[mask], n_x, n_y)
            b_grid = _apply_idw(gather, b[mask], n_x, n_y)
            ranges = _write_tif_rgb(
                r_grid, g_grid, b_grid,
                tif_path, x_min, y_min, y_max, resolution, epsg,
            )
            _write_qml_multiband(tif_path, *ranges)
        else:
            grid = _bin_with_idw(x[mask], y[mask], values[mask], **_kw)
            _write_tif_singleband(
                grid, tif_path, x_min, y_min, y_max, resolution, epsg,
            )

        results.append({
            "path":     tif_path,
            "z_from":   round(z_from, 6),
            "z_to":     round(z_to,   6),
            "z_center": round(float(z_lev), 6),
            "index":    iz,
            "name":     os.path.splitext(tif_name)[0],
        })

    return results
