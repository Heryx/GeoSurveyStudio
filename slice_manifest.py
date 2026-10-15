# -*- coding: utf-8 -*-
"""``sliceset.json`` — the manifest that travels next to a generated slice stack.

Fase B of the time-slice redesign. The generator already knows every slice's
index, depth window, CRS and extent; instead of throwing that away and letting
``TimesliceDepthParser`` guess it back from file names, we persist it here.

Format ``rasterlinker_sliceset_v1``::

    {
      "format": "rasterlinker_sliceset_v1",
      "created": "2026-09-08T12:34:56Z",
      "crs_epsg": 32633,
      "source": {"kind": "ogpr", "variant_id": "bandpass", "n_profiles": 42},
      "spec": { ...SliceSpec.to_dict()... },
      "extent": [xmin, ymin, xmax, ymax],
      "slices": [
        {"index": 0, "z_from": 0.0, "z_to": 0.1, "z_center": 0.05,
         "grid": "slice_0000_z0_0500.tif",
         "stats": {"min": .., "max": .., "p2": .., "p98": .., "fill": 0.87}}
      ]
    }
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os

from .slice_spec import SliceSpec, SliceSpecError

FORMAT = "rasterlinker_sliceset_v1"
MANIFEST_FILENAME = "sliceset.json"


class SliceSetError(ValueError):
    """Raised for a missing or structurally invalid ``sliceset.json``."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def compute_slice_stats(array) -> dict:
    """min / max / p2 / p98 / fill fraction for one grid (NaN = nodata)."""
    import numpy as np

    a = np.asarray(array, dtype=np.float64)
    finite = a[np.isfinite(a)]
    total = int(a.size) or 1
    if finite.size == 0:
        return {"min": None, "max": None, "p2": None, "p98": None, "fill": 0.0}
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "p2": float(np.percentile(finite, 2)),
        "p98": float(np.percentile(finite, 98)),
        "fill": round(float(finite.size) / float(total), 6),
    }


def _slice_entry(item: dict) -> dict:
    """Normalise one slice record for the manifest.

    Accepts the dicts returned by ``write_grids_to_tifs`` (``path`` / ``name`` /
    ``z_from`` / ``z_to`` / ``z_center`` / ``index``) plus an optional ``array``
    (the in-memory grid) or precomputed ``stats``.
    """
    if "grid" in item:
        grid_name = str(item["grid"])
    elif item.get("path"):
        grid_name = os.path.basename(str(item["path"]))
    elif item.get("name"):
        grid_name = f"{item['name']}.tif"
    else:
        raise SliceSetError("slice entry needs one of 'grid', 'path', 'name'")

    stats = item.get("stats")
    if stats is None and item.get("array") is not None:
        stats = compute_slice_stats(item["array"])

    entry = {
        "index": int(item["index"]),
        "z_from": round(float(item["z_from"]), 6),
        "z_to": round(float(item["z_to"]), 6),
        "z_center": round(float(item.get("z_center", (item["z_from"] + item["z_to"]) / 2.0)), 6),
        "grid": grid_name,
    }
    if stats is not None:
        entry["stats"] = stats
    return entry


def write_sliceset(
    output_dir: str,
    spec: SliceSpec,
    slices: list[dict],
    source: dict,
    *,
    extent: "list | tuple | None" = None,
    crs_epsg: int | None = None,
    created: str | None = None,
) -> str:
    """Serialise a ``sliceset.json`` into ``output_dir``. Returns its path."""
    if not isinstance(spec, SliceSpec):
        raise SliceSetError("spec must be a SliceSpec instance")
    spec.validate()

    if crs_epsg is None:
        crs_epsg = spec.crs_epsg

    manifest = {
        "format": FORMAT,
        "created": str(created) if created else _utc_now_iso(),
        "crs_epsg": int(crs_epsg) if crs_epsg is not None else None,
        "source": dict(source or {}),
        "spec": spec.to_dict(),
        "extent": [float(v) for v in extent] if extent is not None else None,
        "slices": [_slice_entry(s) for s in (slices or [])],
    }

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, MANIFEST_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=True, sort_keys=True)
    return path


def sliceset_path(output_dir: str) -> str:
    return os.path.join(output_dir, MANIFEST_FILENAME)


def slice_catalog_from_sliceset(directory: str) -> list[dict]:
    """Build a viewer ``_slice_catalog`` from the authoritative ``sliceset.json``.

    Returns rows shaped like ``GprProfileViewer._scan_slice_catalog`` output —
    ``path`` (absolute, ``directory/<grid>``), ``index``, ``name``, ``z_center``,
    ``z_top`` / ``z_bot`` (from the manifest's ``z_from`` / ``z_to``) — plus the
    manifest ``stats`` and the raw ``z_from`` / ``z_to``. Rows whose grid file is
    missing on disk are skipped. Order follows the manifest (sorted by depth).

    Raises ``SliceSetError`` when there is no readable manifest.
    """
    data = read_sliceset(directory)
    base = directory
    if os.path.isfile(base):
        base = os.path.dirname(base)
    rows: list[dict] = []
    for i, s in enumerate(data.get("slices") or []):
        grid = str(s.get("grid") or "").strip()
        if not grid:
            continue
        path = os.path.normpath(os.path.join(base, grid))
        if not os.path.isfile(path):
            continue
        try:
            z_from = float(s.get("z_from"))
            z_to = float(s.get("z_to"))
        except (TypeError, ValueError):
            continue
        z_center = s.get("z_center")
        try:
            z_center = float(z_center)
        except (TypeError, ValueError):
            z_center = 0.5 * (z_from + z_to)
        lo, hi = (z_from, z_to) if z_from <= z_to else (z_to, z_from)
        try:
            idx = int(s.get("index", i))
        except (TypeError, ValueError):
            idx = i
        row = {
            "path": path,
            "index": idx,
            "name": os.path.splitext(os.path.basename(path))[0],
            "z_center": float(z_center),
            "z_top": float(lo),
            "z_bot": float(hi),
            "z_from": float(z_from),
            "z_to": float(z_to),
        }
        stats = s.get("stats")
        if isinstance(stats, dict):
            row["stats"] = stats
        rows.append(row)
    return rows


def format_depth(z: float) -> str:
    """Profondita' per i nomi file: 2-3 decimali senza zeri inutili (0.30, 0.125, 1.00)."""
    s = f"{abs(float(z)):.3f}"
    if s.endswith("0"):
        s = s[:-1]
    return ("n" if float(z) < 0 else "") + s


def slice_file_name(index: int, z_from: float, z_to: float) -> str:
    """Nome file di una time-slice con la FINESTRA di profondita' dentro:
    ``slice_0003_0.30-0.40m.tif``. L'indice davanti tiene l'ordinamento e
    rende il nome univoco nel set; ``parse_depth_from_filename`` lo legge."""
    lo, hi = sorted((float(z_from), float(z_to)))
    return f"slice_{int(index):04d}_{format_depth(lo)}-{format_depth(hi)}m.tif"


def depth_window_for_file(path: str):
    """``(z_from, z_to)`` in metri di una time-slice, o None se non si trova.

    Ordine: ``sliceset.json`` accanto al file, tag GeoTIFF ``GPR_Z_FROM/TO``
    (viaggiano con la copia del file), poi il nome file."""
    path = str(path or "").split("|")[0].strip()
    if not path:
        return None
    folder = os.path.dirname(os.path.normpath(path))
    grid = os.path.basename(path)
    try:
        if folder and has_sliceset(folder):
            with open(sliceset_path(folder), "r", encoding="utf-8") as fh:
                data = json.load(fh)
            for s in data.get("slices") or []:
                if str(s.get("grid") or "") == grid:
                    return float(s["z_from"]), float(s["z_to"])
    except Exception:
        pass
    try:
        from osgeo import gdal
        ds = gdal.Open(path)
        if ds is not None:
            md = ds.GetMetadata() or {}
            if md.get("GPR_Z_FROM") not in (None, "", "None") and md.get("GPR_Z_TO") not in (None, "", "None"):
                return float(md["GPR_Z_FROM"]), float(md["GPR_Z_TO"])
    except Exception:
        pass
    try:
        from .project_catalog import parse_depth_from_filename
        d = parse_depth_from_filename(grid)
        if d.get("depth_from") is not None and d.get("depth_to") is not None:
            return float(d["depth_from"]), float(d["depth_to"])
    except Exception:
        pass
    return None


def has_sliceset(output_dir: str) -> bool:
    return os.path.isfile(sliceset_path(output_dir))


def read_sliceset(output_dir: str) -> dict:
    """Load ``sliceset.json`` from ``output_dir``.

    Returns the parsed manifest with ``spec`` rehydrated into a ``SliceSpec``.
    Raises ``SliceSetError`` when the file is missing or malformed.
    """
    path = output_dir
    if os.path.isdir(output_dir) or not output_dir.endswith(".json"):
        path = sliceset_path(output_dir)
    if not os.path.isfile(path):
        raise SliceSetError(f"no {MANIFEST_FILENAME} in {output_dir!r}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise SliceSetError(f"cannot read {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise SliceSetError(f"{path}: manifest root is not an object")
    if data.get("format") != FORMAT:
        raise SliceSetError(
            f"{path}: unexpected format {data.get('format')!r} (expected {FORMAT})"
        )
    if not isinstance(data.get("slices"), list):
        raise SliceSetError(f"{path}: 'slices' must be a list")

    try:
        data["spec"] = SliceSpec.from_dict(data.get("spec") or {})
    except SliceSpecError as exc:
        raise SliceSetError(f"{path}: invalid embedded spec: {exc}") from exc

    # Sort slices by depth for consumers that assume ordering.
    data["slices"].sort(key=lambda s: (float(s.get("z_center", 0.0)), int(s.get("index", 0))))
    return data
