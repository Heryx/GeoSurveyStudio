# -*- coding: utf-8 -*-
"""Time-slice per la vista 3D dei profili: scoperta dei gruppi e lettura di un raster (senza Qt).

Le time-slice stanno in ``<progetto>/Processing/timeslices/<gruppo>/`` con un ``sliceset.json`` che
dice, per ogni GeoTIFF, la finestra di profondita' (``z_from`` / ``z_to``) e la variante di origine.
Nella vista 3D una time-slice e' un piano orizzontale alla profondita' centrale della sua finestra.
"""

from __future__ import annotations

import os

import numpy as np

from .gpr_fence_geometry import downsample
from .raster_io import read_embedded_style, read_tif_singleband, slice_stretch
from .slice_manifest import SliceSetError, has_sliceset, read_sliceset, slice_catalog_from_sliceset

MAX_PLANE_PX = 2048


def timeslices_dir_for(root: str) -> str:
    """Cartella ``timeslices`` di un progetto: accetta la radice, ``Processing`` o la stessa cartella."""
    root = os.path.normpath(str(root or "").strip())
    if not root or root == ".":
        return ""
    for cand in (os.path.join(root, "Processing", "timeslices"), os.path.join(root, "timeslices"), root):
        if os.path.basename(cand).lower() == "timeslices" and os.path.isdir(cand):
            return cand
    return ""


def discover_slice_groups(root: str) -> list:
    """Gruppi di time-slice del progetto: ``[{"name", "dir", "variant", "epsg", "slices": [...]}, ...]``.

    ``slices`` ha le righe di ``slice_catalog_from_sliceset`` (``path``, ``z_from``, ``z_to``,
    ``z_center``...), in ordine di profondita'. Si saltano le cartelle senza manifest valido o
    senza raster.
    """
    base = timeslices_dir_for(root)
    if not base:
        return []
    groups = []
    for name in sorted(os.listdir(base), key=str.lower):
        folder = os.path.join(base, name)
        if not os.path.isdir(folder) or not has_sliceset(folder):
            continue
        try:
            data = read_sliceset(folder)
            rows = slice_catalog_from_sliceset(folder)
        except SliceSetError:
            continue
        if not rows:
            continue
        source = data.get("source") if isinstance(data.get("source"), dict) else {}
        groups.append({"name": name, "dir": folder, "variant": str(source.get("variant_id") or ""),
                       "epsg": data.get("crs_epsg"), "slices": sorted(rows, key=lambda r: r["z_center"])})
    return groups


def colormap_from_hex(colors):
    """``Colormap`` matplotlib da una lista di colori esadecimali (stile incorporato nel .tif), o None."""
    cols = [c for c in (colors or []) if c]
    if len(cols) < 2:
        return None
    from matplotlib.colors import LinearSegmentedColormap

    try:
        return LinearSegmentedColormap.from_list("slice_style", cols)
    except ValueError:
        return None


def load_slice(path: str, max_px: int = MAX_PLANE_PX) -> dict:
    """Legge una time-slice: ``data`` (nord in alto, ridotta), bordi del raster, EPSG e limiti di colore.

    ``corners`` = angoli esterni (alto-sinistra, alto-destra, basso-destra, basso-sinistra) nel CRS
    del raster. I limiti di colore vengono dallo stile incorporato nel .tif se c'e', altrimenti dal
    2-98 percentile.
    """
    grid, x_min, y_min, y_max, res, epsg = read_tif_singleband(path)
    n_y, n_x = grid.shape
    left, right = x_min - res / 2.0, x_min + (n_x - 1) * res + res / 2.0
    top, bottom = y_max + res / 2.0, y_min - res / 2.0
    data = downsample(np.flipud(grid), max_px, max_px)
    style = read_embedded_style(path)
    if style is not None:
        vmin, vmax, colors = style
    else:
        vmin, vmax = slice_stretch(grid)
        colors = []
    return {"data": data, "corners": [(left, top), (right, top), (right, bottom), (left, bottom)],
            "epsg": int(epsg or 0), "vmin": float(vmin), "vmax": float(vmax), "colors": list(colors or [])}


def _norm(path: str) -> str:
    return os.path.normcase(os.path.normpath(str(path or "").split("|")[0].strip()))


def _catalog_rows(root: str) -> list:
    """Time-slice del catalogo di progetto con percorso esistente e profondita' assegnata."""
    try:
        from .project_catalog import catalog_path, load_catalog
    except Exception:
        return []
    if not root or not os.path.isfile(catalog_path(root)):
        return []
    try:
        records = load_catalog(root).get("timeslices") or []
    except Exception:
        return []
    rows = []
    for rec in records:
        path = str(rec.get("project_path") or "").strip()
        if not path or not os.path.isfile(path):
            continue
        try:
            z0, z1 = float(rec["depth_from"]), float(rec["depth_to"])
        except (KeyError, TypeError, ValueError):
            continue                                    # profondita' non ancora assegnata: non si puo' collocare
        lo, hi = (z0, z1) if z0 <= z1 else (z1, z0)
        rows.append({"path": os.path.normpath(path), "z_from": lo, "z_to": hi, "z_center": 0.5 * (lo + hi),
                     "variant": str(rec.get("variant") or rec.get("variant_id") or ""),
                     "epsg": rec.get("crs_epsg") or rec.get("epsg")})
    return rows


def project_slice_groups(roots, layer_paths=()) -> list:
    """Gruppi di time-slice da tutte le fonti del progetto, senza duplicati.

    Le time-slice non stanno necessariamente sotto la cartella dell'area (nei progetti reali possono
    vivere in un albero separato), quindi si uniscono: le cartelle ``timeslices`` sotto ``roots``, i
    raster del progetto QGIS (``layer_paths``) con ``sliceset.json`` accanto e il catalogo di progetto.
    Stessa forma di ``discover_slice_groups``; si raggruppa per cartella del raster.
    """
    from .radargram_store import slice_info_for_path

    by_dir: dict = {}
    seen: set = set()

    def add(row: dict, folder: str, variant: str, epsg) -> None:
        key = _norm(row["path"])
        if key in seen:
            return
        seen.add(key)
        g = by_dir.setdefault(_norm(folder), {"name": os.path.basename(os.path.normpath(folder)) or folder,
                                              "dir": os.path.normpath(folder), "variant": variant,
                                              "epsg": epsg, "slices": []})
        if not g["variant"] and variant:
            g["variant"] = variant
        g["slices"].append(row)

    uniq_roots = []
    for r in roots:
        if r and _norm(r) not in [_norm(x) for x in uniq_roots]:
            uniq_roots.append(r)
    for root in uniq_roots:
        for g in discover_slice_groups(root):
            for row in g["slices"]:
                add(row, g["dir"], g["variant"], g["epsg"])
    for path in layer_paths:
        info = slice_info_for_path(path)
        if info is None or not os.path.isfile(str(path).split("|")[0]):
            continue
        p = os.path.normpath(str(path).split("|")[0])
        add({"path": p, "z_from": info["z_from"], "z_to": info["z_to"], "z_center": info["z_center"],
             "index": 0, "name": os.path.splitext(os.path.basename(p))[0]},
            info["dir"], info.get("variant") or "", info.get("epsg"))
    for root in uniq_roots:
        for row in _catalog_rows(root):
            add(row, os.path.dirname(row["path"]), row.get("variant") or "", row.get("epsg"))
    groups = list(by_dir.values())
    for g in groups:
        g["slices"].sort(key=lambda r: r["z_center"])
    return sorted(groups, key=lambda g: (g["name"].lower(), g["dir"]))
