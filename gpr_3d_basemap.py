# -*- coding: utf-8 -*-
"""Base satellitare/aerea (tile XYZ) per il 3D viewer.

Riusa le connessioni XYZ gia' configurate dall'utente in QGIS (Browser >
XYZ Tiles), lette dal registro QSettings condiviso con QGIS stesso
(gruppo ``qgis/connections-xyz``) -- nessun URL o provider incorporato nel
plugin, nessuna nuova dipendenza (urllib per lo scaricamento, GDAL per la
decodifica/riproiezione, gia' dipendenze di questo plugin/QGIS).

Le tile XYZ sono sempre in EPSG:3857 (Web Mercator); questo modulo scarica
un mosaico che copre il bbox richiesto (in un CRS arbitrario), lo riproietta
al CRS del volume 3D e lo restituisce come array RGB pronto per essere usato
come texture di un piano nella scena PyVista.
"""

from __future__ import annotations

import math
import time
import urllib.request

import numpy as np

_WEBMERC_R = 6378137.0
_WEBMERC_HALF = math.pi * _WEBMERC_R  # ~20037508.342789244
_TILE_PX = 256

_TEXTURE_CACHE: dict = {}
_TEXTURE_CACHE_MAX = 6


# ---------------------------------------------------------------------------
# Connessioni XYZ configurate in QGIS
# ---------------------------------------------------------------------------

def list_xyz_connections() -> dict:
    """Connessioni XYZ Tiles dell'utente, lette dal QSettings di QGIS.

    Ritorna ``{nome: {"url": ..., "zmin": int, "zmax": int}}``, solo le
    connessioni con un URL non vuoto.
    """
    try:
        from qgis.PyQt.QtCore import QSettings
    except Exception:
        return {}

    settings = QSettings()
    out: dict = {}
    settings.beginGroup("qgis/connections-xyz")
    try:
        names = settings.childGroups()
        for name in names:
            settings.beginGroup(name)
            try:
                url = str(settings.value("url", "") or "").strip()
                if not url:
                    continue
                try:
                    zmin = int(settings.value("zmin", -1))
                except Exception:
                    zmin = -1
                try:
                    zmax = int(settings.value("zmax", -1))
                except Exception:
                    zmax = -1
                out[str(name)] = {"url": url, "zmin": zmin, "zmax": zmax}
            finally:
                settings.endGroup()
    finally:
        settings.endGroup()
    return out


# ---------------------------------------------------------------------------
# Geometria tile Web Mercator
# ---------------------------------------------------------------------------

def _meters_to_tile_xy(mx: float, my: float, zoom: int) -> tuple[float, float]:
    n = 2 ** int(zoom)
    tile_size = (2 * _WEBMERC_HALF) / n
    tx = (mx + _WEBMERC_HALF) / tile_size
    ty = (_WEBMERC_HALF - my) / tile_size
    return tx, ty


def _tile_bounds_meters(tx: int, ty: int, zoom: int) -> tuple[float, float, float, float]:
    n = 2 ** int(zoom)
    tile_size = (2 * _WEBMERC_HALF) / n
    xmin = tx * tile_size - _WEBMERC_HALF
    xmax = (tx + 1) * tile_size - _WEBMERC_HALF
    ymax = _WEBMERC_HALF - ty * tile_size
    ymin = _WEBMERC_HALF - (ty + 1) * tile_size
    return xmin, ymin, xmax, ymax


def _choose_zoom(width_m: float, height_m: float, max_px: int, zmin: int = 0, zmax: int = 19) -> int:
    zmin = max(0, int(zmin))
    zmax = max(zmin, int(zmax))
    width_m = max(float(width_m), 1e-6)
    height_m = max(float(height_m), 1e-6)
    for z in range(zmax, zmin - 1, -1):
        tile_size = (2 * _WEBMERC_HALF) / (2 ** z)
        px_per_m = _TILE_PX / tile_size
        w_px = width_m * px_per_m
        h_px = height_m * px_per_m
        if max(w_px, h_px) <= max_px:
            return z
    return zmin


# ---------------------------------------------------------------------------
# Scaricamento e decodifica tile
# ---------------------------------------------------------------------------

def _fetch_tile_bytes(url_template: str, z: int, x: int, y: int, timeout: float = 10.0) -> bytes:
    url = (
        url_template.replace("{z}", str(int(z)))
        .replace("{x}", str(int(x)))
        .replace("{y}", str(int(y)))
    )
    req = urllib.request.Request(url, headers={"User-Agent": "rasterlinker-gpr-plugin/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _decode_tile_to_array(data: bytes) -> np.ndarray:
    from osgeo import gdal

    path = f"/vsimem/gpr3d_tile_{time.time_ns()}_{id(data)}.bin"
    gdal.FileFromMemBuffer(path, data)
    ds = None
    try:
        ds = gdal.Open(path)
        if ds is None:
            raise ValueError("tile non decodificabile (formato immagine non supportato)")
        arr = ds.ReadAsArray()
        if arr is None:
            raise ValueError("tile vuota")
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=0)
        arr = np.transpose(arr[:3], (1, 2, 0))
        return np.ascontiguousarray(arr, dtype=np.uint8)
    finally:
        ds = None
        try:
            gdal.Unlink(path)
        except Exception:
            pass


def build_mosaic_3857(
    url_template: str,
    west: float,
    south: float,
    east: float,
    north: float,
    zoom: int,
    timeout: float = 10.0,
    max_tiles: int = 64,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Scarica e ricompone in un unico mosaico numpy le tile che coprono il
    bbox richiesto (in metri EPSG:3857) al livello di zoom dato."""
    tx0f, ty0f = _meters_to_tile_xy(west, north, zoom)
    tx1f, ty1f = _meters_to_tile_xy(east, south, zoom)
    n = 2 ** int(zoom)
    tx0 = max(0, min(int(math.floor(min(tx0f, tx1f))), n - 1))
    tx1 = max(0, min(int(math.floor(max(tx0f, tx1f))), n - 1))
    ty0 = max(0, min(int(math.floor(min(ty0f, ty1f))), n - 1))
    ty1 = max(0, min(int(math.floor(max(ty0f, ty1f))), n - 1))

    n_cols = tx1 - tx0 + 1
    n_rows = ty1 - ty0 + 1
    if n_cols * n_rows > max_tiles:
        raise ValueError(
            f"Area troppo estesa per lo zoom {zoom}: servirebbero {n_cols * n_rows} tile "
            f"(massimo {max_tiles}). Riduci l'estensione o la risoluzione richiesta."
        )

    mosaic = np.zeros((n_rows * _TILE_PX, n_cols * _TILE_PX, 3), dtype=np.uint8)
    for row, ty in enumerate(range(ty0, ty1 + 1)):
        for col, tx in enumerate(range(tx0, tx1 + 1)):
            data = _fetch_tile_bytes(url_template, zoom, tx, ty, timeout=timeout)
            tile_arr = _decode_tile_to_array(data)
            th = min(_TILE_PX, tile_arr.shape[0])
            tw = min(_TILE_PX, tile_arr.shape[1])
            mosaic[row * _TILE_PX: row * _TILE_PX + th, col * _TILE_PX: col * _TILE_PX + tw, :] = (
                tile_arr[:th, :tw, :3]
            )

    xmin, _, _, ymax = _tile_bounds_meters(tx0, ty0, zoom)
    _, ymin, xmax, _ = _tile_bounds_meters(tx1, ty1, zoom)
    return mosaic, (xmin, ymin, xmax, ymax)


# ---------------------------------------------------------------------------
# Riproiezione EPSG:3857 -> CRS del volume
# ---------------------------------------------------------------------------

def _reproject_mosaic(
    mosaic_rgb: np.ndarray,
    mosaic_bounds_3857: tuple[float, float, float, float],
    dst_epsg: int,
    dst_bbox: tuple[float, float, float, float],
    out_px: int = 1024,
) -> np.ndarray:
    from osgeo import gdal, osr

    h, w = mosaic_rgb.shape[:2]
    xmin, ymin, xmax, ymax = mosaic_bounds_3857
    px_w = (xmax - xmin) / max(w, 1)
    px_h = (ymax - ymin) / max(h, 1)

    mem_drv = gdal.GetDriverByName("MEM")
    src_ds = mem_drv.Create("", w, h, 3, gdal.GDT_Byte)
    src_ds.SetGeoTransform((xmin, px_w, 0.0, ymax, 0.0, -px_h))
    src_srs = osr.SpatialReference()
    src_srs.ImportFromEPSG(3857)
    src_ds.SetProjection(src_srs.ExportToWkt())
    for b in range(3):
        src_ds.GetRasterBand(b + 1).WriteArray(mosaic_rgb[:, :, b])

    dxmin, dymin, dxmax, dymax = dst_bbox
    dst_w = max(2, int(out_px))
    aspect = (dymax - dymin) / max((dxmax - dxmin), 1e-9)
    dst_h = max(2, int(round(dst_w * aspect)))
    dst_ds = mem_drv.Create("", dst_w, dst_h, 3, gdal.GDT_Byte)
    dst_px_w = (dxmax - dxmin) / dst_w
    dst_px_h = (dymax - dymin) / dst_h
    dst_ds.SetGeoTransform((dxmin, dst_px_w, 0.0, dymax, 0.0, -dst_px_h))
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromEPSG(int(dst_epsg))
    dst_ds.SetProjection(dst_srs.ExportToWkt())

    gdal.ReprojectImage(
        src_ds, dst_ds, src_srs.ExportToWkt(), dst_srs.ExportToWkt(), gdal.GRA_Bilinear
    )
    out = dst_ds.ReadAsArray()
    if out.ndim == 2:
        out = np.stack([out, out, out], axis=0)
    out = np.transpose(out[:3], (1, 2, 0))
    return np.ascontiguousarray(out, dtype=np.uint8)


# ---------------------------------------------------------------------------
# API di alto livello
# ---------------------------------------------------------------------------

def fetch_basemap_texture(
    url_template: str,
    bbox_dst: tuple[float, float, float, float],
    dst_epsg: int,
    max_px: int = 1536,
    zmin: int = 0,
    zmax: int = 19,
    timeout: float = 10.0,
    use_cache: bool = True,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Scarica un mosaico satellitare per ``bbox_dst`` (xmin, ymin, xmax,
    ymax) nel CRS ``dst_epsg`` del volume, e lo riproietta a quel CRS.

    Ritorna ``(rgb_array, bbox_dst)``: l'array copre esattamente il bbox
    richiesto, pronto per essere applicato come texture a un piano
    posizionato su quei 4 angoli.
    """
    bbox_dst = tuple(float(v) for v in bbox_dst)
    cache_key = (
        str(url_template),
        int(dst_epsg),
        tuple(round(v, 3) for v in bbox_dst),
        int(max_px),
    )
    if use_cache and cache_key in _TEXTURE_CACHE:
        return _TEXTURE_CACHE[cache_key]

    from osgeo import osr

    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromEPSG(int(dst_epsg))
    web_srs = osr.SpatialReference()
    web_srs.ImportFromEPSG(3857)
    for srs in (dst_srs, web_srs):
        try:
            srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        except Exception:
            pass
    xform = osr.CoordinateTransformation(dst_srs, web_srs)

    xmin, ymin, xmax, ymax = bbox_dst
    corners = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    xs, ys = [], []
    for (x, y) in corners:
        wx, wy, _ = xform.TransformPoint(x, y)
        xs.append(wx)
        ys.append(wy)
    west, east = min(xs), max(xs)
    south, north = min(ys), max(ys)

    zoom = _choose_zoom(east - west, north - south, max_px, zmin=zmin, zmax=zmax)
    mosaic, mosaic_bounds = build_mosaic_3857(url_template, west, south, east, north, zoom, timeout=timeout)
    out = _reproject_mosaic(mosaic, mosaic_bounds, dst_epsg, bbox_dst, out_px=max_px)

    result = (out, bbox_dst)
    if use_cache:
        if len(_TEXTURE_CACHE) >= _TEXTURE_CACHE_MAX:
            try:
                _TEXTURE_CACHE.pop(next(iter(_TEXTURE_CACHE)))
            except StopIteration:
                pass
        _TEXTURE_CACHE[cache_key] = result
    return result


def clear_texture_cache():
    _TEXTURE_CACHE.clear()
