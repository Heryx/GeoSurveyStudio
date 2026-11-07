# -*- coding: utf-8 -*-
"""Shared GeoTIFF / QML sidecar writers for the slice generators.

Single source of truth for the raster I/O previously duplicated across
``gpr_las_slicer.py``, ``gpr_csv_slicer.py`` and ``gpr_ogpr_slicer.py``.

GDAL / OSR are imported lazily inside each function so this module can be
imported in headless environments without the ``osgeo`` package.
"""

from __future__ import annotations

import os
import textwrap

import numpy as np

NODATA = -9999.0


# ---------------------------------------------------------------------------
# GeoTIFF writers
# ---------------------------------------------------------------------------


def make_geotransform(x_min, y_max, resolution):
    """Pixel-centre aligned GDAL geotransform for a north-up grid."""
    return (x_min - resolution / 2.0, resolution, 0.0,
            y_max + resolution / 2.0, 0.0, -resolution)


def set_projection(ds, epsg):
    """Assign an EPSG projection to an open GDAL dataset."""
    from osgeo import osr
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(int(epsg))
    ds.SetProjection(srs.ExportToWkt())


def write_tif_singleband(grid, out_path, x_min, y_min, y_max, resolution, epsg):
    """Write a single-band float32 GeoTIFF (NaN -> NODATA, rows flipped)."""
    from osgeo import gdal
    n_y, n_x = grid.shape
    data = np.where(np.isnan(grid), NODATA, grid).astype(np.float32)
    ds = gdal.GetDriverByName("GTiff").Create(
        out_path, n_x, n_y, 1, gdal.GDT_Float32,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    ds.SetGeoTransform(make_geotransform(x_min, y_max, resolution))
    if epsg:
        set_projection(ds, epsg)
    bnd = ds.GetRasterBand(1)
    bnd.WriteArray(np.flipud(data))
    bnd.SetNoDataValue(NODATA)
    bnd.FlushCache()
    ds.FlushCache()
    ds = None


def read_tif_singleband(path):
    """Read band 1 of a GeoTIFF back into the internal (y-ascending) grid.

    Returns ``(grid, x_min, y_min, y_max, resolution, epsg)`` with NODATA -> NaN
    and rows flipped so ``grid[0]`` is the southernmost row (matches
    ``write_tif_singleband``'s input convention). ``epsg`` is 0 when unknown.
    """
    from osgeo import gdal, osr

    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        raise IOError(f"impossibile aprire il raster: {path}")
    bnd = ds.GetRasterBand(1)
    arr = np.asarray(bnd.ReadAsArray(), dtype=np.float64)
    nod = bnd.GetNoDataValue()
    if nod is not None:
        arr = np.where(arr == nod, np.nan, arr)
    arr = np.where(arr == NODATA, np.nan, arr)
    arr = np.flipud(arr)                       # north-up -> y-ascending

    gt = ds.GetGeoTransform() or (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)
    res = float(abs(gt[1])) or 1.0
    n_y, n_x = arr.shape
    x_min = float(gt[0] + res / 2.0)
    y_top = float(gt[3] - res / 2.0)           # centre of the top (north) pixel
    y_max = y_top
    y_min = float(y_top - (n_y - 1) * res)

    epsg = 0
    try:
        wkt = ds.GetProjection()
        if wkt:
            srs = osr.SpatialReference()
            srs.ImportFromWkt(wkt)
            code = srs.GetAuthorityCode(None)
            if code:
                epsg = int(code)
    except Exception:
        epsg = 0
    ds = None
    return arr, x_min, y_min, y_max, res, epsg


def write_tif_rgb(r_grid, g_grid, b_grid, out_path, x_min, y_min, y_max,
                  resolution, epsg):
    """Write a 3-band float32 GeoTIFF and return per-band (min, max) ranges."""
    from osgeo import gdal
    n_y, n_x = r_grid.shape

    def _prep(arr):
        return np.where(np.isnan(arr), NODATA, arr).astype(np.float32)

    bands_data = [_prep(r_grid), _prep(g_grid), _prep(b_grid)]
    ds = gdal.GetDriverByName("GTiff").Create(
        out_path, n_x, n_y, 3, gdal.GDT_Float32,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    ds.SetGeoTransform(make_geotransform(x_min, y_max, resolution))
    if epsg:
        set_projection(ds, epsg)
    for idx, data in enumerate(bands_data, start=1):
        bnd = ds.GetRasterBand(idx)
        bnd.WriteArray(np.flipud(data))
        bnd.SetNoDataValue(NODATA)
        bnd.FlushCache()
    ds.FlushCache()
    ds = None

    def _range(arr):
        v = arr[arr != NODATA]
        return (float(v.min()), float(v.max())) if v.size else (0.0, 1.0)

    mn_r, mx_r = _range(bands_data[0])
    mn_g, mx_g = _range(bands_data[1])
    mn_b, mx_b = _range(bands_data[2])
    return mn_r, mx_r, mn_g, mx_g, mn_b, mx_b


# ---------------------------------------------------------------------------
# QML sidecars
# ---------------------------------------------------------------------------

_QML_SINGLEBAND = """\
<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.0" styleCategories="AllStyleCategories">
  <pipe>
    <provider><resampling enabled="false" maxOversampling="2"
      zoomedInResamplingMethod="nearestNeighbour"
      zoomedOutResamplingMethod="nearestNeighbour"/></provider>
    <rasterrenderer type="singlebandgray" opacity="1" alphaBand="-1"
                    grayBand="1" gradient="BlackToWhite">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>CumulativeCut</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Estimated</statAccuracy>
        <cumulativeCutLower>0.02</cumulativeCutLower>
        <cumulativeCutUpper>0.98</cumulativeCutUpper>
      </minMaxOrigin>
      <contrastEnhancement>
        <minValue>{vmin}</minValue>
        <maxValue>{vmax}</maxValue>
        <algorithm>StretchToMinimumMaximum</algorithm>
      </contrastEnhancement>
    </rasterrenderer>
    <brightnesscontrast brightness="0" contrast="0" gamma="1"/>
    <huesaturation saturation="0" grayscaleMode="0" colorizeOn="0"/>
    <rasterresampler maxOversampling="2"/>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
"""

_QML_MULTIBAND = textwrap.dedent("""\
    <!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
    <qgis version="3.0" styleCategories="AllStyleCategories">
      <flags>
        <Identifiable>1</Identifiable><Removable>1</Removable>
        <Searchable>0</Searchable>
      </flags>
      <pipe>
        <provider>
          <resampling enabled="false" maxOversampling="2"
                      zoomedInResamplingMethod="nearestNeighbour"
                      zoomedOutResamplingMethod="nearestNeighbour"/>
        </provider>
        <rasterrenderer type="multibandcolor" opacity="1" alphaBand="-1"
                        redBand="1" greenBand="2" blueBand="3" nodataColor="">
          <rasterTransparency/>
          <minMaxOrigin>
            <limits>MinMax</limits><extent>WholeRaster</extent>
            <statAccuracy>Estimated</statAccuracy>
            <cumulativeCutLower>0.02</cumulativeCutLower>
            <cumulativeCutUpper>0.98</cumulativeCutUpper>
            <stdDevFactor>2</stdDevFactor>
          </minMaxOrigin>
          <redContrastEnhancement>
            <minValue>{min_r}</minValue><maxValue>{max_r}</maxValue>
            <algorithm>StretchToMinimumMaximum</algorithm>
          </redContrastEnhancement>
          <greenContrastEnhancement>
            <minValue>{min_g}</minValue><maxValue>{max_g}</maxValue>
            <algorithm>StretchToMinimumMaximum</algorithm>
          </greenContrastEnhancement>
          <blueContrastEnhancement>
            <minValue>{min_b}</minValue><maxValue>{max_b}</maxValue>
            <algorithm>StretchToMinimumMaximum</algorithm>
          </blueContrastEnhancement>
        </rasterrenderer>
        <brightnesscontrast brightness="0" contrast="0" gamma="1"/>
        <huesaturation saturation="0" grayscaleMode="0" colorizeOn="0"
                       colorizeRed="255" colorizeGreen="128" colorizeBlue="128"
                       colorizeStrength="100" invertColors="0"/>
        <rasterresampler maxOversampling="2"/>
        <resamplingStage>resamplingFilter</resamplingStage>
      </pipe>
      <blendMode>0</blendMode>
    </qgis>
""")


_QML_PSEUDOCOLOR = """\
<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.0" styleCategories="AllStyleCategories">
  <pipe>
    <provider><resampling enabled="false" maxOversampling="2"
      zoomedInResamplingMethod="nearestNeighbour"
      zoomedOutResamplingMethod="nearestNeighbour"/></provider>
    <rasterrenderer type="singlebandpseudocolor" band="1" opacity="1" alphaBand="-1"
                    classificationMin="{vmin}" classificationMax="{vmax}" nodataColor="">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>None</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Estimated</statAccuracy>
        <cumulativeCutLower>0.02</cumulativeCutLower>
        <cumulativeCutUpper>0.98</cumulativeCutUpper>
        <stdDevFactor>2</stdDevFactor>
      </minMaxOrigin>
      <rastershader>
        <colorrampshader colorRampType="INTERPOLATED" classificationMode="1" clip="0"
                         minimumValue="{vmin}" maximumValue="{vmax}" labelPrecision="4">
{items}
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
    <brightnesscontrast brightness="0" contrast="0" gamma="1"/>
    <huesaturation saturation="0" grayscaleMode="0" colorizeOn="0"/>
    <rasterresampler maxOversampling="2"/>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
"""


def colormap_hex_stops(name: str, n: int = 11):
    """``n`` evenly spaced ``#rrggbb`` colours of a matplotlib colormap.

    Returns ``None`` when the colormap cannot be resolved (matplotlib missing or
    unknown name), so callers can fall back to the grayscale style.
    """
    try:
        import matplotlib
        try:
            cm = matplotlib.colormaps[str(name)]
        except Exception:
            from matplotlib import cm as _cm
            cm = _cm.get_cmap(str(name))
        return [
            "#%02x%02x%02x" % tuple(int(round(255 * c)) for c in cm(i / (n - 1))[:3])
            for i in range(int(n))
        ]
    except Exception:
        return None


def slice_stretch(grid: np.ndarray, clip_lo: float = 2.0, clip_hi: float = 98.0):
    """``(vmin, vmax)`` from percentile clipping, always ``vmax > vmin``."""
    finite = grid[np.isfinite(grid)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.clip(clip_lo, 0.0, 49.0))
    hi = float(np.clip(clip_hi, 51.0, 100.0))
    vmin = float(np.percentile(finite, lo))
    vmax = float(np.percentile(finite, hi))
    if not vmax > vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def _fixed_limits(style: dict):
    """``(vmin, vmax)`` when the style carries finite fixed limits, else None."""
    try:
        lo = style.get("vmin")
        hi = style.get("vmax")
        if lo is None or hi is None:
            return None
        lo, hi = float(lo), float(hi)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return None
    if not hi > lo:
        hi = lo + 1e-6
    return lo, hi


def write_qml_singleband(tif_path: str, grid: np.ndarray, style: dict | None = None) -> None:
    """Write the .qml sidecar of a slice GeoTIFF.

    ``style`` (optional) is ``{"cmap": <matplotlib name>, "clip_lo": p, "clip_hi": p}``:
    the stretch is the ``clip_lo``-``clip_hi`` percentile range (default 2-98) and,
    when ``cmap`` resolves, the raster is rendered with that colour ramp so QGIS
    shows the slice the way it looked in the viewer. Without a usable ``cmap`` the
    classic grayscale style is written.

    ``style["vmin"]`` / ``style["vmax"]`` (both, finite) fix the limits in data
    units instead: every slice of a set then shares one colour scale.
    """
    style = dict(style or {})
    fixed = _fixed_limits(style)
    if fixed is not None:
        vmin, vmax = fixed
    else:
        vmin, vmax = slice_stretch(
            grid, float(style.get("clip_lo", 2.0)), float(style.get("clip_hi", 98.0))
        )
    stops = colormap_hex_stops(style["cmap"]) if style.get("cmap") else None
    if stops:
        n = len(stops)
        items = "\n".join(
            '          <item alpha="255" value="{v!r}" label="{v:.4g}" color="{c}"/>'.format(
                v=float(vmin + (vmax - vmin) * i / (n - 1)), c=c
            )
            for i, c in enumerate(stops)
        )
        qml = _QML_PSEUDOCOLOR.format(vmin=vmin, vmax=vmax, items=items)
    else:
        qml = _QML_SINGLEBAND.format(vmin=vmin, vmax=vmax)
    qml_path = os.path.splitext(tif_path)[0] + ".qml"
    with open(qml_path, "w", encoding="utf-8") as f:
        f.write(qml)
    # Lo stile viaggia anche dentro il GeoTIFF: il .qml si perde quando il file
    # viene copiato nel progetto o spostato.
    embed_style_in_tif(tif_path, vmin, vmax, stops)


_STYLE_KEYS = ("GPR_STYLE_VMIN", "GPR_STYLE_VMAX", "GPR_STYLE_COLORS")


def embed_style_in_tif(tif_path: str, vmin: float, vmax: float, colors=None) -> bool:
    """Store the display stretch (and colour stops) as GDAL metadata inside the
    GeoTIFF itself. ``colors`` = list of ``#rrggbb`` (None -> grayscale)."""
    try:
        from osgeo import gdal
        ds = gdal.Open(str(tif_path), gdal.GA_Update)
        if ds is None:
            return False
        ds.SetMetadataItem("GPR_STYLE_VMIN", repr(float(vmin)))
        ds.SetMetadataItem("GPR_STYLE_VMAX", repr(float(vmax)))
        ds.SetMetadataItem("GPR_STYLE_COLORS", ",".join(colors) if colors else "")
        ds.FlushCache()
        ds = None
        return True
    except Exception:
        return False


def embed_tif_metadata(tif_path: str, items: dict) -> bool:
    """Write string metadata items (``None`` values skipped) into the GeoTIFF
    itself, so what produced the values travels with the file."""
    try:
        from osgeo import gdal
        ds = gdal.Open(str(tif_path), gdal.GA_Update)
        if ds is None:
            return False
        for key, value in (items or {}).items():
            if value is not None:
                ds.SetMetadataItem(str(key), str(value))
        ds.FlushCache()
        ds = None
        return True
    except Exception:
        return False


def read_embedded_style(tif_path: str):
    """``(vmin, vmax, colors)`` embedded by :func:`embed_style_in_tif`, or None."""
    try:
        from osgeo import gdal
        ds = gdal.Open(str(tif_path), gdal.GA_ReadOnly)
        if ds is None:
            return None
        md = ds.GetMetadata() or {}
        ds = None
        if "GPR_STYLE_VMIN" not in md or "GPR_STYLE_VMAX" not in md:
            return None
        cols = [c for c in (md.get("GPR_STYLE_COLORS") or "").split(",") if c]
        return float(md["GPR_STYLE_VMIN"]), float(md["GPR_STYLE_VMAX"]), cols
    except Exception:
        return None


def apply_embedded_style(layer) -> bool:
    """Apply the style embedded in the layer's GeoTIFF (pseudocolor ramp or
    grayscale stretch). Returns False when the file carries no style."""
    try:
        src = str(layer.source()).split("|")[0]
        info = read_embedded_style(src)
        if not info:
            return False
        vmin, vmax, cols = info
        if not vmax > vmin:
            vmax = vmin + 1e-6
        from qgis.core import (
            QgsColorRampShader, QgsRasterShader, QgsSingleBandPseudoColorRenderer,
        )
        from PyQt5.QtGui import QColor
        if not cols:
            cols = ["#000000", "#ffffff"]
        n = len(cols)
        items = [
            QgsColorRampShader.ColorRampItem(
                float(vmin + (vmax - vmin) * i / (n - 1)), QColor(c),
                "{:.4g}".format(vmin + (vmax - vmin) * i / (n - 1)),
            )
            for i, c in enumerate(cols)
        ]
        ramp = QgsColorRampShader(vmin, vmax)
        ramp.setColorRampType(QgsColorRampShader.Interpolated)
        ramp.setClip(False)
        ramp.setColorRampItemList(items)
        shader = QgsRasterShader()
        shader.setRasterShaderFunction(ramp)
        renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader)
        renderer.setClassificationMin(vmin)
        renderer.setClassificationMax(vmax)
        layer.setRenderer(renderer)
        return True
    except Exception:
        return False


def write_qml_multiband(tif_path, mn_r, mx_r, mn_g, mx_g, mn_b, mx_b):
    """Write a multibandcolor .qml sidecar; return its path."""
    qml = _QML_MULTIBAND.format(
        min_r=mn_r, max_r=mx_r,
        min_g=mn_g, max_g=mx_g,
        min_b=mn_b, max_b=mx_b,
    )
    qml_path = os.path.splitext(tif_path)[0] + ".qml"
    with open(qml_path, "w", encoding="utf-8") as f:
        f.write(qml)
    return qml_path
