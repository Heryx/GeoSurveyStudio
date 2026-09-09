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


def write_qml_singleband(tif_path: str, grid: np.ndarray) -> None:
    """Write .qml sidecar using a 2-98 percentile stretch of the grid."""
    finite = grid[np.isfinite(grid)]
    if finite.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.percentile(finite, 2))
        vmax = float(np.percentile(finite, 98))
        if vmax <= vmin:
            vmax = vmin + 1e-6
    qml = _QML_SINGLEBAND.format(vmin=vmin, vmax=vmax)
    qml_path = os.path.splitext(tif_path)[0] + ".qml"
    with open(qml_path, "w", encoding="utf-8") as f:
        f.write(qml)


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
