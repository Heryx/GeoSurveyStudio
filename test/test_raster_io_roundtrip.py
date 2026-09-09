# -*- coding: utf-8 -*-
"""Fase 5 passo UI: round-trip write/read GeoTIFF single-band (raster_io)."""

import os
import tempfile
import unittest

import numpy as np

try:
    from osgeo import gdal  # noqa: F401
    HAS_GDAL = True
except Exception:
    HAS_GDAL = False

from raster_io import write_tif_singleband, read_tif_singleband


@unittest.skipUnless(HAS_GDAL, "osgeo/GDAL non disponibile")
class RasterIoRoundtripTest(unittest.TestCase):
    def test_write_read_preserves_grid_and_georef(self):
        rng = np.random.default_rng(0)
        grid = (rng.standard_normal((20, 30)) * 5.0)
        grid[3:6, 4:7] = np.nan                       # nodata patch
        x_min, y_min, res, epsg = 443570.0, 4548956.0, 0.10, 32633
        y_max = y_min + (grid.shape[0] - 1) * res
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "slice_0000.tif")
            write_tif_singleband(grid, p, x_min, y_min, y_max, res, epsg)
            self.assertTrue(os.path.isfile(p))
            g2, x2, ymin2, ymax2, r2, e2 = read_tif_singleband(p)

        self.assertEqual(g2.shape, grid.shape)
        self.assertAlmostEqual(x2, x_min, places=4)
        self.assertAlmostEqual(ymin2, y_min, places=4)
        self.assertAlmostEqual(ymax2, y_max, places=4)
        self.assertAlmostEqual(r2, res, places=6)
        self.assertEqual(e2, epsg)
        # NaN mask preserved; finite values match
        self.assertTrue(np.array_equal(np.isnan(g2), np.isnan(grid)))
        np.testing.assert_allclose(
            g2[np.isfinite(g2)], grid[np.isfinite(grid)], rtol=1e-5, atol=1e-4
        )

    def test_read_then_grid_filter_then_write(self):
        from grid_filters import lowpass
        grid = np.pad(np.ones((10, 10)), 4) * np.nan
        grid[4:14, 4:14] = np.arange(100).reshape(10, 10).astype(float)
        x_min, y_min, res = 0.0, 0.0, 1.0
        y_max = y_min + (grid.shape[0] - 1) * res
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.tif")
            write_tif_singleband(grid, p, x_min, y_min, y_max, res, 0)
            g2, xm, ym, yM, r, e = read_tif_singleband(p)
            filt = lowpass(g2, 3)
            q = os.path.join(d, "s_filt.tif")
            write_tif_singleband(filt, q, xm, ym, yM, r, e or 0)
            g3, *_ = read_tif_singleband(q)
        self.assertEqual(g3.shape, grid.shape)
        # nodata border still nodata, interior finite
        self.assertTrue(np.isnan(g3[0, 0]))
        self.assertTrue(np.isfinite(g3[9, 9]))


if __name__ == "__main__":
    unittest.main()
