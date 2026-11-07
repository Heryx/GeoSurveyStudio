# -*- coding: utf-8 -*-
"""Unit tests for the pure tile-math helpers of the 3D viewer basemap.

Non tocca rete o GDAL/QGIS (import differiti dentro le funzioni che li
usano), cosi' da poter girare in un ambiente Python qualsiasi.
"""

import math
import unittest

from gpr_3d_basemap import (
    _WEBMERC_HALF,
    _choose_zoom,
    _meters_to_tile_xy,
    _tile_bounds_meters,
)


class TileMathTest(unittest.TestCase):
    def test_origin_is_tile_0_0_at_zoom_0(self):
        tx, ty = _meters_to_tile_xy(0.0, 0.0, 0)
        self.assertAlmostEqual(tx, 0.5, places=6)
        self.assertAlmostEqual(ty, 0.5, places=6)

    def test_tile_bounds_roundtrip(self):
        zoom = 5
        for tx in (0, 3, 10, 31):
            for ty in (0, 3, 10, 31):
                xmin, ymin, xmax, ymax = _tile_bounds_meters(tx, ty, zoom)
                self.assertLess(xmin, xmax)
                self.assertLess(ymin, ymax)
                cx = 0.5 * (xmin + xmax)
                cy = 0.5 * (ymin + ymax)
                tx2, ty2 = _meters_to_tile_xy(cx, cy, zoom)
                self.assertAlmostEqual(tx2, tx + 0.5, places=4)
                self.assertAlmostEqual(ty2, ty + 0.5, places=4)

    def test_world_extent_matches_web_mercator_half(self):
        xmin, ymin, xmax, ymax = _tile_bounds_meters(0, 0, 0)
        self.assertAlmostEqual(xmin, -_WEBMERC_HALF, places=2)
        self.assertAlmostEqual(ymax, _WEBMERC_HALF, places=2)

    def test_choose_zoom_increases_with_smaller_area(self):
        z_large = _choose_zoom(200000.0, 200000.0, max_px=1024, zmin=0, zmax=19)
        z_small = _choose_zoom(200.0, 200.0, max_px=1024, zmin=0, zmax=19)
        self.assertLess(z_large, z_small)

    def test_choose_zoom_respects_max_px_bound(self):
        max_px = 1024
        zoom = _choose_zoom(5000.0, 5000.0, max_px=max_px, zmin=0, zmax=19)
        tile_size = (2 * _WEBMERC_HALF) / (2 ** zoom)
        px = 5000.0 * (256.0 / tile_size)
        self.assertLessEqual(px, max_px + 1e-6)

    def test_choose_zoom_clamped_to_zmax_when_area_tiny(self):
        zoom = _choose_zoom(0.001, 0.001, max_px=1024, zmin=0, zmax=10)
        self.assertEqual(zoom, 10)

    def test_choose_zoom_clamped_to_zmin_when_area_huge(self):
        zoom = _choose_zoom(1e9, 1e9, max_px=256, zmin=2, zmax=19)
        self.assertEqual(zoom, 2)


if __name__ == "__main__":
    unittest.main()
