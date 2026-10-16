# -*- coding: utf-8 -*-
"""gpr_fence_geometry: tende verticali, texture e incroci tra profili (senza Qt)."""

import importlib
import pathlib
import sys
import unittest

import numpy as np

_PLUGIN_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(_PLUGIN_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR.parent))

_F = importlib.import_module("rasterlinker_feature_interpolation.gpr_fence_geometry")
_G = importlib.import_module("rasterlinker_feature_interpolation.gpr_ert_georef")


class TrackVerticesTest(unittest.TestCase):
    def test_keeps_valid_traces_and_decimates(self):
        n = 1000
        d = np.arange(n) * 0.05
        e = 500000.0 + d
        nn = np.full(n, 10.0)
        e[3] = np.nan
        xy, dd = _F.track_vertices(e, nn, d, max_pts=100)
        self.assertLessEqual(len(dd), 100)
        self.assertEqual(float(dd[0]), 0.0)
        self.assertAlmostEqual(float(dd[-1]), d[-1])
        self.assertTrue(np.all(np.diff(dd) > 0))
        self.assertTrue(np.isfinite(xy).all())

    def test_unsorted_and_repeated_distances(self):
        xy, dd = _F.track_vertices([0, 2, 1, 1], [0, 0, 0, 0], [0.0, 2.0, 1.0, 1.0])
        self.assertEqual(dd.tolist(), [0.0, 1.0, 2.0])

    def test_too_few_traces(self):
        with self.assertRaises(ValueError):
            _F.track_vertices([1.0, np.nan], [1.0, 1.0], [0.0, 1.0])


class PolylineVerticesTest(unittest.TestCase):
    def test_control_points_are_kept(self):
        g = _G.ErtGeoref([(0, 0, 0), (10, 10, 0), (20, 10, 10)])
        xy, d = _F.polyline_vertices(g, 0.0, 20.0, step_m=3.0)
        self.assertIn(10.0, d.tolist())
        self.assertEqual(xy[np.argmin(np.abs(d - 10.0))].tolist(), [10.0, 0.0])


class CurtainTest(unittest.TestCase):
    def setUp(self):
        self.xy = np.array([[100.0, 200.0], [110.0, 200.0], [120.0, 205.0]])
        self.d = np.array([0.0, 10.0, 21.2])

    def test_geometry_and_orientation(self):
        c = _F.curtain(self.xy, self.d, (0.0, 21.2), 0.0, 3.0, origin=(100.0, 200.0))
        p = c["points"]
        self.assertEqual(p.shape, (6, 3))
        np.testing.assert_allclose(p[:3, 2], 0.0)             # in alto: superficie
        np.testing.assert_allclose(p[3:, 2], -3.0)            # in basso: 3 m sotto
        np.testing.assert_allclose(p[:3, :2], p[3:, :2])      # a piombo
        self.assertEqual(p[0, :2].tolist(), [0.0, 0.0])        # origine traslata
        self.assertEqual(len(c["faces"]), 2 * 5)               # 2 quad da 4 vertici + conteggio

    def test_texture_coordinates(self):
        c = _F.curtain(self.xy, self.d, (0.0, 21.2), 0.0, 3.0)
        tc = c["tcoords"]
        np.testing.assert_allclose(tc[:3, 1], 1.0)
        np.testing.assert_allclose(tc[3:, 1], 0.0)
        self.assertAlmostEqual(float(tc[0, 0]), 0.0)
        self.assertAlmostEqual(float(tc[2, 0]), 1.0)

    def test_texture_span_larger_than_vertices(self):
        c = _F.curtain(self.xy, self.d, (-10.0, 30.0), 0.5, 4.0)
        self.assertAlmostEqual(float(c["tcoords"][0, 0]), 0.25)
        np.testing.assert_allclose(c["points"][:3, 2], -0.5)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            _F.curtain(self.xy[:1], self.d[:1], (0, 1), 0, 1)
        with self.assertRaises(ValueError):
            _F.curtain(self.xy, self.d, (1.0, 1.0), 0, 1)


class TextureTest(unittest.TestCase):
    def test_nan_is_transparent_and_range_maps_to_colours(self):
        a = np.array([[0.0, 1.0], [np.nan, 0.5]])
        t = _F.rgba_from_array(a, "viridis", 0.0, 1.0)
        self.assertEqual(t.shape, (2, 2, 4))
        self.assertEqual(t.dtype, np.uint8)
        self.assertEqual(int(t[1, 0, 3]), 0)
        self.assertEqual(int(t[0, 0, 3]), 255)
        self.assertNotEqual(t[0, 0, :3].tolist(), t[0, 1, :3].tolist())

    def test_flat_range_does_not_divide_by_zero(self):
        t = _F.rgba_from_array(np.ones((2, 2)), "gray", 1.0, 1.0)
        self.assertEqual(t.shape, (2, 2, 4))

    def test_downsample_respects_limits_and_ignores_nan(self):
        a = np.ones((1000, 5000), dtype=np.float32)
        a[0, 0] = np.nan
        out = _F.downsample(a, max_rows=500, max_cols=1000)
        self.assertLessEqual(out.shape[0], 500)
        self.assertLessEqual(out.shape[1], 1000)
        self.assertTrue(np.isfinite(out).all())
        self.assertAlmostEqual(float(out[0, 0]), 1.0)

    def test_downsample_small_array_untouched(self):
        a = np.arange(6, dtype=np.float32).reshape(2, 3)
        np.testing.assert_array_equal(_F.downsample(a), a)


class CrossingsTest(unittest.TestCase):
    def test_orthogonal_lines_cross_once_with_both_distances(self):
        a_xy = np.array([[0.0, 0.0], [40.0, 0.0]])
        a_d = np.array([0.0, 40.0])
        b_xy = np.array([[12.0, -20.0], [12.0, 20.0]])
        b_d = np.array([0.0, 40.0])
        hits = _F.crossings(a_xy, a_d, b_xy, b_d)
        self.assertEqual(len(hits), 1)
        da, db, x, y = hits[0]
        self.assertAlmostEqual(da, 12.0)
        self.assertAlmostEqual(db, 20.0)
        self.assertAlmostEqual(x, 12.0)
        self.assertAlmostEqual(y, 0.0)

    def test_parallel_and_disjoint_lines_do_not_cross(self):
        a = (np.array([[0.0, 0.0], [10.0, 0.0]]), np.array([0.0, 10.0]))
        self.assertEqual(_F.crossings(*a, np.array([[0.0, 5.0], [10.0, 5.0]]), np.array([0.0, 10.0])), [])
        self.assertEqual(_F.crossings(*a, np.array([[20.0, -5.0], [20.0, 5.0]]), np.array([0.0, 10.0])), [])

    def test_crossing_at_a_shared_vertex_is_reported_once(self):
        a_xy = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
        a_d = np.array([0.0, 10.0, 20.0])
        b_xy = np.array([[10.0, -5.0], [10.0, 5.0]])
        self.assertEqual(len(_F.crossings(a_xy, a_d, b_xy, np.array([0.0, 10.0]))), 1)

    def test_real_p9_geometry_versus_a_gpr_line(self):
        # P9 (ERT) corre quasi a nord; un profilo GPR lo attraversa ad angolo retto
        g = _G.ErtGeoref.from_endpoints((451270.0, 4049823.4), (451276.0, 4049894.1), 71.0)
        exy, ed = _F.polyline_vertices(g, 0.0, 71.0)
        gxy = np.array([[451250.0, 4049860.0], [451300.0, 4049860.0]])
        hits = _F.crossings(gxy, np.array([0.0, 50.0]), exy, ed)
        self.assertEqual(len(hits), 1)
        self.assertAlmostEqual(hits[0][1], 36.4 * 1.0, delta=1.0)     # a meta' circa della linea ERT


if __name__ == "__main__":
    unittest.main()
