# -*- coding: utf-8 -*-
"""Fase 5 §5.4: geometria di griglie (grid_geometry.py)."""

import unittest

import numpy as np

from grid_geometry import rotate_translate, append_grids, edge_match


def _ramp(n_y=20, n_x=30):
    yy, xx = np.mgrid[0:n_y, 0:n_x]
    return (xx + 10.0 * yy).astype(np.float64)


class RotateTranslateTest(unittest.TestCase):
    def test_zero_rotation_translate_origin(self):
        g = {"array": _ramp(), "x0": 0.0, "y0": 0.0, "cell": 1.0}
        out = rotate_translate(g, 0.0, out_origin=(100.0, 200.0), mode="translate_origin")
        self.assertAlmostEqual(out["x0"], 100.0, places=5)
        self.assertAlmostEqual(out["y0"], 200.0, places=5)
        self.assertEqual(out["cell"], 1.0)
        # content preserved (same shape, same values up to resample)
        self.assertEqual(out["array"].shape, g["array"].shape)
        np.testing.assert_allclose(
            np.nan_to_num(out["array"]), g["array"].astype(np.float32), rtol=1e-4, atol=1e-3
        )

    def test_fix_origin_places_sw_corner(self):
        g = {"array": _ramp(10, 10), "x0": 0.0, "y0": 0.0, "cell": 2.0}
        out = rotate_translate(g, 90.0, out_origin=(1000.0, 2000.0), mode="fix_origin")
        # after a 90° rotation the source pixel [0,0] must land at out_origin;
        # it is the nearest finite cell to that world point
        oy = int(round((2000.0 - out["y0"]) / out["cell"]))
        ox = int(round((1000.0 - out["x0"]) / out["cell"]))
        oy = np.clip(oy, 0, out["array"].shape[0] - 1)
        ox = np.clip(ox, 0, out["array"].shape[1] - 1)
        self.assertAlmostEqual(
            float(out["array"][oy, ox]), float(g["array"][0, 0]), delta=1.0
        )

    def test_360_rotation_is_identity(self):
        g = {"array": _ramp(16, 16), "x0": 5.0, "y0": 7.0, "cell": 1.0}
        out = rotate_translate(g, 360.0, out_origin=(5.0, 7.0), mode="fix_origin")
        # central region round-trips closely
        a = g["array"]
        b = out["array"][: a.shape[0], : a.shape[1]]
        inner = slice(3, -3)
        np.testing.assert_allclose(
            np.nan_to_num(b[inner, inner]), a[inner, inner].astype(np.float32),
            rtol=1e-3, atol=0.2,
        )

    def test_rotation_changes_footprint(self):
        g = {"array": np.ones((10, 40)), "x0": 0.0, "y0": 0.0, "cell": 1.0}
        out = rotate_translate(g, 45.0)
        # a rotated wide rectangle needs a bigger bounding raster
        self.assertGreater(out["array"].shape[0], 10)
        self.assertGreater(out["array"].shape[1], 10)


class AppendGridsTest(unittest.TestCase):
    def test_append_side_by_side(self):
        a = {"array": np.full((10, 10), 1.0), "x0": 0.0, "y0": 0.0, "cell": 1.0}
        b = {"array": np.full((10, 10), 2.0), "x0": 10.0, "y0": 0.0, "cell": 1.0}
        out = append_grids([a, b])
        self.assertEqual(out["array"].shape, (10, 20))
        self.assertAlmostEqual(out["x0"], 0.0)
        self.assertAlmostEqual(float(np.nanmean(out["array"][:, :10])), 1.0)
        self.assertAlmostEqual(float(np.nanmean(out["array"][:, 10:])), 2.0)

    def test_append_overlap_mean(self):
        a = {"array": np.full((10, 10), 1.0), "x0": 0.0, "y0": 0.0, "cell": 1.0}
        b = {"array": np.full((10, 10), 3.0), "x0": 5.0, "y0": 0.0, "cell": 1.0}
        out = append_grids([a, b], overlap="mean")
        self.assertEqual(out["array"].shape, (10, 15))
        # overlap columns 5..9 -> mean of 1 and 3 = 2
        self.assertAlmostEqual(float(np.nanmean(out["array"][:, 5:10])), 2.0)
        self.assertAlmostEqual(float(np.nanmean(out["array"][:, :5])), 1.0)
        self.assertAlmostEqual(float(np.nanmean(out["array"][:, 10:])), 3.0)

    def test_append_cell_mismatch_raises(self):
        a = {"array": np.zeros((4, 4)), "x0": 0.0, "y0": 0.0, "cell": 1.0}
        b = {"array": np.zeros((4, 4)), "x0": 4.0, "y0": 0.0, "cell": 2.0}
        with self.assertRaises(ValueError):
            append_grids([a, b])

    def test_append_offset_grid_gap_is_nan(self):
        a = {"array": np.full((5, 5), 1.0), "x0": 0.0, "y0": 0.0, "cell": 1.0}
        b = {"array": np.full((5, 5), 2.0), "x0": 20.0, "y0": 0.0, "cell": 1.0}
        out = append_grids([a, b])
        self.assertEqual(out["array"].shape, (5, 25))
        self.assertTrue(np.all(np.isnan(out["array"][:, 10:15])))


class EdgeMatchTest(unittest.TestCase):
    def test_edge_match_removes_seam_jump(self):
        a = {"array": np.full((10, 10), 5.0), "x0": 0.0, "y0": 0.0, "cell": 1.0}
        b = {"array": np.full((10, 10), 8.0), "x0": 10.0, "y0": 0.0, "cell": 1.0}
        fixed = edge_match(a, b, axis="x", overlap_cells=3)
        self.assertAlmostEqual(fixed["shift"], -3.0, places=5)
        self.assertAlmostEqual(float(np.nanmean(fixed["array"])), 5.0, places=5)


if __name__ == "__main__":
    unittest.main()
