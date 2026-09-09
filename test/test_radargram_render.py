# -*- coding: utf-8 -*-
"""Passo 9: foglia di disegno condivisa radargram_render."""

import unittest

import numpy as np

try:
    from matplotlib.figure import Figure
    HAS_MPL = hasattr(Figure, "add_subplot")
except Exception:  # pragma: no cover
    HAS_MPL = False

from radargram_render import downsample_radargram, robust_vlim, draw_radargram


class DownsampleTest(unittest.TestCase):
    def test_no_op_when_small(self):
        a = np.zeros((10, 20), dtype=np.float32)
        out = downsample_radargram(a, max_rows=700, max_cols=1400)
        self.assertEqual(out.shape, (10, 20))

    def test_strides_large(self):
        a = np.random.rand(2100, 4200).astype(np.float32)
        out = downsample_radargram(a, max_rows=700, max_cols=1400)
        self.assertLessEqual(out.shape[0], 700)
        self.assertLessEqual(out.shape[1], 1400)

    def test_empty_and_1d(self):
        self.assertEqual(downsample_radargram(np.zeros((0, 0))).size, 0)
        self.assertEqual(downsample_radargram(np.zeros(5)).shape, (5,))


class RobustVlimTest(unittest.TestCase):
    def test_symmetric(self):
        lo, hi = robust_vlim(np.array([-3.0, -1.0, 0.0, 1.0, 3.0]))
        self.assertAlmostEqual(lo, -hi)
        self.assertGreater(hi, 0.0)

    def test_all_nan(self):
        lo, hi = robust_vlim(np.full(4, np.nan))
        self.assertEqual((lo, hi), (-1.0, 1.0))


@unittest.skipUnless(HAS_MPL, "matplotlib richiesto")
class DrawRadargramTest(unittest.TestCase):
    def setUp(self):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)

    def test_draws_image_and_markers(self):
        data = np.random.rand(120, 400).astype(np.float32) - 0.5
        img, vlines = draw_radargram(
            self.ax, data, marker_traces=[0, 100, 200, 399], n_traces_full=400,
        )
        self.assertIsNotNone(img)
        self.assertEqual(len(vlines), 4)
        # asse X in tracce: xlim = (0, 400)
        self.assertAlmostEqual(self.ax.get_xlim()[0], 0.0)
        self.assertAlmostEqual(self.ax.get_xlim()[1], 400.0)

    def test_marker_x_scaled_to_metres(self):
        data = np.random.rand(50, 101).astype(np.float32)
        _, vlines = draw_radargram(
            self.ax, data, dist_max=10.0, depth_max=2.0,
            marker_traces=[0, 50, 100], n_traces_full=101,
        )
        xs = sorted(l.get_xdata()[0] for l in vlines)
        np.testing.assert_allclose(xs, [0.0, 5.0, 10.0], atol=1e-6)

    def test_empty_data_no_crash(self):
        img, vlines = draw_radargram(self.ax, np.zeros((0, 0)))
        self.assertIsNone(img)
        self.assertEqual(vlines, [])


if __name__ == "__main__":
    unittest.main()
