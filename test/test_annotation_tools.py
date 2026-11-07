# -*- coding: utf-8 -*-
"""gpr_annotation_tools: aggancio al picco e tracciamento di un riflettore (senza Qt)."""

import importlib
import pathlib
import sys
import unittest

import numpy as np

_PLUGIN_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(_PLUGIN_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR.parent))

_AT = importlib.import_module("rasterlinker_feature_interpolation.gpr_annotation_tools")


def _wavelet(n, center, width=3.0, amp=1.0):
    x = np.arange(n, dtype=np.float64)
    return amp * np.exp(-0.5 * ((x - center) / width) ** 2)


class LocalExtremumTest(unittest.TestCase):
    def test_snaps_to_the_nearest_positive_peak(self):
        tr = _wavelet(100, 40)
        self.assertEqual(_AT.local_extremum_row(tr, 37, 6), 40)
        self.assertEqual(_AT.local_extremum_row(tr, 44, 6), 40)

    def test_follows_the_sign_of_the_starting_sample(self):
        tr = _wavelet(100, 40) - _wavelet(100, 50)         # picco a 40, ventre a 50
        self.assertEqual(_AT.local_extremum_row(tr, 48, 8), 50)    # parte da un valore negativo: cerca il ventre
        self.assertEqual(_AT.local_extremum_row(tr, 42, 8), 40)
        self.assertEqual(_AT.local_extremum_row(tr, 48, 8, sign=1), 40)

    def test_outside_the_window_or_flat_signal_returns_the_row_unchanged(self):
        tr = _wavelet(100, 40)
        self.assertEqual(_AT.local_extremum_row(tr, 60, 5), 60)
        self.assertEqual(_AT.local_extremum_row(np.zeros(50), 20, 6), 20)
        self.assertEqual(_AT.local_extremum_row(np.linspace(0, 1, 50), 20, 6), 20)   # monotono

    def test_nearest_wins_over_stronger_and_ties_go_to_the_stronger(self):
        tr = _wavelet(100, 35, amp=0.5) + _wavelet(100, 45, amp=1.0)
        self.assertEqual(_AT.local_extremum_row(tr, 36, 12), 35)
        self.assertEqual(_AT.local_extremum_row(tr, 40, 12), 45)    # equidistanti: vince il piu' forte

    def test_degenerate_inputs_do_not_crash(self):
        self.assertEqual(_AT.local_extremum_row([], 3, 4), 0)
        self.assertEqual(_AT.local_extremum_row([1.0, 2.0], 5, 4), 1)
        self.assertEqual(_AT.local_extremum_row(_wavelet(10, 5), 99, 3), 9)
        self.assertEqual(_AT.local_extremum_row(_wavelet(50, 20), 18, 0), 18)


class TraceBetweenTest(unittest.TestCase):
    @staticmethod
    def _sloped(rows=120, traces=60, r_start=30, slope=0.5):
        data = np.zeros((rows, traces))
        for t in range(traces):
            data[:, t] = _wavelet(rows, r_start + slope * t, width=2.5)
        return data

    def test_follows_a_dipping_reflector(self):
        data = self._sloped()
        path = _AT.trace_between(data, 0, 30, 40, 50, half_window=4, step=5)
        self.assertEqual(path[0], (0, 30))
        self.assertEqual(path[-1], (40, 50))
        for t, r in path[1:-1]:
            self.assertAlmostEqual(r, 30 + 0.5 * t, delta=1.01)

    def test_path_is_sampled_every_step_traces_and_ends_exactly_at_the_target(self):
        data = self._sloped()
        path = _AT.trace_between(data, 0, 30, 23, 41, half_window=4, step=5)
        self.assertEqual([t for t, _r in path], [0, 5, 10, 15, 20, 23])

    def test_works_right_to_left(self):
        data = self._sloped()
        path = _AT.trace_between(data, 40, 50, 0, 30, half_window=4, step=10)
        self.assertEqual(path[0], (40, 50))
        self.assertEqual(path[-1], (0, 30))
        self.assertEqual([t for t, _r in path], [40, 30, 20, 10, 0])

    def test_does_not_jump_to_a_neighbouring_reflector(self):
        data = self._sloped()
        for t in range(data.shape[1]):                       # secondo riflettore piu' forte, 12 campioni sotto
            data[:, t] += _wavelet(data.shape[0], 42 + 0.5 * t, width=2.5, amp=1.8)
        path = _AT.trace_between(data, 0, 30, 40, 50, half_window=4, step=5)
        for t, r in path[1:-1]:
            self.assertAlmostEqual(r, 30 + 0.5 * t, delta=1.01)

    def test_drift_is_limited_around_the_straight_line(self):
        data = np.zeros((100, 30))
        data[70, :] = 1.0                                    # un riflettore lontano dalla retta
        path = _AT.trace_between(data, 0, 30, 29, 30, half_window=50, step=1, max_drift=4)
        for _t, r in path:
            self.assertLessEqual(abs(r - 30), 4)

    def test_same_trace_and_out_of_range_inputs(self):
        data = self._sloped()
        self.assertEqual(_AT.trace_between(data, 5, 20, 5, 20, 4), [(5, 20)])
        self.assertEqual(_AT.trace_between(data, 5, 20, 5, 25, 4), [(5, 20), (5, 25)])
        path = _AT.trace_between(data, -3, 500, 999, -4, 4, step=20)
        self.assertEqual(path[0], (0, 119))
        self.assertEqual(path[-1], (59, 0))


if __name__ == "__main__":
    unittest.main()
