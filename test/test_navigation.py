# -*- coding: utf-8 -*-
"""Fase 5b: modello marker + dead-reckoning + reverse/stagger/geometria."""

import unittest

import numpy as np

from navigation import (
    Marker,
    artificial_markers,
    interval_markers,
    field_markers,
    gps_trace_markers,
    dead_reckon_along,
    dead_reckon_coords,
    reverse_along,
    stagger_shift,
    line_plan_coords,
)


class MarkerModelTest(unittest.TestCase):
    def test_marker_validation(self):
        m = Marker(3.0, 1.5, kind="artificial")
        self.assertEqual(m.trace_idx, 3)
        with self.assertRaises(ValueError):
            Marker(0, kind="bogus")

    def test_artificial_markers_default_count(self):
        # 10 m line, 201 scans -> 11 marks at 0..10 m
        ms = artificial_markers(201, 10.0)
        self.assertEqual(len(ms), 11)
        self.assertEqual(ms[0].trace_idx, 0)
        self.assertEqual(ms[-1].trace_idx, 200)
        self.assertAlmostEqual(ms[-1].pos_m, 10.0)
        self.assertTrue(all(m.kind == "artificial" for m in ms))

    def test_artificial_markers_explicit_count(self):
        ms = artificial_markers(100, 5.0, n_marks=6)
        self.assertEqual(len(ms), 6)
        self.assertAlmostEqual(ms[2].pos_m, 2.0)

    def test_artificial_markers_degenerate(self):
        self.assertEqual(artificial_markers(1, 10.0), [])
        self.assertEqual(artificial_markers(100, 0.0), [])

    def test_interval_markers_and_leftover(self):
        ms, leftover = interval_markers(201, 10, unit_m=1.0)
        self.assertEqual(ms[0].trace_idx, 0)
        self.assertEqual(ms[1].trace_idx, 10)
        self.assertAlmostEqual(ms[1].pos_m, 1.0)
        self.assertEqual(ms[-1].trace_idx, 200)
        self.assertEqual(leftover, 0)
        ms2, lo2 = interval_markers(105, 10)
        self.assertEqual(ms2[-1].trace_idx, 100)
        self.assertEqual(lo2, 4)

    def test_field_markers_from_flags(self):
        ms = field_markers([0, 48, 100, 152], length_m=3.0)
        self.assertEqual([m.trace_idx for m in ms], [0, 48, 100, 152])
        self.assertAlmostEqual(ms[0].pos_m, 0.0)
        self.assertAlmostEqual(ms[-1].pos_m, 3.0)
        self.assertAlmostEqual(ms[1].pos_m, 1.0)
        # no length -> unit spacing
        ms2 = field_markers([5, 9, 14], unit_m=0.5)
        self.assertAlmostEqual(ms2[2].pos_m, 1.0)

    def test_gps_trace_markers(self):
        ms = gps_trace_markers([(0, 100.0, 200.0, 5.0), (50, 110.0, 205.0)])
        self.assertEqual(ms[0].coord, (100.0, 200.0, 5.0))
        self.assertEqual(ms[1].coord, (110.0, 205.0, 0.0))
        self.assertTrue(all(m.kind == "gps" for m in ms))


class DeadReckonTest(unittest.TestCase):
    def test_along_linear_between_marks(self):
        ms = artificial_markers(101, 10.0, n_marks=11)  # mark every 10 traces / 1 m
        along = dead_reckon_along(ms, 101)
        self.assertEqual(along.shape, (101,))
        self.assertAlmostEqual(along[0], 0.0)
        self.assertAlmostEqual(along[100], 10.0)
        self.assertAlmostEqual(along[50], 5.0, places=6)   # halfway
        self.assertTrue(np.all(np.diff(along) > 0))         # monotonic

    def test_along_extrapolates_constant(self):
        ms = [Marker(20, 2.0, kind="field"), Marker(80, 8.0, kind="field")]
        along = dead_reckon_along(ms, 100)
        self.assertAlmostEqual(along[0], 2.0)               # flat before first
        self.assertAlmostEqual(along[99], 8.0)              # flat after last
        self.assertAlmostEqual(along[50], 5.0, places=6)

    def test_along_edge_cases(self):
        self.assertTrue(np.all(np.isnan(dead_reckon_along([], 10))))
        one = dead_reckon_along([Marker(3, 4.0)], 10)
        self.assertTrue(np.allclose(one, 4.0))

    def test_coords_interpolation(self):
        ms = gps_trace_markers([(0, 0.0, 0.0, 10.0), (100, 100.0, 50.0, 20.0)])
        c = dead_reckon_coords(ms, 101)
        self.assertEqual(c.shape, (101, 3))
        np.testing.assert_allclose(c[50], [50.0, 25.0, 15.0], atol=1e-6)


class GeometryTest(unittest.TestCase):
    def test_reverse_along(self):
        a = np.linspace(0, 10, 11)
        r = reverse_along(a, 10.0)
        np.testing.assert_allclose(r, a[::-1])

    def test_stagger_shift(self):
        a = np.array([0.0, 1.0, 2.0])
        np.testing.assert_allclose(stagger_shift(a, 0.9), [0.9, 1.9, 2.9])

    def test_line_plan_coords_axis_aligned(self):
        along = np.array([0.0, 5.0, 10.0])
        e, n = line_plan_coords(100.0, 200.0, 110.0, 200.0, along)
        np.testing.assert_allclose(e, [100.0, 105.0, 110.0])
        np.testing.assert_allclose(n, [200.0, 200.0, 200.0])

    def test_line_plan_coords_ang_diagonal_and_rescale(self):
        # segment length 10 (3-4-5 *2), along goes 0..8 -> projected onto segment
        along = np.linspace(0.0, 8.0, 9)
        e, n = line_plan_coords(0.0, 0.0, 6.0, 8.0, along)
        # endpoints land on the segment ends
        np.testing.assert_allclose([e[0], n[0]], [0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose([e[-1], n[-1]], [6.0, 8.0], atol=1e-9)
        # all points on the line y = (8/6) x
        np.testing.assert_allclose(n, (8.0 / 6.0) * e, atol=1e-9)

    def test_line_plan_coords_zero_length(self):
        e, n = line_plan_coords(5.0, 5.0, 5.0, 5.0, np.array([0.0, 1.0]))
        np.testing.assert_allclose(e, [5.0, 5.0])
        np.testing.assert_allclose(n, [5.0, 5.0])


if __name__ == "__main__":
    unittest.main()
