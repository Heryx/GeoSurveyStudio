# -*- coding: utf-8 -*-
"""gpr_annotation_analysis: metriche, contrasto d'ampiezza, anomalie, campioni in pianta (senza Qt)."""

import importlib
import math
import pathlib
import sys
import unittest

import numpy as np

_PLUGIN_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(_PLUGIN_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR.parent))

_AA = importlib.import_module("rasterlinker_feature_interpolation.gpr_annotation_analysis")


class SectionMetricsTest(unittest.TestCase):
    def test_point_has_no_metrics(self):
        self.assertEqual(_AA.section_metrics("point", [1.0], [0.5]), {})

    def test_line_length_and_width(self):
        m = _AA.section_metrics("line", [0.0, 3.0, 3.0], [0.0, 0.0, 4.0])
        self.assertAlmostEqual(m["length_m"], 7.0)
        self.assertAlmostEqual(m["width_m"], 3.0)
        self.assertNotIn("area_m2", m)

    def test_polygon_area_perimeter_and_width(self):
        m = _AA.section_metrics("polygon", [1.0, 4.0, 4.0, 1.0], [0.1, 0.1, 0.3, 0.3])
        self.assertAlmostEqual(m["area_m2"], 3.0 * 0.2)
        self.assertAlmostEqual(m["length_m"], 2 * (3.0 + 0.2))
        self.assertAlmostEqual(m["width_m"], 3.0)

    def test_area_is_independent_of_vertex_order(self):
        a = _AA.section_metrics("polygon", [0, 2, 2, 0], [0, 0, 1, 1])["area_m2"]
        b = _AA.section_metrics("polygon", [0, 0, 2, 2], [0, 1, 1, 0])["area_m2"]
        self.assertAlmostEqual(a, b)

    def test_degenerate_input(self):
        self.assertEqual(_AA.section_metrics("line", [1.0], [1.0]), {})
        self.assertEqual(_AA.section_metrics("line", [1.0, 2.0], [1.0]), {})


class PolygonAmplitudeTest(unittest.TestCase):
    def test_points_in_polygon(self):
        xs, zs = [0, 4, 4, 0], [0, 0, 2, 2]
        inside = _AA.points_in_polygon([1, 5, 2, -1], [1, 1, 3, 1], xs, zs)
        self.assertEqual(list(inside), [True, False, False, False])

    def test_concave_polygon(self):
        xs, zs = [0, 4, 4, 2, 0], [0, 0, 4, 1, 4]            # una "V" rovesciata: il vano e' fuori
        inside = _AA.points_in_polygon([2.0, 0.5, 3.5], [3.0, 3.0, 3.0], xs, zs)
        self.assertEqual(list(inside), [False, True, True])

    def _scene(self, inside_amp, outside_amp):
        rng = np.random.default_rng(3)
        data = rng.normal(0, outside_amp, (60, 200))
        data[20:40, 80:120] = rng.normal(0, inside_amp, (20, 40))
        dist = np.arange(200) * 0.05
        return data, dist

    def test_an_attenuated_zone_has_negative_contrast(self):
        data, dist = self._scene(inside_amp=0.25, outside_amp=1.0)
        xs = [80 * 0.05, 120 * 0.05 - 0.01, 120 * 0.05 - 0.01, 80 * 0.05]
        zs = [20 * 0.01, 20 * 0.01, 40 * 0.01, 40 * 0.01]
        out = _AA.region_amplitude(data, dist, 0.01, xs, zs)
        self.assertAlmostEqual(out["amp_rms"], 0.25, delta=0.03)
        self.assertAlmostEqual(out["contrast_db"], 20 * math.log10(0.25), delta=1.0)
        self.assertLess(out["contrast_db"], -9.0)

    def test_a_strong_zone_has_positive_contrast(self):
        data, dist = self._scene(inside_amp=2.0, outside_amp=1.0)
        xs = [4.0, 5.99, 5.99, 4.0]
        zs = [0.2, 0.2, 0.4, 0.4]
        self.assertGreater(_AA.region_amplitude(data, dist, 0.01, xs, zs)["contrast_db"], 3.0)

    def test_zone_at_the_edge_uses_the_available_side_only(self):
        data, dist = self._scene(inside_amp=1.0, outside_amp=1.0)
        out = _AA.region_amplitude(data, dist, 0.01, [0.0, 1.0, 1.0, 0.0], [0.1, 0.1, 0.3, 0.3])
        self.assertIn("contrast_db", out)

    def test_whole_profile_has_amplitude_but_no_reference(self):
        data, dist = self._scene(inside_amp=1.0, outside_amp=1.0)
        out = _AA.region_amplitude(data, dist, 0.01, [0.0, 10.0, 10.0, 0.0], [0.0, 0.0, 0.6, 0.6])
        self.assertIn("amp_rms", out)
        self.assertNotIn("contrast_db", out)

    def test_empty_or_outside_polygons(self):
        data, dist = self._scene(1.0, 1.0)
        self.assertEqual(_AA.region_amplitude(data, dist, 0.01, [50.0, 51.0, 51.0], [0.1, 0.1, 0.2]), {})
        self.assertEqual(_AA.region_amplitude(data, dist, 0.01, [1.0, 2.0], [0.1, 0.1]), {})
        self.assertEqual(_AA.region_amplitude(data, dist, 0.0, [1, 2, 2], [0.1, 0.1, 0.2]), {})
        self.assertEqual(_AA.region_amplitude(np.zeros((10, 0)), np.zeros(0), 0.01, [1, 2, 2], [0.1, 0.1, 0.2]), {})


class GroupAnomaliesTest(unittest.TestCase):
    @staticmethod
    def _f(code, stem, ch, fid, **kw):
        return dict(anomaly_id=code, profile_stem=stem, channel=ch, id=fid, **kw)

    def test_same_code_on_several_profiles_is_one_object(self):
        feats = [
            self._f("M1", "Nord2", 0, "a", class_code="struttura", depth_from=0.4, depth_to=0.6, contrast_db=-6.0),
            self._f("M1", "Nord1", 3, "b", class_code="struttura", depth_from=0.3, depth_to=0.5, contrast_db=-2.0),
            self._f("M1", "Nord1", 1, "c", class_code="riflettore", depth_from=0.35, depth_to=0.9),
            self._f("T4", "Sud1", 0, "d", class_code="iperbole"),
        ]
        rows, without = _AA.group_anomalies(feats)
        self.assertEqual(without, 0)
        self.assertEqual([r["code"] for r in rows], ["M1", "T4"])
        m1 = rows[0]
        self.assertEqual((m1["n"], m1["profiles"]), (3, 2))
        self.assertEqual(m1["classes"], ["riflettore", "struttura"])
        self.assertEqual((m1["depth_from"], m1["depth_to"]), (0.3, 0.9))
        self.assertAlmostEqual(m1["contrast_db"], -4.0)
        self.assertEqual(m1["first"], {"profile_stem": "Nord1", "channel": 1, "id": "c"})
        self.assertEqual(sorted(m1["ids"]), ["a", "b", "c"])
        self.assertIsNone(rows[1]["contrast_db"])

    def test_codes_are_sorted_naturally_and_blank_codes_are_counted(self):
        feats = [self._f(c, "P", 0, c) for c in ("M10", "M2", "A1", " ", "", None)]
        rows, without = _AA.group_anomalies(feats)
        self.assertEqual([r["code"] for r in rows], ["A1", "M2", "M10"])
        self.assertEqual(without, 3)

    def test_empty_input(self):
        self.assertEqual(_AA.group_anomalies([]), ([], 0))


class PlanSamplesTest(unittest.TestCase):
    def test_polyline_is_densified_and_interpolates_the_value(self):
        out = _AA.plan_samples([(0.0, 0.0, 1.0), (10.0, 0.0, 3.0)], closed=False, step=2.5)
        self.assertEqual(len(out), 5)
        self.assertEqual(out[0], (0.0, 0.0, 1.0))
        self.assertEqual(out[-1], (10.0, 0.0, 3.0))
        self.assertAlmostEqual(out[2][2], 2.0)

    def test_original_vertices_are_always_kept(self):
        pts = [(0.0, 0.0, 0.0), (3.0, 0.0, 1.0), (3.0, 4.0, 2.0)]
        out = _AA.plan_samples(pts, closed=False, step=100.0)
        self.assertEqual(out, pts)

    def test_ring_includes_the_closing_edge_without_repeating_the_first_point(self):
        sq = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 2.0, 0.0)]
        out = _AA.plan_samples(sq, closed=True, step=1.0)
        self.assertEqual(len(out), 8)
        self.assertNotEqual(out[-1][:2], out[0][:2])
        self.assertIn((0.0, 1.0, 0.0), [tuple(round(v, 9) for v in p) for p in out])

    def test_single_point_and_empty(self):
        self.assertEqual(_AA.plan_samples([(1.0, 2.0, 3.0)], False, 1.0), [(1.0, 2.0, 3.0)])
        self.assertEqual(_AA.plan_samples([], False, 1.0), [])


if __name__ == "__main__":
    unittest.main()
