# -*- coding: utf-8 -*-
"""gpr_ert_georef: georiferimento a 2 o N punti, controlli di scala, KML e riproiezione."""

import importlib
import math
import os
import pathlib
import sys
import tempfile
import unittest

import numpy as np

_PLUGIN_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(_PLUGIN_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR.parent))

_G = importlib.import_module("rasterlinker_feature_interpolation.gpr_ert_georef")

_REAL_KML = pathlib.Path(r"C:\TIPAZA\GEOPHYSICS\ERT\ERT_TIPASA\ERT P8 à P10\doc.kml")

_KML = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
<Placemark><name>P1</name><Point><coordinates>2.0,36.0,0</coordinates></Point></Placemark>
<Placemark><name>P1</name><Point><coordinates>2.0,36.001,0</coordinates></Point></Placemark>
<Placemark><name>p1</name><LineString><coordinates>
2.0,36.0,0 2.0,36.0005,0 2.0,36.001,0</coordinates></LineString></Placemark>
<Placemark><name>P2</name><Point><coordinates>3.0,37.0,0</coordinates></Point></Placemark>
<Placemark><Point><coordinates>9,9,0</coordinates></Point></Placemark>
</Document></kml>"""


def _flat_projection(lon, lat):
    """Proiezione equirettangolare in metri attorno a (2.0, 36.0): basta per i test del nucleo."""
    return ((lon - 2.0) * 111320.0 * math.cos(math.radians(36.0)), (lat - 36.0) * 110574.0)


class GeorefCoreTest(unittest.TestCase):
    def test_two_points_are_a_straight_line(self):
        g = _G.ErtGeoref.from_endpoints((100.0, 200.0), (100.0, 270.0), 70.0, "EPSG:32631")
        self.assertEqual(g.locate(0.0), (100.0, 200.0))
        x, y = g.locate(35.0)
        self.assertAlmostEqual(x, 100.0)
        self.assertAlmostEqual(y, 235.0)
        self.assertEqual(g.check(), [])

    def test_locate_array_and_extrapolation(self):
        g = _G.ErtGeoref.from_endpoints((0.0, 0.0), (10.0, 0.0), 10.0)
        xs, ys = g.locate(np.array([-2.0, 5.0, 12.0]))
        np.testing.assert_allclose(xs, [-2.0, 5.0, 12.0])
        np.testing.assert_allclose(ys, [0.0, 0.0, 0.0])

    def test_polyline_follows_the_bend(self):
        # 10 m verso est, poi 10 m verso nord: a 15 m si e' a meta' del secondo tratto
        g = _G.ErtGeoref([(0, 0, 0), (10, 10, 0), (20, 10, 10)])
        x, y = g.locate(15.0)
        self.assertAlmostEqual(x, 10.0)
        self.assertAlmostEqual(y, 5.0)

    def test_from_polyline_scales_to_section_length(self):
        g = _G.ErtGeoref.from_polyline([(0, 0), (30, 0), (30, 40)], length_m=70.0)
        self.assertAlmostEqual(g.points[-1].distance, 70.0)
        self.assertAlmostEqual(g.points[1].distance, 30.0)
        self.assertEqual(g.check(), [])

    def test_scale_warning_when_lengths_disagree(self):
        g = _G.ErtGeoref.from_endpoints((0, 0), (0, 80), 70.0)
        warns = g.check()
        self.assertEqual(len(warns), 1)
        self.assertIn("stirata", warns[0])
        short = _G.ErtGeoref.from_endpoints((0, 0), (0, 60), 70.0)
        self.assertIn("compressa", short.check()[0])

    def test_one_point_per_electrode_is_exact(self):
        pts = [(float(i), 5.0 * i, 0.0) for i in range(72)]
        g = _G.ErtGeoref(pts)
        self.assertEqual(g.segment_scales()[0], 5.0)
        x, _ = g.locate(10.5)
        self.assertAlmostEqual(x, 52.5)

    def test_invalid_points(self):
        with self.assertRaises(_G.ErtGeorefError):
            _G.ErtGeoref([(0, 0, 0)])
        with self.assertRaises(_G.ErtGeorefError):
            _G.ErtGeoref([(0, 0, 0), (0, 1, 1)])
        with self.assertRaises(_G.ErtGeorefError):
            _G.ErtGeoref([(0, 5, 5), (10, 5, 5)])

    def test_flipped_reverses_direction(self):
        g = _G.ErtGeoref.from_endpoints((0, 0), (0, 70), 70.0)
        f = g.flipped(70.0)
        self.assertEqual(f.locate(0.0), (0.0, 70.0))
        self.assertEqual(f.locate(70.0), (0.0, 0.0))

    def test_dict_roundtrip(self):
        g = _G.ErtGeoref([(0, 1, 2), (10, 3, 4), (25, 5, 9)], "EPSG:32631")
        h = _G.ErtGeoref.from_dict(g.to_dict())
        self.assertEqual(h.crs_authid, "EPSG:32631")
        self.assertEqual(h.points, g.points)
        with self.assertRaises(_G.ErtGeorefError):
            _G.ErtGeoref.from_dict({"points": [{"distance": 0}]})

    def test_project_is_the_inverse_of_locate(self):
        g = _G.ErtGeoref([(0, 0, 0), (10, 10, 0), (20, 10, 10)])
        d = np.array([0.0, 4.0, 10.0, 15.0, 20.0])
        x, y = g.locate(d)
        back, off = g.project(x, y)
        np.testing.assert_allclose(back, d, atol=1e-9)
        np.testing.assert_allclose(off, 0.0, atol=1e-9)

    def test_project_reports_the_lateral_offset(self):
        g = _G.ErtGeoref.from_endpoints((0, 0), (50, 0), 50.0)
        d, off = g.project(np.array([10.0, 30.0]), np.array([3.0, -4.0]))
        np.testing.assert_allclose(d, [10.0, 30.0])
        np.testing.assert_allclose(off, [3.0, 4.0])

    def test_project_extends_beyond_the_ends(self):
        g = _G.ErtGeoref.from_endpoints((0, 0), (50, 0), 50.0)
        d, _ = g.project(np.array([-5.0, 60.0]), np.array([0.0, 0.0]))
        np.testing.assert_allclose(d, [-5.0, 60.0])

    def test_project_scalar(self):
        g = _G.ErtGeoref.from_endpoints((0, 0), (50, 0), 50.0)
        d, off = g.project(20.0, 2.0)
        self.assertAlmostEqual(d, 20.0)
        self.assertAlmostEqual(off, 2.0)

    def test_polyline_for_the_map(self):
        g = _G.ErtGeoref.from_endpoints((0, 0), (10, 0), 10.0)
        line = g.polyline(step_m=2.5)
        self.assertEqual(len(line), 5)
        self.assertEqual(line[0], (0.0, 0.0))
        self.assertEqual(line[-1], (10.0, 0.0))


class ManualPointsTest(unittest.TestCase):
    def test_parse_xy_accepts_the_usual_spellings(self):
        self.assertEqual(_G.parse_xy("451270.5 4049823.4"), (451270.5, 4049823.4))
        self.assertEqual(_G.parse_xy("451270.5; 4049823.4"), (451270.5, 4049823.4))
        self.assertEqual(_G.parse_xy("451270,5; 4049823,4"), (451270.5, 4049823.4))
        self.assertEqual(_G.parse_xy("  1  2  "), (1.0, 2.0))

    def test_parse_xy_rejects_what_is_not_a_pair(self):
        for bad in ("", None, "12", "1 2 3", "a; b", "1;2;3"):
            self.assertIsNone(_G.parse_xy(bad), bad)

    def test_two_points_make_a_straight_line(self):
        g = _G.georef_from_xy_points([(0, 0), (0, 71)], 71.0, "EPSG:32631")
        self.assertEqual(len(g.points), 2)
        self.assertEqual(g.check(), [])

    def test_n_points_follow_the_polyline_with_proportional_distance(self):
        g = _G.georef_from_xy_points([(0, 0), (30, 0), (30, 40)], 70.0)
        self.assertAlmostEqual(g.points[1].distance, 30.0)
        self.assertAlmostEqual(g.points[2].distance, 70.0)

    def test_invert_reverses_the_order(self):
        g = _G.georef_from_xy_points([(0, 0), (0, 50)], 50.0, invert=True)
        self.assertEqual(g.locate(0.0), (0.0, 50.0))
        self.assertEqual(g.locate(50.0), (0.0, 0.0))


class KmlTest(unittest.TestCase):
    def _profiles(self):
        fd, path = tempfile.mkstemp(suffix=".kml")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(_KML)
        try:
            return _G.read_kml_profiles(path)
        finally:
            os.remove(path)

    def test_groups_line_and_points_by_name_ignoring_case(self):
        profs = self._profiles()
        self.assertEqual([p.name for p in profs], ["P1", "P2"])
        p1 = profs[0]
        self.assertEqual(len(p1.points), 2)
        self.assertEqual(len(p1.line), 3)

    def test_endpoints_prefer_the_line(self):
        a, b = self._profiles()[0].endpoints()
        self.assertEqual(a, (2.0, 36.0))
        self.assertEqual(b, (2.0, 36.001))

    def test_single_point_profile_has_no_endpoints(self):
        with self.assertRaises(_G.ErtGeorefError):
            self._profiles()[1].endpoints()

    def test_kml_to_georef_with_a_projection(self):
        a, b = self._profiles()[0].endpoints()
        pa, pb = _flat_projection(*a), _flat_projection(*b)
        g = _G.ErtGeoref.from_endpoints(pa, pb, 110.574)
        self.assertEqual(g.check(), [])


@unittest.skipUnless(_REAL_KML.exists(), "KML di prova non presente")
class RealTipazaTest(unittest.TestCase):
    def test_p9_length_matches_the_71_m_section(self):
        profs = {p.name.lower(): p for p in _G.read_kml_profiles(str(_REAL_KML))}
        self.assertEqual({"p8", "p9", "p10"}, set(profs))
        a, b = profs["p9"].endpoints()
        g = _G.ErtGeoref.from_endpoints(_flat_projection_at(a), _flat_projection_at(b), 71.0)
        # 72 elettrodi a 1 m = 71 m: la linea del KML deve essere entro la tolleranza
        self.assertAlmostEqual(g.map_length(), 71.0, delta=71.0 * _G.SCALE_TOLERANCE)
        self.assertEqual(g.check(), [])


def _flat_projection_at(lonlat):
    lon, lat = lonlat
    return ((lon - 2.455) * 111320.0 * math.cos(math.radians(36.59)),
            (lat - 36.59) * 110900.0)


class TransformerTest(unittest.TestCase):
    def test_utm_zone_31n_for_tipaza(self):
        try:
            import qgis.core  # noqa: F401
        except ImportError:
            self.skipTest("QGIS non disponibile in questo interprete")
        tr = _G.lonlat_transformer("EPSG:32631")
        x, y = tr(2.455219387383009, 36.59264668942385)
        # Meridiano centrale della zona 31 = 3 E, falso est 500 km: a 2.455 E si sta ~48.7 km a ovest
        # (0.545 gradi * 111.32 km * cos(36.6)). y ~ 36.59 gradi * ~110.9 km.
        self.assertAlmostEqual(x, 500000.0 - 0.5448 * 111320.0 * math.cos(math.radians(36.59)),
                               delta=500.0)
        self.assertAlmostEqual(y, 4050000.0, delta=5000.0)
        x2, y2 = tr(2.455281963592022, 36.59328703539677)
        self.assertAlmostEqual(math.hypot(x2 - x, y2 - y), 71.4, delta=1.0)


if __name__ == "__main__":
    unittest.main()
