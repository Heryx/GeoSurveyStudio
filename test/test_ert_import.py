# -*- coding: utf-8 -*-
"""gpr_ert_import: lettura delle sezioni ERT 2D di ResIPY e griglia regolare (senza Qt)."""

import importlib
import os
import pathlib
import sys
import tempfile
import unittest

import numpy as np

_PLUGIN_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(_PLUGIN_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR.parent))

_ERT = importlib.import_module("rasterlinker_feature_interpolation.gpr_ert_import")

# Export reale di prova (Tipaza, P9, 72 elettrodi a 1 m): salta se non e' su questo PC.
_REAL = pathlib.Path(r"C:\TIPAZA\GEOPHYSICS\ERT\ERT_TIPASA\ERT P8 à P10\P9_inverted1.csv")


def _write(text, suffix=".csv"):
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def _synthetic_csv(z_offset=0.0):
    """Celle su una griglia 1 m x 0.5 m, resistivita' crescente con la profondita'."""
    lines = ["x,y,z,param,elm_id,region,cellType,Resistivity(ohm.m),Resistivity(log10)"]
    k = 0
    for xi in np.arange(0.5, 20.0, 1.0):
        for zi in np.arange(-0.25, -5.0, -0.5):
            k += 1
            res = 10.0 ** (1.0 + abs(zi) / 5.0)
            lines.append(f"{xi},0.0,{zi + z_offset},{k},{k},1.0,5.0,{res},{np.log10(res)}")
    return "\n".join(lines)


class LoadSectionTest(unittest.TestCase):
    def test_reads_resipy_header_and_columns(self):
        path = _write(_synthetic_csv())
        try:
            sec = _ERT.load_ert_section(path)
        finally:
            os.remove(path)
        self.assertEqual(len(sec.x), 20 * 10)
        self.assertIn("resistivity", sec.values)
        self.assertIn("log10_res", sec.values)
        self.assertAlmostEqual(sec.length_m, 19.0)

    def test_flat_survey_is_depth_mode(self):
        path = _write(_synthetic_csv())
        try:
            sec = _ERT.load_ert_section(path)
        finally:
            os.remove(path)
        self.assertEqual(sec.z_mode, "depth")
        self.assertAlmostEqual(float(sec.depth.min()), 0.25)

    def test_positive_elevations_are_elevation_mode(self):
        path = _write(_synthetic_csv(z_offset=120.0))
        try:
            sec = _ERT.load_ert_section(path)
        finally:
            os.remove(path)
        self.assertEqual(sec.z_mode, "elevation")
        self.assertAlmostEqual(float(sec.depth.min()), 0.0)

    def test_explicit_z_mode_wins_over_auto(self):
        path = _write(_synthetic_csv(z_offset=120.0))
        try:
            sec = _ERT.load_ert_section(path, z_mode="depth")
        finally:
            os.remove(path)
        self.assertEqual(sec.z_mode, "depth")

    def test_headerless_three_columns(self):
        path = _write("1.0 -0.5 30.0\n2.0 -0.5 40.0\n1.5 -1.5 50.0\n", ".xyz")
        try:
            sec = _ERT.load_ert_section(path)
        finally:
            os.remove(path)
        self.assertEqual(len(sec.x), 3)
        self.assertAlmostEqual(float(sec.values["log10_res"][0]), np.log10(30.0))

    def test_rejects_3d_data(self):
        path = _write("x,y,z,Resistivity(ohm.m)\n0,0,-1,10\n1,5,-1,12\n2,9,-1,15\n")
        try:
            with self.assertRaises(_ERT.ErtImportError):
                _ERT.load_ert_section(path)
        finally:
            os.remove(path)

    def test_missing_resistivity_column(self):
        path = _write("x,y,z,foo\n0,0,-1,10\n")
        try:
            with self.assertRaises(_ERT.ErtImportError):
                _ERT.load_ert_section(path)
        finally:
            os.remove(path)

    def test_log10_only_file_gets_resistivity(self):
        path = _write("x,y,z,Resistivity(log10)\n0,0,-1,2.0\n1,0,-1,1.0\n0.5,0,-2,1.5\n")
        try:
            sec = _ERT.load_ert_section(path)
        finally:
            os.remove(path)
        np.testing.assert_allclose(sec.values["resistivity"], [100.0, 10.0, 10 ** 1.5])


class GridTest(unittest.TestCase):
    def setUp(self):
        path = _write(_synthetic_csv())
        try:
            self.sec = _ERT.load_ert_section(path)
        finally:
            os.remove(path)

    def test_grid_axes_run_top_to_bottom(self):
        g = _ERT.to_grid(self.sec, dx=0.5, dz=0.25)
        self.assertEqual(g.data.shape, (len(g.z_axis), len(g.x_axis)))
        self.assertTrue(np.all(np.diff(g.z_axis) < 0))
        self.assertAlmostEqual(float(g.depth_axis[0]), 0.25)

    def test_values_follow_the_synthetic_model(self):
        g = _ERT.to_grid(self.sec, dx=0.5, dz=0.25)
        iz = int(np.argmin(np.abs(g.depth_axis - 2.0)))
        ix = int(np.argmin(np.abs(g.x_axis - 10.0)))
        self.assertAlmostEqual(float(g.data[iz, ix]), 1.0 + 2.0 / 5.0, places=1)

    def test_no_data_outside_the_coverage(self):
        # copertura a trapezio: in basso a destra le celle mancano (come il fondo di un'inversione)
        keep = (self.sec.depth < 1.0) | (self.sec.x < 10.0)
        sec = _ERT.ErtSection(x=self.sec.x[keep], z=self.sec.z[keep],
                              values={k: v[keep] for k, v in self.sec.values.items()},
                              z_mode=self.sec.z_mode)
        g = _ERT.to_grid(sec, dx=0.5, dz=0.25)
        corner = g.data[-1, -1]
        self.assertTrue(np.isnan(corner))
        self.assertTrue(np.isfinite(g.data[0, 0]))
        self.assertTrue(np.isfinite(g.data).any())

    def test_default_spacing_is_positive_and_fine(self):
        step = _ERT.default_spacing(self.sec)
        self.assertGreater(step, 0.0)
        self.assertLessEqual(step, 0.5)

    def test_unknown_quantity(self):
        with self.assertRaises(_ERT.ErtImportError):
            _ERT.to_grid(self.sec, quantity="nope")


@unittest.skipUnless(_REAL.exists(), "export ResIPY di prova non presente")
class RealP9Test(unittest.TestCase):
    def test_p9_section(self):
        sec = _ERT.load_ert_section(str(_REAL))
        self.assertEqual(len(sec.x), 5040)
        self.assertEqual(sec.z_mode, "depth")
        # 72 elettrodi a 1 m: la linea e' lunga 71 m
        self.assertAlmostEqual(sec.length_m, 71.0, delta=0.1)
        self.assertGreater(float(sec.depth.max()), 11.0)
        g = _ERT.to_grid(sec)
        frac = float(np.isfinite(g.data).mean())
        self.assertGreater(frac, 0.5)
        vals = 10.0 ** g.data[np.isfinite(g.data)]
        self.assertGreaterEqual(float(vals.min()), 2.0)
        self.assertLessEqual(float(vals.max()), 421.0)


if __name__ == "__main__":
    unittest.main()
