# -*- coding: utf-8 -*-
"""gpr_hyperbola_math: inversione (apice, punto) -> velocita'/RDP (senza Qt)."""

import importlib
import pathlib
import sys
import unittest

import numpy as np

_PLUGIN_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(_PLUGIN_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR.parent))

_HM = importlib.import_module("rasterlinker_feature_interpolation.gpr_hyperbola_math")


class TestHyperbolaMath(unittest.TestCase):
    def test_rdp_velocity_roundtrip(self):
        for rdp in (1.0, 4.0, 9.0, 16.0, 25.0, 40.0):
            v = _HM.rdp_to_velocity_m_s(rdp)
            self.assertAlmostEqual(_HM.velocity_to_rdp(v), rdp, places=6)

    def test_monostatic_recovers_known_rdp(self):
        v_ref = 1.0e8
        x0, z0 = 3.0, 1.2
        t0 = (2.0 * z0) / v_ref
        for true_rdp in (2.0, 9.0, 16.0, 30.0):
            v_true = _HM.rdp_to_velocity_m_s(true_rdp)
            for dx in (0.3, 0.8, 1.5, -1.1):
                x1 = x0 + dx
                z1 = float(_HM.hyperbola_curve_depth(np.array([dx]), t0, v_true, v_ref, 0.0)[0])
                if not (z1 > z0):
                    continue
                got = _HM.velocity_from_two_points(x0, z0, x1, z1, v_ref, s=0.0)
                self.assertIsNotNone(got, f"nessuna soluzione per rdp={true_rdp} dx={dx}")
                v_fit, rdp_fit = got
                self.assertAlmostEqual(v_fit, v_true, delta=max(1.0, abs(v_true) * 1e-4))
                self.assertAlmostEqual(rdp_fit, true_rdp, delta=max(1e-3, true_rdp * 1e-4))

    def test_bistatic_recovers_known_rdp(self):
        v_ref = 1.0e8
        x0, z0, s = 2.0, 0.9, 0.4
        t0 = (2.0 * z0) / v_ref
        true_rdp = 12.0
        v_true = _HM.rdp_to_velocity_m_s(true_rdp)
        dx = 1.0
        x1 = x0 + dx
        z1 = float(_HM.hyperbola_curve_depth(np.array([dx]), t0, v_true, v_ref, s)[0])
        got = _HM.velocity_from_two_points(x0, z0, x1, z1, v_ref, s=s)
        self.assertIsNotNone(got)
        v_fit, rdp_fit = got
        self.assertAlmostEqual(rdp_fit, true_rdp, delta=0.05)

    def test_point_above_apex_has_no_solution(self):
        # z1 <= z0: il punto non e' su un'ala dell'iperbole sotto l'apice.
        self.assertIsNone(_HM.velocity_from_two_points(3.0, 1.5, 3.5, 1.4, 1.0e8))
        self.assertIsNone(_HM.velocity_from_two_points(3.0, 1.5, 3.5, 1.5, 1.0e8))

    def test_rdp_out_of_bounds_is_rejected(self):
        # Un punto quasi in verticale sotto l'apice implica un RDP fuori scala.
        v_ref = 1.0e8
        got = _HM.velocity_from_two_points(0.0, 1.0, 5.0, 1.0001, v_ref)
        self.assertIsNone(got)


if __name__ == "__main__":
    unittest.main()
