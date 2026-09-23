# -*- coding: utf-8 -*-
"""gpr_velocity_store: sidecar velocity_overrides.json (senza Qt)."""

import importlib
import json
import os
import pathlib
import sys
import tempfile
import unittest

_PLUGIN_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(_PLUGIN_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR.parent))

_VS = importlib.import_module("rasterlinker_feature_interpolation.gpr_velocity_store")
_HM = importlib.import_module("rasterlinker_feature_interpolation.gpr_hyperbola_math")


class TestVelocityStore(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.variant_dir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_default_fallback_when_nothing_saved(self):
        got = _VS.resolve_velocity_m_ns(self.variant_dir, "SwathX")
        self.assertEqual(got["source"], "default")
        self.assertAlmostEqual(got["velocity_m_ns"], _HM.DEFAULT_VELOCITY_M_NS, places=9)
        self.assertAlmostEqual(
            got["rdp"], _HM.velocity_to_rdp(_HM.m_ns_to_velocity_m_s(_HM.DEFAULT_VELOCITY_M_NS)), places=6)

    def test_per_profile_override_wins_over_default(self):
        _VS.write_velocity_override(self.variant_dir, velocity_m_ns=0.08, source="hyperbola_fit")
        _VS.write_velocity_override(
            self.variant_dir, profile_stem="SwathX", velocity_m_ns=0.095, source="manual")

        specific = _VS.resolve_velocity_m_ns(self.variant_dir, "SwathX")
        self.assertEqual(specific["source"], "manual")
        self.assertAlmostEqual(specific["velocity_m_ns"], 0.095, places=9)

        other = _VS.resolve_velocity_m_ns(self.variant_dir, "SwathY")
        self.assertEqual(other["source"], "hyperbola_fit")
        self.assertAlmostEqual(other["velocity_m_ns"], 0.08, places=9)

    def test_write_persists_to_disk_as_readable_json(self):
        path = _VS.write_velocity_override(
            self.variant_dir, profile_stem="Swath001", velocity_m_ns=0.1, rdp=8.99, source="manual")
        self.assertTrue(os.path.isfile(path))
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self.assertAlmostEqual(data["profiles"]["Swath001"]["velocity_m_ns"], 0.1, places=9)
        self.assertAlmostEqual(data["profiles"]["Swath001"]["rdp"], 8.99, places=6)

    def test_rdp_is_derived_when_not_given(self):
        _VS.write_velocity_override(self.variant_dir, velocity_m_ns=0.06, source="hyperbola_fit")
        got = _VS.resolve_velocity_m_ns(self.variant_dir, None)
        expected_rdp = _HM.velocity_to_rdp(_HM.m_ns_to_velocity_m_s(0.06))
        self.assertAlmostEqual(got["rdp"], expected_rdp, places=6)

    def test_second_write_updates_same_profile_in_place(self):
        _VS.write_velocity_override(self.variant_dir, profile_stem="Swath001", velocity_m_ns=0.1, source="manual")
        _VS.write_velocity_override(
            self.variant_dir, profile_stem="Swath001", velocity_m_ns=0.09, source="hyperbola_fit")
        got = _VS.resolve_velocity_m_ns(self.variant_dir, "Swath001")
        self.assertAlmostEqual(got["velocity_m_ns"], 0.09, places=9)
        self.assertEqual(got["source"], "hyperbola_fit")


if __name__ == "__main__":
    unittest.main()
