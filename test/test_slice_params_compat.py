# -*- coding: utf-8 -*-
"""Fase D: SliceSpec -> legacy slice-parameter dict bridge."""

import importlib
import pathlib
import sys
import unittest

_HERE = pathlib.Path(__file__).resolve()
for _p in (str(_HERE.parents[1].parent), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

compat = importlib.import_module("rasterlinker_feature_interpolation.slice_params_compat")
slice_spec = importlib.import_module("rasterlinker_feature_interpolation.slice_spec")
SliceSpec = slice_spec.SliceSpec


def _resolved(**kw):
    base = dict(z_max_m=2.0, search_radius_m=0.5)
    base.update(kw)
    return SliceSpec(**base)


class BuildParamsTest(unittest.TestCase):
    def test_requires_resolved_spec(self):
        with self.assertRaises(ValueError):
            compat.spec_to_build_params(SliceSpec())  # z_max_m None

    def test_maps_geometry(self):
        s = _resolved(z_min_m=0.1, z_max_m=3.0, thickness_m=0.25, resolution_m=0.15,
                      search_radius_m=0.9)
        p = compat.spec_to_build_params(s, channel=2, combine_method="max",
                                        group_name="siteA")
        self.assertEqual(p["channel"], 2)
        self.assertEqual(p["combine_method"], "max")
        self.assertEqual(p["z_min"], 0.1)
        self.assertEqual(p["z_max"], 3.0)
        self.assertEqual(p["z_step"], 0.25)
        self.assertEqual(p["resolution"], 0.15)
        self.assertEqual(p["radius"], 0.9)
        self.assertEqual(p["group_name"], "siteA")
        self.assertEqual(p["advanced"], {})

    def test_auto_radius_becomes_zero_in_build_params(self):
        p = compat.spec_to_build_params(_resolved(search_radius_m=None))
        self.assertEqual(p["radius"], 0.0)


class ExtraParamsTest(unittest.TestCase):
    def test_amplitude_mode_maps_to_extraction(self):
        e = compat.spec_to_extra_slice_params(_resolved(amplitude_mode="env_peak"))
        self.assertEqual(e["extraction_mode"], "envelope")
        self.assertTrue(e["use_hilbert"])
        e = compat.spec_to_extra_slice_params(_resolved(amplitude_mode="signed_mean"))
        self.assertEqual(e["extraction_mode"], "signed")
        self.assertFalse(e["use_hilbert"])

    def test_dropped_knobs_have_fixed_defaults(self):
        e = compat.spec_to_extra_slice_params(_resolved())
        self.assertFalse(e["use_processing"])
        self.assertFalse(e["topographic_correction"])
        self.assertEqual(e["stack_n"], 1)
        self.assertEqual(e["idw_mode"], "quality")
        self.assertFalse(e["use_anisotropic_idw"])
        self.assertIsNone(e["amplitude_sigma"])

    def test_overlap_blanking_smooth(self):
        e = compat.spec_to_extra_slice_params(
            _resolved(overlap=0.25, blanking_radius_m=0.4, smooth_lowpass=3)
        )
        self.assertEqual(e["overlap_fraction"], 0.25)
        self.assertEqual(e["blanking_distance"], 0.4)
        self.assertAlmostEqual(e["smooth_sigma"], 1.0)

    def test_auto_radius_flag(self):
        self.assertTrue(compat.spec_to_extra_slice_params(_resolved(search_radius_m=None))["auto_radius"])
        e = compat.spec_to_extra_slice_params(_resolved(search_radius_m=0.7))
        self.assertFalse(e["auto_radius"])
        self.assertEqual(e["radius"], 0.7)


class LegacyParamsTest(unittest.TestCase):
    def test_full_dict_shape_and_spec_passthrough(self):
        s = _resolved(z_min_m=0.0, z_max_m=1.2, thickness_m=0.3, resolution_m=0.2,
                      search_radius_m=0.4, crs_epsg=32633)
        d = compat.spec_to_legacy_params(
            s, channel=-1, combine_method="mean", group_name="g",
            open_3d_on_complete=True, output_dir="/tmp/x",
        )
        # legacy keys present
        for k in ("extraction_mode", "use_hilbert", "overlap_fraction", "blanking_distance",
                  "smooth_sigma", "idw_mode", "fill_nodata", "z_min", "z_max", "z_step",
                  "resolution", "channel", "combine_method"):
            self.assertIn(k, d, k)
        self.assertEqual(d["z_max"], 1.2)
        self.assertEqual(d["resolution"], 0.2)
        self.assertEqual(d["radius"], 0.4)
        self.assertEqual(d["crs_epsg"], 32633)
        self.assertIs(d["_spec"], s)
        self.assertTrue(d["open_3d_on_complete"])
        self.assertEqual(d["output_dir"], "/tmp/x")


if __name__ == "__main__":
    unittest.main()
