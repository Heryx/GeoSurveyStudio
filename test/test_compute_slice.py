# -*- coding: utf-8 -*-
"""Fase C: entry point spec-native unico `compute_slice` / `compute_slice_preview`."""

import importlib
import pathlib
import sys
import unittest

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_PLUGIN_DIR = _HERE.parents[1]
_PLUGIN_PARENT = _PLUGIN_DIR.parent
_TEST_DIR = _HERE.parent
for _p in (str(_PLUGIN_PARENT), str(_TEST_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ogpr = importlib.import_module("rasterlinker_feature_interpolation.gpr_ogpr_slicer")
slice_spec = importlib.import_module("rasterlinker_feature_interpolation.slice_spec")
_ogpr_test = importlib.import_module("test_gpr_ogpr_slicer")

SliceSpec = slice_spec.SliceSpec
compute_slice = ogpr.compute_slice
compute_slice_preview = ogpr.compute_slice_preview
_make_profiles = _ogpr_test._make_profiles

try:
    import scipy  # noqa: F401

    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


@unittest.skipUnless(HAS_SCIPY, "SciPy required for IDW gridding")
class ComputeSliceTest(unittest.TestCase):
    def setUp(self):
        self.profiles = _make_profiles()
        # small, deterministic spec
        self.spec = SliceSpec(
            z_min_m=0.0,
            z_max_m=1.2,
            thickness_m=0.3,
            overlap=0.0,
            resolution_m=0.2,
            search_radius_m=0.4,
            blanking_radius_m=0.0,
            smooth_lowpass=0,
        ).resolve(self.profiles)

    def test_type_guard(self):
        with self.assertRaises(TypeError):
            compute_slice(self.profiles, {"z_min_m": 0.0})

    def test_multi_slice_matches_spec_geometry(self):
        grids, meta = compute_slice(
            self.profiles, self.spec,
            fill_nodata=False, smooth_sigma=0.0, emit_diagnostics=False,
            parallel_profiles=False,
        )
        self.assertGreater(len(grids), 0)
        # z-levels are CENTRES from z_min..z_max stepping by thickness*(1-overlap):
        # 0.0, 0.3, 0.6, 0.9, 1.2 -> 5 slices
        self.assertEqual(len(grids), 5)
        centres = [float(g["z_lev"]) for g in grids]
        self.assertTrue(all(b - a > 0 for a, b in zip(centres, centres[1:])))
        for a, b in zip(centres, centres[1:]):
            self.assertAlmostEqual(b - a, 0.3, places=6)
        # window thickness == spec thickness
        self.assertAlmostEqual(
            float(grids[0]["z_to"]) - float(grids[0]["z_from"]), 0.3, places=6
        )
        self.assertAlmostEqual(float(meta["resolution"]), 0.2, places=6)
        self.assertIn("slice_spec", meta)

    def test_single_window_produces_one_slice(self):
        grids, meta = compute_slice(
            self.profiles, self.spec, single_window=(0.30, 0.60),
            fill_nodata=False, smooth_sigma=0.0, emit_diagnostics=False,
            parallel_profiles=False,
        )
        self.assertEqual(len(grids), 1)
        self.assertAlmostEqual(float(grids[0]["z_from"]), 0.30, places=5)
        self.assertAlmostEqual(float(grids[0]["z_to"]), 0.60, places=5)

    def test_resolution_override(self):
        grids_a, meta_a = compute_slice(
            self.profiles, self.spec, single_window=(0.30, 0.60),
            fill_nodata=False, smooth_sigma=0.0, emit_diagnostics=False,
            parallel_profiles=False,
        )
        grids_b, meta_b = compute_slice(
            self.profiles, self.spec, single_window=(0.30, 0.60),
            resolution_override=0.5,
            fill_nodata=False, smooth_sigma=0.0, emit_diagnostics=False,
            parallel_profiles=False,
        )
        self.assertAlmostEqual(float(meta_a["resolution"]), 0.2, places=6)
        self.assertAlmostEqual(float(meta_b["resolution"]), 0.5, places=6)
        # coarser cells -> not more rows/cols
        self.assertLessEqual(grids_b[0]["grid"].shape[0], grids_a[0]["grid"].shape[0])

    def test_preview_matches_single_window_grid(self):
        z0, z1 = 0.30, 0.60
        grids, meta = compute_slice(
            self.profiles, self.spec, single_window=(z0, z1),
            fill_nodata=False, smooth_sigma=0.0, emit_diagnostics=False,
            parallel_profiles=False,
        )
        grid_direct = np.asarray(grids[0]["grid"], dtype=np.float32)

        grid_prev, extent = compute_slice_preview(
            self.profiles, self.spec, z0, z1,
            fill_nodata=False, smooth_sigma=0.0, emit_diagnostics=False,
            parallel_profiles=False,
        )
        self.assertIsNotNone(grid_prev)
        self.assertIsNotNone(extent)
        self.assertEqual(grid_prev.shape, grid_direct.shape)
        np.testing.assert_allclose(
            np.nan_to_num(grid_prev), np.nan_to_num(grid_direct), rtol=1e-5, atol=1e-4
        )
        # extent is a 4-tuple, x_left < x_right, y_bottom < y_top
        xl, xr, yb, yt = extent
        self.assertLess(xl, xr)
        self.assertLess(yb, yt)

    def test_preview_bad_window_returns_none(self):
        g, e = compute_slice_preview(self.profiles, self.spec, float("nan"), 0.5)
        self.assertIsNone(g)
        self.assertIsNone(e)


if __name__ == "__main__":
    unittest.main()
