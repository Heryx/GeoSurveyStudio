# -*- coding: utf-8 -*-
"""Fase B tests: SliceSpec dataclass + sliceset.json manifest."""

import importlib
import json
import pathlib
import sys
import tempfile
import unittest

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_PLUGIN_DIR = _HERE.parents[1]
_PLUGIN_PARENT = _PLUGIN_DIR.parent
_TEST_DIR = _HERE.parent
for _p in (str(_PLUGIN_PARENT), str(_TEST_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

slice_spec = importlib.import_module("rasterlinker_feature_interpolation.slice_spec")
slice_manifest = importlib.import_module("rasterlinker_feature_interpolation.slice_manifest")
ogpr = importlib.import_module("rasterlinker_feature_interpolation.gpr_ogpr_slicer")
_ogpr_test = importlib.import_module("test_gpr_ogpr_slicer")

SliceSpec = slice_spec.SliceSpec
SliceSpecError = slice_spec.SliceSpecError
SliceSetError = slice_manifest.SliceSetError


try:
    import scipy  # noqa: F401

    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


class SliceSpecBasicsTest(unittest.TestCase):
    def test_defaults_validate(self):
        SliceSpec().validate()  # must not raise

    def test_rejects_bad_enums_and_sizes(self):
        for kw in (
            {"amplitude_mode": "bogus"},
            {"interp": "spline"},
            {"thickness_m": 0.0},
            {"overlap": 0.95},
            {"resolution_m": -1.0},
            {"smooth_lowpass": 4},
            {"z_min_m": 1.0, "z_max_m": 0.5},
            {"search_radius_m": 0.0},
            {"blanking_radius_m": -0.1},
        ):
            with self.assertRaises(SliceSpecError, msg=kw):
                SliceSpec(**kw).validate()

    def test_derived_geometry(self):
        s = SliceSpec(thickness_m=0.2, overlap=0.5, smooth_lowpass=3)
        self.assertAlmostEqual(s.stride_m, 0.1)
        self.assertAlmostEqual(s.smooth_sigma, 1.0)
        self.assertEqual(SliceSpec(smooth_lowpass=0).smooth_sigma, 0.0)

    def test_json_roundtrip(self):
        s = SliceSpec(
            z_min_m=0.1, z_max_m=3.0, thickness_m=0.25, overlap=0.3,
            amplitude_mode="env_peak", source_variant="bandpass",
            resolution_m=0.15, search_radius_m=0.9, blanking_radius_m=0.4,
            smooth_lowpass=5, crs_epsg=32633,
        )
        self.assertEqual(SliceSpec.from_json(s.to_json()), s)
        d = s.to_dict()
        self.assertEqual(d["format"], slice_spec.FORMAT)
        self.assertEqual(SliceSpec.from_dict(d), s)

    def test_from_dict_rejects_unknown_format_and_ignores_extra_keys(self):
        with self.assertRaises(SliceSpecError):
            SliceSpec.from_dict({"format": "something_else"})
        s = SliceSpec.from_dict({"resolution_m": 0.2, "not_a_field": 123})
        self.assertEqual(s.resolution_m, 0.2)

    def test_to_compute_kwargs_amplitude_and_radius(self):
        k = SliceSpec(amplitude_mode="abs_mean").to_compute_kwargs()
        self.assertEqual(k["extraction_mode"], "las_like")
        self.assertFalse(k["use_hilbert"])
        self.assertTrue(k["auto_radius"])
        self.assertNotIn("radius", k)

        k = SliceSpec(amplitude_mode="env_peak", search_radius_m=0.5).to_compute_kwargs()
        self.assertEqual(k["extraction_mode"], "envelope")
        self.assertTrue(k["use_hilbert"])
        self.assertEqual(k["radius"], 0.5)
        self.assertFalse(k["auto_radius"])

        k = SliceSpec(amplitude_mode="signed_mean").to_compute_kwargs()
        self.assertEqual(k["extraction_mode"], "signed")


@unittest.skipUnless(HAS_SCIPY, "SciPy required for resolve() geometry helpers")
class SliceSpecResolveTest(unittest.TestCase):
    def setUp(self):
        self.profiles = _ogpr_test._make_profiles()

    def test_resolve_fills_survey_fields(self):
        s = SliceSpec().resolve(self.profiles)
        self.assertIsNotNone(s.z_max_m)
        self.assertAlmostEqual(s.z_max_m, 2.5, places=6)  # _FakeProfile depth_max_m
        self.assertIsNotNone(s.search_radius_m)
        self.assertGreater(s.search_radius_m, 0.0)
        self.assertEqual(s.crs_epsg, 32633)
        self.assertTrue(s.is_resolved())

    def test_resolve_keeps_explicit_values(self):
        s = SliceSpec(z_max_m=1.0, search_radius_m=0.42, crs_epsg=4326).resolve(self.profiles)
        self.assertEqual(s.z_max_m, 1.0)
        self.assertEqual(s.search_radius_m, 0.42)
        self.assertEqual(s.crs_epsg, 4326)

    def test_resolve_without_profiles_needs_z_max(self):
        with self.assertRaises(SliceSpecError):
            SliceSpec().resolve([])
        SliceSpec(z_max_m=2.0).resolve([])  # ok

    def test_kriging_not_implemented(self):
        with self.assertRaises(SliceSpecError):
            SliceSpec(interp="kriging").resolve(self.profiles)


class ManifestTest(unittest.TestCase):
    def _slices(self):
        g0 = np.array([[1.0, 2.0], [np.nan, 4.0]], dtype=np.float32)
        g1 = np.full((2, 2), np.nan, dtype=np.float32)
        return [
            {"index": 1, "z_from": 0.1, "z_to": 0.2, "z_center": 0.15,
             "grid": "slice_0001.tif", "array": g1},
            {"index": 0, "z_from": 0.0, "z_to": 0.1, "z_center": 0.05,
             "path": "/x/slice_0000.tif", "array": g0},
        ]

    def test_write_read_roundtrip(self):
        spec = SliceSpec(z_max_m=1.0, search_radius_m=0.5, crs_epsg=32633)
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(slice_manifest.has_sliceset(d))
            p = slice_manifest.write_sliceset(
                d, spec, self._slices(),
                source={"kind": "ogpr", "variant_id": "raw", "n_profiles": 3},
                extent=[0.0, 0.0, 10.0, 5.0],
            )
            self.assertTrue(slice_manifest.has_sliceset(d))
            man = slice_manifest.read_sliceset(d)

        self.assertEqual(man["format"], slice_manifest.FORMAT)
        self.assertEqual(man["crs_epsg"], 32633)
        self.assertEqual(man["extent"], [0.0, 0.0, 10.0, 5.0])
        self.assertIsInstance(man["spec"], SliceSpec)
        self.assertEqual(man["spec"].z_max_m, 1.0)
        # slices sorted by depth
        self.assertEqual([s["index"] for s in man["slices"]], [0, 1])
        s0 = man["slices"][0]
        self.assertEqual(s0["grid"], "slice_0000.tif")
        self.assertAlmostEqual(s0["stats"]["min"], 1.0)
        self.assertAlmostEqual(s0["stats"]["max"], 4.0)
        self.assertAlmostEqual(s0["stats"]["fill"], 0.75)
        self.assertEqual(man["slices"][1]["stats"]["fill"], 0.0)

    def test_slice_catalog_from_sliceset(self):
        spec = SliceSpec(z_max_m=1.0, search_radius_m=0.5, crs_epsg=32633)
        with tempfile.TemporaryDirectory() as d:
            slice_manifest.write_sliceset(
                d, spec, self._slices(),
                source={"kind": "ogpr"},
                extent=[0.0, 0.0, 10.0, 5.0],
            )
            # grid files referenced by the manifest must exist on disk
            for name in ("slice_0000.tif", "slice_0001.tif"):
                open(pathlib.Path(d) / name, "wb").close()
            rows = slice_manifest.slice_catalog_from_sliceset(d)

        self.assertEqual([r["index"] for r in rows], [0, 1])          # depth order
        r0 = rows[0]
        self.assertTrue(r0["path"].endswith("slice_0000.tif"))
        self.assertEqual(r0["name"], "slice_0000")
        self.assertAlmostEqual(r0["z_top"], 0.0)
        self.assertAlmostEqual(r0["z_bot"], 0.1)
        self.assertAlmostEqual(r0["z_center"], 0.05)
        self.assertIn("stats", r0)

    def test_slice_catalog_from_sliceset_skips_missing_files(self):
        spec = SliceSpec(z_max_m=1.0)
        with tempfile.TemporaryDirectory() as d:
            slice_manifest.write_sliceset(
                d, spec, self._slices(), source={"kind": "ogpr"},
            )
            open(pathlib.Path(d) / "slice_0000.tif", "wb").close()  # only one on disk
            rows = slice_manifest.slice_catalog_from_sliceset(d)
        self.assertEqual([r["name"] for r in rows], ["slice_0000"])

    def test_read_errors(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SliceSetError):
                slice_manifest.read_sliceset(d)
            with open(slice_manifest.sliceset_path(d), "w", encoding="utf-8") as f:
                json.dump({"format": "wrong"}, f)
            with self.assertRaises(SliceSetError):
                slice_manifest.read_sliceset(d)

    def test_from_compute_meta(self):
        meta = {
            "z_min": 0.0, "z_max": 2.4, "z_step": 0.2, "overlap_fraction": 0.5,
            "resolution": 0.1, "radius": 0.7, "blanking_distance_m": 0.3,
            "extraction_mode": "envelope", "use_hilbert": True,
        }
        s = SliceSpec.from_compute_meta(meta, crs_epsg=32633)
        self.assertEqual(s.amplitude_mode, "env_peak")
        self.assertEqual(s.z_max_m, 2.4)
        self.assertEqual(s.search_radius_m, 0.7)
        self.assertEqual(s.blanking_radius_m, 0.3)
        self.assertEqual(s.crs_epsg, 32633)


@unittest.skipUnless(HAS_SCIPY, "SciPy required for end-to-end compute")
class ComputeWithSpecTest(unittest.TestCase):
    def setUp(self):
        self.profiles = _ogpr_test._make_profiles()

    def test_spec_matches_equivalent_kwargs(self):
        # interp="idw" pins the legacy engine this test is about (default is "bin").
        spec = SliceSpec(
            z_min_m=0.0, z_max_m=1.2, thickness_m=0.30, overlap=0.0,
            amplitude_mode="abs_mean", resolution_m=0.20, search_radius_m=0.40,
            blanking_radius_m=0.0, smooth_lowpass=0, interp="idw",
        )
        grids_spec, meta_spec = ogpr.compute_ogpr_slice_grids(
            profiles=self.profiles, idw_mode="fast", min_points=1,
            fill_nodata=False, emit_diagnostics=False, parallel_profiles=False,
            spec=spec,
        )
        grids_kw, _ = ogpr.compute_ogpr_slice_grids(
            profiles=self.profiles, channel=-1, combine_method="mean",
            resolution=0.20, z_step=0.30, overlap_fraction=0.0,
            z_min=0.0, z_max=1.2, radius=0.40, auto_radius=False,
            extraction_mode="las_like", use_hilbert=False, use_processing=False,
            idw_mode="fast", min_points=1, fill_nodata=False, smooth_sigma=0.0,
            blanking_distance=0.0, emit_diagnostics=False, parallel_profiles=False,
        )
        self.assertEqual(len(grids_spec), len(grids_kw))
        self.assertGreater(len(grids_spec), 0)
        for a, b in zip(grids_spec, grids_kw):
            aa = np.asarray(a["grid"], dtype=np.float64)
            bb = np.asarray(b["grid"], dtype=np.float64)
            self.assertEqual(aa.shape, bb.shape)
            self.assertTrue(np.array_equal(np.isnan(aa), np.isnan(bb)))
            self.assertTrue(np.allclose(aa[~np.isnan(aa)], bb[~np.isnan(bb)], rtol=1e-6, atol=1e-6))

        self.assertIn("slice_spec", meta_spec)
        self.assertEqual(meta_spec["slice_spec"]["z_max_m"], 1.2)
        self.assertEqual(meta_spec["slice_spec"]["search_radius_m"], 0.40)

    def test_spec_resolves_none_fields_from_profiles(self):
        spec = SliceSpec(thickness_m=0.4, resolution_m=0.25)  # z_max_m / radius None
        grids, meta = ogpr.compute_ogpr_slice_grids(
            profiles=self.profiles, idw_mode="fast", emit_diagnostics=False,
            fill_nodata=False, parallel_profiles=False, spec=spec,
        )
        self.assertGreater(len(grids), 0)
        self.assertAlmostEqual(meta["slice_spec"]["z_max_m"], 2.5, places=6)
        self.assertIsNotNone(meta["slice_spec"]["search_radius_m"])

    def test_write_run_sliceset_glue(self):
        """Fase B step 4: slice_ogpr_to_tifs manifest writer (no GDAL needed)."""
        spec = SliceSpec(
            z_min_m=0.0, z_max_m=1.2, thickness_m=0.3, overlap=0.0,
            resolution_m=0.2, search_radius_m=0.4,
        )
        grids, meta = ogpr.compute_ogpr_slice_grids(
            profiles=self.profiles, idw_mode="fast", emit_diagnostics=False,
            fill_nodata=False, parallel_profiles=False, spec=spec,
        )
        results = [
            {"path": f"/out/slice_{g['index']:04d}.tif", "z_from": g["z_from"],
             "z_to": g["z_to"], "z_center": round(g["z_lev"], 6),
             "index": g["index"], "name": f"slice_{g['index']:04d}"}
            for g in grids
        ]
        with tempfile.TemporaryDirectory() as d:
            ogpr._write_run_sliceset(
                d, meta, grids, results,
                spec=spec, epsg=32633, source_variant="bandpass",
                n_profiles=len(self.profiles),
            )
            man = slice_manifest.read_sliceset(d)
        self.assertEqual(man["source"], {"kind": "ogpr", "variant_id": "bandpass",
                                         "n_profiles": len(self.profiles)})
        self.assertEqual(man["crs_epsg"], 32633)
        self.assertEqual(len(man["slices"]), len(grids))
        self.assertTrue(all("stats" in s for s in man["slices"]))
        self.assertEqual(len(man["extent"]), 4)
        self.assertEqual(man["spec"].crs_epsg, 32633)


if __name__ == "__main__":
    unittest.main()


class TestGridBg(unittest.TestCase):
    def test_apply_grid_bg_removes_row_and_column_bands(self):
        import numpy as np
        from grid_filters import apply_grid_bg
        rng = np.random.default_rng(1)
        base = rng.normal(0, 0.1, (60, 80))
        rows = rng.normal(0, 2.0, (60, 1))
        cols = rng.normal(0, 2.0, (1, 80))
        g = base + rows + cols
        h = apply_grid_bg(g, "h")
        self.assertLess(np.std(h.mean(axis=1)), 0.05)
        v = apply_grid_bg(g, "v")
        self.assertLess(np.std(v.mean(axis=0)), 0.05)
        hv = apply_grid_bg(g, "hv@95")
        self.assertLess(float(np.std(hv)), float(np.std(g)) * 0.2)
        self.assertTrue(np.array_equal(apply_grid_bg(g, ""), g))

    def test_spec_exposes_grid_bg_kwarg(self):
        from slice_spec import SliceSpec
        sp = SliceSpec(z_max_m=1.0, search_radius_m=0.5, bg_horizontal=True, bg_vertical=True)
        self.assertEqual(sp.to_compute_kwargs()["grid_bg"], "hv@95")
        self.assertEqual(SliceSpec(z_max_m=1.0, search_radius_m=0.5).to_compute_kwargs()["grid_bg"], "")
