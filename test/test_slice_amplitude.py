# -*- coding: utf-8 -*-
"""Ampiezze per singola time-slice (non distruttive): impostazioni, filtri sulla
griglia, file allegato e raster derivato. L'originale non deve mai cambiare."""

import hashlib
import importlib
import os
import pathlib
import sys
import tempfile
import unittest

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_PLUGIN_PARENT = _HERE.parents[1].parent
if str(_PLUGIN_PARENT) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_PARENT))

PKG = "rasterlinker_feature_interpolation"
amp = importlib.import_module(PKG + ".slice_amplitude")
AS = amp.AmplitudeSettings

try:
    from osgeo import gdal
    HAS_GDAL = True
except Exception:  # pragma: no cover
    HAS_GDAL = False


def _noisy(seed=0, n=40):
    rng = np.random.default_rng(seed)
    g = np.abs(rng.normal(1.0, 0.2, (n, n)))
    g[10, 10] = 60.0                      # picco isolato
    g[:3, :] = np.nan                     # contorno
    return g


def _write_tif(path, grid, res=0.1):
    n_y, n_x = grid.shape
    ds = gdal.GetDriverByName("GTiff").Create(path, n_x, n_y, 1, gdal.GDT_Float32)
    ds.SetGeoTransform((100.0, res, 0.0, 200.0, 0.0, -res))
    ds.SetMetadataItem("GPR_STYLE_VMIN", "0")
    b = ds.GetRasterBand(1)
    b.WriteArray(np.where(np.isnan(grid), -9999.0, grid).astype(np.float32))
    b.SetNoDataValue(-9999.0)
    b.FlushCache()
    ds = None


class SettingsTests(unittest.TestCase):
    def test_default_is_inert(self):
        s = AS()
        self.assertTrue(s.is_default)
        self.assertFalse(s.pixel_tools_active)
        g = _noisy()
        out = amp.process_grid(g, s, 0.1)
        self.assertTrue(np.array_equal(np.isnan(out), np.isnan(g)))
        np.testing.assert_allclose(out[np.isfinite(g)], g[np.isfinite(g)], rtol=1e-6)

    def test_roundtrip_and_validation(self):
        s = AS(despike_pct=99.0, normalize="median", amp_scale="db", despeckle=3,
               smooth_lowpass=5, highpass_m=0.5, limits_mode="fixed", vmin=-1.0, vmax=3.0)
        back = AS.from_dict(s.to_dict())
        self.assertEqual(s, back)
        self.assertTrue(back.pixel_tools_active)
        for bad in ({"despike_pct": 10}, {"normalize": "z"}, {"amp_scale": "log"},
                    {"despeckle": 4}, {"smooth_lowpass": 4}, {"highpass_m": -1},
                    {"limits_mode": "x"}, {"limits_mode": "fixed", "vmin": 2, "vmax": 1}):
            with self.assertRaises(amp.AmplitudeSettingsError):
                AS.from_dict(bad)

    def test_unknown_keys_ignored(self):
        self.assertEqual(AS.from_dict({"nope": 1, "despeckle": 3}).despeckle, 3)

    def test_view_only_settings_need_no_derived(self):
        self.assertFalse(AS(limits_mode="percentile").pixel_tools_active)


class GridTests(unittest.TestCase):
    def test_despike_caps_the_spike_and_keeps_outline(self):
        g = _noisy()
        out = amp.process_grid(g, AS(despike_pct=99.0), 0.1)
        self.assertLess(np.nanmax(out), 5.0)
        self.assertTrue(np.array_equal(np.isnan(out), np.isnan(g)))

    def test_despeckle_removes_spike(self):
        out = amp.process_grid(_noisy(), AS(despeckle=3), 0.1)
        self.assertLess(out[10, 10], 5.0)

    def test_scale_skipped_for_signed_data(self):
        g = _noisy() - 1.0                 # ha valori negativi
        out = amp.process_grid(g, AS(amp_scale="db"), 0.1)
        np.testing.assert_allclose(out[np.isfinite(g)], g[np.isfinite(g)], rtol=1e-5)

    def test_db_changes_positive_data(self):
        g = _noisy()
        out = amp.process_grid(g, AS(amp_scale="db"), 0.1)
        self.assertFalse(np.allclose(out[np.isfinite(g)], g[np.isfinite(g)]))

    def test_all_nan(self):
        out = amp.process_grid(np.full((5, 5), np.nan), AS(despeckle=3), 0.1)
        self.assertTrue(np.isnan(out).all())

    def test_display_limits_modes(self):
        g = np.arange(101, dtype=float).reshape(1, -1)
        self.assertEqual(amp.display_limits(AS(), g), (0.0, 100.0))
        lo, hi = amp.display_limits(AS(limits_mode="percentile", p_lo=10, p_hi=90), g)
        self.assertAlmostEqual(lo, 10.0)
        self.assertAlmostEqual(hi, 90.0)
        self.assertEqual(amp.display_limits(AS(limits_mode="fixed", vmin=-2, vmax=5), g), (-2.0, 5.0))
        self.assertIsNone(amp.display_limits(AS(), np.full((2, 2), np.nan)))


@unittest.skipUnless(HAS_GDAL, "GDAL non disponibile")
class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        os.makedirs(os.path.join(self.root, "slices"))
        self.src = os.path.join(self.root, "slices", "s_0.50.tif")
        self.grid = _noisy()
        _write_tif(self.src, self.grid)

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def _md5(p):
        with open(p, "rb") as fh:
            return hashlib.md5(fh.read()).hexdigest()

    def test_store_roundtrip_and_default_removes_entry(self):
        s = AS(despeckle=3, limits_mode="percentile")
        amp.set_settings(self.root, self.src, s)
        self.assertEqual(amp.get_settings(self.root, self.src), s)
        self.assertIn("slices/s_0.50.tif", amp.load_store(self.root))
        amp.set_settings(self.root, self.src, AS())
        self.assertIsNone(amp.get_settings(self.root, self.src))
        self.assertEqual(amp.load_store(self.root), {})

    def test_corrupt_store_is_empty(self):
        with open(amp.store_path(self.root), "w") as fh:
            fh.write("{not json")
        self.assertEqual(amp.load_store(self.root), {})
        self.assertIsNone(amp.get_settings(self.root, self.src))

    def test_derived_leaves_original_untouched(self):
        before = self._md5(self.src)
        out = amp.ensure_derived(self.root, self.src, AS(despeckle=3))
        self.assertNotEqual(os.path.normcase(out), os.path.normcase(self.src))
        self.assertTrue(os.path.exists(out))
        self.assertEqual(self._md5(self.src), before)
        got, _ = amp.read_band(out)
        orig, _ = amp.read_band(self.src)
        self.assertEqual(got.shape, orig.shape)
        self.assertTrue(np.array_equal(np.isnan(got), np.isnan(orig)))
        self.assertLess(got[10, 10], 5.0)
        ds = gdal.Open(out)
        self.assertEqual(ds.GetGeoTransform(), (100.0, 0.1, 0.0, 200.0, 0.0, -0.1))
        self.assertEqual(ds.GetMetadataItem("GPR_STYLE_VMIN"), "0")
        ds = None

    def test_view_only_uses_original(self):
        out = amp.ensure_derived(self.root, self.src, AS(limits_mode="percentile"))
        self.assertEqual(out, self.src)
        self.assertFalse(os.path.exists(amp.derived_path(self.root, self.src)))

    def test_derived_regenerated_when_original_newer_or_forced(self):
        out = amp.ensure_derived(self.root, self.src, AS(despeckle=3))
        m1 = os.path.getmtime(out)
        os.utime(out, (m1 + 100, m1 + 100))          # fresco: non si rigenera
        amp.ensure_derived(self.root, self.src, AS(despeckle=3))
        self.assertAlmostEqual(os.path.getmtime(out), m1 + 100, places=0)
        amp.ensure_derived(self.root, self.src, AS(despeckle=5), force=True)
        self.assertNotAlmostEqual(os.path.getmtime(out), m1 + 100, places=0)
        os.utime(self.src, (m1 + 500, m1 + 500))     # originale piu' recente
        amp.ensure_derived(self.root, self.src, AS(despeckle=5))
        self.assertGreaterEqual(os.path.getmtime(out), m1 + 400 - 1000)

    def test_is_derived_of(self):
        out = amp.ensure_derived(self.root, self.src, AS(despeckle=3))
        self.assertTrue(amp.is_derived_of(self.root, out + "|layerid=0", self.src))
        self.assertFalse(amp.is_derived_of(self.root, self.src, self.src))


if __name__ == "__main__":
    unittest.main()
