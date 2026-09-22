# -*- coding: utf-8 -*-
"""Strumenti di ampiezza per le time-slice: media dell'inviluppo, taglio dei
picchi, normalizzazione/scala, despeckle, alto-passa e la loro catena nel motore
"bin" (`slice_engine.finish_grid`). Con i valori predefiniti il risultato deve
restare identico a quello di prima che gli strumenti esistessero."""

import importlib
import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_PLUGIN_PARENT = _HERE.parents[1].parent
_TEST_DIR = _HERE.parent
for _p in (str(_PLUGIN_PARENT), str(_TEST_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PKG = "rasterlinker_feature_interpolation"
slicing = importlib.import_module(PKG + ".slicing")
engine = importlib.import_module(PKG + ".slice_engine")
slice_spec = importlib.import_module(PKG + ".slice_spec")
grid_filters = importlib.import_module(PKG + ".grid_filters")
raster_io = importlib.import_module(PKG + ".raster_io")
slicer = importlib.import_module(PKG + ".gpr_ogpr_slicer")
SliceSpec = slice_spec.SliceSpec
SliceSpecError = slice_spec.SliceSpecError
import test_slice_engine as _base  # noqa: E402  (helpers: _array_survey)

try:
    import scipy  # noqa: F401
    HAS_SCIPY = True
except Exception:  # pragma: no cover
    HAS_SCIPY = False

try:
    from osgeo import gdal
    HAS_GDAL = True
except Exception:  # pragma: no cover
    gdal = None
    HAS_GDAL = False


def _roughness(g):
    """Std of the 5-point Laplacian of the finite interior (noise level proxy)."""
    a = np.where(np.isfinite(g), g, np.nanmedian(g)).astype(np.float64)
    lap = (4 * a[1:-1, 1:-1] - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:])
    return float(lap.std())


@unittest.skipUnless(HAS_SCIPY, "SciPy required")
class PrimitivesTest(unittest.TestCase):
    def test_clip_outliers_is_off_at_100(self):
        v = np.array([1.0, 2.0, 500.0])
        self.assertTrue(np.array_equal(slicing.clip_outliers(v, 100.0), v))

    def test_clip_outliers_caps_a_single_spike(self):
        v = np.ones(1000)
        v[5] = 100.0
        out = slicing.clip_outliers(v, 99.0)
        self.assertEqual(float(out.max()), 1.0)
        self.assertEqual(int(np.count_nonzero(out != v)), 1)

    def test_clip_outliers_symmetric_caps_both_tails(self):
        v = np.concatenate([np.zeros(998), [-100.0, 100.0]])
        one = slicing.clip_outliers(v, 99.0)
        both = slicing.clip_outliers(v, 99.0, symmetric=True)
        self.assertEqual(float(one.min()), -100.0)     # only the top is capped
        self.assertEqual(float(both.min()), 0.0)
        self.assertEqual(float(both.max()), 0.0)

    def test_clip_outliers_keeps_nan(self):
        v = np.array([1.0, np.nan, 2.0, 100.0] * 50)
        out = slicing.clip_outliers(v, 90.0)
        self.assertTrue(np.array_equal(np.isnan(out), np.isnan(v)))

    def test_scale_sqrt_and_db(self):
        g = np.array([[1.0, 4.0], [100.0, np.nan]])
        s = slicing.scale_amplitude(g, "sqrt")
        self.assertTrue(np.allclose(s[:, 0], [1.0, 10.0]) and s[0, 1] == 2.0 and np.isnan(s[1, 1]))
        d = slicing.scale_amplitude(g, "db")
        self.assertTrue(np.allclose([d[0, 0], d[0, 1], d[1, 0]], [0.0, 20 * np.log10(4.0), 40.0]))
        self.assertTrue(np.isnan(d[1, 1]))
        self.assertTrue(np.array_equal(slicing.scale_amplitude(g, "linear"), g, equal_nan=True))

    def test_db_never_returns_minus_infinity(self):
        d = slicing.scale_amplitude(np.array([[0.0, 10.0]]), "db")
        self.assertTrue(np.isfinite(d).all())
        self.assertAlmostEqual(float(d[0, 0]), 20 * np.log10(10.0 * 1e-6), places=6)

    def test_normalize_median(self):
        g = np.array([[2.0, 4.0, 6.0], [8.0, np.nan, 10.0]])
        n = slicing.normalize_grid(g, "median")
        self.assertAlmostEqual(float(np.nanmedian(n)), 1.0)
        self.assertTrue(np.array_equal(slicing.normalize_grid(g, "none"), g, equal_nan=True))
        zero = np.zeros((3, 3))
        self.assertTrue(np.array_equal(slicing.normalize_grid(zero, "median"), zero))

    def test_median_despeckle_removes_a_spike_and_keeps_an_edge(self):
        g = np.zeros((20, 20))
        g[:, 10:] = 1.0                       # sharp step
        g[5, 4] = 50.0                        # isolated speckle
        out = slicing.median_despeckle(g, 3)
        self.assertEqual(float(out[5, 4]), 0.0)
        self.assertTrue(np.array_equal(out[:, :10].round(6), np.zeros((20, 10))))
        self.assertTrue(np.array_equal(out[:, 10:].round(6), np.ones((20, 10))))  # edge not blurred

    def test_median_despeckle_respects_the_outline(self):
        g = np.full((15, 15), 2.0)
        g[:, :5] = np.nan
        g[7, 9] = 40.0
        out = slicing.median_despeckle(g, 5)
        self.assertTrue(np.array_equal(np.isnan(out), np.isnan(g)))
        self.assertEqual(float(out[7, 9]), 2.0)

    def test_median_despeckle_edge_cases(self):
        allnan = np.full((6, 6), np.nan)
        self.assertTrue(np.isnan(slicing.median_despeckle(allnan, 3)).all())
        g = np.arange(36, dtype=float).reshape(6, 6)
        self.assertTrue(np.array_equal(slicing.median_despeckle(g, 0), g))

    def test_highpass_removes_the_trend_and_keeps_the_target(self):
        yy, xx = np.mgrid[0:80, 0:80]
        trend = 0.05 * xx                     # slow ramp
        g = trend.copy()
        g[38:42, 38:42] += 5.0                # small target
        out = slicing.highpass_background(g, 10.0)
        self.assertLess(abs(float(out[10, 40])), 0.1)                # far from the target: ramp gone
        self.assertGreater(float(out[40, 40]), 3.0)                  # target survives

    def test_highpass_is_a_noop_when_off(self):
        g = np.random.default_rng(0).normal(size=(10, 10))
        self.assertTrue(np.array_equal(slicing.highpass_background(g, 0.0), g))

    def test_highpass_keeps_nan(self):
        g = np.random.default_rng(1).normal(size=(30, 30))
        g[:, :8] = np.nan
        out = slicing.highpass_background(g, 4.0)
        self.assertTrue(np.array_equal(np.isnan(out), np.isnan(g)))


@unittest.skipUnless(HAS_SCIPY, "SciPy required")
class EnvelopeMeanTest(unittest.TestCase):
    def setUp(self):
        t = np.arange(200)
        self.data = (2.0 * np.sin(2 * np.pi * t / 16.0))[:, None].astype(np.float32).repeat(2, axis=1)

    def test_pure_tone(self):
        v = slicing.window_amplitude(self.data, 40, 104, "env_mean")
        self.assertTrue(np.allclose(v, 2.0, atol=0.1))

    def test_less_sensitive_to_a_spike_than_the_peak(self):
        d = self.data.copy()
        d[70, 0] = 60.0
        peak = slicing.window_amplitude(d, 40, 104, "env_peak")
        mean = slicing.window_amplitude(d, 40, 104, "env_mean")
        self.assertGreater(float(peak[0]) / float(mean[0]), 8.0)
        self.assertLess(float(mean[1]), 2.2)                     # the clean trace is untouched

    def test_is_a_registered_mode(self):
        self.assertIn("env_mean", slicing.AMPLITUDE_WINDOW_MODES)
        self.assertIn("env_mean", slice_spec.AMPLITUDE_MODES)
        self.assertIn("env_mean", slice_spec._AMPLITUDE_MAP)


class SpecFieldsTest(unittest.TestCase):
    def test_defaults_are_off(self):
        s = SliceSpec()
        self.assertEqual((s.despike_pct, s.normalize, s.amp_scale, s.despeckle, s.highpass_m),
                         (100.0, "none", "linear", 0, 0.0))
        self.assertFalse(s.tools_active)

    def test_tools_active(self):
        for kw in ({"despike_pct": 99.0}, {"normalize": "median"}, {"amp_scale": "db"},
                   {"despeckle": 3}, {"highpass_m": 1.0}):
            self.assertTrue(SliceSpec(**kw).tools_active, kw)

    def test_roundtrip(self):
        s = SliceSpec(amplitude_mode="env_mean", despike_pct=99.5, normalize="median",
                      amp_scale="sqrt", despeckle=5, highpass_m=0.8, smooth_lowpass=9)
        back = SliceSpec.from_json(s.to_json())
        self.assertEqual(back, s)

    def test_old_spec_without_the_fields_loads_with_defaults(self):
        old = {"format": "rasterlinker_slicespec_v1", "thickness_m": 0.2, "balance": "channel",
               "interp": "bin", "smooth_lowpass": 3}
        s = SliceSpec.from_dict(old)
        self.assertFalse(s.tools_active)
        self.assertEqual(s.smooth_lowpass, 3)

    def test_validation(self):
        for kw in ({"despike_pct": 40.0}, {"despike_pct": 101.0}, {"normalize": "zscore"},
                   {"amp_scale": "log"}, {"despeckle": 4}, {"highpass_m": -1.0},
                   {"smooth_lowpass": 4}):
            with self.assertRaises(SliceSpecError, msg=str(kw)):
                SliceSpec(**kw).validate()
        SliceSpec(smooth_lowpass=15, despeckle=5, amp_scale="db").validate()


@unittest.skipUnless(HAS_SCIPY, "SciPy required")
class FinishGridTest(unittest.TestCase):
    """finish_grid = the whole grid-domain chain."""

    @staticmethod
    def _old_chain(grid, spec):
        keep = np.isfinite(grid)
        if spec.grid_bg_spec:
            grid = grid_filters.apply_grid_bg(grid, spec.grid_bg_spec).astype(np.float32, copy=False)
        if spec.smooth_sigma > 0.0:
            grid = slicing._smooth_grid_gaussian(grid, sigma=spec.smooth_sigma)
        return np.where(keep, grid, np.nan).astype(np.float32, copy=False)

    def _grid(self, seed=0):
        rng = np.random.default_rng(seed)
        g = np.abs(rng.normal(1.0, 0.4, (60, 45))).astype(np.float32)
        g[:, :6] = np.nan
        g[:4, :] = np.nan
        return g

    def test_defaults_reproduce_the_previous_chain_exactly(self):
        for lp in (0, 3, 5):
            for bg in ((False, False), (True, False), (True, True)):
                spec = SliceSpec(smooth_lowpass=lp, bg_horizontal=bg[0], bg_vertical=bg[1])
                g = self._grid()
                got = engine.finish_grid(g.copy(), spec, 0.1)
                exp = self._old_chain(g.copy(), spec)
                self.assertTrue(np.array_equal(got, exp, equal_nan=True), (lp, bg))
                self.assertEqual(got.dtype, np.float32)

    def test_order_normalize_then_scale(self):
        g = self._grid()
        spec = SliceSpec(smooth_lowpass=0, normalize="median", amp_scale="db")
        got = engine.finish_grid(g.copy(), spec, 0.1)
        ratio = g / np.nanmedian(g)
        exp = 20 * np.log10(np.maximum(ratio, np.nanmax(ratio) * 1e-6))
        self.assertTrue(np.allclose(got[np.isfinite(g)], exp[np.isfinite(g)], atol=1e-4))
        self.assertTrue(np.array_equal(np.isnan(got), np.isnan(g)))

    def test_normalize_gives_unit_median(self):
        spec = SliceSpec(smooth_lowpass=0, normalize="median")
        got = engine.finish_grid(self._grid(3) * 7.5, spec, 0.1)
        self.assertAlmostEqual(float(np.nanmedian(got)), 1.0, places=5)

    def test_signed_mean_ignores_scale_and_normalize(self):
        g = self._grid() - 1.0
        plain = SliceSpec(amplitude_mode="signed_mean", smooth_lowpass=0)
        tooled = SliceSpec(amplitude_mode="signed_mean", smooth_lowpass=0,
                           normalize="median", amp_scale="db")
        a = engine.finish_grid(g.copy(), plain, 0.1)
        b = engine.finish_grid(g.copy(), tooled, 0.1)
        self.assertTrue(np.array_equal(a, b, equal_nan=True))

    def test_despeckle_reduces_the_noise(self):
        g = self._grid(5)
        base = engine.finish_grid(g.copy(), SliceSpec(smooth_lowpass=0), 0.1)
        med = engine.finish_grid(g.copy(), SliceSpec(smooth_lowpass=0, despeckle=3), 0.1)
        self.assertLess(_roughness(med), _roughness(base))

    def test_more_smoothing_means_less_noise(self):
        g = self._grid(6)
        rough = [_roughness(engine.finish_grid(g.copy(), SliceSpec(smooth_lowpass=lp), 0.1))
                 for lp in (0, 3, 7, 15)]
        self.assertEqual(rough, sorted(rough, reverse=True))

    def test_highpass_centres_the_slice_and_keeps_the_outline(self):
        g = self._grid(8) + np.linspace(0, 3, 45)[None, :].astype(np.float32)   # add a trend
        spec = SliceSpec(smooth_lowpass=0, highpass_m=0.8)
        got = engine.finish_grid(g.copy(), spec, 0.1)
        self.assertTrue(np.array_equal(np.isnan(got), np.isnan(g)))
        self.assertLess(abs(float(np.nanmean(got))), 0.25 * float(np.nanstd(got)))
        # the ramp is gone: left and right halves have the same mean level
        left = float(np.nanmean(got[:, 6:25])); right = float(np.nanmean(got[:, 25:]))
        self.assertLess(abs(left - right), 0.3)

    def test_highpass_uses_metres_not_cells(self):
        """sigma of 0.4 m is 4 cells at 0.1 m but 2 cells at 0.2 m: same physical size."""
        g = self._grid(9)
        spec = SliceSpec(smooth_lowpass=0, highpass_m=0.4)
        a = engine.finish_grid(g.copy(), spec, 0.1)
        ref = g - slicing._smooth_grid_gaussian(g, sigma=4.0)
        self.assertTrue(np.allclose(a[np.isfinite(g)], ref[np.isfinite(g)], atol=1e-5))


@unittest.skipUnless(HAS_SCIPY, "SciPy required")
class EngineIntegrationTest(unittest.TestCase):
    def _profs(self, spike=False):
        def fill(p, c, n_s, n_t):
            rng = np.random.default_rng(100 * p + c)
            d = rng.normal(0.0, 1.0, (n_s, n_t))
            if spike and p == 2 and c == 3:
                d[100:110, 30] = 80.0        # one very noisy trace
            return d
        return _base._array_survey(n_prof=5, n_ch=6, fill=fill)

    def _spec(self, **kw):
        base = dict(z_min_m=0.9, z_max_m=1.3, thickness_m=0.1, resolution_m=0.1,
                    search_radius_m=0.5, smooth_lowpass=0, balance="none",
                    amplitude_mode="abs_mean")
        base.update(kw)
        return SliceSpec(**base)

    def _grid(self, profs, **kw):
        grids, meta = engine.compute_grids(profs, self._spec(**kw).resolve(profs),
                                           single_window=(1.0, 1.1), emit_diagnostics=False)
        return grids[0]["grid"], meta

    def test_default_spec_is_unchanged_by_the_tools_code(self):
        profs = self._profs()
        spec = self._spec().resolve(profs)
        pts = engine.window_points(profs, engine.collect_geometry(profs)[0], spec, 1.0, 1.1)
        x_min, y_min, n_x, n_y, _ = engine._grid_params(engine.collect_geometry(profs)[0], 0.1)
        raw = slicing.grid_binned(*pts, x_min, y_min, n_x, n_y, 0.1, reach_m=0.5, sigma_cells=1.0)
        got, _ = self._grid(profs)
        self.assertTrue(np.array_equal(got, raw, equal_nan=True))

    def test_despike_tames_the_noisy_trace(self):
        profs = self._profs(spike=True)
        raw, _ = self._grid(profs)
        cut, _ = self._grid(profs, despike_pct=99.0)
        self.assertGreater(float(np.nanmax(raw)), float(np.nanmax(cut)) * 1.5)

    def test_env_mean_runs_end_to_end(self):
        profs = self._profs()
        g, meta = self._grid(profs, amplitude_mode="env_mean")
        self.assertGreater(int(np.isfinite(g).sum()), 50)
        self.assertEqual(meta["amplitude_mode"], "env_mean")

    def test_env_mean_is_calmer_than_env_peak(self):
        profs = self._profs()
        peak, _ = self._grid(profs, amplitude_mode="env_peak")
        mean, _ = self._grid(profs, amplitude_mode="env_mean")
        cv = lambda a: float(np.nanstd(a) / np.nanmean(a))       # noqa: E731
        self.assertLess(cv(mean), cv(peak))

    def test_tools_reach_the_saved_grid_and_the_manifest_spec(self):
        profs = self._profs()
        g, meta = self._grid(profs, normalize="median", amp_scale="sqrt", despeckle=3)
        plain, _ = self._grid(profs)
        self.assertFalse(np.array_equal(g, plain, equal_nan=True))
        d = meta["slice_spec"]
        self.assertEqual((d["normalize"], d["amp_scale"], d["despeckle"]), ("median", "sqrt", 3))

    def test_preview_and_batch_agree_with_the_tools_on(self):
        """The live preview (single window) and the saved set go through the same chain."""
        profs = self._profs()
        spec = self._spec(z_min_m=1.0, z_max_m=1.1, thickness_m=0.1, normalize="median",
                          amp_scale="db", despeckle=3, highpass_m=0.5, smooth_lowpass=5,
                          despike_pct=99.0).resolve(profs)
        one, _ = engine.compute_grids(profs, spec, single_window=(1.0, 1.1), emit_diagnostics=False)
        batch, _ = engine.compute_grids(profs, spec, emit_diagnostics=False)
        self.assertEqual(len(batch), 1)
        self.assertTrue(np.array_equal(one[0]["grid"], batch[0]["grid"], equal_nan=True))


@unittest.skipUnless(HAS_SCIPY, "SciPy required")
class PointsCacheTest(unittest.TestCase):
    """The live preview reuses the extracted amplitudes while only the tools change."""

    def setUp(self):
        self.profs = _base._array_survey(n_prof=4, n_ch=5, noise=1.0, seed=3)
        self.calls = []
        real = engine.window_amplitude

        def spy(*a, **k):
            self.calls.append(1)
            return real(*a, **k)

        p = mock.patch.object(engine, "window_amplitude", spy)
        p.start()
        self.addCleanup(p.stop)

    def _spec(self, **kw):
        base = dict(z_min_m=0.9, z_max_m=1.3, thickness_m=0.1, resolution_m=0.1,
                    search_radius_m=0.5, smooth_lowpass=0, balance="channel",
                    amplitude_mode="env_mean")
        base.update(kw)
        return SliceSpec(**base).resolve(self.profs)

    def _run(self, cache, window=(1.0, 1.1), **kw):
        grids, _ = engine.compute_grids(self.profs, self._spec(**kw), single_window=window,
                                        emit_diagnostics=False, points_cache=cache)
        return grids[0]["grid"]

    def test_cached_result_is_identical_to_uncached(self):
        kw = dict(despike_pct=99.0, normalize="median", amp_scale="db", despeckle=3,
                  highpass_m=0.5, smooth_lowpass=5)
        plain = self._run(None, **kw)
        cache = {}
        first = self._run(cache, **kw)
        again = self._run(cache, **kw)
        self.assertTrue(np.array_equal(plain, first, equal_nan=True))
        self.assertTrue(np.array_equal(plain, again, equal_nan=True))

    def test_changing_the_tools_does_not_repeat_the_extraction(self):
        cache = {}
        self._run(cache)
        n = len(self.calls)
        self.assertGreater(n, 0)
        for kw in ({"smooth_lowpass": 7}, {"despike_pct": 99.0}, {"normalize": "median"},
                   {"amp_scale": "sqrt"}, {"despeckle": 5}, {"highpass_m": 0.6},
                   {"bg_horizontal": True}, {"resolution_m": 0.2}, {"search_radius_m": 0.3}):
            self._run(cache, **kw)
        self.assertEqual(len(self.calls), n, "the extraction must be reused")

    def test_extraction_settings_invalidate(self):
        cache = {}
        self._run(cache)
        n = len(self.calls)
        for kw in ({"amplitude_mode": "abs_mean"}, {"velocity_m_s": 0.6e8}, {"balance": "none"}):
            before = len(self.calls)
            self._run(cache, **kw)
            self.assertGreater(len(self.calls), before, kw)
        before = len(self.calls)
        self._run(cache, window=(1.1, 1.2))
        self.assertGreater(len(self.calls), before)
        self.assertGreater(len(self.calls), n)

    def test_cache_is_bounded(self):
        cache = {}
        for i in range(6):
            self._run(cache, window=(0.8 + 0.1 * i, 0.9 + 0.1 * i))
        self.assertLessEqual(len(cache), engine._POINTS_CACHE_MAX)

    def test_despike_never_mutates_the_cached_points(self):
        cache = {}
        self._run(cache)
        (key, pts), = cache.items()
        before = pts[2].copy()
        self._run(cache, despike_pct=90.0)
        self.assertTrue(np.array_equal(cache[key][2], before))

    def test_no_cache_means_no_state(self):
        self._run(None)
        self._run(None)
        self.assertEqual(len(self.calls), 2 * len(engine.collect_geometry(self.profs)[0]))


@unittest.skipUnless(HAS_SCIPY, "SciPy required")
class ThreadedExtractionTest(unittest.TestCase):
    """The envelope modes extract channels in a thread pool: same numbers, same order."""

    def setUp(self):
        self.profs = _base._array_survey(n_prof=4, n_ch=6, noise=1.0, seed=7)
        self.geoms = engine.collect_geometry(self.profs)[0]

    def _points(self, mode, min_jobs):
        spec = SliceSpec(z_min_m=0.9, z_max_m=1.3, thickness_m=0.1, resolution_m=0.1,
                         search_radius_m=0.5, balance="channel", amplitude_mode=mode).resolve(self.profs)
        with mock.patch.object(engine, "_PARALLEL_MIN_JOBS", min_jobs):
            return engine.window_points_raw(self.profs, self.geoms, spec, 1.0, 1.1)

    def test_pool_gives_bit_identical_points(self):
        for mode in ("env_peak", "env_mean"):
            seq = self._points(mode, 10 ** 9)
            par = self._points(mode, 1)
            for a, b in zip(seq, par):
                self.assertTrue(np.array_equal(a, b), mode)

    def test_pool_is_only_used_for_the_envelope_modes(self):
        used = []
        real = engine.ThreadPoolExecutor

        class Spy(real):
            def __init__(self, *a, **k):
                used.append(1)
                super().__init__(*a, **k)

        with mock.patch.object(engine, "ThreadPoolExecutor", Spy):
            self._points("abs_mean", 1)
            self.assertEqual(used, [])
            self._points("env_mean", 1)
        self.assertEqual(used, [1])

    def test_small_jobs_stay_sequential(self):
        used = []
        real = engine.ThreadPoolExecutor

        class Spy(real):
            def __init__(self, *a, **k):
                used.append(1)
                super().__init__(*a, **k)

        with mock.patch.object(engine, "ThreadPoolExecutor", Spy):
            self._points("env_mean", 10 ** 6)
        self.assertEqual(used, [])

    def test_an_error_in_a_worker_is_not_swallowed(self):
        def boom(data, lo, hi, mode):
            raise RuntimeError("bad channel")

        with mock.patch.object(engine, "window_amplitude", boom):
            with self.assertRaises(RuntimeError):
                self._points("env_mean", 1)


@unittest.skipUnless(HAS_GDAL, "GDAL required")
class ExportedStyleTest(unittest.TestCase):
    """What the viewer shows is what lands in the .qml and inside the GeoTIFF."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tif = os.path.join(self._tmp.name, "s.tif")
        rng = np.random.default_rng(0)
        self.grid = (1.0 + 0.3 * rng.standard_normal((20, 30))).astype(np.float32)
        raster_io.write_tif_singleband(self.grid, self.tif, 0.0, 0.0, 1.9, 0.1, 32633)

    def _qml(self):
        return open(os.path.splitext(self.tif)[0] + ".qml", encoding="utf-8").read()

    def test_fixed_limits_reach_the_qml_and_the_tif(self):
        raster_io.write_qml_singleband(self.tif, self.grid,
                                       {"cmap": "viridis", "clip_lo": 5.0, "clip_hi": 95.0,
                                        "vmin": 0.25, "vmax": 1.75})
        self.assertIn('classificationMin="0.25"', self._qml())
        self.assertIn('classificationMax="1.75"', self._qml())
        vmin, vmax, cols = raster_io.read_embedded_style(self.tif)
        self.assertEqual((vmin, vmax), (0.25, 1.75))
        self.assertEqual(len(cols), 11)

    def test_every_slice_of_a_set_gets_the_same_scale(self):
        other = os.path.join(self._tmp.name, "t.tif")
        raster_io.write_tif_singleband(self.grid * 3.0, other, 0.0, 0.0, 1.9, 0.1, 32633)
        st = {"cmap": "jet", "vmin": 0.5, "vmax": 2.0}
        raster_io.write_qml_singleband(self.tif, self.grid, st)
        raster_io.write_qml_singleband(other, self.grid * 3.0, st)
        self.assertEqual(raster_io.read_embedded_style(self.tif)[:2], raster_io.read_embedded_style(other)[:2])

    def test_without_fixed_limits_the_percentiles_are_used_as_before(self):
        raster_io.write_qml_singleband(self.tif, self.grid, {"cmap": "viridis", "clip_lo": 5.0, "clip_hi": 95.0})
        exp = raster_io.slice_stretch(self.grid, 5.0, 95.0)
        got = raster_io.read_embedded_style(self.tif)[:2]
        self.assertAlmostEqual(got[0], exp[0], places=5)
        self.assertAlmostEqual(got[1], exp[1], places=5)

    def test_unusable_fixed_limits_fall_back(self):
        exp = raster_io.slice_stretch(self.grid, 2.0, 98.0)
        for bad in ({"vmin": float("nan"), "vmax": 1.0}, {"vmin": 0.5}, {"vmax": 1.0},
                    {"vmin": "x", "vmax": 1.0}):
            raster_io.write_qml_singleband(self.tif, self.grid, dict(bad))
            got = raster_io.read_embedded_style(self.tif)[:2]
            self.assertAlmostEqual(got[0], exp[0], places=5, msg=str(bad))
        raster_io.write_qml_singleband(self.tif, self.grid, {"vmin": 2.0, "vmax": 1.0})   # inverted
        vmin, vmax, _ = raster_io.read_embedded_style(self.tif)
        self.assertEqual(vmin, 2.0)
        self.assertGreater(vmax, vmin)


@unittest.skipUnless(HAS_GDAL and HAS_SCIPY, "GDAL and SciPy required")
class ProvenanceInTheTifTest(unittest.TestCase):
    def test_saved_slices_carry_their_spec_and_window(self):
        profs = _base._array_survey(n_prof=4, n_ch=5, noise=1.0, seed=2)
        spec = SliceSpec(z_min_m=0.8, z_max_m=1.2, thickness_m=0.1, resolution_m=0.1,
                         search_radius_m=0.5, amplitude_mode="env_mean", normalize="median",
                         amp_scale="db", despeckle=3, highpass_m=0.6, smooth_lowpass=7,
                         despike_pct=99.5).resolve(profs)
        grids, meta = engine.compute_grids(profs, spec, emit_diagnostics=False)
        self.assertGreaterEqual(len(grids), 2)
        with tempfile.TemporaryDirectory() as d:
            res = slicer.write_grids_to_tifs(grids, meta, d, 32633, style={"cmap": "jet", "vmin": -8.0, "vmax": 8.0})
            self.assertEqual(len(res), len(grids))
            for item, g in zip(res, grids):
                ds = gdal.Open(item["path"])
                md = ds.GetMetadata()
                ds = None
                back = json.loads(md["GPR_SLICE_SPEC"])
                self.assertEqual(back["amplitude_mode"], "env_mean")
                self.assertEqual((back["normalize"], back["amp_scale"], back["despeckle"]), ("median", "db", 3))
                self.assertAlmostEqual(back["highpass_m"], 0.6)
                self.assertAlmostEqual(back["despike_pct"], 99.5)
                self.assertAlmostEqual(float(md["GPR_Z_FROM"]), g["z_from"], places=5)
                self.assertAlmostEqual(float(md["GPR_Z_TO"]), g["z_to"], places=5)
                self.assertEqual((float(md["GPR_STYLE_VMIN"]), float(md["GPR_STYLE_VMAX"])), (-8.0, 8.0))
                # the values in the file are the finished (dB) values, not the raw amplitude
                arr, *_ = raster_io.read_tif_singleband(item["path"])
                self.assertTrue(np.allclose(arr[np.isfinite(arr)], g["grid"][np.isfinite(g["grid"])], atol=1e-4))

    def test_no_spec_no_crash(self):
        grids = [{"z_lev": 0.05, "index": 0, "grid": np.ones((4, 5), dtype=np.float32),
                  "z_from": 0.0, "z_to": 0.1}]
        meta = {"x_min": 0.0, "y_min": 0.0, "y_max": 0.3, "resolution": 0.1, "n_x": 5, "n_y": 4, "x_max": 0.4}
        with tempfile.TemporaryDirectory() as d:
            res = slicer.write_grids_to_tifs(grids, meta, d, 32633)
            md = gdal.Open(res[0]["path"]).GetMetadata()
            self.assertNotIn("GPR_SLICE_SPEC", md)
            self.assertIn("GPR_Z_FROM", md)


if __name__ == "__main__":
    unittest.main()
