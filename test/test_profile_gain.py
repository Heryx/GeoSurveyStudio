# -*- coding: utf-8 -*-
"""TVG per profilo: curve proprie, preset adattivo, uso nel salvataggio/slicer."""

import dataclasses
import unittest
from types import SimpleNamespace

import numpy as np

from .test_profile_viewer_interaction import (  # imposta anche il sys.path del pacchetto
    HAS_MPL,
    HAS_VIEWER,
    _ViewerTestBase,
    _make_profile,
)

import importlib

_G = importlib.import_module("rasterlinker_feature_interpolation.gpr_profile_gain")
BY_PROFILE_KEY = _G.BY_PROFILE_KEY
adapt_curve = _G.adapt_curve
adaptive_gain_curves = _G.adaptive_gain_curves
attenuation_index = _G.attenuation_index
profile_gain_key = _G.profile_gain_key
resolve_profile_params = _G.resolve_profile_params
_CM = importlib.import_module("rasterlinker_feature_interpolation.gpr_viewer_common")


def _prof(path, decay, n_s=400, n_t=60, seed=0):
    """Profilo sintetico: rumore con ampiezza che decade come exp(-decay*x)."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n_s)[:, None]
    data = (rng.normal(0, 1.0, (n_s, n_t)) * np.exp(-decay * x)).astype(np.float32)
    ch = SimpleNamespace(data=data)
    return SimpleNamespace(path=path, n_channels=1, channel=lambda i, ch=ch: ch)


BASE = [[0.0, 10.0], [0.5, 100.0], [1.0, 1000.0]]


class ResolveTest(unittest.TestCase):
    def test_key_is_file_stem(self):
        self.assertEqual(profile_gain_key("C:/x/Swath001_Array02.ogpr"), "Swath001_Array02")
        self.assertEqual(profile_gain_key(_prof("a/b/p1.npz", 1.0)), "p1")

    def test_profile_curve_replaces_base_and_map_is_removed(self):
        params = {"pre_agc_gain": True, "pre_agc_breakpoints": BASE,
                  BY_PROFILE_KEY: {"p1": [[0.0, 2.0], [1.0, 50.0]]}}
        out = resolve_profile_params(params, "p1")
        self.assertNotIn(BY_PROFILE_KEY, out)
        self.assertEqual(out["pre_agc_breakpoints"], [[0.0, 2.0], [1.0, 50.0]])
        self.assertEqual(out["pre_agc_surface_gain"], 2.0)
        self.assertEqual(out["pre_agc_deep_gain"], 50.0)
        self.assertIn(BY_PROFILE_KEY, params)  # l'originale non cambia

    def test_unknown_profile_keeps_base(self):
        params = {"pre_agc_breakpoints": BASE, BY_PROFILE_KEY: {"p1": [[0, 1], [1, 2]]}}
        out = resolve_profile_params(params, "other")
        self.assertEqual(out["pre_agc_breakpoints"], BASE)
        self.assertNotIn(BY_PROFILE_KEY, out)


class AdaptiveTest(unittest.TestCase):
    def test_more_attenuated_profile_has_lower_index(self):
        strong = attenuation_index(_prof("a.ogpr", decay=5.0))
        weak = attenuation_index(_prof("b.ogpr", decay=1.0))
        self.assertLess(strong, weak)

    def test_attenuated_profile_gets_more_deep_gain(self):
        profs = [_prof("a.ogpr", 1.0), _prof("b.ogpr", 3.0, seed=1), _prof("c.ogpr", 5.0, seed=2)]
        curves, info = adaptive_gain_curves(profs, BASE)
        deep = {k: v[-1][1] for k, v in curves.items()}
        self.assertLess(deep["a"], deep["b"])
        self.assertLess(deep["b"], deep["c"])
        # il profilo mediano resta alla curva del preset
        self.assertAlmostEqual(deep["b"], 1000.0, delta=1.0)
        # in superficie nessuna modifica
        for v in curves.values():
            self.assertAlmostEqual(v[0][1], 10.0)

    def test_single_profile_is_the_preset_curve(self):
        curves, _ = adaptive_gain_curves([_prof("a.ogpr", 4.0)], BASE)
        np.testing.assert_allclose([p[1] for p in curves["a"]][0::1][-1], 1000.0, rtol=1e-6)
        y = np.interp([0.0, 0.5, 1.0], [p[0] for p in curves["a"]], [p[1] for p in curves["a"]])
        np.testing.assert_allclose(y, [10.0, 100.0, 1000.0], rtol=1e-6)

    def test_factor_is_capped_and_points_fit_the_editor(self):
        pts = adapt_curve(BASE, 50.0)
        self.assertLessEqual(len(pts), 16)
        self.assertAlmostEqual(pts[-1][1] / 1000.0, 6.0, places=3)
        many = [[i / 19.0, 1.0 + i] for i in range(20)]
        self.assertLessEqual(len(adapt_curve(many, 0.3)), 16)

    def test_unmeasurable_profile_gets_preset_curve(self):
        empty = SimpleNamespace(path="z.ogpr", n_channels=0, channel=lambda i: None)
        curves, _ = adaptive_gain_curves([empty, _prof("a.ogpr", 2.0)], BASE)
        self.assertAlmostEqual(curves["z"][-1][1], 1000.0)


class AutoGainTest(unittest.TestCase):
    @staticmethod
    def _env(n=500, burst=0.08, slope_db=-70.0):
        x = np.linspace(0, 1, n)
        env = 10.0 + slope_db * x
        env[x < burst] = 60.0          # onda diretta molto piu' forte
        return env

    def test_log_interpolation_is_exponential_between_points(self):
        proc = importlib.import_module("rasterlinker_feature_interpolation.gpr_processing")
        pts = [[0.0, 1.0], [1.0, 1000.0]]
        lin = proc._build_pre_agc_gain(101, {"pre_agc_breakpoints": pts})[:, 0]
        log = proc._build_pre_agc_gain(101, {"pre_agc_breakpoints": pts, "pre_agc_interp": "log"})[:, 0]
        self.assertAlmostEqual(float(lin[50]), 500.5, places=1)
        self.assertAlmostEqual(float(log[50]), 1000.0 ** 0.5, delta=0.05)   # meta' in dB
        self.assertAlmostEqual(float(log[-1]), 1000.0, delta=0.01)

    def test_ground_wave_end_is_found_after_the_burst(self):
        end = _G.ground_wave_end(self._env(burst=0.08))
        self.assertGreater(end, 0.07)
        self.assertLess(end, 0.2)

    def test_auto_curve_is_flat_on_the_ground_wave_and_compensates_decay(self):
        env = self._env()
        pts = np.asarray(_G.auto_gain_points(env, start=0.1, strength=1.0, max_db=80.0, n_points=12))
        self.assertLessEqual(len(pts), 16)
        self.assertTrue(np.all(pts[pts[:, 0] <= 0.1, 1] == 1.0))
        self.assertTrue(np.all(np.diff(pts[:, 1]) >= -1e-9))
        # la crescita e' progressiva (esponenziale), non un gradino: dopo l'inizio
        # i guadagni salgono per piu' punti, in dB a passi regolari
        db = 20 * np.log10(pts[pts[:, 0] >= 0.1, 1])
        self.assertGreaterEqual(int(np.sum(np.diff(db) > 1.0)), 5)
        # attenuazione ~ 63 dB tra 10% e 100%: compensata (a meno del livellamento)
        self.assertGreater(20 * np.log10(pts[-1, 1]), 45.0)

    def test_strength_and_ceiling_are_respected(self):
        env = self._env()
        half = np.asarray(_G.auto_gain_points(env, start=0.1, strength=0.5, max_db=80.0))
        full = np.asarray(_G.auto_gain_points(env, start=0.1, strength=1.0, max_db=80.0))
        self.assertAlmostEqual(20 * np.log10(half[-1, 1]), 0.5 * 20 * np.log10(full[-1, 1]), delta=0.5)
        capped = np.asarray(_G.auto_gain_points(env, start=0.1, strength=1.0, max_db=20.0))
        self.assertLessEqual(20 * np.log10(capped[-1, 1]), 20.0 + 1e-6)

    def test_auto_for_profiles_follows_each_profile(self):
        profs = [_prof("a.ogpr", 1.0), _prof("b.ogpr", 5.0, seed=1)]
        curves = _G.auto_gain_for_profiles(profs, {"start": 0.1, "strength": 1.0, "max_db": 90.0})
        self.assertEqual(set(curves), {"a", "b"})
        self.assertGreater(curves["b"][-1][1], curves["a"][-1][1])


class AutoAgcTest(unittest.TestCase):
    @staticmethod
    def _data(n_s=512, n_t=40, period=16, burst=40):
        rng = np.random.default_rng(2)
        t = np.arange(n_s)[:, None]
        sig = np.sin(2 * np.pi * t / period) * np.exp(-3.0 * t / n_s)
        d = sig * (1 + 0.1 * rng.normal(size=(1, n_t))) + 0.01 * rng.normal(size=(n_s, n_t))
        d[:burst] *= 200.0            # onda diretta molto piu' forte
        return d

    def test_period_and_window_follow_the_signal(self):
        d = self._data(period=16)
        self.assertAlmostEqual(_G.dominant_period_samples(d, 60), 16.0, delta=2.0)
        p = _G.auto_agc_params(d, periods=4.0)
        self.assertEqual(p["agc_win"] % 2, 0)
        self.assertAlmostEqual(p["agc_win"], 64, delta=10)
        self.assertGreater(p["agc_sample_start"], 20)       # dopo l'onda diretta
        self.assertGreaterEqual(p["agc_max_gain"], 2.0)

    def test_limit_noise_can_be_turned_off(self):
        p = _G.auto_agc_params(self._data(), periods=3.0, limit_noise=False)
        self.assertEqual(p["agc_max_gain"], 0.0)

    def test_resolve_applies_the_profiles_own_agc(self):
        params = {"agc": True, "agc_win": 128, "agc_lateral": 0,
                  _G.AGC_BY_PROFILE_KEY: {"p1": {"agc_win": 48, "agc_max_gain": 20.0}}}
        out = resolve_profile_params(params, "p1")
        self.assertEqual((out["agc_win"], out["agc_max_gain"], out["agc_lateral"]), (48, 20.0, 0))
        self.assertNotIn(_G.AGC_BY_PROFILE_KEY, out)
        self.assertEqual(resolve_profile_params(params, "other")["agc_win"], 128)


@unittest.skipUnless(HAS_MPL and HAS_VIEWER, "matplotlib/viewer non disponibili")
class ViewerPerProfileGainTest(_ViewerTestBase):
    """Wiring sul viewer reale (offscreen)."""

    def setUp(self):
        super().setUp()
        v = self.v
        p1 = _make_profile()
        p2 = dataclasses.replace(_make_profile(), path="second.ogpr")
        v._profiles = [p1, p2]
        v._prof_idx = 0
        v._load_current_profile()
        self._pump(150)

    def test_edit_stays_on_its_profile_and_follows_navigation(self):
        v = self.v
        v._chk_range_gain.setChecked(True)
        v._spin_gain_deep.setValue(40.0)
        self.assertAlmostEqual(v._current_processing_params()["pre_agc_breakpoints"][-1][1], 40.0)
        v._select_profile(1, deferred=False)
        self._pump(100)
        self.assertNotAlmostEqual(v._spin_gain_deep.value(), 40.0)
        v._spin_gain_deep.setValue(90.0)
        v._select_profile(0, deferred=False)
        self._pump(100)
        self.assertAlmostEqual(v._spin_gain_deep.value(), 40.0)
        by = v._current_processing_params()[BY_PROFILE_KEY]
        self.assertAlmostEqual(by["synthetic"][-1][1], 40.0)
        self.assertAlmostEqual(by["second"][-1][1], 90.0)

    def test_copy_to_all_profiles(self):
        v = self.v
        v._chk_range_gain.setChecked(True)
        v._spin_gain_deep.setValue(25.0)
        v._apply_current_range_gain_to_all()
        self.assertNotIn(BY_PROFILE_KEY, v._current_processing_params())
        v._select_profile(1, deferred=False)
        self._pump(100)
        self.assertAlmostEqual(v._spin_gain_deep.value(), 25.0)

    def test_preset_gives_each_profile_its_own_curve(self):
        v = self.v
        # secondo profilo piu' attenuato
        d = np.asarray(v._profiles[1].channel(0).data)
        x = np.linspace(0, 1, d.shape[0])[:, None]
        v._profiles[1].channel(0).data = (d * np.exp(-4.0 * x)).astype(np.float32)
        v._apply_named_preset("opera_duo")
        self._pump(300)
        by = v._current_processing_params()[BY_PROFILE_KEY]
        self.assertEqual(set(by), {"synthetic", "second"})
        self.assertGreater(by["second"][-1][1], by["synthetic"][-1][1])

    def test_variant_params_resolve_per_profile(self):
        v = self.v
        v._chk_range_gain.setChecked(True)
        v._spin_gain_deep.setValue(40.0)
        v._select_profile(1, deferred=False)
        self._pump(100)
        v._spin_gain_deep.setValue(90.0)
        params = v._current_processing_params()
        self.assertEqual(resolve_profile_params(params, "synthetic")["pre_agc_deep_gain"], 40.0)
        self.assertEqual(resolve_profile_params(params, "second")["pre_agc_deep_gain"], 90.0)

    def test_agc_edit_stays_on_its_profile_and_follows_navigation(self):
        v = self.v
        v._chk_agc.setChecked(True)
        v._spin_agc.setValue(48)
        v._select_profile(1, deferred=False)
        self._pump(100)
        self.assertEqual(v._spin_agc.value(), 128)           # l'altro profilo non cambia
        v._spin_agc.setValue(200)
        v._select_profile(0, deferred=False)
        self._pump(100)
        self.assertEqual(v._spin_agc.value(), 48)
        by = v._current_processing_params()[_G.AGC_BY_PROFILE_KEY]
        self.assertEqual(by["synthetic"]["agc_win"], 48)
        self.assertEqual(by["second"]["agc_win"], 200)
        resolved = resolve_profile_params(v._current_processing_params(), "second")
        self.assertEqual(resolved["agc_win"], 200)

    def test_agc_copy_to_all_profiles(self):
        v = self.v
        v._chk_agc.setChecked(True)
        v._spin_agc.setValue(64)
        v._apply_current_agc_to_all()
        self.assertNotIn(_G.AGC_BY_PROFILE_KEY, v._current_processing_params())
        v._select_profile(1, deferred=False)
        self._pump(100)
        self.assertEqual(v._spin_agc.value(), 64)

    def test_agc_editor_panel_apply_and_cancel(self):
        v = self.v
        v._chk_agc.setChecked(True)
        self._pump(50)
        v._open_agc_editor()
        self._pump(100)
        self.assertIsInstance(v._gain_panel, _CM.AgcPanel)
        self.assertFalse(v._gain_curve_host.isHidden())
        self.assertEqual(len(v._ax_wiggle.lines), 3)          # zero, prima, dopo
        v._gain_panel._spin_win.setValue(80)
        self.assertEqual(v._spin_agc.value(), 80)             # live nei controlli
        v._gain_panel.finished_with.emit("cancel")
        self._pump(50)
        self.assertEqual(v._spin_agc.value(), 128)
        self.assertNotIn(v._current_gain_key(), v._agc_by_profile)
        v._open_agc_editor()
        v._gain_panel._spin_win.setValue(96)
        v._gain_panel.finished_with.emit("one")
        self._pump(50)
        self.assertEqual(v._agc_by_profile[v._current_gain_key()]["agc_win"], 96)
        self.assertTrue(v._gain_curve_host.isHidden())

    def test_agc_auto_for_all_profiles_fills_every_profile(self):
        import os
        v = self.v
        v._chk_agc.setChecked(True)
        os.environ["RASTERLINKER_GPR_WORKERS"] = "1"
        self.addCleanup(lambda: os.environ.pop("RASTERLINKER_GPR_WORKERS", None))
        mine = v._auto_agc_all_profiles({"periods": 4.0, "limit_noise": True})
        self.assertIsNotNone(mine)
        self.assertEqual(set(v._agc_by_profile), {"synthetic", "second"})
        for d in v._agc_by_profile.values():
            self.assertEqual(d["agc_win"] % 2, 0)

    def test_project_state_roundtrip_keeps_per_profile_tvg_and_agc(self):
        import json
        v = self.v
        v._chk_range_gain.setChecked(True)
        v._chk_agc.setChecked(True)
        v._spin_gain_deep.setValue(40.0)
        v._spin_agc.setValue(48)
        v._select_profile(1, deferred=False)
        self._pump(100)
        v._spin_gain_deep.setValue(90.0)
        v._spin_agc.setValue(200)
        state = json.loads(json.dumps(v._collect_full_processing_state()))
        v._range_gain_by_profile = {}
        v._agc_by_profile = {}
        v._apply_full_processing_state(state)
        self._pump(100)
        self.assertAlmostEqual(v._range_gain_by_profile["synthetic"][-1][1], 40.0)
        self.assertAlmostEqual(v._range_gain_by_profile["second"][-1][1], 90.0)
        self.assertEqual(v._agc_by_profile["synthetic"]["agc_win"], 48)
        self.assertEqual(v._agc_by_profile["second"]["agc_win"], 200)


if __name__ == "__main__":
    unittest.main()
