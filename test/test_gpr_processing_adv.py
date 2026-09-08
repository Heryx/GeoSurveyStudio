import unittest
import numpy as np

from gpr_processing import (
    stolt_migration_2d,
    background_removal,
    agc_gain,
    apply_pipeline,
    DEFAULT_PIPELINE,
    DEFAULT_CHAIN_ORDER,
    FILTER_REGISTRY,
    _resolve_chain_order,
    dcshift,
    boxcar_smooth,
    notch_filter,
    spectral_whitening,
    amplitude_equalization,
    radar_threshold,
    power_gain,
    bandpass_filter,
    eigen_filter,
    median_filter2d,
    fk_filter,
    fk_dip_filter,
    kirchhoff_migration_2d,
    instantaneous_attributes,
    nmo_correction,
    cmp_semblance,
    multifreq_blend,
)


class TestGprProcessingAdvanced(unittest.TestCase):

    def test_stolt_migration_diffraction_focus(self):
        n_s, n_t = 256, 128
        dt_ns = 0.5
        dx_m = 0.05
        v = 1e8  # 0.1 m/ns
        t0_ns = 60.0
        x0_m = 64 * dx_m

        f0 = 100e6
        t_wave = np.arange(-20, 21) * (dt_ns * 1e-9)
        ricker = (1.0 - 2.0 * (np.pi * f0 * t_wave) ** 2) * np.exp(-(np.pi * f0 * t_wave) ** 2)

        synth = np.zeros((n_s, n_t), dtype=np.float64)
        for it in range(n_t):
            x = it * dx_m
            t = np.sqrt(t0_ns ** 2 + (4.0 * (x - x0_m) ** 2) / (0.1 ** 2))
            samp = int(round(t / dt_ns))
            for iw, w_val in enumerate(ricker):
                s_idx = samp + iw - len(ricker) // 2
                if 0 <= s_idx < n_s:
                    synth[s_idx, it] += w_val

        mig = stolt_migration_2d(synth, dt_ns=dt_ns, dx_m=dx_m, velocity_m_s=v)
        self.assertEqual(mig.shape, (n_s, n_t))
        self.assertTrue(np.isfinite(mig).all())

        # Check peak location
        max_idx = np.unravel_index(np.argmax(mig), mig.shape)
        apex_samp = int(round(t0_ns / dt_ns))
        self.assertAlmostEqual(max_idx[0], apex_samp, delta=3)
        self.assertAlmostEqual(max_idx[1], 64, delta=2)

    def test_median_background_removal(self):
        # Array with constant horizontal clutter and one strong localized anomaly
        data = np.ones((50, 40), dtype=np.float32) * 10.0
        data[25, 20] = 500.0  # Strong peak

        bg_mean = background_removal(data, method="mean")
        bg_med = background_removal(data, method="median")

        # Mean subtraction creates negative shadow on other traces on row 25
        self.assertLess(bg_mean[25, 0], -1.0)
        # Median subtraction is unaffected by single outlier peak
        self.assertAlmostEqual(float(bg_med[25, 0]), 0.0, places=4)
        self.assertAlmostEqual(float(bg_med[25, 20]), 490.0, places=4)

    def test_agc_noise_clamping(self):
        noise = np.random.randn(200, 20).astype(np.float32)
        noise[160:, :] *= 1e-7  # deep quiet floor
        agc_raw = agc_gain(noise, window=32)
        agc_clamp = agc_gain(noise, window=32, max_gain_factor=5.0)

        # Deep zone amplitude should be kept under control by max_gain_factor
        self.assertLess(float(agc_clamp[160:, :].std()), float(agc_raw[160:, :].std()))

    def test_pipeline_stage_caching(self):
        data = np.random.randn(100, 30).astype(np.float32)
        cache = {}
        p = dict(DEFAULT_PIPELINE)
        p["dewow"] = True
        p["bg_removal"] = True
        p["agc"] = True
        p["agc_win"] = 64

        out1 = apply_pipeline(data, p, dt_ns=0.1, stage_cache=cache)
        self.assertGreater(len(cache), 0)

        # Record cached array for first step (dewow)
        cached_step0 = cache[0][1]

        # Change downstream parameter (AGC window)
        p["agc_win"] = 128
        out2 = apply_pipeline(data, p, dt_ns=0.1, stage_cache=cache)
        cached_step0_after = cache[0][1]

        # First step output array object is identical (instantaneous cache hit)
        self.assertIs(cached_step0, cached_step0_after)

    def test_chain_order_migration(self):
        self.assertIn("migration", DEFAULT_CHAIN_ORDER)
        self.assertIn("migration", FILTER_REGISTRY)
        order = _resolve_chain_order(None, DEFAULT_PIPELINE)
        self.assertIn("migration", order)


class TestPhase1Filters(unittest.TestCase):

    def _synth(self, n_s=200, n_t=40):
        rng = np.random.default_rng(0)
        return (rng.standard_normal((n_s, n_t)) * 3.0).astype(np.float32)

    def test_dcshift_removes_per_trace_offset(self):
        data = self._synth()
        data[:, 5] += 12.0  # DC offset on one trace
        out = dcshift(data)
        self.assertEqual(out.shape, data.shape)
        self.assertAlmostEqual(float(out.mean(axis=0)[5]), 0.0, places=3)
        self.assertAlmostEqual(float(out.mean(axis=0).max()), 0.0, places=3)

    def test_boxcar_reduces_variance_and_keeps_shape(self):
        data = self._synth()
        out = boxcar_smooth(data, width=2, height=2)
        self.assertEqual(out.shape, data.shape)
        self.assertLess(float(out.std()), float(data.std()))
        # width=0 height=0 is a no-op
        np.testing.assert_allclose(boxcar_smooth(data, 0, 0), data)

    def test_notch_attenuates_target_frequency(self):
        dt_ns = 0.4
        n_s, n_t = 512, 8
        t = np.arange(n_s) * dt_ns * 1e-9
        f_noise_hz = 200e6
        base = np.random.default_rng(1).standard_normal((n_s, n_t)).astype(np.float32)
        tone = np.sin(2 * np.pi * f_noise_hz * t).astype(np.float32)[:, None]
        data = base + 5.0 * tone
        out = notch_filter(data, dt_ns, freq_mhz=200.0, bandwidth_mhz=40.0)
        spec_in = np.abs(np.fft.rfft(data[:, 0]))
        spec_out = np.abs(np.fft.rfft(out[:, 0]))
        k = int(round(f_noise_hz / (np.fft.rfftfreq(n_s, dt_ns * 1e-9)[1])))
        self.assertLess(spec_out[k], 0.5 * spec_in[k])
        # freq_mhz <= 0 -> passthrough
        np.testing.assert_allclose(notch_filter(data, dt_ns, 0.0), data)

    def test_whitening_flattens_spectrum(self):
        dt_ns = 0.4
        n_s, n_t = 512, 6
        rng = np.random.default_rng(2)
        # red-ish noise: cumulative sum has a steep spectrum
        data = np.cumsum(rng.standard_normal((n_s, n_t)), axis=0).astype(np.float32)
        out = spectral_whitening(data, dt_ns, 0.0, 0.0, gain=1.0)
        self.assertEqual(out.shape, data.shape)
        self.assertTrue(np.isfinite(out).all())
        s_in = np.abs(np.fft.rfft(data[:, 0]))[1:]
        s_out = np.abs(np.fft.rfft(out[:, 0]))[1:]
        # coefficient of variation of the magnitude spectrum should drop
        cv_in = s_in.std() / (s_in.mean() + 1e-12)
        cv_out = s_out.std() / (s_out.mean() + 1e-12)
        self.assertLess(cv_out, cv_in)

    def test_amplitude_equalization_balances_depth(self):
        data = self._synth(n_s=300)
        data[150:, :] *= 0.02  # weak deep zone
        out = amplitude_equalization(data, window=64)
        self.assertEqual(out.shape, data.shape)
        shallow = float(np.sqrt((out[:150] ** 2).mean()))
        deep = float(np.sqrt((out[150:] ** 2).mean()))
        self.assertGreater(deep / shallow, 0.3)  # was ~0.02 before

    def test_radar_threshold_zeros_low_amplitudes(self):
        data = self._synth()
        amax = float(np.max(np.abs(data)))
        out = radar_threshold(data, neg_fraction=0.5, pos_fraction=0.5)
        kept = out[out != 0.0]
        self.assertTrue(np.all(np.abs(kept) >= 0.5 * amax - 1e-4))
        self.assertGreater(int(np.count_nonzero(out == 0.0)), 0)

    def test_power_gain_amplifies_with_depth(self):
        data = np.ones((200, 10), dtype=np.float32)
        out = power_gain(data, dt_ns=0.4, mode="power", exponent=2.0)
        self.assertEqual(out.shape, data.shape)
        self.assertGreater(float(np.abs(out[-1]).mean()), float(np.abs(out[10]).mean()))

    def test_phase1_filters_registered_and_default_off(self):
        for sid in ("dcshift", "boxcar", "notch", "whiten", "amp_eq",
                    "rad_threshold", "power_gain"):
            self.assertIn(sid, FILTER_REGISTRY)
            self.assertIn(sid, DEFAULT_PIPELINE)
            self.assertFalse(DEFAULT_PIPELINE[sid], f"{sid} should default OFF")

    def test_apply_pipeline_runs_phase1_step_via_chain_order(self):
        data = self._synth(n_s=128, n_t=24)
        p = dict(DEFAULT_PIPELINE)
        for k in ("dewow", "bg_removal", "agc"):
            p[k] = False
        p["boxcar"] = True
        p["boxcar_width"] = 2
        p["boxcar_height"] = 2
        p["chain_order"] = ["boxcar"]
        out = apply_pipeline(data, p, dt_ns=0.4, normalize_output=False)
        self.assertEqual(out.shape, data.shape)
        self.assertFalse(np.allclose(out, data))
        # disabled -> identity (within normalize-free path)
        p["boxcar"] = False
        out_off = apply_pipeline(data, p, dt_ns=0.4, normalize_output=False)
        np.testing.assert_allclose(out_off, data, rtol=1e-5, atol=1e-4)

    def test_apply_pipeline_phase1_stage_cache_roundtrip(self):
        data = self._synth(n_s=128, n_t=20)
        p = dict(DEFAULT_PIPELINE)
        p["notch"] = True
        p["notch_freq_mhz"] = 150.0
        p["chain_order"] = ["notch"]
        cache = {}
        o1 = apply_pipeline(data, p, dt_ns=0.4, stage_cache=cache, normalize_output=False)
        o2 = apply_pipeline(data, p, dt_ns=0.4, stage_cache=cache, normalize_output=False)
        np.testing.assert_array_equal(o1, o2)

    def test_bandpass_taper_frac_default_and_effect(self):
        dt_ns = 0.4
        n_s, n_t = 512, 6
        rng = np.random.default_rng(7)
        data = rng.standard_normal((n_s, n_t)).astype(np.float32)
        # default keyword == explicit 0.12
        a = bandpass_filter(data, dt_ns, 100.0, 600.0)
        b = bandpass_filter(data, dt_ns, 100.0, 600.0, taper_frac=0.12)
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)
        # taper is clamped and both extremes stay finite and shape-stable
        sharp = bandpass_filter(data, dt_ns, 100.0, 600.0, taper_frac=0.0)
        soft = bandpass_filter(data, dt_ns, 100.0, 600.0, taper_frac=0.5)
        self.assertEqual(sharp.shape, data.shape)
        self.assertEqual(soft.shape, data.shape)
        self.assertTrue(np.isfinite(sharp).all() and np.isfinite(soft).all())
        # different taper -> different output
        self.assertFalse(np.allclose(sharp, soft))
        # out-of-range taper does not raise (clamped internally)
        _ = bandpass_filter(data, dt_ns, 100.0, 600.0, taper_frac=9.0)

    def test_bandpass_taper_in_pipeline_fingerprint(self):
        data = self._synth(n_s=256, n_t=16)
        base = dict(DEFAULT_PIPELINE)
        base["bandpass"] = True
        base["chain_order"] = ["bandpass"]
        o1 = apply_pipeline(data, {**base, "bp_taper_frac": 0.05},
                            dt_ns=0.4, normalize_output=False)
        o2 = apply_pipeline(data, {**base, "bp_taper_frac": 0.40},
                            dt_ns=0.4, normalize_output=False)
        self.assertFalse(np.allclose(o1, o2))

    def test_phase1_steps_in_default_chain_order(self):
        for sid in ("dcshift", "boxcar", "notch", "whiten", "amp_eq",
                    "rad_threshold", "power_gain"):
            self.assertIn(sid, DEFAULT_CHAIN_ORDER)
        order = _resolve_chain_order(list(DEFAULT_CHAIN_ORDER), DEFAULT_PIPELINE)
        for sid in ("dcshift", "notch", "power_gain"):
            self.assertIn(sid, order)

    def test_default_chain_order_all_off_is_near_identity(self):
        # With DEFAULT_PIPELINE the only ON steps are dewow+bg_removal+agc;
        # the 7 new phase-1 steps are OFF and must not alter the result vs the
        # old explicit order.
        data = self._synth(n_s=160, n_t=32)
        p = dict(DEFAULT_PIPELINE)
        old_order = ["dewow", "timezero", "bg_removal", "bandpass",
                     "migration", "pre_agc_gain", "agc", "envelope"]
        a = apply_pipeline(data, {**p, "chain_order": old_order}, dt_ns=0.3)
        b = apply_pipeline(data, {**p, "chain_order": list(DEFAULT_CHAIN_ORDER)}, dt_ns=0.3)
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)


class TestPhase4Filters(unittest.TestCase):

    def _synth(self, n_s=160, n_t=48, seed=0):
        rng = np.random.default_rng(seed)
        return (rng.standard_normal((n_s, n_t)) * 2.0).astype(np.float32)

    def test_eigen_filter_removes_coherent_band(self):
        sig = self._synth()
        band = sig + 30.0 * np.ones_like(sig)          # strong coherent horizontal
        band[40, :] += np.linspace(-5, 5, band.shape[1])  # slight tilt
        out = eigen_filter(band, drop=1)
        self.assertEqual(out.shape, band.shape)
        self.assertLess(float(np.abs(out).mean()), float(np.abs(band).mean()))
        # drop >= rank -> zeros
        z = eigen_filter(band, drop=999)
        self.assertTrue(np.allclose(z, 0.0))
        # drop=0 -> reconstruction ~ identity
        r = eigen_filter(band, drop=0)
        np.testing.assert_allclose(r, band, rtol=1e-3, atol=1e-2)

    def test_median_filter2d_despikes(self):
        data = np.zeros((60, 40), dtype=np.float32)
        data[30, 20] = 500.0
        out = median_filter2d(data, w_trace=1, w_sample=1)
        self.assertAlmostEqual(float(out[30, 20]), 0.0, places=4)
        # no-op
        np.testing.assert_allclose(median_filter2d(data, 0, 0), data)

    def test_fk_filter_mask_roundtrip(self):
        data = self._synth(seed=2)
        # mask=None -> identity
        np.testing.assert_allclose(fk_filter(data, 0.4, 0.05, None), data)
        # all-ones mask -> FFT/IFFT round-trip (within numeric noise)
        ones = np.ones_like(data, dtype=np.float64)
        rt = fk_filter(data, 0.4, 0.05, ones)
        np.testing.assert_allclose(rt, data, rtol=1e-4, atol=1e-3)
        # all-zeros mask -> ~zeros
        zeros = np.zeros_like(data, dtype=np.float64)
        self.assertLess(float(np.abs(fk_filter(data, 0.4, 0.05, zeros)).max()), 1e-3)

    def test_fk_dip_filter_shape_and_modes_are_complementary(self):
        data = self._synth(seed=3)
        rej = fk_dip_filter(data, 0.4, 0.05, v_min=0.0, v_max=5e7, mode="reject", taper=0.0)
        pas = fk_dip_filter(data, 0.4, 0.05, v_min=0.0, v_max=5e7, mode="pass", taper=0.0)
        self.assertEqual(rej.shape, data.shape)
        self.assertTrue(np.isfinite(rej).all() and np.isfinite(pas).all())
        # reject + pass should reconstruct the (DC-inclusive) input closely
        np.testing.assert_allclose(rej + pas, data, rtol=1e-3, atol=1e-2)

    def test_kirchhoff_focuses_diffraction(self):
        n_s, n_t = 256, 128
        dt_ns, dx_m, v = 0.5, 0.05, 1e8
        t0_ns = 60.0
        x0 = 64 * dx_m
        f0 = 100e6
        tw = np.arange(-20, 21) * (dt_ns * 1e-9)
        ricker = (1.0 - 2.0 * (np.pi * f0 * tw) ** 2) * np.exp(-(np.pi * f0 * tw) ** 2)
        synth = np.zeros((n_s, n_t), dtype=np.float64)
        for it in range(n_t):
            x = it * dx_m
            t = np.sqrt(t0_ns ** 2 + (4.0 * (x - x0) ** 2) / (0.1 ** 2))
            samp = int(round(t / dt_ns))
            for iw, wv in enumerate(ricker):
                s = samp + iw - len(ricker) // 2
                if 0 <= s < n_s:
                    synth[s, it] += wv
        mig = kirchhoff_migration_2d(synth, dt_ns, dx_m, v, aperture_m=n_t * dx_m)
        self.assertEqual(mig.shape, (n_s, n_t))
        self.assertTrue(np.isfinite(mig).all())
        peak = np.unravel_index(np.argmax(np.abs(mig)), mig.shape)
        self.assertAlmostEqual(peak[0], int(round(t0_ns / dt_ns)), delta=6)
        self.assertAlmostEqual(peak[1], 64, delta=4)

    def test_kirchhoff_accepts_velocity_profile(self):
        data = self._synth(n_s=128, n_t=32, seed=9)
        v_of_t = np.linspace(0.8e8, 1.2e8, 128)
        out = kirchhoff_migration_2d(data, 0.4, 0.05, v_of_t, aperture_m=0.8)
        self.assertEqual(out.shape, data.shape)
        self.assertTrue(np.isfinite(out).all())

    def test_phase4_registered_default_off_and_in_chain(self):
        for sid in ("eigen_filter", "median2d", "fk_dip", "kirchhoff"):
            self.assertIn(sid, FILTER_REGISTRY)
            self.assertIn(sid, DEFAULT_PIPELINE)
            self.assertFalse(DEFAULT_PIPELINE[sid])
            self.assertIn(sid, DEFAULT_CHAIN_ORDER)

    def test_phase4_pipeline_on_off(self):
        data = self._synth(n_s=96, n_t=24)
        p = dict(DEFAULT_PIPELINE)
        for k in ("dewow", "bg_removal", "agc"):
            p[k] = False
        p["chain_order"] = ["median2d"]
        p["median2d"] = True
        p["median2d_w_trace"] = 1
        p["median2d_w_sample"] = 1
        on = apply_pipeline(data, p, dt_ns=0.4, normalize_output=False)
        self.assertFalse(np.allclose(on, data))
        p["median2d"] = False
        off = apply_pipeline(data, p, dt_ns=0.4, normalize_output=False)
        np.testing.assert_allclose(off, data, rtol=1e-5, atol=1e-4)


class TestPhase6Filters(unittest.TestCase):

    def _synth(self, n_s=200, n_t=32, seed=0):
        rng = np.random.default_rng(seed)
        return (rng.standard_normal((n_s, n_t)) * 2.0).astype(np.float32)

    def test_instantaneous_attributes_kinds(self):
        d = self._synth(seed=1)
        ph = instantaneous_attributes(d, 0.4, "phase")
        self.assertEqual(ph.shape, d.shape)
        self.assertTrue(np.all(ph >= -np.pi - 1e-3) and np.all(ph <= np.pi + 1e-3))
        cp = instantaneous_attributes(d, 0.4, "cos_phase")
        self.assertTrue(np.all(np.abs(cp) <= 1.0 + 1e-5))
        fr = instantaneous_attributes(d, 0.4, "frequency")
        self.assertTrue(np.all(fr >= 0.0) and np.isfinite(fr).all())
        env = instantaneous_attributes(d, 0.4, "envelope")
        self.assertTrue(np.all(env >= 0.0))

    def test_instantaneous_frequency_of_tone(self):
        # pure tone at f0 -> instantaneous frequency ~ f0 (away from edges)
        n_s = 512
        dt_ns = 0.2
        f0_hz = 300e6
        t = np.arange(n_s) * dt_ns * 1e-9
        tone = np.sin(2 * np.pi * f0_hz * t).astype(np.float32)[:, None] * np.ones((1, 4), np.float32)
        fr = instantaneous_attributes(tone, dt_ns, "frequency")
        mid = fr[100:400, 0]
        self.assertAlmostEqual(float(np.median(mid)), 300.0, delta=15.0)  # MHz

    def test_nmo_flattens_gather(self):
        n_s, n_t = 400, 24
        dt_ns, v, t0_ns = 0.4, 8e7, 100.0
        dt = dt_ns * 1e-9
        t0 = t0_ns * 1e-9
        off = np.arange(n_t) * 0.1
        g = np.zeros((n_s, n_t), dtype=np.float32)
        for j in range(n_t):
            s = int(round(np.sqrt(t0 ** 2 + (off[j] / v) ** 2) / dt))
            if s < n_s:
                g[s, j] = 1.0
        nmo = nmo_correction(g, dt_ns, 0.1, v)
        self.assertEqual(nmo.shape, g.shape)
        self.assertEqual(int(np.argmax(nmo.sum(axis=1))), int(round(t0 / dt)))
        # a wrong (too low) velocity does NOT flatten -> lower peak coherence
        bad = nmo_correction(g, dt_ns, 0.1, v * 0.5)
        self.assertGreater(nmo.sum(axis=1).max(), bad.sum(axis=1).max())

    def test_cmp_semblance_peaks_at_true_velocity(self):
        n_s, n_t = 400, 24
        dt_ns, v_true, t0_ns = 0.4, 8e7, 100.0
        dt, t0 = dt_ns * 1e-9, t0_ns * 1e-9
        off = np.arange(n_t) * 0.1
        g = np.zeros((n_s, n_t), dtype=np.float32)
        for j in range(n_t):
            s = int(round(np.sqrt(t0 ** 2 + (off[j] / v_true) ** 2) / dt))
            if s < n_s:
                g[max(s - 1, 0):s + 2, j] = 1.0
        sem, v_axis = cmp_semblance(g, dt_ns, 0.1, v_min=4e7, v_max=1.2e8, n_v=40)
        self.assertEqual(sem.shape, (n_s, v_axis.size))
        row = int(round(t0 / dt))
        v_best = v_axis[int(np.argmax(sem[row]))]
        self.assertAlmostEqual(v_best / 1e7, v_true / 1e7, delta=0.6)

    def test_multifreq_blend_identity_and_weights(self):
        d = self._synth(seed=4)
        b = multifreq_blend([d, d], [0.5, 0.5], [0.4, 0.4])
        np.testing.assert_allclose(b, d, rtol=1e-4, atol=1e-3)
        # single input -> itself
        np.testing.assert_allclose(multifreq_blend([d], [1.0], [0.4]), d, atol=1e-4)
        # resample: coarse dt array upsampled onto fine target keeps shape of target
        coarse = d[::2, :]
        out = multifreq_blend([d, coarse], [0.5, 0.5], [0.4, 0.8], target=0)
        self.assertEqual(out.shape, d.shape)

    def test_inst_attr_registered_and_pipeline(self):
        self.assertIn("inst_attr", FILTER_REGISTRY)
        self.assertIn("inst_attr", DEFAULT_CHAIN_ORDER)
        self.assertFalse(DEFAULT_PIPELINE["inst_attr"])
        data = self._synth(n_s=128, n_t=20)
        p = dict(DEFAULT_PIPELINE)
        for k in ("dewow", "bg_removal", "agc"):
            p[k] = False
        p["inst_attr"] = True
        p["inst_attr_kind"] = "cos_phase"
        p["chain_order"] = ["inst_attr"]
        out = apply_pipeline(data, p, dt_ns=0.4, normalize_output=False)
        self.assertEqual(out.shape, data.shape)
        self.assertTrue(np.all(np.abs(out) <= 1.0 + 1e-4))   # cos_phase range
        p["inst_attr"] = False
        off = apply_pipeline(data, p, dt_ns=0.4, normalize_output=False)
        np.testing.assert_allclose(off, data, rtol=1e-5, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
