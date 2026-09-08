# -*- coding: utf-8 -*-
"""Fase 5a: filtri nel dominio griglia / time-slice (grid_filters.py)."""

import unittest

import numpy as np

from grid_filters import (
    lowpass,
    hipass,
    laplacian,
    gaussian_smooth,
    line_background,
    fft2d_filter,
    fft2d_directional_notch,
    blank,
    mosaic_zero_mean,
    mosaic_histogram_match,
)


def _grid(n_y=40, n_x=60, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n_y, n_x)) * 2.0)


class LowpassHipassTest(unittest.TestCase):
    def test_lowpass_reduces_variance_keeps_shape(self):
        g = _grid()
        lo = lowpass(g, 5)
        self.assertEqual(lo.shape, g.shape)
        self.assertLess(np.nanstd(lo), np.nanstd(g))

    def test_lowpass_hipass_reconstruct(self):
        g = _grid(seed=1)
        np.testing.assert_allclose(lowpass(g, 3) + hipass(g, 3), g, rtol=1e-9, atol=1e-9)

    def test_lowpass_nan_aware(self):
        g = _grid(seed=2)
        g[10:15, 20:25] = np.nan
        lo = lowpass(g, 3)
        # celle originariamente NaN restano NaN; nessuna NaN nuova introdotta
        self.assertTrue(np.all(np.isnan(lo[10:15, 20:25])))
        self.assertEqual(np.isnan(lo).sum(), np.isnan(g).sum())

    def test_laplacian_shape_and_nan(self):
        g = _grid(seed=3)
        g[0, 0] = np.nan
        out = laplacian(g)
        self.assertEqual(out.shape, g.shape)
        self.assertTrue(np.isnan(out[0, 0]))
        self.assertTrue(np.isfinite(out[5, 5]))

    def test_gaussian_smooth_reduces_variance(self):
        g = _grid(seed=4)
        sm = gaussian_smooth(g, 1.5)
        self.assertEqual(sm.shape, g.shape)
        self.assertLess(np.nanstd(sm), np.nanstd(g))
        np.testing.assert_allclose(gaussian_smooth(g, 0.0), g)


class LineBackgroundTest(unittest.TestCase):
    def test_removes_per_row_stripe(self):
        g = _grid(seed=5)
        stripe = np.linspace(-3, 3, g.shape[0])[:, None]   # bias per riga
        gs = g + stripe
        out = line_background(gs, axis="x", length=0)       # media intera riga
        self.assertLess(abs(float(np.nanmean(out))), 0.2)
        # media per riga ~ 0
        self.assertLess(float(np.nanmax(np.abs(np.nanmean(out, axis=1)))), 1e-6)

    def test_percentile_clip_protects_targets(self):
        g = np.zeros((10, 40))
        g += 5.0                       # background costante per riga
        g[5, 18:22] = 100.0            # target forte su una riga
        # senza clip il target si "spalma" (media riga alzata) -> residuo negativo altrove
        no_clip = line_background(g, axis="x", length=0)
        with_clip = line_background(g, axis="x", length=0, clip_hi_pct=90.0)
        self.assertLess(no_clip[5, 0], -1.0)               # ombra
        self.assertAlmostEqual(with_clip[5, 0], 0.0, places=6)  # niente ombra
        self.assertGreater(with_clip[5, 20], 90.0)         # target preservato

    def test_axis_y_columns(self):
        g = _grid(seed=6)
        col_bias = np.linspace(-2, 2, g.shape[1])[None, :]
        out = line_background(g + col_bias, axis="y", length=0)
        self.assertLess(float(np.nanmax(np.abs(np.nanmean(out, axis=0)))), 1e-6)


class Fft2dTest(unittest.TestCase):
    def test_fft2d_filter_roundtrip_and_blank(self):
        g = _grid(seed=7)
        ones = np.ones_like(g)
        np.testing.assert_allclose(fft2d_filter(g, ones), g, rtol=1e-6, atol=1e-6)
        zeros = np.zeros_like(g)
        self.assertLess(float(np.nanmax(np.abs(fft2d_filter(g, zeros)))), 1e-6)

    def test_fft2d_filter_shape_guard(self):
        g = _grid(seed=8)
        with self.assertRaises(ValueError):
            fft2d_filter(g, np.ones((3, 3)))

    def test_directional_notch_attenuates_striping(self):
        ny, nx = 64, 64
        yy, xx = np.mgrid[0:ny, 0:nx]
        # strisce diagonali ~45°: pattern periodico
        stripes = np.sin(2 * np.pi * (xx + yy) / 6.0)
        g = _grid(ny, nx, seed=9) * 0.2 + 3.0 * stripes
        out = fft2d_directional_notch(g, angle_deg=45.0 + 90.0, halfwidth_deg=15.0, mode="reject")
        self.assertEqual(out.shape, g.shape)
        self.assertLess(np.nanstd(out), np.nanstd(g))
        # reject + pass ~ input (DC a parte, tolleranza ampia)
        keep = fft2d_directional_notch(g, angle_deg=135.0, halfwidth_deg=15.0, mode="pass")
        np.testing.assert_allclose(out + keep, g, rtol=1e-3, atol=1e-2)

    def test_blank(self):
        g = _grid(seed=10)
        mk = np.zeros(g.shape, dtype=bool)
        mk[0:5, 0:5] = True
        out = blank(g, mk)
        self.assertTrue(np.all(np.isnan(out[0:5, 0:5])))
        self.assertTrue(np.isfinite(out[20, 20]))


class MosaicTest(unittest.TestCase):
    def test_zero_mean_grid(self):
        gs = [_grid(seed=11) + 5.0, _grid(seed=12) - 3.0]
        out = mosaic_zero_mean(gs, scope="grid")
        for a in out:
            self.assertAlmostEqual(float(np.nanmean(a)), 0.0, places=6)

    def test_zero_mean_line(self):
        g = _grid(seed=13) + np.linspace(0, 10, 40)[:, None]
        out = mosaic_zero_mean([g], scope="line", axis="x")[0]
        self.assertLess(float(np.nanmax(np.abs(np.nanmean(out, axis=1)))), 1e-6)

    def test_histogram_match(self):
        rng = np.random.default_rng(14)
        ref = rng.normal(10.0, 3.0, size=(50, 50))
        src = rng.normal(0.0, 1.0, size=(40, 40))
        out = mosaic_histogram_match(src, ref)
        self.assertEqual(out.shape, src.shape)
        self.assertAlmostEqual(float(np.nanmean(out)), float(np.nanmean(ref)), delta=0.6)
        self.assertAlmostEqual(float(np.nanstd(out)), float(np.nanstd(ref)), delta=0.6)


if __name__ == "__main__":
    unittest.main()
