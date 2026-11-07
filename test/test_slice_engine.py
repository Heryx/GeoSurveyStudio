# -*- coding: utf-8 -*-
"""Motore time-slice "bin": binned gridding, geometria per canale, profondita'
da dt/velocita', equalizzazione dei canali, stile QML."""

import importlib
import os
import pathlib
import sys
import tempfile
import threading
import unittest

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
slicer = importlib.import_module(PKG + ".gpr_ogpr_slicer")
raster_io = importlib.import_module(PKG + ".raster_io")
slice_spec = importlib.import_module(PKG + ".slice_spec")
SliceSpec = slice_spec.SliceSpec

try:
    import scipy  # noqa: F401
    HAS_SCIPY = True
except Exception:  # pragma: no cover
    HAS_SCIPY = False

try:
    import matplotlib  # noqa: F401
    HAS_MPL = True
except Exception:  # pragma: no cover
    HAS_MPL = False


class _Ch:
    def __init__(self, data, east, north):
        self.data = np.asarray(data, dtype=np.float32)
        self.easting = np.asarray(east, dtype=np.float64)
        self.northing = np.asarray(north, dtype=np.float64)
        self.altitude = np.zeros(self.data.shape[1])
        self.distances = None


class _Prof:
    """Minimal OgprProfile look-alike (channels, dt, velocity, geometry)."""

    def __init__(self, path, chans, dt_ns=0.2, velocity=1.0e8, depth_max_m=None):
        self.path = path
        self._ch = chans
        self.n_channels = len(chans)
        self.n_samples = int(chans[0].data.shape[0])
        self.n_slices = int(chans[0].data.shape[1])
        self.dt_ns = dt_ns
        self.velocity_m_s = velocity
        self.depth_max_m = (
            depth_max_m if depth_max_m is not None
            else self.n_samples * dt_ns * 1e-9 * velocity / 2.0
        )
        self.sampling_step_m = 0.05
        self.epsg = 32633

    def channel(self, i):
        return self._ch[i]


def _array_survey(
    n_prof=6, n_ch=8, pitch=0.04, line_gap=None, n_t=140, n_s=400,
    gains=None, noise=0.0, seed=0, fill=None, pulse_rows=None,
):
    """Parallel N-S lines; each profile is an array of ``n_ch`` channels at
    lateral offsets ``c*pitch``. ``line_gap`` defaults to the swath width so the
    swaths tile the ground. ``fill(p, c, n_s, n_t)`` can override the data."""
    rng = np.random.default_rng(seed)
    swath = n_ch * pitch
    gap = swath if line_gap is None else line_gap
    profs = []
    for p in range(n_prof):
        chans = []
        y = 1000.0 + np.arange(n_t) * 0.05
        for c in range(n_ch):
            x = 500.0 + p * gap + c * pitch + np.zeros(n_t)
            if fill is not None:
                data = fill(p, c, n_s, n_t)
            else:
                data = rng.normal(0.0, noise, (n_s, n_t)) if noise else np.zeros((n_s, n_t))
            if gains is not None:
                data = data * gains[c]
            if pulse_rows is not None:
                a, b = pulse_rows
                data[a:b, :] += 1.0 * (gains[c] if gains is not None else 1.0)
            chans.append(_Ch(data, x, y))
        profs.append(_Prof(f"p{p}.ogpr", chans))
    return profs


def _bead_ratio(g):
    """How much rougher the roughest column is than a typical one.

    Roughness = |cell - mean(4 neighbours)|. IDW on the collapsed centre line makes
    the cells ON each track far noisier than the blended cells between tracks, so
    one column per track stands out (large ratio); a uniform field gives ~1-2."""
    inner = g[1:-1, 1:-1]
    nb = (g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]) / 4.0
    d = np.abs(inner - nb)
    col = np.nanmean(d[10:-10], axis=0)   # skip the ragged ends of the lines
    col = col[np.isfinite(col)]
    return float(col.max() / np.median(col))


class DepthMappingTest(unittest.TestCase):
    def test_depth_per_sample(self):
        self.assertAlmostEqual(slicing.depth_per_sample_m(0.125, 1e8), 0.00625)
        self.assertEqual(slicing.depth_per_sample_m(0.0, 1e8), 0.0)
        self.assertEqual(slicing.depth_per_sample_m(0.1, None), 0.0)
        self.assertEqual(slicing.depth_per_sample_m(float("nan"), 1e8), 0.0)

    def test_window_sample_range_ignores_nominal_depth(self):
        # 0.01 m per row: 1.50-1.60 m is rows 150..160 whatever the matrix height.
        self.assertEqual(slicing.window_sample_range(1.5, 1.6, 0.01, 400), (150, 160))
        self.assertEqual(slicing.window_sample_range(1.5, 1.6, 0.01, 155), (150, 155))
        self.assertEqual(slicing.window_sample_range(9.0, 9.1, 0.01, 400), (0, 0))
        lo, hi = slicing.window_sample_range(0.0, 0.0001, 0.01, 400)
        self.assertEqual((lo, hi), (0, 1))  # never an empty window

    def test_legacy_mapping_was_stretched(self):
        """Documents the bug the new mapping fixes: matrix cropped to 866 rows of a
        985-row nominal record puts 4.08 m ~12% too shallow with the old rule."""
        old = slicing._depth_to_sample_range(4.08, 4.18, 6.15625, 866)[0]
        new = slicing.window_sample_range(4.08, 4.18, 0.00625, 866)[0]
        self.assertEqual(new, 652)
        self.assertLess(old, 600)


@unittest.skipUnless(HAS_SCIPY, "SciPy required")
class WindowAmplitudeTest(unittest.TestCase):
    def setUp(self):
        n = 200
        t = np.arange(n)
        self.data = (2.0 * np.sin(2 * np.pi * t / 16.0))[:, None].astype(np.float32).repeat(3, axis=1)

    def test_modes(self):
        lo, hi = 40, 104  # 4 full periods
        wa = slicing.window_amplitude
        self.assertTrue(np.allclose(wa(self.data, lo, hi, "signed_mean"), 0.0, atol=1e-5))
        self.assertTrue(np.allclose(wa(self.data, lo, hi, "max_abs"), 2.0, atol=1e-3))
        self.assertTrue(np.allclose(wa(self.data, lo, hi, "rms"), 2.0 / np.sqrt(2), atol=1e-3))
        self.assertTrue(np.allclose(wa(self.data, lo, hi, "abs_mean"), 4.0 / np.pi, atol=3e-2))
        self.assertTrue(np.allclose(wa(self.data, lo, hi, "env_peak"), 2.0, atol=0.1))

    def test_signed_mean_keeps_sign(self):
        d = np.full((50, 2), -3.0, dtype=np.float32)
        self.assertTrue(np.allclose(slicing.window_amplitude(d, 10, 20, "signed_mean"), -3.0))

    def test_env_peak_only_looks_at_padded_window(self):
        d = np.zeros((300, 1), dtype=np.float32)
        d[280, 0] = 100.0  # far outside the window and its context
        self.assertLess(float(slicing.window_amplitude(d, 50, 70, "env_peak")[0]), 1e-3)

    def test_bad_mode(self):
        with self.assertRaises(ValueError):
            slicing.window_amplitude(self.data, 0, 10, "nope")


@unittest.skipUnless(HAS_SCIPY, "SciPy required")
class GridBinnedTest(unittest.TestCase):
    def _line_points(self, xs, y0=0.0, y1=6.0, step=0.05, value=1.0):
        y = np.arange(y0, y1, step)
        x = np.repeat(np.asarray(xs, dtype=float), y.size)
        return x, np.tile(y, len(xs)), np.full(x.size, value)

    def test_gap_between_lines_is_bridged_but_not_beyond_reach(self):
        res = 0.1
        # two lines 0.6 m apart: gap <= 2*reach(0.5) -> filled
        x, y, v = self._line_points([1.0, 1.6])
        g = slicing.grid_binned(x, y, v, 0.0, 0.0, 30, 70, res, reach_m=0.5)
        mid = g[30, 13]  # x = 1.3 m, y = 3.0 m
        self.assertTrue(np.isfinite(mid))
        self.assertAlmostEqual(float(mid), 1.0, places=3)
        # two lines 2.4 m apart: middle stays empty
        x, y, v = self._line_points([0.5, 2.9])
        g = slicing.grid_binned(x, y, v, 0.0, 0.0, 40, 70, res, reach_m=0.5)
        self.assertTrue(np.isnan(g[30, 17]))  # x = 1.7 m

    def test_no_halo_around_the_survey(self):
        res = 0.1
        x, y, v = self._line_points([1.0, 1.4])
        g = slicing.grid_binned(x, y, v, 0.0, 0.0, 30, 70, res, reach_m=0.5)
        cols = np.where(np.isfinite(g).any(axis=0))[0]
        # data span x = 1.0..1.4 (cols 10..14); the edge may grow ~1.5 cells only
        self.assertGreaterEqual(int(cols.min()), 8)
        self.assertLessEqual(int(cols.max()), 16)

    def test_body_reaches_line_ends_at_grid_border(self):
        # Rilievo che tocca il bordo della griglia: il riempimento fra le linee
        # deve arrivare fino alla fine delle linee (prima la chiusura veniva
        # erosa dal bordo dell'array e le code delle linee sporgevano).
        res = 0.1
        ys = np.arange(0.0, 6.0, 0.02)
        x = np.concatenate([np.full(ys.size, c) for c in (1.0, 1.5)])
        y = np.concatenate([ys, ys])
        v = np.ones(x.size)
        g = slicing.grid_binned(x, y, v, 0.0, 0.0, 30, 62, res, reach_m=1.0)
        # colonna 12 = x 1.2 m (fra le due linee), righe vicine alla fine (y = 5.9)
        self.assertTrue(np.isfinite(g[58, 12]))
        self.assertTrue(np.isfinite(g[2, 12]))

    def test_blanking(self):
        x, y, v = self._line_points([1.0, 2.0])
        g = slicing.grid_binned(x, y, v, 0.0, 0.0, 30, 70, 0.1, reach_m=1.0, blanking_m=0.2)
        self.assertTrue(np.isnan(g[30, 15]))      # 0.5 m from both lines
        self.assertTrue(np.isfinite(g[30, 10]))   # on a line

    def test_empty_and_nonfinite_input(self):
        g = slicing.grid_binned([], [], [], 0.0, 0.0, 5, 5, 0.1, reach_m=0.5)
        self.assertEqual(g.shape, (5, 5))
        self.assertTrue(np.isnan(g).all())
        g = slicing.grid_binned([1.0, np.nan], [1.0, 1.0], [1.0, 2.0], 0.0, 0.0, 30, 30, 0.1, reach_m=0.5)
        self.assertTrue(np.isfinite(g).any())

    def test_constant_field_stays_constant(self):
        x, y, v = self._line_points([1.0, 1.3, 1.6], value=7.5)
        g = slicing.grid_binned(x, y, v, 0.0, 0.0, 30, 70, 0.1, reach_m=0.5)
        self.assertTrue(np.allclose(g[np.isfinite(g)], 7.5, atol=1e-4))


@unittest.skipUnless(HAS_SCIPY, "SciPy required")
class EngineTest(unittest.TestCase):
    def _spec(self, **kw):
        base = dict(z_min_m=0.0, z_max_m=3.0, thickness_m=0.1, overlap=0.0,
                    resolution_m=0.1, search_radius_m=0.5, smooth_lowpass=0)
        base.update(kw)
        return SliceSpec(**base)

    # -- windows ----------------------------------------------------------
    def test_build_windows(self):
        w = engine.build_windows(self._spec(z_max_m=1.0))
        self.assertEqual(len(w), 10)
        self.assertAlmostEqual(w[0][0], 0.0)
        self.assertAlmostEqual(w[-1][1], 1.0)
        w = engine.build_windows(self._spec(z_max_m=1.0, overlap=0.5))
        self.assertEqual(len(w), 19)
        self.assertAlmostEqual(w[1][0] - w[0][0], 0.05)
        w = engine.build_windows(self._spec(z_max_m=0.05))  # thinner than a window
        self.assertEqual(len(w), 1)

    # -- geometry: every channel at its own position ------------------------
    def test_every_channel_keeps_its_own_position(self):
        profs = _array_survey(n_prof=2, n_ch=8, pitch=0.04)
        geoms, diag = engine.collect_geometry(profs, -1)
        self.assertEqual(len(geoms), 16)
        self.assertFalse(diag["using_synthetic_coords"])
        xs0 = sorted(g.x[0] for g in geoms if g.prof_idx == 0)
        self.assertAlmostEqual(xs0[-1] - xs0[0], 7 * 0.04, places=6)
        only = engine.collect_geometry(profs, 3)[0]
        self.assertEqual(len(only), 2)

    def test_lateral_structure_inside_a_swath_survives(self):
        """Channel c carries amplitude c. The old centre-line averaging flattened
        this to one value per swath; every-channel gridding keeps the gradient."""
        def fill(p, c, n_s, n_t):
            return np.full((n_s, n_t), float(c + 1))

        profs = _array_survey(n_prof=1, n_ch=16, pitch=0.05, fill=fill)
        spec = self._spec(balance="none", z_max_m=1.0, smooth_lowpass=0)
        g, _ = slicer.compute_slice_preview(profs, spec.resolve(profs), 0.5, 0.6, resolution=0.05)
        row = np.nanmean(g, axis=0)
        row = row[np.isfinite(row)]
        self.assertGreater(float(row.max() - row.min()), 8.0)  # ~1 -> ~16

    # -- depth: window follows dt*v/2, not nominal depth ---------------------
    def test_reflector_lands_in_the_right_window(self):
        # 0.2 ns, 1e8 m/s -> 0.01 m/row; pulse at rows 150..154 = 1.50..1.54 m
        profs = _array_survey(n_prof=3, pulse_rows=(150, 155), noise=0.01, seed=3)
        # crop one channel's matrix (as time-zero alignment does) and lie about
        # the nominal depth: the mapping must not care.
        for p in profs:
            p._ch[2].data = p._ch[2].data[:380]
            p.depth_max_m = 9.99
        spec = self._spec(balance="none", z_max_m=3.0)
        hit, _ = slicer.compute_slice_preview(profs, spec.resolve(profs), 1.50, 1.55, resolution=0.1)
        miss, _ = slicer.compute_slice_preview(profs, spec.resolve(profs), 1.20, 1.25, resolution=0.1)
        self.assertGreater(float(np.nanmean(hit)), 0.5)
        self.assertLess(float(np.nanmean(miss)), 0.1)

    def test_velocity_override(self):
        profs = _array_survey(n_prof=3, pulse_rows=(150, 155), noise=0.01, seed=3)
        # at 0.5e8 m/s a row is 0.005 m: the same rows now sit at 0.75..0.77 m
        spec = self._spec(balance="none", velocity_m_s=0.5e8)
        hit, _ = slicer.compute_slice_preview(profs, spec.resolve(profs), 0.75, 0.77, resolution=0.1)
        miss, _ = slicer.compute_slice_preview(profs, spec.resolve(profs), 1.50, 1.55, resolution=0.1)
        self.assertGreater(float(np.nanmean(hit)), 0.5)
        self.assertLess(float(np.nanmean(miss)), 0.1)

    # -- channel balance ------------------------------------------------------
    def test_balance_channel_removes_stripes(self):
        gains = np.array([1.0, 3.0, 1.0, 2.0, 1.0, 3.5, 1.0, 1.5])
        rng = np.random.default_rng(1)
        profs = _array_survey(n_prof=4, n_ch=8, gains=gains,
                              fill=lambda p, c, n_s, n_t: rng.normal(0, 1.0, (n_s, n_t)))
        def stripe(balance):
            spec = self._spec(balance=balance, z_max_m=1.0)
            g, _ = slicer.compute_slice_preview(profs, spec.resolve(profs), 0.5, 0.6, resolution=0.04)
            # column means across the swath: stripes along the tracks show up here
            return float(np.nanstd(np.nanmean(g, axis=0)))
        raw, bal = stripe("none"), stripe("channel")
        self.assertGreater(raw, 0.1)
        self.assertLess(bal, raw * 0.2)

    def test_balance_signed_mean_uses_offset(self):
        def fill(p, c, n_s, n_t):
            return np.full((n_s, n_t), 5.0 * c)  # per-channel DC offset
        profs = _array_survey(n_prof=2, n_ch=6, fill=fill)
        spec = self._spec(balance="channel", amplitude_mode="signed_mean")
        pts = engine.window_points(profs, engine.collect_geometry(profs, -1)[0], spec.resolve(profs), 0.5, 0.6)
        self.assertLess(float(np.ptp(pts[2])), 1e-6)

    # -- the dots ---------------------------------------------------------------
    def test_no_beads_along_survey_lines(self):
        """Dense array swaths: the legacy path collapses each swath to its centre
        line and IDW-blends in between, leaving a string of beads along every
        track. The bin engine keeps every channel and must not."""
        rng = np.random.default_rng(5)
        profs = _array_survey(
            n_prof=8, n_ch=16, pitch=0.05, n_t=160,
            fill=lambda p, c, n_s, n_t: rng.normal(0, 1.0, (n_s, n_t)),
        )
        common = dict(z_min_m=0.0, z_max_m=1.0, thickness_m=0.1, overlap=0.0,
                      resolution_m=0.1, search_radius_m=0.8, smooth_lowpass=0, balance="none")
        s_bin = SliceSpec(interp="bin", **common).resolve(profs)
        s_idw = SliceSpec(interp="idw", **common).resolve(profs)
        g_bin, _ = slicer.compute_slice_preview(profs, s_bin, 0.5, 0.6, resolution=0.1)
        g_idw, _ = slicer.compute_slice_preview(
            profs, s_idw, 0.5, 0.6, resolution=0.1, idw_mode="fast", fill_nodata=False,
            emit_diagnostics=False,
        )
        r_idw, r_bin = _bead_ratio(g_idw), _bead_ratio(g_bin)
        self.assertGreater(r_idw, 4.0)          # the artifact is real in the legacy path
        self.assertLess(r_bin, 0.5 * r_idw)

    # -- contract with the rest of the plugin ---------------------------------
    def test_grids_and_meta_are_compatible(self):
        profs = _array_survey(n_prof=3)
        spec = self._spec(z_max_m=0.5)
        grids, meta = slicer.compute_ogpr_slice_grids(
            profs, channel=-1, spec=spec, emit_diagnostics=False)
        self.assertEqual(len(grids), 5)
        for k in ("x_min", "y_min", "x_max", "y_max", "n_x", "n_y", "resolution", "z_min",
                  "z_max", "z_step", "z_levels", "slice_spec", "timing_s", "profiles_total"):
            self.assertIn(k, meta)
        for i, g in enumerate(grids):
            self.assertEqual(g["index"], i)
            self.assertEqual(g["grid"].shape, (meta["n_y"], meta["n_x"]))
            self.assertAlmostEqual(g["z_to"] - g["z_from"], 0.1, places=6)
            self.assertAlmostEqual(g["z_lev"], 0.5 * (g["z_from"] + g["z_to"]), places=6)
            slicer._validate_grid_georef_meta(meta, g["grid"])
        # georeferencing is exact: y_max is the top row's centre
        self.assertAlmostEqual(meta["y_max"], meta["y_min"] + (meta["n_y"] - 1) * meta["resolution"])
        # the manifest writer accepts the engine output
        results = [{"path": f"/o/s{g['index']}.tif", "z_from": g["z_from"], "z_to": g["z_to"],
                    "z_center": g["z_lev"], "index": g["index"], "name": f"s{g['index']}"} for g in grids]
        slice_manifest = importlib.import_module(PKG + ".slice_manifest")
        with tempfile.TemporaryDirectory() as d:
            slicer._write_run_sliceset(d, meta, grids, results, spec=spec, epsg=32633,
                                       source_variant="raw", n_profiles=3)
            man = slice_manifest.read_sliceset(d)
        self.assertEqual(man["spec"].interp, "bin")
        self.assertEqual(len(man["slices"]), 5)

    def test_preview_equals_single_window_and_respects_resolution(self):
        profs = _array_survey(n_prof=4, noise=0.5, seed=2)
        spec = self._spec(z_max_m=2.0).resolve(profs)
        grids, meta = slicer.compute_slice(profs, spec, single_window=(0.5, 0.6))
        self.assertEqual(len(grids), 1)
        prev, extent = slicer.compute_slice_preview(profs, spec, 0.5, 0.6)
        np.testing.assert_allclose(np.nan_to_num(prev), np.nan_to_num(grids[0]["grid"]))
        xl, xr, yb, yt = extent
        self.assertLess(xl, xr)
        self.assertLess(yb, yt)
        coarse, _ = slicer.compute_slice_preview(profs, spec, 0.5, 0.6, resolution=0.3)
        self.assertLess(coarse.shape[0], prev.shape[0])

    def test_bad_window_gives_none(self):
        profs = _array_survey(n_prof=2)
        spec = self._spec().resolve(profs)
        g, e = slicer.compute_slice_preview(profs, spec, float("nan"), 1.0)
        self.assertIsNone(g)
        self.assertIsNone(e)
        # window below the data -> no points -> (None, None)
        g, e = slicer.compute_slice_preview(profs, spec, 50.0, 50.1)
        self.assertIsNone(g)

    def test_progress_and_cancel(self):
        profs = _array_survey(n_prof=2)
        spec = self._spec(z_max_m=1.0)
        seen = []
        slicer.compute_ogpr_slice_grids(profs, spec=spec, emit_diagnostics=False,
                                        progress_callback=lambda d, t: seen.append((d, t)))
        self.assertEqual(seen[-1], (10, 10))
        ev = threading.Event()
        ev.set()
        grids, _ = slicer.compute_ogpr_slice_grids(profs, spec=spec, emit_diagnostics=False, cancel_event=ev)
        self.assertEqual(grids, [])

    def test_profiles_without_geometry_use_local_coordinates(self):
        profs = _array_survey(n_prof=2)
        for p in profs:
            for ch in p._ch:
                ch.easting = np.zeros_like(ch.easting)
                ch.northing = np.zeros_like(ch.northing)
        grids, meta = slicer.compute_ogpr_slice_grids(
            profs, spec=self._spec(z_max_m=0.2), emit_diagnostics=False)
        self.assertTrue(meta["using_synthetic_coords"])
        self.assertGreater(len(grids), 0)

    def test_smoothing_keeps_the_mask(self):
        profs = _array_survey(n_prof=3, noise=1.0, seed=4)
        a, _ = slicer.compute_slice_preview(profs, self._spec(smooth_lowpass=0).resolve(profs), 0.5, 0.6)
        b, _ = slicer.compute_slice_preview(profs, self._spec(smooth_lowpass=5).resolve(profs), 0.5, 0.6)
        self.assertTrue(np.array_equal(np.isnan(a), np.isnan(b)))
        self.assertLess(float(np.nanstd(b)), float(np.nanstd(a)))


class SpecMigrationTest(unittest.TestCase):
    def test_defaults(self):
        s = SliceSpec()
        self.assertEqual(s.interp, "bin")
        self.assertEqual(s.balance, "channel")
        self.assertIsNone(s.velocity_m_s)
        self.assertEqual(s.smooth_lowpass, 3)

    def test_old_saved_spec_moves_to_bin(self):
        old = {"format": slice_spec.FORMAT, "interp": "idw", "velocity_m_s": 1e8,
               "smooth_lowpass": 0, "z_min_m": 0.0, "thickness_m": 0.1}
        s = SliceSpec.from_dict(old)
        self.assertEqual(s.interp, "bin")
        self.assertEqual(s.velocity_m_s, 1e8)   # explicit value is kept
        self.assertEqual(s.smooth_lowpass, 0)   # so is the user's low-pass

    def test_explicit_idw_survives_roundtrip(self):
        s = SliceSpec(interp="idw", z_max_m=1.0, search_radius_m=0.5)
        s2 = SliceSpec.from_json(s.to_json())
        self.assertEqual(s2.interp, "idw")
        self.assertEqual(s2, s)

    def test_validation(self):
        with self.assertRaises(slice_spec.SliceSpecError):
            SliceSpec(balance="bogus").validate()
        with self.assertRaises(slice_spec.SliceSpecError):
            SliceSpec(velocity_m_s=-1.0).validate()

    def test_bin_needs_a_radius_to_be_resolved(self):
        self.assertFalse(SliceSpec(z_max_m=1.0).is_resolved())
        self.assertTrue(SliceSpec(z_max_m=1.0, search_radius_m=0.5).is_resolved())

    def test_from_compute_meta_reads_engine_meta(self):
        s = SliceSpec.from_compute_meta({"engine": "bin", "amplitude_mode": "rms", "z_min": 0.0,
                                         "z_max": 1.0, "z_step": 0.1})
        self.assertEqual((s.interp, s.amplitude_mode), ("bin", "rms"))


class QmlStyleTest(unittest.TestCase):
    def setUp(self):
        self.grid = np.linspace(-1.0, 1.0, 1000).reshape(20, 50)

    def _qml(self, style):
        with tempfile.TemporaryDirectory() as d:
            tif = os.path.join(d, "s.tif")
            raster_io.write_qml_singleband(tif, self.grid, style)
            with open(os.path.join(d, "s.qml"), encoding="utf-8") as f:
                return f.read()

    def test_default_is_the_classic_grayscale(self):
        q = self._qml(None)
        self.assertIn("singlebandgray", q)
        self.assertNotIn("pseudocolor", q)

    def test_stretch_uses_clip_percentiles(self):
        lo, hi = raster_io.slice_stretch(self.grid, 10, 90)
        self.assertAlmostEqual(lo, np.percentile(self.grid, 10))
        self.assertAlmostEqual(hi, np.percentile(self.grid, 90))
        self.assertTrue(raster_io.slice_stretch(np.full((3, 3), 2.0))[1] > 2.0)
        self.assertEqual(raster_io.slice_stretch(np.full((3, 3), np.nan)), (0.0, 1.0))

    @unittest.skipUnless(HAS_MPL, "matplotlib required")
    def test_colormap_is_written_as_pseudocolor(self):
        q = self._qml({"cmap": "RdBu_r", "clip_lo": 5, "clip_hi": 95})
        self.assertIn("singlebandpseudocolor", q)
        self.assertEqual(q.count("<item "), 11)
        lo, hi = raster_io.slice_stretch(self.grid, 5, 95)
        self.assertIn(repr(float(lo)), q)
        self.assertIn(repr(float(hi)), q)
        # RdBu_r: blue at the low end, red at the high end
        first = q.split('<item ')[1]
        last = q.split('<item ')[-1]
        self.assertIn('color="#', first)
        r0 = int(first.split('color="#')[1][:2], 16)
        r1 = int(last.split('color="#')[1][:2], 16)
        self.assertLess(r0, r1)

    def test_unknown_colormap_falls_back_to_grayscale(self):
        q = self._qml({"cmap": "definitely_not_a_colormap"})
        self.assertIn("singlebandgray", q)

    def test_write_grids_to_tifs_passes_style(self):
        import unittest.mock as mock
        grids = [{"z_lev": 0.05, "index": 0, "grid": np.ones((4, 5), dtype=np.float32),
                  "z_from": 0.0, "z_to": 0.1}]
        meta = {"x_min": 0.0, "y_min": 0.0, "y_max": 0.3, "resolution": 0.1, "n_x": 5, "n_y": 4,
                "x_max": 0.4}
        seen = {}
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(slicer, "_write_tif_singleband", lambda *a, **k: None), \
                mock.patch.object(slicer, "_write_qml_singleband",
                                  lambda tif, grid, style=None: seen.update(style=style)):
            slicer.write_grids_to_tifs(grids, meta, d, 32633, style={"cmap": "viridis"})
        self.assertEqual(seen["style"], {"cmap": "viridis"})


if __name__ == "__main__":
    unittest.main()
