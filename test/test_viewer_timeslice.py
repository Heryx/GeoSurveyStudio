# -*- coding: utf-8 -*-
"""Tab Slicing del profile viewer: colormap raggiungibile, spessore coerente col
pannello, preview con il motore "bin" e lettura della variante fuori dal thread
GUI (richiede Qt offscreen e QGIS; si salta altrimenti)."""

import importlib
import pathlib
import sys
import tempfile
import time
import unittest

import numpy as np

_TEST_DIR = pathlib.Path(__file__).resolve().parent
if str(_TEST_DIR) not in sys.path:  # importabile sia con "discover -s test" sia come test.test_*
    sys.path.insert(0, str(_TEST_DIR))

import test_profile_viewer_interaction as _base  # noqa: E402

HAS_MPL, HAS_VIEWER = _base.HAS_MPL, _base.HAS_VIEWER
_ViewerTestBase = _base._ViewerTestBase
_RD = getattr(_base, "_RD", None)  # undefined when the viewer could not be imported

if HAS_VIEWER:
    _NPZ = importlib.import_module("rasterlinker_feature_interpolation.gpr_npz_io")


def _array_profile(idx, n_ch=6, n_s=200, n_t=120, pitch=0.05, gap=0.3, dt_ns=0.2):
    """Piccolo array: n_ch canali a 5 cm l'uno dall'altro, swath adiacenti."""
    rng = np.random.default_rng(idx)
    y = 4500000.0 + np.arange(n_t) * 0.05
    chans = []
    for c in range(n_ch):
        x = 500000.0 + idx * gap + c * pitch + 0 * y
        data = rng.normal(0, 1, (n_s, n_t)).astype(np.float32)
        chans.append(_RD.OgprChannel(c, data, x, y, 0 * y, 0 * y, np.arange(n_t) * 0.05))
    return _RD.OgprProfile(
        path=f"synthetic_{idx}.ogpr", version_major=1, version_minor=0, swath_name="s",
        swath_id=str(idx), array_id=0, n_samples=n_s, n_channels=n_ch, n_slices=n_t,
        sampling_step_m=0.05, sampling_time_ns=dt_ns, velocity_m_s=1.0e8, frequency_mhz=600.0,
        polarization="HH", epsg=32633, value_type="f4", channels=chans, header_raw={},
    )


@unittest.skipUnless(HAS_MPL and HAS_VIEWER, "matplotlib/viewer non disponibili")
class SlicingTabTest(_ViewerTestBase):
    def setUp(self):
        super().setUp()
        self.v._profiles = [_array_profile(i) for i in range(4)]
        self.v._prof_idx = 0
        self.v._load_current_profile()
        self._pump(100)
        self.panel = self.v._slice_spec_panel

    def _wait(self, cond, timeout_s=15.0):
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            self._pump(20)
            if cond():
                return True
        return False

    # -- form ----------------------------------------------------------------
    def test_colormap_and_contrast_are_reachable(self):
        tab = self.v._right_tabs.widget(1)  # "Slicing"
        for w in (self.v._cb_slice_cmap, self.v._lbl_slice_cmap_preview,
                  self.v._spin_slice_vmin_pct, self.v._spin_slice_vmax_pct):
            self.assertTrue(w.isVisibleTo(tab), w)
        self.assertTrue(self.panel.isVisibleTo(tab))

    def test_new_defaults_in_the_form(self):
        spec = self.panel.get_spec()
        self.assertEqual(spec.interp, "bin")
        self.assertEqual(spec.balance, "channel")
        self.assertIsNone(spec.velocity_m_s)
        self.assertEqual(spec.smooth_lowpass, 3)
        self.assertFalse(self.panel.sp_velocity.isEnabled())
        self.panel.chk_velocity_auto.setChecked(False)
        self.assertTrue(self.panel.sp_velocity.isEnabled())
        self.assertAlmostEqual(self.panel.get_spec().velocity_m_s, 1.0e8)

    def test_old_project_spec_is_upgraded_and_roundtrips(self):
        from rasterlinker_feature_interpolation.slice_spec import SliceSpec
        old = {"format": "rasterlinker_slicespec_v1", "interp": "idw", "velocity_m_s": 1e8,
               "smooth_lowpass": 0, "thickness_m": 0.2}
        self.panel.set_spec(SliceSpec.from_dict(old))
        got = self.panel.get_spec()
        self.assertEqual(got.interp, "bin")
        self.assertEqual(got.smooth_lowpass, 0)
        self.assertEqual(got.velocity_m_s, 1e8)
        self.assertAlmostEqual(got.thickness_m, 0.2)

    def test_thickness_follows_the_panel(self):
        self.panel.sp_thickness.setValue(0.25)
        self.assertAlmostEqual(self.v._spin_slice_thickness.value(), 0.25, places=3)
        z0, z1 = self.v._preview_depth_window()
        self.assertAlmostEqual(z1 - z0, 0.25, places=3)

    def test_display_style_carries_colormap_and_clip(self):
        cb = self.v._cb_slice_cmap
        idx = cb.findText("viridis")
        self.assertGreaterEqual(idx, 0)
        cb.setCurrentIndex(idx)
        self.v._spin_slice_vmin_pct.setValue(5.0)
        self.v._spin_slice_vmax_pct.setValue(95.0)
        style = self.v._slice_display_style()
        self.assertEqual(style, {"cmap": "viridis", "clip_lo": 5.0, "clip_hi": 95.0})
        self.assertEqual(self.v.get_slice_params()["style"], style)

    def test_display_style_survives_project_state(self):
        cb = self.v._cb_slice_cmap
        cb.setCurrentIndex(cb.findText("magma"))
        self.v._spin_slice_vmax_pct.setValue(90.0)
        saved = {"slice_style": self.v._slice_display_style()}
        cb.setCurrentIndex(cb.findText("viridis"))
        self.v._spin_slice_vmax_pct.setValue(99.0)
        self.v._apply_gpr_processing_ui_state(saved)
        self.assertEqual(cb.currentText(), "magma")
        self.assertAlmostEqual(self.v._spin_slice_vmax_pct.value(), 90.0)

    # -- preview -----------------------------------------------------------------
    def test_preview_end_to_end_with_bin_engine(self):
        self.v._btn_preview_slice.setChecked(True)
        self.assertTrue(self._wait(lambda: self.v._preview_slice_grid is not None))
        grid = self.v._preview_slice_grid
        self.assertEqual(grid.ndim, 2)
        self.assertGreater(int(np.isfinite(grid).sum()), 100)
        ext = self.v._preview_slice_extent
        self.assertLess(ext[0], ext[1])
        self.assertFalse(self.v._preview_busy)

    def test_preview_reloads_when_the_parameters_change(self):
        self.v._btn_preview_slice.setChecked(True)
        self.assertTrue(self._wait(lambda: self.v._preview_slice_grid is not None))
        first = self.v._preview_slice_grid.copy()
        self.v._preview_slice_grid = None
        self.panel.cb_smooth.setCurrentIndex(self.panel.cb_smooth.findData(5))
        self.assertTrue(self._wait(lambda: self.v._preview_slice_grid is not None))
        self.assertEqual(self.v._preview_slice_grid.shape, first.shape)
        self.assertFalse(np.allclose(np.nan_to_num(self.v._preview_slice_grid), np.nan_to_num(first)))

    # -- variant read off the GUI thread -------------------------------------------
    def test_variant_is_read_once_in_the_worker_thread(self):
        import threading
        with tempfile.TemporaryDirectory() as d:
            for p in self.v._profiles:
                _NPZ.save_processed_profile_npz(
                    d, p, [p.channel(c).data for c in range(p.n_channels)])
            calls = []
            gui = threading.get_ident()
            real_load = self.v._load_variant_profiles

            def slow_load(paths):
                calls.append(threading.get_ident())
                time.sleep(0.4)
                return real_load(paths)

            self.v._load_variant_profiles = slow_load
            self.v._variant_dir_for_name = lambda name: d if name == "myvar" else ""
            self.panel.set_variant_choices(["myvar"])
            self.panel.cb_source_variant.setCurrentText("myvar")
            self.v._preview_variant_cache = None

            self.v._btn_preview_slice.setChecked(True)
            self._pump(50)
            # the GUI thread is free: keep asking for previews while the worker reads
            for z in (0.3, 0.4, 0.5):
                self.v._cursor_z = z
                self.v._request_preview_update(immediate=True)
                self._pump(30)
            self.assertTrue(self._wait(lambda: self.v._preview_slice_grid is not None, 20.0))
            self._pump(300)
            self.assertEqual(len(calls), 1, "la variante va letta una volta sola")
            self.assertNotEqual(calls[0], gui, "la lettura non deve avvenire nel thread GUI")
            self.assertIsNotNone(self.v._preview_variant_cache)
            self.assertIsNone(getattr(self.v, "_preview_loading_key", None))
            # una nuova richiesta usa la cache: nessuna nuova lettura
            self.v._cursor_z = 0.9
            self.v._preview_slice_grid = None
            self.v._request_preview_update(immediate=True)
            self.assertTrue(self._wait(lambda: self.v._preview_slice_grid is not None))
            self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
