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

    # -- amplitude tools ---------------------------------------------------------
    def test_amplitude_tools_roundtrip_in_the_form(self):
        from rasterlinker_feature_interpolation.slice_spec import SliceSpec
        want = SliceSpec(amplitude_mode="env_mean", despike_pct=99.0, normalize="median",
                         amp_scale="db", despeckle=5, highpass_m=0.8, smooth_lowpass=9,
                         bg_horizontal=True)
        self.panel.set_spec(want)
        got = self.panel.get_spec()
        for f in ("amplitude_mode", "despike_pct", "normalize", "amp_scale", "despeckle",
                  "highpass_m", "smooth_lowpass", "bg_horizontal"):
            self.assertEqual(getattr(got, f), getattr(want, f), f)
        self.assertTrue(self.panel.chk_despike.isChecked())
        self.panel.chk_despike.setChecked(False)
        self.assertEqual(self.panel.get_spec().despike_pct, 100.0)

    def test_untouched_form_keeps_every_tool_off(self):
        spec = self.panel.get_spec()
        self.assertFalse(spec.tools_active)
        self.assertEqual(spec.amplitude_mode, "abs_mean")

    def test_tools_that_do_not_apply_are_greyed_out(self):
        idx = self.panel.cb_interp.findData("idw")
        self.panel.cb_interp.setCurrentIndex(idx)
        for w in (self.panel.chk_despike, self.panel.cb_despeckle, self.panel.sp_highpass,
                  self.panel.cb_normalize, self.panel.cb_scale):
            self.assertFalse(w.isEnabled(), w)
        self.panel.cb_interp.setCurrentIndex(self.panel.cb_interp.findData("bin"))
        self.panel.cb_amplitude.setCurrentIndex(self.panel.cb_amplitude.findData("signed_mean"))
        self.assertFalse(self.panel.cb_scale.isEnabled())       # dB of a signed value: no
        self.assertFalse(self.panel.cb_normalize.isEnabled())
        self.assertTrue(self.panel.cb_despeckle.isEnabled())

    def test_tools_reach_the_preview(self):
        self.v._btn_preview_slice.setChecked(True)
        self.assertTrue(self._wait(lambda: self.v._preview_slice_grid is not None))
        first = self.v._preview_slice_grid.copy()
        self.v._preview_slice_grid = None
        self.panel.cb_scale.setCurrentIndex(self.panel.cb_scale.findData("db"))
        self.assertTrue(self._wait(lambda: self.v._preview_slice_grid is not None))
        db = self.v._preview_slice_grid
        self.assertEqual(db.shape, first.shape)
        ok = np.isfinite(first) & np.isfinite(db)
        self.assertTrue(ok.sum() > 100)
        self.assertFalse(np.allclose(db[ok], first[ok]))

    # -- colour limits, histogram, probe ---------------------------------------------
    def _show_preview(self):
        self.v._btn_preview_slice.setChecked(True)
        self.assertTrue(self._wait(lambda: self.v._preview_slice_grid is not None))
        self._pump(50)

    def test_percentile_mode_is_the_default_and_matches_the_old_behaviour(self):
        self.assertEqual(self.v._slice_scale_mode(), "percentile")
        self.assertFalse(self.v._spin_slice_vmin_abs.isEnabled())
        self.assertTrue(self.v._spin_slice_vmin_pct.isEnabled())
        arr = np.array([[0.0, 1.0, 2.0, 3.0, 100.0]] * 4, dtype=np.float32)
        lo, hi = self.v._slice_color_limits(arr)
        f = arr[np.isfinite(arr)]
        self.assertAlmostEqual(lo, float(np.percentile(f, 2.0)), places=5)
        self.assertAlmostEqual(hi, float(np.percentile(f, 98.0)), places=5)
        self.assertNotIn("vmin", self.v._slice_display_style())

    def test_fixed_limits_drive_display_and_style(self):
        self._show_preview()
        cb = self.v._cb_slice_scale_mode
        cb.setCurrentIndex(cb.findData("fixed"))
        self.v._spin_slice_vmin_abs.setValue(0.25)
        self.v._spin_slice_vmax_abs.setValue(1.75)
        self.assertEqual(self.v._slice_color_limits(np.array([[0.0, 9.0]])), (0.25, 1.75))
        self.assertEqual(tuple(self.v._slice_im.get_clim()), (0.25, 1.75))
        style = self.v._slice_display_style()
        self.assertEqual((style["vmin"], style["vmax"]), (0.25, 1.75))
        self.assertEqual(self.v.get_slice_params()["style"], style)
        self.assertFalse(self.v._spin_slice_vmin_pct.isEnabled())
        self.assertTrue(self.v._spin_slice_vmin_abs.isEnabled())

    def test_switching_to_fixed_starts_from_what_is_on_screen(self):
        self._show_preview()
        shown = tuple(float(x) for x in self.v._slice_im.get_clim())
        cb = self.v._cb_slice_scale_mode
        cb.setCurrentIndex(cb.findData("fixed"))
        got = (self.v._spin_slice_vmin_abs.value(), self.v._spin_slice_vmax_abs.value())
        self.assertAlmostEqual(got[0], shown[0], places=4)
        self.assertAlmostEqual(got[1], shown[1], places=4)
        cb.setCurrentIndex(cb.findData("percentile"))
        self.assertNotIn("vmin", self.v._slice_display_style())

    def test_histogram_shows_the_slice_and_dragging_fixes_the_limits(self):
        from qgis.PyQt.QtCore import QEvent, QPointF, Qt
        from qgis.PyQt.QtGui import QMouseEvent
        self._show_preview()
        h = self.v._hist_slice
        self.assertIsNotNone(h._counts)
        self.assertEqual(int(h._counts.sum()), int(np.isfinite(self.v._preview_slice_grid).sum()))
        self.assertIn("mediana", self.v._lbl_slice_stats.text())
        self.assertEqual(self.v._slice_scale_mode(), "percentile")
        h.resize(300, 84)
        vmin, vmax = h.limits
        target = vmin + 0.4 * (vmax - vmin)
        y = 30.0

        def ev(kind, x, b):
            return QMouseEvent(kind, QPointF(x, y), Qt.LeftButton, b, Qt.NoModifier)

        h.mousePressEvent(ev(QEvent.MouseButtonPress, h.x_of(vmin), Qt.LeftButton))
        h.mouseMoveEvent(ev(QEvent.MouseMove, h.x_of(target), Qt.LeftButton))
        h.mouseReleaseEvent(ev(QEvent.MouseButtonRelease, h.x_of(target), Qt.NoButton))
        self.assertEqual(self.v._slice_scale_mode(), "fixed")
        self.assertAlmostEqual(self.v._spin_slice_vmin_abs.value(), target, places=3)
        self.assertAlmostEqual(self.v._slice_im.get_clim()[0], target, places=3)
        self.assertAlmostEqual(self.v._slice_im.get_clim()[1], vmax, places=3)
        self.assertEqual(self.v._slice_display_style()["vmin"], self.v._spin_slice_vmin_abs.value())

    def test_histogram_takes_the_colormap(self):
        cb = self.v._cb_slice_cmap
        self.assertGreaterEqual(cb.findText("jet"), 0)            # the IQMaps colormap is offered
        cb.setCurrentIndex(cb.findText("jet"))
        self.assertEqual(len(self.v._hist_slice._colors), 32)
        self.assertEqual(self.v._hist_slice._colors[0], "#000080")

    def test_fixed_limits_survive_project_state(self):
        cb = self.v._cb_slice_scale_mode
        cb.setCurrentIndex(cb.findData("fixed"))
        self.v._spin_slice_vmin_abs.setValue(0.5)
        self.v._spin_slice_vmax_abs.setValue(2.5)
        saved = {"slice_style": self.v._slice_display_style()}
        cb.setCurrentIndex(cb.findData("percentile"))
        self.v._apply_gpr_processing_ui_state(saved)
        self.assertEqual(self.v._slice_scale_mode(), "fixed")
        self.assertEqual((self.v._spin_slice_vmin_abs.value(), self.v._spin_slice_vmax_abs.value()), (0.5, 2.5))
        # a project saved in percentile mode brings it back
        self.v._apply_gpr_processing_ui_state({"slice_style": {"cmap": "gray", "clip_lo": 3.0, "clip_hi": 97.0}})
        self.assertEqual(self.v._slice_scale_mode(), "percentile")

    def test_value_probe_reads_the_cell_under_the_cursor(self):
        self._show_preview()
        grid = self.v._preview_slice_grid
        x0, x1, y0, y1 = self.v._preview_slice_extent
        n_r, n_c = grid.shape
        cells = np.argwhere(np.isfinite(grid))
        rr, cc = cells[len(cells) // 2]
        x = x0 + (cc + 0.5) * (x1 - x0) / n_c
        y = y0 + (rr + 0.5) * (y1 - y0) / n_r
        self.assertAlmostEqual(self.v._slice_value_at(x, y), float(grid[rr, cc]), places=5)
        self.assertIsNone(self.v._slice_value_at(x0 - 100.0, y))              # outside the picture
        holes = np.argwhere(~np.isfinite(grid))
        if len(holes):
            r2, c2 = holes[0]
            self.assertIsNone(self.v._slice_value_at(x0 + (c2 + 0.5) * (x1 - x0) / n_c,
                                                     y0 + (r2 + 0.5) * (y1 - y0) / n_r))

    def test_changing_a_tool_reuses_the_extraction_in_the_preview(self):
        from unittest import mock
        eng = importlib.import_module("rasterlinker_feature_interpolation.slice_engine")
        real = eng.window_amplitude
        calls = []

        def spy(*a, **k):
            calls.append(1)
            return real(*a, **k)

        self.panel.cb_amplitude.setCurrentIndex(self.panel.cb_amplitude.findData("env_mean"))
        with mock.patch.object(eng, "window_amplitude", spy):
            self.v._btn_preview_slice.setChecked(True)
            self.assertTrue(self._wait(lambda: self.v._preview_slice_grid is not None))
            self._pump(100)
            first_calls = len(calls)
            self.assertGreater(first_calls, 0)
            first = self.v._preview_slice_grid.copy()
            self.v._preview_slice_grid = None
            self.panel.cb_despeckle.setCurrentIndex(self.panel.cb_despeckle.findData(3))
            self.assertTrue(self._wait(lambda: self.v._preview_slice_grid is not None))
            self._pump(100)
            self.assertEqual(len(calls), first_calls, "only the tools changed: no new extraction")
            self.assertFalse(np.array_equal(np.nan_to_num(self.v._preview_slice_grid), np.nan_to_num(first)))
            self.v._preview_slice_grid = None
            self.panel.cb_amplitude.setCurrentIndex(self.panel.cb_amplitude.findData("rms"))
            self.assertTrue(self._wait(lambda: self.v._preview_slice_grid is not None))
            self.assertGreater(len(calls), first_calls, "another amplitude mode must be extracted again")

    def test_preview_cache_is_dropped_when_the_data_can_have_changed(self):
        c1 = self.v._preview_points_cache_for("", None)
        c1["k"] = 1
        self.assertIs(self.v._preview_points_cache_for("", None), c1)
        self.v._geometry_rev = int(getattr(self.v, "_geometry_rev", 0)) + 1     # geometry edited
        c2 = self.v._preview_points_cache_for("", None)
        self.assertIsNot(c2, c1)
        self.assertEqual(c2, {})
        self.v._profiles = [_array_profile(i, n_ch=4) for i in range(3)]          # other profiles
        self.assertIsNot(self.v._preview_points_cache_for("", None), c2)
        v1 = self.v._preview_points_cache_for("v1", ("d", 3, 1.0, 0))
        self.assertIs(self.v._preview_points_cache_for("v1", ("d", 3, 1.0, 0)), v1)
        self.assertIsNot(self.v._preview_points_cache_for("v1", ("d", 3, 2.0, 0)), v1)  # file changed
        self.assertIsNot(self.v._preview_points_cache_for("v2", ("d", 3, 1.0, 0)), v1)  # other variant

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


@unittest.skipUnless(HAS_VIEWER, "gpr_profile_viewer non importabile")
class AmplitudeHistogramWidgetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from qgis.core import QgsApplication
        from qgis.PyQt.QtWidgets import QApplication
        cls._own_app = None
        if QApplication.instance() is None:
            cls._own_app = QgsApplication([], False)
            cls._own_app.initQgis()

    @classmethod
    def tearDownClass(cls):
        if cls._own_app is not None:
            cls._own_app.exitQgis()

    def setUp(self):
        from rasterlinker_feature_interpolation.amplitude_histogram import AmplitudeHistogram
        self.h = AmplitudeHistogram()
        self.h.resize(300, 84)
        rng = np.random.default_rng(0)
        self.vals = np.concatenate([rng.normal(1.0, 0.2, 5000), rng.normal(3.0, 0.3, 300)])
        self.h.set_data(self.vals, 0.6, 3.5)
        self.got = []
        self.h.limits_changed.connect(lambda a, b: self.got.append((a, b)))

    def _ev(self, kind, x, released=False):
        from qgis.PyQt.QtCore import QEvent, QPointF, Qt
        from qgis.PyQt.QtGui import QMouseEvent
        k = {"press": QEvent.MouseButtonPress, "move": QEvent.MouseMove,
             "release": QEvent.MouseButtonRelease}[kind]
        held = Qt.NoButton if (released or kind == "release") else Qt.LeftButton
        return QMouseEvent(k, QPointF(float(x), 30.0), Qt.LeftButton if kind != "move" else Qt.NoButton,
                           held, Qt.NoModifier)

    def _drag(self, marker_value, to_value):
        self.h.mousePressEvent(self._ev("press", self.h.x_of(marker_value)))
        self.h.mouseMoveEvent(self._ev("move", self.h.x_of(to_value)))
        self.h.mouseReleaseEvent(self._ev("release", self.h.x_of(to_value)))

    def test_dragging_the_low_marker_moves_only_it(self):
        self._drag(0.6, 1.1)
        self.assertTrue(self.got)
        lo, hi = self.got[-1]
        self.assertAlmostEqual(lo, 1.1, delta=0.03)
        self.assertEqual(hi, 3.5)
        self.assertEqual(self.h.limits, (lo, hi))

    def test_dragging_the_high_marker_moves_only_it(self):
        self._drag(3.5, 2.4)
        lo, hi = self.got[-1]
        self.assertEqual(lo, 0.6)
        self.assertAlmostEqual(hi, 2.4, delta=0.03)

    def test_markers_never_cross(self):
        self._drag(0.6, 3.4)       # low marker dragged nearly past the high one
        self.assertLess(self.got[-1][0], self.got[-1][1])
        self.got.clear()
        self.h.set_data(self.vals, 0.6, 3.5)
        self._drag(3.5, 0.0)       # high marker dragged far below the low one
        self.assertLess(self.got[-1][0], self.got[-1][1])

    def test_a_click_far_from_both_markers_grabs_nothing(self):
        self.h.mousePressEvent(self._ev("press", self.h.x_of(2.0)))
        self.h.mouseMoveEvent(self._ev("move", self.h.x_of(2.4)))
        self.h.mouseReleaseEvent(self._ev("release", self.h.x_of(2.4)))
        self.assertEqual(self.got, [])
        self.assertEqual(self.h.limits, (0.6, 3.5))

    def test_new_data_is_ignored_while_dragging(self):
        self.h.mousePressEvent(self._ev("press", self.h.x_of(0.6)))
        self.h.set_data(self.vals * 10.0, 5.0, 50.0)          # must not re-scale under the hand
        self.assertEqual(self.h.limits, (0.6, 3.5))
        self.h.mouseReleaseEvent(self._ev("release", self.h.x_of(0.6)))
        self.h.set_data(self.vals * 10.0, 5.0, 50.0)
        self.assertEqual(self.h.limits, (5.0, 50.0))

    def test_set_limits_is_silent(self):
        self.h.set_limits(0.9, 2.0)
        self.assertEqual(self.got, [])
        self.assertEqual(self.h.limits, (0.9, 2.0))

    def test_histogram_counts_everything(self):
        self.assertEqual(int(self.h._counts.sum()), self.vals.size)

    def test_it_paints_with_and_without_data(self):
        self.h.set_colors(["#000000", "#ffffff"])
        self.assertFalse(self.h.grab().isNull())
        self.h.set_data(np.array([]), 0.0, 1.0)
        self.assertFalse(self.h.grab().isNull())
        self.h.set_data(np.full((4, 4), np.nan), 0.0, 1.0)
        self.assertFalse(self.h.grab().isNull())


if __name__ == "__main__":
    unittest.main()
