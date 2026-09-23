# -*- coding: utf-8 -*-
"""Profile viewer: cursore/blitting, pick da profilo, pipeline asincrona.

La parte "viewer" richiede Qt (offscreen) e QGIS; si salta se non disponibili.
"""

import importlib
import os
import pathlib
import sys
import tempfile
import time
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_PLUGIN_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(_PLUGIN_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR.parent))

try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.backend_bases import MouseEvent
    HAS_MPL = True
except Exception:  # pragma: no cover
    HAS_MPL = False

try:
    _GV = importlib.import_module("rasterlinker_feature_interpolation.gpr_profile_viewer")
    _RD = importlib.import_module("rasterlinker_feature_interpolation.gpr_ogpr_reader")
    # Dopo la divisione in mixin ogni funzione va sostituita nel modulo che la USA.
    _CM = importlib.import_module("rasterlinker_feature_interpolation.gpr_viewer_common")
    _RV = importlib.import_module("rasterlinker_feature_interpolation.gpr_viewer_radargram")
    _PROC = importlib.import_module("rasterlinker_feature_interpolation.gpr_processing")
    HAS_VIEWER = True
except Exception:  # pragma: no cover - ambiente senza QGIS
    HAS_VIEWER = False


@unittest.skipUnless(HAS_VIEWER, "gpr_profile_viewer non importabile")
class ResampleToUniformGridTest(unittest.TestCase):
    """Il resampling vettorizzato deve coincidere con np.interp riga per riga."""

    @staticmethod
    def _reference(disp, dist):
        d = np.asarray(dist, dtype=np.float64)
        uniform = np.linspace(d[0], d[-1], disp.shape[1])
        return np.stack([np.interp(uniform, d, row) for row in disp]).astype(np.float32)

    def _check(self, dist):
        rng = np.random.default_rng(11)
        disp = rng.uniform(-1, 1, (24, len(dist))).astype(np.float32)
        got = _GV.GprProfileViewer._resample_disp_to_uniform_grid(disp, dist)
        self.assertEqual(got.dtype, np.float32)
        self.assertLess(float(np.abs(got - self._reference(disp, dist)).max()), 1e-5)

    def test_smooth_non_uniform(self):
        n = 800
        d = np.cumsum(0.02 * (1 + 0.5 * np.sin(np.arange(n) / 30.0)))
        self._check(d - d[0])

    def test_repeated_distances_stopped_vehicle(self):
        n = 800
        d = np.cumsum(0.02 * (1 + 0.5 * np.sin(np.arange(n) / 30.0)))
        d[200:240] = d[200]   # tratto fermo
        d[-20:] = d[-21]      # coda ferma
        d[:15] = d[0]         # testa ferma
        self._check(d - d[0])

    def test_uniform_grid_is_returned_untouched(self):
        d = np.arange(300) * 0.02
        disp = np.zeros((4, 300), dtype=np.float32)
        self.assertIs(_GV.GprProfileViewer._resample_disp_to_uniform_grid(disp, d), disp)


class NearestIndexTest(unittest.TestCase):
    @unittest.skipUnless(HAS_VIEWER, "gpr_profile_viewer non importabile")
    def test_picks_closest_not_right_neighbour(self):
        d = np.array([0.0, 1.0, 2.0, 3.0])
        f = _GV.GprProfileViewer._nearest_index
        self.assertEqual(f(d, 1.2), 1)   # searchsorted darebbe 2
        self.assertEqual(f(d, 1.8), 2)
        self.assertEqual(f(d, -5.0), 0)
        self.assertEqual(f(d, 99.0), 3)
        self.assertEqual(f(np.array([7.0]), 1.0), 0)


@unittest.skipUnless(HAS_MPL and HAS_VIEWER, "matplotlib/viewer non disponibili")
class BlitManagerTest(unittest.TestCase):
    def _setup(self):
        fig = Figure(figsize=(3, 2), dpi=80)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        ax.imshow(np.random.default_rng(0).random((40, 60)), aspect="auto")
        line = ax.axvline(10, color="yellow")
        mgr = _GV._BlitManager(canvas)
        mgr.set_artists([line])
        return fig, canvas, ax, line, mgr

    def test_blit_matches_full_draw(self):
        _fig, canvas, _ax, line, mgr = self._setup()
        canvas.draw()
        line.set_xdata([35])
        mgr.update()
        blitted = np.asarray(canvas.buffer_rgba()).copy()
        canvas.draw()
        full = np.asarray(canvas.buffer_rgba()).copy()
        self.assertTrue(np.array_equal(blitted, full))

    def test_falls_back_to_full_draw_without_background(self):
        _fig, canvas, _ax, _line, mgr = self._setup()
        calls = []
        canvas.draw_idle = lambda: calls.append(1)
        mgr.update()  # nessun draw ancora: niente sfondo salvato
        self.assertEqual(calls, [1])

    def test_artists_of_hidden_axes_are_not_drawn(self):
        _fig, canvas, ax, line, mgr = self._setup()
        line.set_visible(False)
        canvas.draw()
        no_line = np.asarray(canvas.buffer_rgba()).copy()

        line.set_visible(True)
        mgr.update()  # controllo: con l'asse visibile la linea compare
        with_line = np.asarray(canvas.buffer_rgba()).copy()
        self.assertFalse(np.array_equal(no_line, with_line))

        ax.set_visible(False)
        mgr.update()  # asse nascosto: la linea animata non va disegnata
        self.assertTrue(np.array_equal(no_line, np.asarray(canvas.buffer_rgba())))

    def test_resize_invalidates_background(self):
        _fig, canvas, _ax, _line, mgr = self._setup()
        canvas.draw()
        self.assertIsNotNone(mgr._bg)
        mgr.invalidate()
        self.assertIsNone(mgr._bg)


def _make_profile(n_s=128, n_t=400, n_channels=1):
    rng = np.random.default_rng(3)
    data = rng.normal(0, 0.05, (n_s, n_t)).astype(np.float32)
    t = np.arange(n_s)[:, None]
    x = np.arange(n_t)[None, :]
    data += (np.exp(-((t - 40 - 0.002 * (x - 200) ** 2) ** 2) / 20.0) * np.cos(0.5 * t)).astype(np.float32)
    steps = 0.02 * (1 + 0.5 * np.sin(np.arange(n_t) / 20.0))  # velocita' non uniforme
    dist = np.concatenate([[0.0], np.cumsum(steps[1:])])
    ch = _RD.OgprChannel(0, data, 500000 + dist, 4500000 + 0 * dist, 0 * dist, 0 * dist, dist)
    return _RD.OgprProfile(
        path="synthetic.ogpr", version_major=1, version_minor=0, swath_name="s", swath_id="1",
        array_id=0, n_samples=n_s, n_channels=n_channels, n_slices=n_t, sampling_step_m=0.02,
        sampling_time_ns=0.117, velocity_m_s=1.0e8, frequency_mhz=400.0, polarization="HH",
        epsg=32633, value_type="f4", channels=[ch], header_raw={},
    )


class _FakeIface:
    def mapCanvas(self):
        if "_c" not in self.__dict__:
            from qgis.gui import QgsMapCanvas
            self._c = QgsMapCanvas()
        return self._c

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *a, **k: None


class _ViewerTestBase(unittest.TestCase):
    """Costruisce una finestra GprProfileViewer offscreen con un profilo sintetico."""

    PROFILE: dict = {}

    @classmethod
    def setUpClass(cls):
        from qgis.core import QgsApplication
        from qgis.PyQt.QtWidgets import QApplication
        cls._own_app = None
        if QApplication.instance() is None:
            cls._own_app = QgsApplication([], False)
            cls._own_app.initQgis()
        cls._qapp = QApplication.instance()

    @classmethod
    def tearDownClass(cls):
        if cls._own_app is not None:
            cls._own_app.exitQgis()

    def _pump(self, ms=0):
        t0 = time.time()
        while True:
            self._qapp.processEvents()
            if (time.time() - t0) * 1000 >= ms:
                break
            time.sleep(0.005)

    def setUp(self):
        self.v = _GV.GprProfileViewer(_FakeIface())
        self.v._profiles = [_make_profile(**self.PROFILE)]
        self.v._prof_idx = 0
        self.v.resize(1300, 800)
        self.v.show()
        self._pump(100)
        self.v._load_current_profile()
        self._pump(200)
        self.cv = self.v._canvas_mpl
        self.cv.draw()

    def tearDown(self):
        self.v._allow_close = True
        self.v.close()
        self._pump(50)

    def _event(self, x, z):
        px, py = self.v._ax.transData.transform((x, z))
        return MouseEvent("motion_notify_event", self.cv, px, py)


@unittest.skipUnless(HAS_MPL and HAS_VIEWER, "matplotlib/viewer non disponibili")
class ViewerInteractionTest(_ViewerTestBase):
    def test_pick_sample_uses_displayed_depth_axis(self):
        v = self.v
        depth = v._depth_max_display(v._profiles[0])
        n_s = v._disp_data.shape[0]
        self.assertEqual(v._sample_from_ydata(0.0), 0)
        self.assertEqual(v._sample_from_ydata(depth), n_s - 1)
        self.assertEqual(v._sample_from_ydata(depth / 2), int(round((n_s - 1) / 2)))
        # coerente con la lettura del cursore
        v._cursor_x, v._cursor_z = 2.0, depth / 3
        self.assertEqual(v._cursor_metrics()["idx_sample"], v._sample_from_ydata(depth / 3))

    def test_cursor_move_does_not_redraw_the_image(self):
        v = self.v
        draws = []
        orig = type(self.cv).draw
        type(self.cv).draw = lambda s: (draws.append(1), orig(s))[1]
        try:
            for x in np.linspace(0.5, 7.0, 15):
                v._on_mouse_move(self._event(x, 0.5))
                self._pump(0)
        finally:
            type(self.cv).draw = orig
        self.assertEqual(draws, [], "il cursore deve usare il blitting, non un draw completo")
        self.assertTrue(v._vline.get_visible())

    def test_status_text_does_not_resize_canvas(self):
        v = self.v
        sizes = []
        self.cv.mpl_connect("resize_event", lambda e: sizes.append((e.width, e.height)))
        for x in np.linspace(0.5, 7.0, 10):
            v._on_mouse_move(self._event(x, 0.5))
            self._pump(0)
        self.assertEqual(sizes, [])

    def test_vertical_zoom_survives_trace_change(self):
        v = self.v
        depth = v._depth_max_display(v._profiles[0])
        lim = (depth * 0.5, depth * 0.2)
        v._view_ylim = lim
        v._ax.set_ylim(*lim)
        self.cv.draw()
        for x in (1.0, 3.0, 5.0):
            v._on_mouse_move(self._event(x, depth * 0.3))
        self.assertEqual(tuple(v._ax.get_ylim()), lim)
        self.assertEqual(tuple(v._ax_wiggle.get_ylim()), lim)

    def test_gain_change_reuses_image_and_matches_full_redraw(self):
        v = self.v
        im0 = v._im
        v._spin_gain.setValue(3.0)
        v._apply_gain_only()
        self.cv.draw()
        self.assertIs(v._im, im0)
        fast = np.asarray(self.cv.buffer_rgba()).copy()
        v._redraw()
        self.cv.draw()
        self.assertTrue(np.array_equal(fast, np.asarray(self.cv.buffer_rgba())))

    def _expected_display(self, data, apply_depth_gain):
        """Display atteso: [range-gain in profondita'] -> normalizza -> gain scalare -> clip."""
        v = self.v
        norm = _PROC.normalize_display
        d = np.asarray(data, dtype=np.float32)
        if apply_depth_gain:
            d = d * v._build_depth_range_gain(d.shape[0])
        return np.clip(norm(d) * float(v._spin_gain.value()), -1.0, 1.0)

    def _run_with_stubbed_pipeline(self, fake):
        v = self.v
        for w in (v._chk_range_gain, v._chk_dewow):
            w.blockSignals(True)
            w.setChecked(True)
            w.blockSignals(False)
        v._param_timer.stop()
        real = _CM.apply_pipeline
        _CM.apply_pipeline = fake
        try:
            v._apply_processing()
        finally:
            _CM.apply_pipeline = real

    def test_range_gain_flag_describes_the_data_when_pipeline_result_is_used(self):
        v = self.v
        raw = np.asarray(v._raw_data, dtype=np.float32)
        self._run_with_stubbed_pipeline(lambda data, *a, **k: (np.asarray(data) * 0.5).astype(np.float32))
        self.assertTrue(v._proc_pre_agc)  # pipeline con pre_agc_gain: gain incorporato
        got = v._compose_display_data()
        # il gain e' gia' nel dato: non va applicato una seconda volta
        self.assertTrue(np.allclose(got, self._expected_display(raw * 0.5, apply_depth_gain=False), atol=1e-6))

    def test_range_gain_is_applied_for_display_after_raw_fallback(self):
        v = self.v
        raw = np.asarray(v._raw_data, dtype=np.float32)
        # risultato degenere -> _finish_processing ripiega sul dato grezzo
        self._run_with_stubbed_pipeline(lambda data, *a, **k: np.zeros_like(np.asarray(data), dtype=np.float32))
        self.assertTrue(v._last_pipeline_params.get("pre_agc_gain"))  # i parametri dicono "applicato"...
        self.assertFalse(v._proc_pre_agc)                            # ...ma il dato e' grezzo
        got = v._compose_display_data()
        self.assertTrue(np.allclose(got, self._expected_display(raw, apply_depth_gain=True), atol=1e-6))

    def test_swallowed_error_is_logged_once_per_cause(self):
        v = self.v
        logged = []
        real = _GV._log_gpr
        _GV._log_gpr = lambda msg, level=None: logged.append(msg)
        try:
            for _ in range(5):
                try:
                    raise ValueError("stessa causa")
                except ValueError as exc:
                    v._note_swallowed("prova", exc)
            try:
                raise ValueError("causa diversa")
            except ValueError as exc:
                v._note_swallowed("prova", exc)
        finally:
            _GV._log_gpr = real
        self.assertEqual(len(logged), 2)
        self.assertIn("stessa causa", logged[0])
        self.assertIn("prova", logged[0])

    def test_manual_gain_failure_is_visible_in_log_and_display_survives(self):
        v = self.v
        logged = []
        real_log, real_mg = _GV._log_gpr, _RV.manual_gain
        _GV._log_gpr = lambda msg, level=None: logged.append(msg)
        _RV.manual_gain = lambda *a, **k: (_ for _ in ()).throw(ValueError("mg boom"))
        v._chk_manual_gain.blockSignals(True)
        v._chk_manual_gain.setChecked(True)
        v._chk_manual_gain.blockSignals(False)
        try:
            first = v._compose_display_data()
            v._compose_display_data()
        finally:
            _GV._log_gpr, _RV.manual_gain = real_log, real_mg
        self.assertEqual(len(logged), 1)  # una volta sola, non a ogni ridisegno
        self.assertIn("mg boom", logged[0])
        self.assertIn("manual gain", logged[0])
        self.assertEqual(first.shape, v._disp_data.shape)
        self.assertTrue(np.isfinite(first).all())

    def test_display_base_cache_follows_processed_data(self):
        v = self.v
        first = v._compose_display_data()
        cache = v._display_base_cache
        v._spin_gain.setValue(2.0)
        v._compose_display_data()
        self.assertIs(v._display_base_cache, cache)          # gain: riuso
        v._proc_data = (v._proc_data * 0.5).astype(np.float32)
        v._compose_display_data()
        self.assertIsNot(v._display_base_cache, cache)       # nuovo dato: ricalcolo
        self.assertEqual(first.shape, v._compose_display_data().shape)

    def _enable_filters(self):
        v = self.v
        for w in (v._chk_dewow, v._chk_agc):
            w.blockSignals(True)
            w.setChecked(True)
            w.blockSignals(False)
        v._param_timer.stop()

    def _wait_idle(self, timeout=60):
        t0 = time.time()
        while (self.v._proc_thread_running() or self.v._proc_busy) and time.time() - t0 < timeout:
            self._pump(10)
        self._pump(100)

    def test_async_pipeline_matches_sync(self):
        v = self.v
        self._enable_filters()
        v._apply_processing()
        ref = np.asarray(v._proc_data).copy()
        v._proc_data = None
        v._stage_cache.clear()
        v._apply_processing_async()
        self.assertTrue(v._proc_busy)
        self._wait_idle()
        self.assertIsNotNone(v._proc_data)
        self.assertTrue(np.array_equal(np.asarray(v._proc_data), ref))
        self.assertFalse(v._proc_busy)

    def test_async_requests_during_a_job_collapse_into_one_rerun(self):
        v = self.v
        self._enable_filters()
        v._stage_cache.clear()
        started = []
        real = v._begin_processing
        v._begin_processing = lambda deferred: (lambda j: (started.append(j is not None), j)[1])(real(deferred))
        v._apply_processing_async()
        v._apply_processing_async()
        v._apply_processing_async()
        self._wait_idle()
        self.assertEqual(sum(started), 2)  # job in volo + un solo rerun
        self.assertIsNotNone(v._proc_data)

    def test_result_of_superseded_job_is_discarded(self):
        v = self.v
        self._enable_filters()
        v._stage_cache.clear()
        v._proc_data = None
        v._apply_processing_async()
        v._proc_job_seq += 1  # come cambiare profilo/canale mentre il job gira
        self._wait_idle()
        self.assertIsNone(v._proc_data)
        self.assertFalse(v._proc_busy)

    def test_deferred_reload_hides_stale_data_until_the_new_result_arrives(self):
        v = self.v
        self._enable_filters()
        v._stage_cache.clear()
        self.assertIsNotNone(v._disp_data)
        v._reload_data(deferred=True)
        self.assertIsNone(v._disp_data)      # il dato mostrato era del profilo/canale precedente
        self.assertTrue(v._proc_busy)
        v._on_mouse_move(self._event(1.0, 0.5))  # senza dato il cursore non deve rompersi
        self._wait_idle()
        self.assertIsNotNone(v._disp_data)
        self.assertEqual(v._disp_data.shape[1], v._raw_data.shape[1])

    def test_next_profile_loads_asynchronously_and_ends_on_the_new_profile(self):
        v = self.v
        v._profiles.append(_make_profile(n_s=128, n_t=300))
        self._enable_filters()
        v._stage_cache.clear()
        v._next_profile()
        self.assertEqual(v._prof_idx, 1)
        self.assertIsNone(v._disp_data)
        self._wait_idle()
        self.assertIsNotNone(v._disp_data)
        self.assertEqual(v._disp_data.shape[1], 300)
        self.assertIn("synthetic.ogpr", v._lbl_profile.text())

    def test_gain_change_while_a_job_runs_does_not_block_the_gui(self):
        import threading
        v = self.v
        self._enable_filters()
        release = threading.Event()
        real = _CM.apply_pipeline

        def slow(data, *a, **k):
            release.wait(10)
            return (np.asarray(data) * 0.5).astype(np.float32)

        _CM.apply_pipeline = slow
        try:
            v._stage_cache.clear()
            v._proc_data = None  # come subito dopo un reload
            v._apply_processing_async()
            self.assertTrue(v._proc_thread_running())
            t0 = time.time()
            v._apply_gain_only()  # prima: _apply_processing() sincrono, bloccato fino a fine job
            self.assertLess(time.time() - t0, 1.0)
            self.assertTrue(v._proc_thread_running())
            release.set()
            self._wait_idle()
        finally:
            release.set()
            _CM.apply_pipeline = real
        self.assertIsNotNone(v._disp_data)

    def test_bandpass_controls_use_the_async_path(self):
        v = self.v
        calls = []
        v._apply_processing_async = lambda *a: calls.append("async")
        v._apply_processing = lambda *a: calls.append("sync")
        v._on_bp_controls_changed(True)
        self.assertEqual(calls, ["async"])

    def test_worker_error_is_reported_and_busy_cleared(self):
        v = self.v
        self._enable_filters()
        v._stage_cache.clear()
        errors = []
        v._show_processing_error = lambda exc: errors.append(str(exc))
        real = _CM.apply_pipeline
        _CM.apply_pipeline = lambda *a, **k: (_ for _ in ()).throw(ValueError("boom"))
        try:
            v._apply_processing_async()
            self._wait_idle()
        finally:
            _CM.apply_pipeline = real
        self.assertEqual(errors, ["boom"])
        self.assertFalse(v._proc_busy)


@unittest.skipUnless(HAS_MPL and HAS_VIEWER, "matplotlib/viewer non disponibili")
class RestoredProcessingStateAppliesFiltersTest(_ViewerTestBase):
    """Riaprire un progetto con processing salvato deve mostrare subito il
    radargramma filtrato, senza dover deflaggare/flaggare un filtro a mano
    (bug: _apply_gain_hyper_preset non specchiava l'enabled-map sulle caselle
    _chk_* quando chiamato durante il restore, con _syncing_filter_chain=True)."""

    def test_restoring_a_state_with_dewow_enabled_checks_the_box_and_reprocesses(self):
        v = self.v
        self.assertFalse(v._chk_dewow.isChecked())    # baseline: nessun filtro attivo
        raw_before = np.array(v._disp_data, copy=True) if v._disp_data is not None else None

        state = {
            "gain_hyper": {
                "format": "gpr_gain_hyper_preset",
                "pipeline_chain": {"enabled": {"dewow": True}},
            },
            "params": {},
        }
        v._apply_full_processing_state(state)
        self._pump(50)

        self.assertTrue(v._chk_dewow.isChecked())     # specchiato subito, non solo dopo un toggle manuale
        self.assertNotIn("RAW", v._lbl_status.text().upper())
        if raw_before is not None and v._disp_data is not None:
            self.assertFalse(np.array_equal(v._disp_data, raw_before))   # il dewow ha davvero cambiato l'immagine


@unittest.skipUnless(HAS_MPL and HAS_VIEWER, "matplotlib/viewer non disponibili")
class HyperbolaVelocityFieldTest(_ViewerTestBase):
    """Campo velocita' in m/ns accanto all'RDP (item 8) e salvataggio del fit
    come preimpostato di variante (item 7, gpr_velocity_store.py)."""

    def test_rdp_and_velocity_spin_stay_in_sync_both_ways(self):
        v = self.v
        HM = importlib.import_module("rasterlinker_feature_interpolation.gpr_hyperbola_math")
        v._spin_hyperbola_rdp.setValue(16.0)
        self.assertAlmostEqual(
            v._spin_hyperbola_vel_mns.value(), HM.rdp_to_velocity_m_s(16.0) * 1e-9, places=4)
        v._spin_hyperbola_vel_mns.setValue(0.12)
        self.assertAlmostEqual(v._spin_hyperbola_rdp.value(), HM.velocity_to_rdp(0.12 * 1e9), places=1)

    def test_save_velocity_refuses_without_a_saved_variant(self):
        v = self.v
        v._variant_parent_ctx = None
        warned = []
        orig = _GV.QMessageBox.information
        _GV.QMessageBox.information = staticmethod(lambda *a, **k: warned.append(a))
        try:
            v._save_hyperbola_velocity_to_variant()
        finally:
            _GV.QMessageBox.information = orig
        self.assertEqual(len(warned), 1)

    def test_save_velocity_writes_the_sidecar_for_the_saved_variant(self):
        v = self.v
        VS = importlib.import_module("rasterlinker_feature_interpolation.gpr_velocity_store")
        with tempfile.TemporaryDirectory() as root:
            v._gpr_workspace_override = root
            v._variant_parent_ctx = "Lavoro"
            v._spin_hyperbola_rdp.setValue(11.0)
            v._save_hyperbola_velocity_to_variant()
            variant_dir = os.path.join(root, "GPR_Processing", "processed", "Lavoro")
            got = VS.resolve_velocity_m_ns(variant_dir, None)
            self.assertEqual(got["source"], "manual")
            self.assertAlmostEqual(got["rdp"], 11.0, places=3)


@unittest.skipUnless(HAS_MPL and HAS_VIEWER, "matplotlib/viewer non disponibili")
class VelocityEstimationTabTest(_ViewerTestBase):
    """Scheda 'Stima velocità' separata dalla pipeline (round 3): i controlli
    dell'iperbole vivono fuori da 'Processing', metodo due-punti, lista di
    stime multiple con aggregato media/mediana, niente più salvataggio come
    parte del preset Gain/Hyperbola."""

    def test_velocity_tab_exists_and_hosts_the_hyperbola_controls(self):
        v = self.v
        titles = [v._right_tabs.tabText(i) for i in range(v._right_tabs.count())]
        self.assertIn("Stima velocità", titles)
        self.assertTrue(v._velocity_group_box.isAncestorOf(v._chk_hyperbola))
        self.assertTrue(v._velocity_group_box.isAncestorOf(v._tbl_hyper_picks))
        self.assertFalse(v._processing_panel_widget.isAncestorOf(v._chk_hyperbola))

    def test_two_point_method_matches_the_math_module(self):
        v = self.v
        HM = importlib.import_module("rasterlinker_feature_interpolation.gpr_hyperbola_math")
        depth = v._depth_max_display(v._profiles[0])
        v._on_mouse_move(self._event(4.0, depth * 0.2))
        v._set_hyperbola_apex_from_cursor()
        self.assertEqual(v._hyperbola_last_method, "Manuale")
        v._on_mouse_move(self._event(4.3, depth * 0.35))
        v._set_hyperbola_two_point()
        self.assertEqual(v._hyperbola_last_method, "Due punti")
        prof = v._profiles[v._prof_idx]
        v_ref = v._profile_reference_velocity(prof)
        s = v._hyperbola_txrx_offset()
        expected = HM.velocity_from_two_points(
            v._hyper_apex_x, v._hyper_apex_depth, v._cursor_x, v._cursor_z, v_ref, s=s)
        self.assertIsNotNone(expected)
        _, expected_rdp = expected
        self.assertAlmostEqual(v._spin_hyperbola_rdp.value(), expected_rdp, places=3)

    def test_two_point_button_requires_an_apex_first(self):
        v = self.v
        v._hyper_apex_x = None
        v._hyper_apex_depth = None
        warned = []
        orig = _GV.QMessageBox.information
        _GV.QMessageBox.information = staticmethod(lambda *a, **k: warned.append(a))
        try:
            v._set_hyperbola_two_point()
        finally:
            _GV.QMessageBox.information = orig
        self.assertEqual(len(warned), 1)

    def test_pick_list_aggregate_mean_median_and_exclusion(self):
        from qgis.PyQt.QtCore import Qt
        v = self.v
        v._hyper_picks = [
            {"depth": 0.2, "method": "Manuale", "v_mns": 0.08, "rdp": 14.0, "included": True},
            {"depth": 0.4, "method": "Due punti", "v_mns": 0.10, "rdp": 9.0, "included": True},
            {"depth": 0.6, "method": "Automatico (ricerca ampiezza)", "v_mns": 0.20, "rdp": 2.25, "included": True},
        ]
        v._refresh_hyper_picks_table()
        self.assertEqual(v._tbl_hyper_picks.rowCount(), 3)
        self.assertIn("media", v._lbl_hyper_aggregate.text())
        self.assertIn(f"v={(0.08 + 0.10 + 0.20) / 3.0:.4f}", v._lbl_hyper_aggregate.text())

        idx = v._cb_hyper_aggregate.findData("median")
        self.assertGreaterEqual(idx, 0)
        v._cb_hyper_aggregate.setCurrentIndex(idx)
        self.assertIn("mediana", v._lbl_hyper_aggregate.text())
        self.assertIn("v=0.1000", v._lbl_hyper_aggregate.text())

        # deselezionare una riga la esclude dall'aggregato
        v._tbl_hyper_picks.item(2, 0).setCheckState(Qt.Unchecked)
        self.assertIn("2 stime", v._lbl_hyper_aggregate.text())
        self.assertFalse(v._hyper_picks[2]["included"])

    def test_add_current_pick_requires_an_apex(self):
        v = self.v
        v._hyper_apex_depth = None
        v._hyper_picks = []
        warned = []
        orig = _GV.QMessageBox.information
        _GV.QMessageBox.information = staticmethod(lambda *a, **k: warned.append(a))
        try:
            v._add_current_hyperbola_pick()
        finally:
            _GV.QMessageBox.information = orig
        self.assertEqual(len(warned), 1)
        self.assertEqual(v._hyper_picks, [])

    def test_save_velocity_uses_the_aggregate_when_picks_are_included(self):
        v = self.v
        VS = importlib.import_module("rasterlinker_feature_interpolation.gpr_velocity_store")
        with tempfile.TemporaryDirectory() as root:
            v._gpr_workspace_override = root
            v._variant_parent_ctx = "Lavoro"
            v._hyper_picks = [
                {"depth": 0.2, "method": "Manuale", "v_mns": 0.08, "rdp": 14.0, "included": True},
                {"depth": 0.4, "method": "Due punti", "v_mns": 0.12, "rdp": 6.25, "included": True},
            ]
            v._save_hyperbola_velocity_to_variant()
            variant_dir = os.path.join(root, "GPR_Processing", "processed", "Lavoro")
            got = VS.resolve_velocity_m_ns(variant_dir, None)
            self.assertEqual(got["source"], "hyperbola_fit_aggregate")
            self.assertAlmostEqual(got["velocity_m_ns"], 0.10, places=6)

    def test_gain_hyper_preset_no_longer_carries_hyperbola_state(self):
        v = self.v
        v._chk_hyperbola.setChecked(True)
        v._spin_hyperbola_rdp.setValue(15.0)
        v._hyper_apex_x, v._hyper_apex_depth = 3.0, 0.4
        preset = v._collect_gain_hyper_preset()
        self.assertNotIn("hyperbola", preset)

    def test_apply_gain_hyper_preset_ignores_a_round2_hyperbola_key(self):
        v = self.v
        v._chk_hyperbola.setChecked(False)
        v._spin_hyperbola_rdp.setValue(9.0)
        old_format_payload = {
            "format": "gpr_gain_hyper_preset",
            "hyperbola": {"enabled": True, "rdp": 30.0, "apex_x": 1.0, "apex_depth": 0.1},
        }
        v._apply_gain_hyper_preset(old_format_payload)   # non deve sollevare, e va ignorato
        self.assertFalse(v._chk_hyperbola.isChecked())
        self.assertAlmostEqual(v._spin_hyperbola_rdp.value(), 9.0, places=3)


@unittest.skipUnless(HAS_VIEWER, "gpr_profile_viewer non importabile")
class ImageLodTest(unittest.TestCase):
    """Logica pura del livello di dettaglio (ritaglio + decimazione) per profili lunghi."""

    L = 200.0

    def _lod(self, n_s=4, n_t=10000):
        src = np.random.default_rng(0).uniform(-1, 1, (n_s, n_t)).astype(np.float32)
        return _RV._ImageLod(src, self.L)

    def test_inactive_below_threshold(self):
        self.assertFalse(self._lod(n_t=_RV._ImageLod.MIN_COLS - 1).active)
        self.assertTrue(self._lod(n_t=_RV._ImageLod.MIN_COLS).active)

    def test_full_view_is_decimated_and_crop_covers_view(self):
        lod = self._lod()
        f, need, want = lod.plan(0.0, self.L, px=1000)
        self.assertGreater(f, 1)
        self.assertTrue(want[0] <= need[0] and want[1] >= need[1])
        arr, (left, right) = lod.crop(f, *want)
        self.assertLess(arr.shape[1], lod.n_cols)
        self.assertLessEqual(left, 0.0 + 1e-9)
        self.assertGreaterEqual(right, self.L - 1e-9)

    def test_zoomed_view_uses_original_resolution(self):
        lod = self._lod()
        f, need, want = lod.plan(90.0, 92.0, px=1000)  # ~100 colonne su ~2000 disponibili
        self.assertEqual(f, 1)
        arr, (left, right) = lod.crop(f, *want)
        self.assertTrue(np.array_equal(arr, lod.src[:, want[0]:want[1]]))
        self.assertLessEqual(left, 90.0)
        self.assertGreaterEqual(right, 92.0)

    def test_levels_are_block_means_including_the_tail(self):
        lod = self._lod(n_t=10003)
        lv = lod.level(4)
        self.assertEqual(lv.shape[1], 2501)
        self.assertTrue(np.allclose(lv[:, :2500], lod.src[:, :10000].reshape(4, 2500, 4).mean(axis=2), atol=1e-6))
        self.assertTrue(np.allclose(lv[:, 2500], lod.src[:, 10000:].mean(axis=1), atol=1e-6))
        self.assertIs(lod.level(4), lv)  # in cache

    def test_crop_extent_matches_columns(self):
        lod = self._lod()
        _arr, (left, right) = lod.crop(4, 100, 300)
        scale = self.L / lod.n_cols
        self.assertAlmostEqual(left, 100 * 4 * scale)
        self.assertAlmostEqual(right, 300 * 4 * scale)

    def test_small_pan_keeps_the_crop_large_pan_reloads_it(self):
        lod = self._lod(n_t=20000)
        f, need, want = lod.plan(80.0, 100.0, px=1000)
        lod.state = (f, want[0], want[1])
        f2, need2, _w = lod.plan(81.0, 101.0, px=1000)   # piccolo spostamento: dentro il margine
        self.assertTrue(lod.covers(f2, need2))
        f3, need3, _w = lod.plan(150.0, 170.0, px=1000)  # lontano
        self.assertFalse(lod.covers(f3, need3))
        f4, need4, _w = lod.plan(80.0, 180.0, px=1000)   # altro fattore di decimazione
        self.assertFalse(lod.covers(f4, need4))


@unittest.skipUnless(HAS_MPL and HAS_VIEWER, "matplotlib/viewer non disponibili")
class ViewerLongProfileTest(_ViewerTestBase):
    """Profilo sopra la soglia: l'immagine mostra solo il ritaglio visibile."""

    PROFILE = dict(n_s=32, n_t=12000)

    def _shown_cols(self):
        return int(self.v._im.get_array().shape[1])

    def _crop_extent(self):
        e = self.v._im.get_extent()
        return e[0], e[1]

    def _zoom_to(self, x0, x1):
        self.v._view_xlim = (x0, x1)
        self.v._ax.set_xlim(x0, x1)

    def test_full_view_shows_a_decimated_crop_covering_the_view(self):
        v = self.v
        self.assertIsNotNone(v._lod)
        self.assertLess(self._shown_cols(), 12000)
        c0, c1 = self._crop_extent()
        x0, x1 = v._ax.get_xlim()
        self.assertLessEqual(c0, x0 + 1e-9)
        self.assertGreaterEqual(c1, x1 - 1e-9)

    def test_zoom_in_switches_to_original_resolution_without_moving_the_view(self):
        v = self.v
        length = v._im_full_extent[1]
        mid = length / 2
        self._zoom_to(mid - length * 0.01, mid + length * 0.01)
        self.assertEqual(v._lod.state[0], 1)
        f, a, b = v._lod.state
        self.assertTrue(np.array_equal(np.asarray(v._im.get_array()), v._lod.src[:, a:b]))
        self.assertEqual(tuple(v._ax.get_xlim()), tuple(v._view_xlim))  # set_extent non sposta la vista

    def test_pan_slider_moves_the_crop(self):
        v = self.v
        length = v._im_full_extent[1]
        self._zoom_to(length * 0.10, length * 0.15)
        self.cv.draw()
        before = v._lod.state
        v._xpan_slider.setValue(900)  # quasi in fondo al profilo
        x0, x1 = v._ax.get_xlim()
        c0, c1 = self._crop_extent()
        self.assertNotEqual(v._lod.state, before)
        self.assertLessEqual(c0, x0 + 1e-9)
        self.assertGreaterEqual(c1, x1 - 1e-9)

    def test_scroll_zoom_out_can_reach_the_whole_profile(self):
        # il vincolo va preso dall'immagine INTERA, non dal ritaglio mostrato
        v = self.v
        length = v._im_full_extent[1]
        mid = length / 2
        self._zoom_to(mid - length * 0.01, mid + length * 0.01)
        self.cv.draw()
        for _ in range(80):
            px, py = v._ax.transData.transform((sum(v._ax.get_xlim()) / 2, sum(v._ax.get_ylim()) / 2))
            ev = MouseEvent("scroll_event", self.cv, px, py, button="down", step=-1)
            v._on_scroll_zoom(ev)
        x0, x1 = v._ax.get_xlim()
        self.assertAlmostEqual(x0, 0.0, places=3)
        self.assertAlmostEqual(x1, length, places=3)

    def test_gain_change_keeps_the_crop_and_updates_pixels(self):
        v = self.v
        old_src = v._lod.src
        v._spin_gain.setValue(3.0)
        v._apply_gain_only()
        self.assertIsNotNone(v._lod)
        self.assertIsNot(v._lod.src, old_src)
        self.assertLess(self._shown_cols(), 12000)
        self.assertGreater(float(np.abs(v._lod.src).max()), float(np.abs(old_src).max()) - 1e-9)

    def test_long_profile_render_is_close_to_full_resolution_render(self):
        v = self.v
        self.cv.draw()
        lod_px = np.asarray(self.cv.buffer_rgba()).astype(int)[:, :, :3]
        saved = _RV._ImageLod.MIN_COLS
        _RV._ImageLod.MIN_COLS = 10 ** 9
        try:
            v._view_xlim = None
            v._reset_zoom()
            self.cv.draw()
            ref_px = np.asarray(self.cv.buffer_rgba()).astype(int)[:, :, :3]
        finally:
            _RV._ImageLod.MIN_COLS = saved
        self.assertLess(float(np.abs(lod_px - ref_px).mean()), 1.0)  # /255


@unittest.skipUnless(HAS_MPL and HAS_VIEWER, "matplotlib/viewer non disponibili")
class ShortProfileKeepsFullImageTest(_ViewerTestBase):
    def test_below_threshold_the_image_is_untouched(self):
        v = self.v
        self.assertIsNone(v._lod)
        self.assertEqual(int(v._im.get_array().shape[1]), v._disp_data.shape[1])


class SanitizeGainBreakpointsTest(unittest.TestCase):
    """Il tetto del range gain (motore) non e' piu' 100."""

    @unittest.skipUnless(HAS_VIEWER, "gpr_processing non importabile")
    def test_gain_above_100_is_kept(self):
        pts = _PROC._sanitize_gain_breakpoints([[0.0, 0.3], [0.5, 800.0], [1.0, 5000.0]])
        self.assertEqual(pts[:, 1].tolist(), [0.3, 800.0, 5000.0])
        g = _PROC._build_pre_agc_gain(
            101, {"pre_agc_breakpoints": [[0.0, 0.3], [1.0, 5000.0]]}
        )
        self.assertAlmostEqual(float(g[-1, 0]), 5000.0, places=1)

    @unittest.skipUnless(HAS_VIEWER, "gpr_processing non importabile")
    def test_ceiling_is_still_enforced(self):
        pts = _PROC._sanitize_gain_breakpoints([[0.0, 1.0], [1.0, 1e12]])
        self.assertEqual(float(pts[-1, 1]), _PROC.RANGE_GAIN_MAX)


@unittest.skipUnless(HAS_MPL and HAS_VIEWER, "matplotlib/viewer non disponibili")
class RangeGainCurveLimitTest(_ViewerTestBase):
    def _dialog(self, **kw):
        dlg = _CM._RangeGainCurveDialog(max_points=16, **kw)
        self._pump(30)   # smaltisce il draw_idle del costruttore prima che il dialogo sparisca
        self.addCleanup(lambda: (dlg.close(), self._pump(30)))
        return dlg

    def test_interior_peak_is_not_flattened_to_the_deep_value(self):
        v = self.v
        v._chk_range_gain.setChecked(True)
        self._pump(50)
        # picco a meta' profondita', fondo piu' basso (per non amplificare il rumore)
        v._on_range_gain_curve_live_changed(np.array([[0.0, 0.3], [0.5, 50.0], [1.0, 20.0]]))
        self._pump(50)
        sent = v._current_processing_params()["pre_agc_breakpoints"]
        self.assertEqual([p[1] for p in sent], [0.3, 50.0, 20.0])
        self.assertAlmostEqual(float(v._build_depth_range_gain(101)[50, 0]), 50.0, places=3)

    def test_gain_above_100_reaches_the_pipeline_and_the_display(self):
        v = self.v
        v._chk_range_gain.setChecked(True)
        self.assertGreaterEqual(v._spin_gain_deep.maximum(), 1e5)
        v._spin_gain_deep.setValue(2500.0)
        self._pump(50)
        self.assertEqual(v._current_processing_params()["pre_agc_breakpoints"][-1][1], 2500.0)
        self.assertAlmostEqual(float(v._build_depth_range_gain(64)[-1, 0]), 2500.0, places=1)

    def test_editor_axis_follows_the_curve_and_keeps_points_above_100(self):
        pts = np.array([[0.0, 0.3], [0.4, 300.0], [1.0, 40.0]])
        dlg = self._dialog(points=pts)
        self.assertEqual(dlg.points()[:, 1].tolist(), [0.3, 300.0, 40.0])   # niente clip a 100
        self.assertGreaterEqual(dlg.gain_max(), 600.0)                       # asse auto, sopra i punti
        self.assertTrue(dlg._chk_log.isChecked())                            # decadi -> scala log
        # il picking funziona anche con l'asse log: ogni punto e' il piu' vicino a se stesso
        dlg._canvas.draw()
        for i, (depth, gain) in enumerate(dlg.points()):
            idx, dist = dlg._nearest_idx(gain, depth)
            self.assertEqual(idx, i)
            self.assertLess(dist, 1e-6)

    def test_editor_stays_linear_for_small_gains(self):
        dlg = self._dialog(points=np.array([[0.0, 1.0], [1.0, 6.0]]))
        self.assertFalse(dlg._chk_log.isChecked())
        self.assertEqual(dlg.gain_max(), 20.0)

    def test_editor_manual_axis_does_not_clip_points(self):
        dlg = self._dialog(points=np.array([[0.0, 1.0], [1.0, 500.0]]), gain_max=50.0)
        self.assertFalse(dlg._chk_auto_axis.isChecked())
        self.assertEqual(float(dlg.points()[-1, 1]), 500.0)
        self.assertGreaterEqual(dlg.gain_max(), 500.0)   # l'asse non nasconde il punto


if __name__ == "__main__":
    unittest.main()
