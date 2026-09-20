# -*- coding: utf-8 -*-
"""Profile viewer: cursore/blitting, pick da profilo, fast path del gain.

La parte "viewer" richiede Qt (offscreen) e QGIS; si salta se non disponibili.
"""

import importlib
import os
import pathlib
import sys
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
    HAS_VIEWER = True
except Exception:  # pragma: no cover - ambiente senza QGIS
    HAS_VIEWER = False


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


def _make_profile(n_s=128, n_t=400):
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
        array_id=0, n_samples=n_s, n_channels=1, n_slices=n_t, sampling_step_m=0.02,
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


@unittest.skipUnless(HAS_MPL and HAS_VIEWER, "matplotlib/viewer non disponibili")
class ViewerInteractionTest(unittest.TestCase):
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
        self.v._profiles = [_make_profile()]
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


if __name__ == "__main__":
    unittest.main()
