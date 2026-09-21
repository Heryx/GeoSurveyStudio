# -*- coding: utf-8 -*-
"""Pannello agganciabile dei radargrammi processati (Qt offscreen + QGIS).

Si lancia come pacchetto con il Python di QGIS:
    python-qgis.bat -m unittest test.test_radargram_dock
"""

import gc
import importlib
import os
import pathlib
import sys
import tempfile
import time
import types
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_PLUGIN_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(_PLUGIN_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR.parent))

try:
    _DOCK = importlib.import_module("rasterlinker_feature_interpolation.gpr_radargram_dock")
    _SM = importlib.import_module("rasterlinker_feature_interpolation.slice_manifest")
    _SS = importlib.import_module("rasterlinker_feature_interpolation.slice_spec")
    _RIO = importlib.import_module("rasterlinker_feature_interpolation.raster_io")
    from qgis.PyQt.QtCore import QObject, Qt, pyqtSignal
    from qgis.PyQt.QtWidgets import QMainWindow
    from qgis.core import QgsCoordinateReferenceSystem, QgsPointXY, QgsProject, QgsRasterLayer
    from qgis.gui import QgsMapCanvas
    from matplotlib.backend_bases import MouseEvent
    HAS_ENV = bool(_DOCK.HAS_MPL)
except Exception:  # pragma: no cover - ambiente senza QGIS/matplotlib
    HAS_ENV = False

from .test_radargram_store import _channel, _profile, _write_variant


class _Iface(QObject if HAS_ENV else object):
    """iface minimo: finestra principale, canvas vero e segnale del livello corrente."""

    if HAS_ENV:
        currentLayerChanged = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self._win = QMainWindow()
        self._canvas = QgsMapCanvas()
        self._active = None
        self.docks = []

    def mainWindow(self):
        return self._win

    def mapCanvas(self):
        return self._canvas

    def activeLayer(self):
        return self._active

    def addDockWidget(self, area, dock):
        self.docks.append((area, dock))
        self._win.addDockWidget(area, dock)

    def removeDockWidget(self, dock):
        self._win.removeDockWidget(dock)


def _write_slice(root, variant, grid, z0, z1):
    """GeoTIFF minuscolo + sliceset.json accanto, come li scrive la generazione time-slice."""
    folder = os.path.join(root, "GPR_Processing", "timeslices", variant)
    os.makedirs(folder, exist_ok=True)
    _RIO.write_tif_singleband(np.ones((4, 4), dtype=np.float32), os.path.join(folder, grid),
                              500000.0, 0.0, 3.0, 1.0, 32632)
    _SM.write_sliceset(
        folder, _SS.SliceSpec(),
        [{"index": 0, "z_from": z0, "z_to": z1, "grid": grid}],
        {"kind": "ogpr", "variant_id": variant, "n_profiles": 2}, crs_epsg=32632)
    return os.path.join(folder, grid)


_OWN_APP = None


def setUpModule():
    global _OWN_APP
    if not HAS_ENV:
        return
    from qgis.core import QgsApplication
    from qgis.PyQt.QtWidgets import QApplication
    if QApplication.instance() is None:
        _OWN_APP = QgsApplication([], False)
        _OWN_APP.initQgis()


def tearDownModule():
    if _OWN_APP is not None:
        _OWN_APP.exitQgis()


@unittest.skipUnless(HAS_ENV, "QGIS/matplotlib non disponibili")
class _DockBase(unittest.TestCase):
    """Pannello aperto su una variante 'Lavoro' e una 'v3' di prova, in una finestra offscreen."""

    @classmethod
    def setUpClass(cls):
        from qgis.PyQt.QtWidgets import QApplication
        cls._qapp = QApplication.instance()

    def _pump(self, ms=0):
        t0 = time.time()
        while True:
            self._qapp.processEvents()
            if (time.time() - t0) * 1000 >= ms:
                break
            time.sleep(0.005)

    @staticmethod
    def _forget_saved_look():
        from qgis.PyQt.QtCore import QSettings
        s = QSettings()
        s.remove("GeoSurveyStudio/radargram_dock")
        s.sync()

    def setUp(self):
        self._forget_saved_look()
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = self._tmp.name
        self._layers = []
        # Swath2 -> indice 0 (nord +10 m); Swath10 -> indice 1 (nord 0)
        self.v_lavoro, _ = _write_variant(self.root, "Lavoro", ["Swath10", "Swath2"])
        self.v_v3, _ = _write_variant(self.root, "v3", ["Swath10", "Swath2"])
        crs = QgsCoordinateReferenceSystem("EPSG:32632")
        QgsProject.instance().setCrs(crs)
        self.iface = _Iface()
        self.iface.mapCanvas().setDestinationCrs(crs)
        self.dock = _DOCK.RadargramDock(
            self.iface, workspace_provider=lambda: self.root, geometry_provider=None,
            parent=self.iface.mainWindow())
        self.iface.addDockWidget(Qt.BottomDockWidgetArea, self.dock)
        self.dock.resize(900, 320)
        self.iface.mainWindow().show()
        self.dock.show()
        self._pump(80)
        self.dock._canvas.draw()

    def tearDown(self):
        self.dock.cleanup()
        self.dock.close()
        self.iface.mainWindow().close()
        self._pump(30)
        self.iface._active = None
        self._layers.clear()             # su Windows un layer aperto blocca il .tif da cancellare
        gc.collect()
        self._tmp.cleanup()
        self._forget_saved_look()

    # -- utilita' --------------------------------------------------------

    def _event(self, name, x_m, z_m, button=1, dblclick=False, step=0):
        """Evento matplotlib alle coordinate dati (distanza m, profondita' m)."""
        cv = self.dock._canvas
        cv.draw()
        px, py = self.dock._ax.transData.transform((x_m, z_m))
        if name == "scroll_event":
            from matplotlib.backend_bases import MouseEvent as ME
            return ME(name, cv, px, py, button="up" if step > 0 else "down", step=step)
        return MouseEvent(name, cv, px, py, button=button, dblclick=dblclick)

    def _layer(self, path):
        lyr = QgsRasterLayer(path, os.path.basename(path))
        self.assertTrue(lyr.isValid(), path)
        self._layers.append(lyr)
        return lyr


@unittest.skipUnless(HAS_ENV, "QGIS/matplotlib non disponibili")
class DockTest(_DockBase):
    # -- apertura -------------------------------------------------------------

    def test_opens_working_variant_first_profile(self):
        d = self.dock
        names = [d._cb_variant.itemText(i) for i in range(d._cb_variant.count())]
        self.assertEqual(len(names), 2)
        self.assertEqual(d._store.name, "Lavoro")            # "Lavoro" e' la variante di default
        self.assertEqual((d._pi, d._ci), (0, 0))
        self.assertEqual(d._cb_profile.count(), 2)
        self.assertEqual(d._cb_channel.count(), 2)
        self.assertIn("Swath2", d._cb_profile.itemText(0))   # ordine naturale
        self.assertIn("Profilo 1/2", d._lbl_status.text())

    def test_axes_are_in_metres_and_depth_is_downwards(self):
        d = self.dock
        ch = d._cur
        self.assertEqual(tuple(d._ax.get_xlim()), (0.0, ch.length_m))
        self.assertEqual(tuple(d._ax.get_ylim()), (ch.depth_max_m, 0.0))
        self.assertAlmostEqual(ch.depth_max_m, 40 * 0.00625)

    def test_empty_workspace_says_so(self):
        with tempfile.TemporaryDirectory() as empty:
            d = _DOCK.RadargramDock(self.iface, workspace_provider=lambda: empty,
                                    parent=self.iface.mainWindow())
            self.assertIsNone(d._cur)
            self.assertEqual(d._cb_variant.count(), 0)
            self.assertIn("Nessuna variante", d._lbl_status.text())
            d.cleanup()
            d.deleteLater()

    # -- profilo, canale ---------------------------------------------------------

    def test_profile_and_channel_selection_load_once(self):
        d = self.dock
        loads = []
        orig = d._store.channel
        d._store.channel = lambda pi, ci: (loads.append((pi, ci)), orig(pi, ci))[1]
        d.select_profile(1)
        self.assertEqual(loads, [(1, 0)])                    # un solo caricamento, non a cascata
        self.assertEqual((d._pi, d._ci), (1, 0))
        self.assertEqual(d._cb_profile.currentIndex(), 1)
        d.select_profile(1, 1)
        self.assertEqual(d._cb_channel.currentIndex(), 1)
        self.assertEqual(loads[-1], (1, 1))
        d.select_profile(1, 1)                               # nessun cambio: nessun ricaricamento
        self.assertEqual(len(loads), 2)
        self.assertIn("Ch 1", d._lbl_status.text())

    def test_navigation_buttons_step_and_stop_at_the_ends(self):
        d = self.dock
        self.assertFalse(d._btn_prev_p.isEnabled())
        self.assertFalse(d._btn_prev_c.isEnabled())
        d._btn_next_p.click()
        d._btn_next_c.click()
        self.assertEqual((d._pi, d._ci), (1, 1))
        self.assertFalse(d._btn_next_p.isEnabled())
        self.assertFalse(d._btn_next_c.isEnabled())
        d.select_profile(99, 99)                              # fuori range: si ferma all'ultimo
        self.assertEqual((d._pi, d._ci), (1, 1))

    def test_combos_change_selection(self):
        d = self.dock
        d._cb_profile.setCurrentIndex(1)
        self.assertEqual(d._pi, 1)
        d._cb_channel.setCurrentIndex(1)
        self.assertEqual((d._pi, d._ci), (1, 1))

    def test_keys_change_profile_channel_and_depth(self):
        d = self.dock
        d._on_key(types.SimpleNamespace(key="pagedown"))
        self.assertEqual(d._pi, 1)
        d._on_key(types.SimpleNamespace(key="pageup"))
        self.assertEqual(d._pi, 0)
        d._on_key(types.SimpleNamespace(key="right"))
        self.assertEqual(d._ci, 1)
        d.set_reference_depth(0.10)
        d._on_key(types.SimpleNamespace(key="down"))
        self.assertAlmostEqual(d._ref_z, 0.15)
        self.assertAlmostEqual(d._spin_depth.value(), 0.15)
        d._on_key(types.SimpleNamespace(key="up"))
        self.assertAlmostEqual(d._ref_z, 0.10)

    def test_reopening_reloads_a_variant_rewritten_on_disk_and_keeps_the_selection(self):
        d = self.dock
        d.select_profile(1, 1)
        old_store = d._store
        target = os.path.join(self.v_lavoro, "Swath10.npz")
        later = os.path.getmtime(target) + 5
        os.utime(target, (later, later))                      # il viewer ha riscritto "Lavoro"
        d.refresh_variants()                                  # come riaprendo il pannello dalla toolbar
        self.assertIsNot(d._store, old_store)                 # ricaricata
        self.assertEqual((d._pi, d._ci), (1, 1))              # stesso profilo (per nome) e canale
        stable = d._store
        d.refresh_variants()
        self.assertIs(d._store, stable)                       # invariata: nessun ricaricamento inutile

    def test_changing_variant_keeps_profile_and_depth(self):
        d = self.dock
        d.select_profile(1, 1)
        d.set_reference_depth(0.12)
        d._cb_variant.setCurrentIndex(d._cb_variant.findText("v3", Qt.MatchStartsWith))
        self.assertEqual(d._store.name, "v3")
        self.assertEqual((d._pi, d._ci), (1, 1))             # stesso profilo (per nome) e stesso canale
        self.assertAlmostEqual(d._ref_z, 0.12)

    # -- profondita' ----------------------------------------------------------------

    def test_depth_spin_draws_reference_line(self):
        d = self.dock
        self.assertFalse(d._ref_line.get_visible())
        d._spin_depth.setValue(0.10)
        self.assertTrue(d._ref_line.get_visible())
        self.assertEqual(list(d._ref_line.get_ydata()), [0.10, 0.10])
        self.assertIn("riferimento 0.100 m", d._lbl_status.text())
        self.assertAlmostEqual(d._spin_depth.maximum(), d._cur.depth_max_m)

    def test_click_on_radargram_sets_reference_depth(self):
        d = self.dock
        d._on_mouse_press(self._event("button_press_event", 1.5, 0.15))
        self.assertAlmostEqual(d._ref_z, 0.15, places=2)
        self.assertAlmostEqual(d._spin_depth.value(), 0.15, places=2)

    def test_reference_depth_survives_profile_change(self):
        d = self.dock
        d.set_reference_depth(0.11)
        d.select_profile(1)
        self.assertAlmostEqual(d._ref_z, 0.11)
        self.assertTrue(d._ref_line.get_visible())            # riga ridisegnata sul nuovo profilo

    def test_mouse_move_reads_depth_time_amplitude_and_position(self):
        d = self.dock
        d._on_mouse_move(self._event("motion_notify_event", 1.0, 0.103))   # meta' della riga 16
        self.assertTrue(d._vline.get_visible() and d._hline.get_visible())
        txt = d._lbl_status.text()
        self.assertIn("traccia 20", txt)                      # x = 1.0 m con passo 5 cm
        self.assertIn("z 0.103 m", txt)
        self.assertIn("2.1 ns", txt)                          # riga 16 -> (16+0.5)*0.125 = 2.06 ns
        r = d._cursor_readout()
        self.assertEqual((r["trace"], r["row"]), (20, 16))
        self.assertAlmostEqual(r["amp"], float(d._cur.data[16, 20]))
        self.assertAlmostEqual(r["time_ns"], 16.5 * 0.125)
        self.assertAlmostEqual(r["pos"][0], 500001.0)
        self.assertAlmostEqual(r["pos"][1], 10.0)

    # -- mappa ---------------------------------------------------------------------------

    def test_track_and_cursor_point_are_drawn_on_the_map(self):
        d = self.dock
        rb = d._map._rb_line
        self.assertIsNotNone(rb)
        self.assertGreaterEqual(rb.numberOfVertices(), 2)
        d._on_mouse_move(self._event("motion_notify_event", 1.0, 0.10))
        d._flush_map_point()                                  # normalmente dopo 50 ms
        self.assertEqual(d._map._rb_pt.numberOfVertices(), 1)

    def test_go_to_world_point_selects_nearest_profile_channel_trace(self):
        d = self.dock
        # Swath10 (indice 1), canale 1 (+4 cm), traccia 30
        self.assertTrue(d.go_to_world_point(500001.5, 0.04 + 0.01))
        self.assertEqual((d._pi, d._ci), (1, 1))
        self.assertAlmostEqual(d._vline.get_xdata()[0], d._cur.dist[30])
        self.assertTrue(d._vline.get_visible())
        self.assertIn("dal punto", d._lbl_status.text())
        self.assertEqual(d._map._rb_pt.numberOfVertices(), 1)

    def test_go_to_world_point_far_away_changes_nothing(self):
        d = self.dock
        self.assertFalse(d.go_to_world_point(500001.5, 500.0))
        self.assertEqual((d._pi, d._ci), (0, 0))
        self.assertIn("Nessun profilo entro", d._lbl_status.text())

    def test_go_to_world_point_scrolls_a_zoomed_view_to_the_trace(self):
        d = self.dock
        d._set_view(0.0, 1.0, d._cur.depth_max_m, 0.0)        # vista sui primi 1.0 m
        d.go_to_world_point(500000.0 + 90 * 0.05, 10.0)       # traccia 90 = 4.5 m
        lo, hi = sorted(d._ax.get_xlim())
        self.assertTrue(lo <= 4.5 <= hi)
        self.assertAlmostEqual(hi - lo, 1.0, places=6)        # la scala non cambia

    def test_map_click_tool_picks_and_ignores_other_buttons(self):
        d = self.dock
        d._btn_pick.setChecked(True)
        self.assertIs(self.iface.mapCanvas().mapTool(), d._map._tool)
        d._map._on_canvas_clicked(QgsPointXY(500001.5, 0.0), Qt.RightButton)
        self.assertEqual(d._pi, 0)                             # tasto destro: ignorato
        d._map._on_canvas_clicked(QgsPointXY(500001.5, 0.0), Qt.LeftButton)
        self.assertEqual((d._pi, d._ci), (1, 0))               # Swath10 (nord 0)
        # scegliendo un altro strumento il pulsante si spegne da solo
        from qgis.gui import QgsMapToolPan
        self.iface.mapCanvas().setMapTool(QgsMapToolPan(self.iface.mapCanvas()))
        self.assertFalse(d._btn_pick.isChecked())

    def test_map_items_follow_dock_visibility_and_checkbox(self):
        d = self.dock
        self.assertTrue(d._map._rb_line.isVisible())
        d._chk_map.setChecked(False)
        self.assertFalse(d._map._rb_line.isVisible())
        d._chk_map.setChecked(True)
        self.assertTrue(d._map._rb_line.isVisible())
        d.hide()
        self._pump(30)
        self.assertFalse(d._map._rb_line.isVisible())          # dock chiuso: niente tracce orfane
        d.show()
        self._pump(30)
        self.assertTrue(d._map._rb_line.isVisible())

    def test_cleanup_removes_map_items_and_disconnects(self):
        d = self.dock
        d.cleanup()
        self.assertIsNone(d._map._rb_line)
        self.assertIsNone(d._map._rb_pt)
        layer = self._layer(_write_slice(self.root, "v3", "s0.tif", 0.1, 0.2))
        self.iface.currentLayerChanged.emit(layer)             # dopo cleanup non deve piu' reagire
        self.assertEqual(d._store.name, "Lavoro")

    # -- time-slice ----------------------------------------------------------------------

    def test_selecting_a_timeslice_shows_its_depth_window_and_variant(self):
        d = self.dock
        layer = self._layer(_write_slice(self.root, "v3", "s0.tif", 0.10, 0.20))
        self.iface._active = layer
        self.iface.currentLayerChanged.emit(layer)
        self.assertEqual(d._store.name, "v3")                  # la variante da cui deriva la slice
        self.assertEqual(d._band, (0.10, 0.20))
        self.assertAlmostEqual(d._ref_z, 0.15)
        self.assertAlmostEqual(d._spin_depth.value(), 0.15)
        self.assertTrue(d._band_patch.get_visible())
        self.assertIn("time-slice 0.10", d._lbl_status.text())
        xy = np.asarray(d._band_patch.get_xy())
        self.assertEqual(sorted(set(np.round(xy[:, 1], 6))), [0.10, 0.20])

    def test_manual_depth_drops_the_timeslice_band(self):
        d = self.dock
        d.set_reference_depth(0.15, band=(0.10, 0.20))
        d._spin_depth.setValue(0.30)
        self.assertIsNone(d._band)
        self.assertFalse(d._band_patch.get_visible())

    def test_follow_can_be_switched_off(self):
        d = self.dock
        d._chk_follow.setChecked(False)
        layer = self._layer(_write_slice(self.root, "v3", "s0.tif", 0.10, 0.20))
        self.iface.currentLayerChanged.emit(layer)
        self.assertIsNone(d._band)
        self.assertEqual(d._store.name, "Lavoro")
        # riaccendendo, si allinea subito al livello attivo
        self.iface._active = layer
        d._chk_follow.setChecked(True)
        self.assertEqual(d._band, (0.10, 0.20))

    def test_foreign_raster_is_ignored(self):
        d = self.dock
        d.set_reference_depth(0.05)
        p = os.path.join(self.root, "esterno.tif")
        _RIO.write_tif_singleband(np.ones((3, 3), dtype=np.float32), p, 0.0, 0.0, 2.0, 1.0, 32632)
        self.iface.currentLayerChanged.emit(self._layer(p))
        self.assertAlmostEqual(d._ref_z, 0.05)
        self.assertIsNone(d._band)

    # -- geometria ---------------------------------------------------------------------------

    def test_geometry_comes_from_open_viewer_when_available_and_says_so(self):
        d = self.dock
        self.assertIn("coordinate: salvate nel .npz", d._lbl_status.text())
        moved = _channel(100, 700000.0 + np.arange(100) * 0.05, np.full(100, 3.0))
        d._geometry_provider = lambda stem: _profile(stem, [moved, moved])
        d._store.geometry_provider = d._geometry_provider
        d.refresh()                                             # "Aggiorna": rilegge le coordinate
        self.assertEqual(d._cur.geometry_source, "viewer")
        self.assertNotIn("salvate nel .npz", d._lbl_status.text())
        self.assertAlmostEqual(float(d._cur.east[0]), 700000.0)

    def test_scroll_zoom_stays_inside_the_data_and_double_click_resets(self):
        d = self.dock
        full = tuple(d._ax.get_xlim())
        d._on_scroll(self._event("scroll_event", 2.0, 0.1, step=1))
        z1 = d._ax.get_xlim()
        self.assertLess(z1[1] - z1[0], full[1] - full[0])
        for _ in range(60):                                   # zoom-out ben oltre il dato, ancorato al centro
            lo, hi = d._ax.get_xlim()
            d._on_scroll(self._event("scroll_event", 0.5 * (lo + hi), 0.1, step=-1))
        self.assertEqual(tuple(d._ax.get_xlim()), full)
        d._on_scroll(self._event("scroll_event", 2.0, 0.1, step=1))
        d._on_mouse_press(self._event("button_press_event", 2.0, 0.1, dblclick=True))
        self.assertEqual(tuple(d._ax.get_xlim()), full)

    def test_gain_and_colormap_redraw_without_losing_the_reference(self):
        d = self.dock
        d.set_reference_depth(0.10)
        d._spin_gain.setValue(4.0)
        self.assertTrue(d._ref_line.get_visible())
        self.assertEqual(list(d._ref_line.get_ydata()), [0.10, 0.10])
        disp = np.asarray(d._im.get_array())
        self.assertLessEqual(float(np.abs(disp).max()), 1.0)    # saturato a +-1
        i = d._cb_cmap.findText("gray")
        d._cb_cmap.setCurrentIndex(i)
        self.assertEqual(d._im.get_cmap().name, "gray")


@unittest.skipUnless(HAS_ENV, "QGIS/matplotlib non disponibili")
class FlowLayoutTest(_DockBase):
    """I controlli vanno a capo: il pannello si aggancia anche a un lato, stretto."""

    def _flow(self, n=5, w=100, h=20):
        from qgis.PyQt.QtWidgets import QWidget
        box = QWidget()
        flow = _DOCK._FlowLayout(box)
        kids = []
        for _ in range(n):
            k = QWidget()
            k.setFixedSize(w, h)
            flow.addWidget(k)
            kids.append(k)
        return box, flow, kids

    def test_height_grows_as_width_shrinks_and_min_width_is_the_widest_item(self):
        box, flow, _kids = self._flow()
        self.assertEqual(flow.heightForWidth(1000), 20)                   # tutti su una riga
        self.assertEqual(flow.heightForWidth(250), 3 * 20 + 2 * 3)        # 2 per riga -> 3 righe
        self.assertEqual(flow.heightForWidth(100), 5 * 20 + 4 * 3)        # uno per riga
        self.assertEqual(flow.minimumSize().width(), 100)                 # si puo' stringere fino a un elemento

    def test_items_are_placed_on_wrapped_lines_without_overlap(self):
        box, flow, kids = self._flow()
        box.resize(250, flow.heightForWidth(250))
        box.show()
        self._pump(30)
        pos = [(k.x(), k.y()) for k in kids]
        self.assertEqual(pos[:3], [(0, 0), (108, 0), (0, 23)])
        self.assertEqual(pos[3], (108, 23))
        self.assertEqual(pos[4], (0, 46))
        box.close()

    def test_dock_can_be_narrow_and_controls_stay_inside(self):
        d = self.dock
        self.assertLess(d.minimumSizeHint().width(), 1000)                # non impone una larghezza enorme
        ctrl = d.widget().layout().itemAt(0).widget()
        h_wide = ctrl.layout().heightForWidth(1400)
        h_narrow = ctrl.layout().heightForWidth(400)
        self.assertGreater(h_narrow, h_wide)                              # stretto: piu' righe
        d.setFloating(True)
        d.resize(420, 500)
        self._pump(80)
        cw = ctrl.width()
        for g in ctrl.findChildren(type(ctrl), options=Qt.FindDirectChildrenOnly):
            self.assertLessEqual(g.x() + g.width(), cw + 1, "gruppo di controlli fuori dal pannello")

    def test_axis_labels_keep_a_fixed_margin_when_narrow(self):
        d = self.dock
        d.setFloating(True)
        d.resize(380, 400)
        self._pump(100)
        d._canvas.draw()
        fig = d._fig
        left_px = d._ax.get_position().x0 * fig.bbox.width
        self.assertGreater(left_px, 25)                                   # spazio per "Profondita' (m)" e i numeri
        self.assertAlmostEqual(left_px, 38 * fig.dpi / 72.0, delta=2.0)   # non e' piu' una frazione della larghezza

    def test_status_tooltip_carries_the_full_text(self):
        d = self.dock
        self.assertEqual(d._lbl_status.toolTip(), d._lbl_status.text())
        d.set_reference_depth(0.2, band=(0.1, 0.3))
        self.assertIn("time-slice 0.10", d._lbl_status.toolTip())


class PluginWiringTest(unittest.TestCase):
    """open_radargram_dock del plugin: crea una volta sola, riusa, apre la variante richiesta."""

    @unittest.skipUnless(HAS_ENV, "QGIS/matplotlib non disponibili")
    def test_open_radargram_dock_creates_once_and_selects_variant(self):
        if True:
            GS = importlib.import_module("rasterlinker_feature_interpolation.geosurvey_studio")
            if not GS._HAS_RADARGRAM_DOCK:
                self.skipTest("dock non importabile")
            with tempfile.TemporaryDirectory() as root:
                _write_variant(root, "Lavoro", ["Swath1"])
                _write_variant(root, "v3", ["Swath1", "Swath2"])
                iface = _Iface()
                fake = types.SimpleNamespace(
                    iface=iface, _radargram_dock=None, _gpr_profile_viewer=None,
                    settings=types.SimpleNamespace(value=lambda *a, **k: root),
                    settings_key_active_project="k")
                fake._radargram_workspace_root = lambda: GS.GeoSurveyStudioPlugin._radargram_workspace_root(fake)
                fake._live_profile_viewer = lambda: GS.GeoSurveyStudioPlugin._live_profile_viewer(fake)
                fake._radargram_geometry_provider = lambda s: GS.GeoSurveyStudioPlugin._radargram_geometry_provider(fake, s)
                open_ = lambda **kw: GS.GeoSurveyStudioPlugin.open_radargram_dock(fake, **kw)
                dock = open_(activate=False)
                self.assertIsNotNone(dock)
                self.assertEqual(len(iface.docks), 1)
                self.assertEqual(iface.docks[0][0], Qt.BottomDockWidgetArea)
                self.assertEqual(dock._store.name, "Lavoro")
                self.assertIs(open_(variant="v3", profile_stem="Swath2"), dock)   # riusato
                self.assertEqual(len(iface.docks), 1)
                self.assertEqual(dock._store.name, "v3")
                self.assertEqual(dock._store.profiles[dock._pi].stem, "Swath2")
                dock.cleanup()
                dock.deleteLater()


if __name__ == "__main__":
    unittest.main()
