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
    _PC = importlib.import_module("rasterlinker_feature_interpolation.project_catalog")
    from qgis.PyQt.QtCore import QObject, Qt, pyqtSignal
    from qgis.PyQt.QtWidgets import QMainWindow
    from qgis.core import QgsCoordinateReferenceSystem, QgsPointXY, QgsProject, QgsRasterLayer, QgsRectangle
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

    def setActiveLayer(self, layer):
        self._active = layer

    def addDockWidget(self, area, dock):
        self.docks.append((area, dock))
        self._win.addDockWidget(area, dock)

    def removeDockWidget(self, dock):
        self._win.removeDockWidget(dock)


def _write_slice(root, variant, grid, z0, z1):
    """GeoTIFF minuscolo + sliceset.json accanto, come li scrive la generazione time-slice."""
    folder = os.path.join(root, "Processing", "timeslices", variant)
    os.makedirs(folder, exist_ok=True)
    _RIO.write_tif_singleband(np.ones((4, 4), dtype=np.float32), os.path.join(folder, grid),
                              500000.0, 0.0, 3.0, 1.0, 32632)
    _SM.write_sliceset(
        folder, _SS.SliceSpec(),
        [{"index": 0, "z_from": z0, "z_to": z1, "grid": grid}],
        {"kind": "ogpr", "variant_id": variant, "n_profiles": 2}, crs_epsg=32632)
    return os.path.join(folder, grid)


def _write_sliceset(root, variant, slices):
    """Piu' time-slice nello stesso gruppo/sliceset.json: slices = [(grid, z0, z1), ...]."""
    folder = os.path.join(root, "Processing", "timeslices", variant)
    os.makedirs(folder, exist_ok=True)
    for grid, _z0, _z1 in slices:
        _RIO.write_tif_singleband(np.ones((4, 4), dtype=np.float32), os.path.join(folder, grid),
                                  500000.0, 0.0, 3.0, 1.0, 32632)
    _SM.write_sliceset(
        folder, _SS.SliceSpec(),
        [{"index": i, "z_from": z0, "z_to": z1, "grid": grid} for i, (grid, z0, z1) in enumerate(slices)],
        {"kind": "ogpr", "variant_id": variant, "n_profiles": 2}, crs_epsg=32632)
    return [os.path.join(folder, grid) for grid, _z0, _z1 in slices]


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
        self._project_layers = []
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
        if self._project_layers:
            QgsProject.instance().removeMapLayers([lyr.id() for lyr in self._project_layers])
            self._project_layers.clear()
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

    def _project_layer(self, path):
        """Come `_layer`, ma registrato in `QgsProject` (serve a `_timeslice_layers_in_project`,
        che legge `QgsProject.instance().mapLayers()`, non solo un `layer` isolato)."""
        lyr = self._layer(path)
        QgsProject.instance().addMapLayer(lyr)
        self._project_layers.append(lyr)
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

    def test_go_to_world_point_uses_the_depth_of_the_visible_timeslice(self):
        d = self.dock
        paths = _write_sliceset(self.root, "v3", [("s0.tif", 0.00, 0.10), ("s1.tif", 0.10, 0.20)])
        layers = [self._project_layer(p) for p in paths]
        d.open_variant("v3")
        root = QgsProject.instance().layerTreeRoot()
        root.findLayer(layers[0].id()).setItemVisibilityChecked(False)
        root.findLayer(layers[1].id()).setItemVisibilityChecked(True)
        d.set_reference_depth(0.05)                          # vecchio riferimento: non deve sopravvivere
        # Swath10 (indice 1), canale 1 (+4 cm), traccia 30 - stessa geometria di "Lavoro"
        self.assertTrue(d.go_to_world_point(500001.5, 0.04 + 0.01))
        self.assertEqual(d._band, (0.10, 0.20))               # fascia della slice visibile, non 0.05
        self.assertAlmostEqual(d._ref_z, 0.15)
        self.assertAlmostEqual(d._hline.get_ydata()[0], 0.15)

    def test_go_to_world_point_keeps_old_reference_when_no_timeslice_is_visible(self):
        d = self.dock
        d.set_reference_depth(0.07)
        self.assertTrue(d.go_to_world_point(500001.5, 0.04 + 0.01))
        self.assertAlmostEqual(d._ref_z, 0.07)                # nessuna slice caricata: resta il riferimento

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

    def test_scroll_zoom_stays_inside_the_data_and_button_resets(self):
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
        d._reset_zoom()                                       # doppio clic ora attiva la time-slice, non piu' questo
        self.assertEqual(tuple(d._ax.get_xlim()), full)

    # -- doppio clic -> attiva la time-slice alla profondita' cliccata ------------------

    def test_double_click_activates_the_matching_timeslice_and_hides_the_others(self):
        d = self.dock
        paths = _write_sliceset(self.root, "v3", [
            ("s0.tif", 0.00, 0.10), ("s1.tif", 0.10, 0.20), ("s2.tif", 0.20, 0.30),
        ])
        layers = [self._project_layer(p) for p in paths]
        d.open_variant("v3")
        root = QgsProject.instance().layerTreeRoot()
        for lyr in layers:
            root.findLayer(lyr.id()).setItemVisibilityChecked(False)
        d._on_mouse_press(self._event("button_press_event", 1.0, 0.15, dblclick=True))
        self.assertFalse(root.findLayer(layers[0].id()).itemVisibilityChecked())
        self.assertTrue(root.findLayer(layers[1].id()).itemVisibilityChecked())    # 0.10-0.20 contiene 0.15
        self.assertFalse(root.findLayer(layers[2].id()).itemVisibilityChecked())
        self.assertIs(self.iface.activeLayer(), layers[1])
        self.assertEqual(d._band, (0.10, 0.20))
        self.assertAlmostEqual(d._ref_z, 0.15)

    def test_double_click_picks_the_nearest_band_when_depth_is_outside_all_slices(self):
        d = self.dock
        paths = _write_sliceset(self.root, "v3", [("s0.tif", 0.00, 0.10), ("s1.tif", 0.10, 0.20)])
        layers = [self._project_layer(p) for p in paths]
        d.open_variant("v3")
        root = QgsProject.instance().layerTreeRoot()
        d._activate_timeslice_at_depth(50.0)      # oltre ogni fascia: vince la piu' vicina per centro
        self.assertTrue(root.findLayer(layers[1].id()).itemVisibilityChecked())
        self.assertFalse(root.findLayer(layers[0].id()).itemVisibilityChecked())

    def test_double_click_without_loaded_timeslices_shows_a_message_and_does_not_crash(self):
        d = self.dock
        d._on_mouse_press(self._event("button_press_event", 1.0, 0.15, dblclick=True))
        self.assertIn("Nessuna time-slice", d._lbl_status.text())

    def test_double_click_works_even_when_the_timeslice_folder_is_unrelated_to_the_variant(self):
        # Dati reali: le time-slice possono vivere in un albero di cartelle scollegato
        # da "processed/<variante>" (es. una cartella di export separata) - il doppio
        # clic deve funzionare comunque, senza alcuna parentela di cartelle.
        d = self.dock
        unrelated = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(unrelated.cleanup)
        paths = _write_sliceset(unrelated.name, "not_a_real_variant", [
            ("s0.tif", 0.00, 0.10), ("s1.tif", 0.10, 0.20),
        ])
        layers = [self._project_layer(p) for p in paths]
        d._on_mouse_press(self._event("button_press_event", 1.0, 0.15, dblclick=True))
        root = QgsProject.instance().layerTreeRoot()
        self.assertTrue(root.findLayer(layers[1].id()).itemVisibilityChecked())
        self.assertEqual(d._band, (0.10, 0.20))

    def test_double_click_activates_a_catalog_only_timeslice_without_sliceset_json(self):
        # Alcune time-slice non hanno sliceset.json accanto ma sono registrate nel
        # catalogo di progetto (project_catalog.json) con la loro fascia di profondita'.
        d = self.dock
        path = os.path.join(self.root, "catalog_only.tif")
        _RIO.write_tif_singleband(np.ones((4, 4), dtype=np.float32), path, 500000.0, 0.0, 3.0, 1.0, 32632)
        _PC.register_timeslice(self.root, {"project_path": path, "depth_from": 0.10, "depth_to": 0.20})
        layer = self._project_layer(path)
        d._on_mouse_press(self._event("button_press_event", 1.0, 0.15, dblclick=True))
        root = QgsProject.instance().layerTreeRoot()
        self.assertTrue(root.findLayer(layer.id()).itemVisibilityChecked())
        self.assertEqual(d._band, (0.10, 0.20))

    def _slice_at(self, tag, x_min, y_min, y_max, z0=0.10, z1=0.20, variant="v3"):
        """Time-slice (con sliceset.json) in una cartella propria, in un punto preciso della mappa."""
        folder = os.path.join(self.root, "slices_" + tag)
        os.makedirs(folder, exist_ok=True)
        grid = tag + ".tif"
        _RIO.write_tif_singleband(np.ones((4, 4), dtype=np.float32), os.path.join(folder, grid),
                                  x_min, y_min, y_max, 1.0, 32632)
        _SM.write_sliceset(
            folder, _SS.SliceSpec(), [{"index": 0, "z_from": z0, "z_to": z1, "grid": grid}],
            {"kind": "ogpr", "variant_id": variant, "n_profiles": 2}, crs_epsg=32632)
        return os.path.join(folder, grid)

    def test_double_click_prefers_the_timeslice_under_the_clicked_point(self):
        # Piu' gruppi di time-slice (aree/varianti) con la stessa fascia: vince quella che
        # copre il punto cliccato, non la prima dell'elenco.
        d = self.dock
        far = self._project_layer(self._slice_at("a_far", 600000.0, 0.0, 4.0))
        near = self._project_layer(self._slice_at("b_near", 500000.0, 8.0, 12.0))
        root = QgsProject.instance().layerTreeRoot()
        root.findLayer(far.id()).setItemVisibilityChecked(False)
        root.findLayer(near.id()).setItemVisibilityChecked(False)
        d._on_mouse_press(self._event("button_press_event", 1.0, 0.15, dblclick=True))   # profilo 0: N = 10
        self.assertTrue(root.findLayer(near.id()).itemVisibilityChecked())
        self.assertIs(self.iface.activeLayer(), near)

    def test_double_click_moves_the_map_to_the_clicked_point(self):
        d = self.dock
        self._project_layer(self._slice_at("s", 500000.0, 8.0, 12.0))
        self.iface.mapCanvas().setExtent(QgsRectangle(-10.0, -10.0, 10.0, 10.0))     # lontano dal profilo
        d._on_mouse_press(self._event("button_press_event", 1.0, 0.15, dblclick=True))
        c = self.iface.mapCanvas().center()
        e, n = d._cur.position(d._cur.trace_at_distance(1.0))
        self.assertAlmostEqual(c.x(), e, delta=0.5)
        self.assertAlmostEqual(c.y(), n, delta=0.5)

    def test_double_click_turns_on_a_hidden_group(self):
        d = self.dock
        lyr = self._layer(self._slice_at("g", 500000.0, 8.0, 12.0))
        QgsProject.instance().addMapLayer(lyr, False)
        self._project_layers.append(lyr)
        root = QgsProject.instance().layerTreeRoot()
        grp = root.addGroup("gruppo time-slice")
        self.addCleanup(lambda: root.removeChildNode(grp))
        grp.addLayer(lyr)
        grp.setItemVisibilityChecked(False)
        d._on_mouse_press(self._event("button_press_event", 1.0, 0.15, dblclick=True))
        self.assertTrue(grp.itemVisibilityChecked())
        self.assertTrue(root.findLayer(lyr.id()).itemVisibilityChecked())

    def test_catalog_is_read_from_the_project_root_not_only_the_workspace(self):
        # Il catalogo sta nella radice del progetto; la cartella di lavoro del pannello e' Processing/.
        d = self.dock
        project = os.path.join(self.root, "progetto")
        os.makedirs(os.path.join(project, "Processing"))
        path = os.path.join(self.root, "solo_catalogo.tif")
        _RIO.write_tif_singleband(np.ones((4, 4), dtype=np.float32), path, 500000.0, 0.0, 3.0, 1.0, 32632)
        _PC.register_timeslice(project, {"project_path": path, "depth_from": 0.10, "depth_to": 0.20})
        d._workspace_provider = lambda: os.path.join(project, "Processing")
        d._project_root_provider = lambda: project
        layer = self._project_layer(path)
        d._on_mouse_press(self._event("button_press_event", 1.0, 0.15, dblclick=True))
        self.assertIs(self.iface.activeLayer(), layer)
        self.assertEqual(d._band, (0.10, 0.20))

    def test_layer_tree_visibility_change_updates_the_reference_line_live(self):
        # Simula lo slider di navigazione del dialogo principale (catalog_group_mixin
        # .update_visibility_with_dial): accende/spegne i raster con
        # setItemVisibilityChecked, senza passare da questo pannello.
        d = self.dock
        paths = _write_sliceset(self.root, "v3", [("s0.tif", 0.00, 0.10), ("s1.tif", 0.10, 0.20)])
        layers = [self._project_layer(p) for p in paths]
        root = QgsProject.instance().layerTreeRoot()
        root.findLayer(layers[0].id()).setItemVisibilityChecked(False)
        root.findLayer(layers[1].id()).setItemVisibilityChecked(False)
        self._pump(30)
        root.findLayer(layers[1].id()).setItemVisibilityChecked(True)
        self._pump(30)
        self.assertEqual(d._band, (0.10, 0.20))
        self.assertAlmostEqual(d._ref_z, 0.15)

    def test_layer_tree_visibility_change_is_ignored_when_follow_is_off(self):
        d = self.dock
        d._chk_follow.setChecked(False)
        paths = _write_sliceset(self.root, "v3", [("s0.tif", 0.00, 0.10)])
        layers = [self._project_layer(p) for p in paths]
        root = QgsProject.instance().layerTreeRoot()
        root.findLayer(layers[0].id()).setItemVisibilityChecked(True)
        self._pump(30)
        self.assertIsNone(d._band)

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

    # -- disegno sul radargramma -----------------------------------------------------

    def test_draw_line_sends_vertices_with_real_geometry_to_feature_sink(self):
        d = self.dock
        captured = []
        d._feature_sink = lambda **kw: captured.append(kw)
        d._btn_draw_line.setChecked(True)
        self.assertEqual(d._draw_mode, "line")
        d._on_mouse_press(self._event("button_press_event", 1.0, 0.05))
        d._on_mouse_press(self._event("button_press_event", 2.0, 0.10))
        self.assertEqual(len(d._draw_vertices), 2)
        self.assertTrue(d._draw_preview_line.get_visible())
        d._finalize_drawn_feature()
        self.assertEqual(len(captured), 1)
        call = captured[0]
        self.assertEqual(call["kind"], "line")
        self.assertEqual(call["variant_name"], d._store.name)
        self.assertEqual(call["profile_stem"], d._store.profiles[d._pi].stem)
        self.assertEqual(call["channel"], d._ci)
        self.assertEqual(call["epsg"], d._store.profiles[d._pi].epsg)
        self.assertEqual(len(call["vertices"]), 2)
        ch = d._cur
        exp0 = ch.position(ch.trace_at_distance(1.0))
        exp1 = ch.position(ch.trace_at_distance(2.0))
        self.assertAlmostEqual(call["vertices"][0]["east"], exp0[0], places=2)
        self.assertAlmostEqual(call["vertices"][0]["north"], exp0[1], places=2)
        self.assertAlmostEqual(call["vertices"][0]["depth_m"], 0.05, places=2)
        self.assertAlmostEqual(call["vertices"][1]["east"], exp1[0], places=2)
        self.assertAlmostEqual(call["vertices"][1]["depth_m"], 0.10, places=2)
        # dopo il salvataggio la modalita' resta attiva: pronta per la prossima geometria
        self.assertEqual(d._draw_mode, "line")
        self.assertEqual(d._draw_vertices, [])
        self.assertFalse(d._draw_preview_line.get_visible())

    def test_draw_point_finalizes_on_single_click(self):
        d = self.dock
        captured = []
        d._feature_sink = lambda **kw: captured.append(kw)
        d._btn_draw_point.setChecked(True)
        d._on_mouse_press(self._event("button_press_event", 1.5, 0.08))
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["kind"], "point")
        self.assertEqual(len(captured[0]["vertices"]), 1)
        self.assertEqual(d._draw_vertices, [])

    def test_draw_undo_and_cancel(self):
        d = self.dock
        d._btn_draw_line.setChecked(True)
        d._on_mouse_press(self._event("button_press_event", 1.0, 0.05))
        d._on_mouse_press(self._event("button_press_event", 1.5, 0.06))
        self.assertEqual(len(d._draw_vertices), 2)
        d._on_key(types.SimpleNamespace(key="backspace"))
        self.assertEqual(len(d._draw_vertices), 1)
        d._on_key(types.SimpleNamespace(key="escape"))
        self.assertEqual(d._draw_vertices, [])
        self.assertIsNone(d._draw_mode)
        self.assertFalse(d._btn_draw_line.isChecked())

    def test_draw_needs_minimum_vertices_before_finalizing(self):
        d = self.dock
        captured = []
        d._feature_sink = lambda **kw: captured.append(kw)
        d._btn_draw_polygon.setChecked(True)
        d._on_mouse_press(self._event("button_press_event", 1.0, 0.05))
        d._on_mouse_press(self._event("button_press_event", 1.5, 0.06))
        d._finalize_drawn_feature()                          # solo 2 vertici: servono almeno 3
        self.assertEqual(captured, [])
        self.assertEqual(len(d._draw_vertices), 2)
        d._on_mouse_press(self._event("button_press_event", 2.0, 0.07))
        d._finalize_drawn_feature()
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["kind"], "polygon")

    def test_polygon_preview_fills_while_drawing_not_just_outline(self):
        d = self.dock
        d._btn_draw_line.setChecked(True)
        d._on_mouse_press(self._event("button_press_event", 1.0, 0.05))
        d._on_mouse_press(self._event("button_press_event", 1.5, 0.06))
        self.assertFalse(d._draw_preview_fill.get_visible())    # una linea non si riempie
        d._cancel_draw()

        d._btn_draw_polygon.setChecked(True)
        d._on_mouse_press(self._event("button_press_event", 1.0, 0.05))
        d._on_mouse_press(self._event("button_press_event", 1.5, 0.06))
        self.assertFalse(d._draw_preview_fill.get_visible())    # solo 2 vertici: non ancora un poligono
        d._on_mouse_move(self._event("motion_notify_event", 1.2, 0.09))
        self.assertTrue(d._draw_preview_fill.get_visible())     # 2 vertici + cursore = 3 punti
        xy = d._draw_preview_fill.get_xy()
        self.assertEqual(len(xy), 4)                            # closed=True: matplotlib chiude l'anello
        self.assertEqual(tuple(xy[0]), tuple(xy[-1]))
        d._on_mouse_press(self._event("button_press_event", 2.0, 0.07))   # 3° vertice reale
        d._finalize_drawn_feature()
        self.assertEqual(len(d._draw_vertices), 0)
        self.assertFalse(d._draw_preview_fill.get_visible())    # pronto per la prossima, ripulito

    def test_draw_and_velocity_modes_are_mutually_exclusive(self):
        d = self.dock
        d._btn_draw_line.setChecked(True)
        self.assertEqual(d._draw_mode, "line")
        d._btn_vel.setChecked(True)
        self.assertTrue(d._vel_mode_active)
        self.assertIsNone(d._draw_mode)
        self.assertFalse(d._btn_draw_line.isChecked())
        d._btn_hyper.setChecked(True)
        self.assertTrue(d._hyper_mode_active)
        self.assertFalse(d._btn_vel.isChecked())        # "Misura velocita'" e "Calcola con iperbole" esclusivi

    def test_switching_profile_cancels_pending_draw_and_hyper_apex(self):
        d = self.dock
        d._btn_draw_line.setChecked(True)
        d._on_mouse_press(self._event("button_press_event", 1.0, 0.05))
        self.assertEqual(len(d._draw_vertices), 1)
        d.select_profile(1)
        self.assertEqual(d._draw_vertices, [])
        d._btn_hyper.setChecked(True)
        d._on_mouse_press(self._event("button_press_event", 1.0, 0.05))
        self.assertIsNotNone(d._hyper_apex)
        d.select_profile(0)
        self.assertIsNone(d._hyper_apex)

    # -- velocita' preimpostata: popup informativo -------------------------------------

    def test_measure_velocity_popup_shows_the_default_preset_without_recomputing(self):
        # Finche' non si clicca un apice, il popup resta la lettura statica del
        # preimpostato di profilo (round 2). Vedi i test sotto per il comportamento
        # dal vivo dopo un clic (round 4).
        d = self.dock
        HM = importlib.import_module("rasterlinker_feature_interpolation.gpr_hyperbola_math")
        d._btn_vel.setChecked(True)
        d._on_mouse_move(self._event("motion_notify_event", 1.0, 0.10))
        self.assertTrue(d._vel_popup.get_visible())
        txt = d._vel_popup.get_text()
        self.assertIn(f"{HM.DEFAULT_VELOCITY_M_NS:.3f}", txt)
        self.assertIn("di base", txt)
        # spostando il cursore altrove il valore resta lo stesso (nessun ricalcolo)
        d._on_mouse_move(self._event("motion_notify_event", 2.0, 0.20))
        self.assertIn(f"{HM.DEFAULT_VELOCITY_M_NS:.3f}", d._vel_popup.get_text())
        d._on_axes_leave(None)
        self.assertFalse(d._vel_popup.get_visible())

    def test_measure_velocity_click_fixes_apex_and_then_updates_live_on_move(self):
        d = self.dock
        HM = importlib.import_module("rasterlinker_feature_interpolation.gpr_hyperbola_math")
        d._btn_vel.setChecked(True)
        d._on_mouse_press(self._event("button_press_event", 2.0, 0.05))
        self.assertAlmostEqual(d._vel_apex[0], 2.0, places=6)
        self.assertAlmostEqual(d._vel_apex[1], 0.05, places=6)
        # (x1=2.12, z1=0.13) scelto cosi' che il fit torni v~=v_ref (fisicamente
        # valido: sqrt(0.13^2-0.05^2) = 0.12 = dx, quindi v_fit = v_ref).
        d._on_mouse_move(self._event("motion_notify_event", 2.12, 0.13))
        txt = d._vel_popup.get_text()
        self.assertIn("dal vivo", txt)
        v_ref = d._reference_velocity_m_s()
        expected = HM.velocity_from_two_points(2.0, 0.05, 2.12, 0.13, v_ref)
        self.assertIsNotNone(expected)
        v_mns = HM.velocity_m_s_to_m_ns(expected[0])
        self.assertIn(f"{v_mns:.3f}", txt)
        self.assertTrue(d._vel_curve_line.get_visible())
        self.assertTrue(d._vel_apex_marker.get_visible())
        # muovendo altrove il valore cambia (a differenza del popup statico pre-clic)
        d._on_mouse_move(self._event("motion_notify_event", 3.0, 0.20))
        self.assertNotIn(f"{v_mns:.3f}", d._vel_popup.get_text())

    def test_measure_velocity_hints_when_cursor_is_above_the_apex(self):
        d = self.dock
        d._btn_vel.setChecked(True)
        d._on_mouse_press(self._event("button_press_event", 2.0, 0.10))
        d._on_mouse_move(self._event("motion_notify_event", 2.5, 0.02))   # sopra l'apice: non valido
        self.assertIn("n/d", d._vel_popup.get_text())
        self.assertFalse(d._vel_curve_line.get_visible())

    def test_measure_velocity_escape_clears_the_apex_and_reverts_to_static(self):
        d = self.dock
        HM = importlib.import_module("rasterlinker_feature_interpolation.gpr_hyperbola_math")
        d._btn_vel.setChecked(True)
        d._on_mouse_press(self._event("button_press_event", 2.0, 0.05))
        d._on_mouse_move(self._event("motion_notify_event", 2.5, 0.12))
        d._on_key(types.SimpleNamespace(key="escape"))
        self.assertIsNone(d._vel_apex)
        self.assertFalse(d._vel_curve_line.get_visible())
        d._on_mouse_move(self._event("motion_notify_event", 2.5, 0.12))
        self.assertIn(f"{HM.DEFAULT_VELOCITY_M_NS:.3f}", d._vel_popup.get_text())

    def test_measure_velocity_apex_does_not_survive_a_profile_change(self):
        d = self.dock
        d._btn_vel.setChecked(True)
        d._on_mouse_press(self._event("button_press_event", 2.0, 0.05))
        self.assertIsNotNone(d._vel_apex)
        d.select_profile(1)
        self.assertIsNone(d._vel_apex)

    def test_measure_velocity_apex_is_cleared_when_switching_to_hyperbola_tool(self):
        d = self.dock
        d._btn_vel.setChecked(True)
        d._on_mouse_press(self._event("button_press_event", 2.0, 0.05))
        self.assertIsNotNone(d._vel_apex)
        d._btn_hyper.setChecked(True)
        self.assertIsNone(d._vel_apex)
        self.assertFalse(d._btn_vel.isChecked())

    def test_editing_the_velocity_field_saves_override_and_updates_depth_axis(self):
        d = self.dock
        VS = importlib.import_module("rasterlinker_feature_interpolation.gpr_velocity_store")
        stem = d._store.profiles[d._pi].stem
        old_depth_max = d._cur.depth_max_m
        d._spin_vel_mns.setValue(0.15)
        got = VS.resolve_velocity_m_ns(d._store.path, stem)
        self.assertEqual(got["source"], "manual")
        self.assertAlmostEqual(got["velocity_m_ns"], 0.15, places=6)
        self.assertNotAlmostEqual(d._cur.depth_max_m, old_depth_max)   # asse ricalcolato subito
        d._btn_vel.setChecked(True)
        d._on_mouse_move(self._event("motion_notify_event", 1.0, 0.05))
        self.assertIn("0.150", d._vel_popup.get_text())
        self.assertIn("manuale", d._vel_popup.get_text())

    # -- fit iperbole a rotellina + blocco (round 3) -----------------------------------

    def test_hyperbola_click_sets_apex_and_initial_velocity_from_preset(self):
        d = self.dock
        d._btn_hyper.setChecked(True)
        d._on_mouse_press(self._event("button_press_event", 1.0, 0.05))
        self.assertIsNotNone(d._hyper_apex)
        self.assertAlmostEqual(d._hyper_v_mns, d._current_preset_velocity()["velocity_m_ns"], places=6)
        self.assertFalse(d._hyper_locked)
        self.assertTrue(d._vel_curve_line.get_visible())
        self.assertTrue(d._btn_hyper_save.isEnabled())
        d._btn_vel_clear.click()
        self.assertEqual(d._lbl_vel.text(), "v=n/d")
        self.assertFalse(d._vel_curve_line.get_visible())
        self.assertIsNone(d._hyper_apex)
        self.assertIsNone(d._hyper_v_mns)
        self.assertFalse(d._btn_hyper_save.isEnabled())

    def test_hyperbola_wheel_adjusts_velocity_no_longer_the_cursor(self):
        d = self.dock
        d._btn_hyper.setChecked(True)
        d._on_mouse_press(self._event("button_press_event", 1.0, 0.05))
        v0 = d._hyper_v_mns
        d._on_scroll(self._event("scroll_event", 1.0, 0.05, step=1))
        self.assertAlmostEqual(d._hyper_v_mns, v0 + _DOCK._HYPER_VEL_STEP_MNS, places=6)
        self.assertIn(f"{d._hyper_v_mns:.4f}", d._lbl_vel.text())
        # muovere il cursore (senza rotellina) non cambia piu' nulla, a differenza del round 2
        v1 = d._hyper_v_mns
        d._on_mouse_move(self._event("motion_notify_event", 1.8, 0.22))
        self.assertEqual(d._hyper_v_mns, v1)
        d._on_scroll(self._event("scroll_event", 1.0, 0.05, step=-1))
        self.assertAlmostEqual(d._hyper_v_mns, v0, places=6)

    def test_hyperbola_right_click_locks_and_blocks_wheel(self):
        d = self.dock
        d._btn_hyper.setChecked(True)
        d._on_mouse_press(self._event("button_press_event", 1.0, 0.05))
        v0 = d._hyper_v_mns
        d._on_mouse_press(self._event("button_press_event", 1.0, 0.05, button=3))
        self.assertTrue(d._hyper_locked)
        self.assertIn("bloccato", d._lbl_vel.text())
        d._on_scroll(self._event("scroll_event", 1.0, 0.05, step=1))
        self.assertEqual(d._hyper_v_mns, v0)          # bloccato: la rotellina non cambia piu' nulla
        self.assertIsNone(d._pan)                     # il tasto destro non deve avviare il pan
        d._on_mouse_press(self._event("button_press_event", 1.0, 0.05, button=3))
        self.assertFalse(d._hyper_locked)
        d._on_scroll(self._event("scroll_event", 1.0, 0.05, step=1))
        self.assertAlmostEqual(d._hyper_v_mns, v0 + _DOCK._HYPER_VEL_STEP_MNS, places=6)

    def test_save_hyperbola_fit_as_preset_persists_and_is_read_back(self):
        d = self.dock
        HM = importlib.import_module("rasterlinker_feature_interpolation.gpr_hyperbola_math")
        VS = importlib.import_module("rasterlinker_feature_interpolation.gpr_velocity_store")
        stem = d._store.profiles[d._pi].stem
        d._btn_hyper.setChecked(True)
        d._on_mouse_press(self._event("button_press_event", 1.0, 0.05))
        d._on_scroll(self._event("scroll_event", 1.0, 0.05, step=1))
        self.assertTrue(d._btn_hyper_save.isEnabled())
        expected_v_mns = d._hyper_v_mns
        expected_rdp = HM.velocity_to_rdp(HM.m_ns_to_velocity_m_s(expected_v_mns))
        d._btn_hyper_save.click()
        got = VS.resolve_velocity_m_ns(d._store.path, stem)
        self.assertEqual(got["source"], "hyperbola_fit")
        self.assertAlmostEqual(got["velocity_m_ns"], expected_v_mns, places=6)
        self.assertAlmostEqual(got["rdp"], expected_rdp, places=3)
        self.assertAlmostEqual(d._spin_vel_mns.value(), expected_v_mns, places=4)   # rispecchiato subito

    # -- niente scelta del materiale; comandi degli strumenti solo quando servono ----------

    def test_no_material_chooser(self):
        d = self.dock
        self.assertFalse(hasattr(d, "_cb_vel_material"))
        self.assertFalse(hasattr(d, "_on_material_preset_chosen"))

    def test_old_preset_source_is_still_readable(self):
        VS = importlib.import_module("rasterlinker_feature_interpolation.gpr_velocity_store")
        d = self.dock
        stem = d._store.profiles[d._pi].stem
        VS.write_velocity_override(d._store.path, profile_stem=stem, velocity_m_ns=0.033,
                                   rdp=81.0, source="preset:acqua_dolce")
        d._store.invalidate_geometry()
        d._sync_velocity_spin()
        self.assertAlmostEqual(d._spin_vel_mns.value(), 0.033, places=4)
        self.assertNotEqual(d._format_velocity_source("preset:acqua_dolce"), "preset:acqua_dolce")

    def test_tool_commands_are_hidden_until_the_tool_is_active(self):
        d = self.dock
        self.assertTrue(d._grp_draw_ctx.isHidden())
        self.assertTrue(d._grp_hyper_ctx.isHidden())
        d._btn_draw_line.setChecked(True)
        self.assertFalse(d._grp_draw_ctx.isHidden())
        d._btn_draw_line.setChecked(False)
        self.assertTrue(d._grp_draw_ctx.isHidden())
        d._btn_hyper.setChecked(True)
        self.assertFalse(d._grp_hyper_ctx.isHidden())
        d._btn_hyper.setChecked(False)
        self.assertTrue(d._grp_hyper_ctx.isHidden())

    def test_zone_chooser_is_hidden_with_a_single_area(self):
        self.assertTrue(self.dock._grp_zone.isHidden())

    # -- overlay persistente delle geometrie gia' disegnate -----------------------------

    def test_previously_drawn_line_reappears_as_overlay_via_feature_query(self):
        d = self.dock
        ch = d._cur
        info = d._store.profiles[d._pi]
        p0 = ch.position(ch.trace_at_distance(1.0))
        p1 = ch.position(ch.trace_at_distance(2.0))
        feature = {
            "kind": "line",
            "vertices": [
                {"east": p0[0], "north": p0[1], "depth_m": 0.05},
                {"east": p1[0], "north": p1[1], "depth_m": 0.09},
            ],
        }
        d._feature_query = lambda variant, stem, channel: (
            [feature] if (stem, channel) == (info.stem, d._ci) else [])
        d._refresh_persisted_features()
        self.assertEqual(len(d._persisted_artists), 1)
        line = d._persisted_artists[0]
        xs, zs = line.get_data()
        self.assertAlmostEqual(min(xs), 1.0, delta=0.06)
        self.assertAlmostEqual(max(xs), 2.0, delta=0.06)
        self.assertEqual(list(zs), [0.05, 0.09])
        # cambiando canale (query filtra su channel) l'overlay sparisce
        d._feature_query = lambda variant, stem, channel: []
        d._refresh_persisted_features()
        self.assertEqual(d._persisted_artists, [])


class _FakeAnnotationLayer:
    """Archivio in memoria con la stessa interfaccia dei ponti verso il layer vero
    (sink / query / update / delete): il layer reale e' coperto da test_radargram_capture."""

    EDITABLE = ("class_code", "class_label", "confidence", "anomaly_id", "interpretation", "notes")

    def __init__(self):
        self.items = {}
        self.n = 0
        self.updates = []
        self.deletes = []

    def sink(self, **kw):
        self.n += 1
        fid = kw.get("trace_id") or f"tr_fake_{self.n}"
        rec = {"id": fid, "kind": kw["kind"], "variant": kw["variant_name"], "stem": kw["profile_stem"],
               "source": kw.get("source") or "radargramma", "metrics": dict(kw.get("metrics") or {}),
               "channel": kw["channel"], "velocity_m_ns": kw.get("velocity_m_ns"), "rdp": kw.get("rdp"),
               "class_code": kw.get("class_code") or "", "class_label": kw.get("class_label") or "",
               "confidence": kw.get("confidence") or "", "anomaly_id": "", "interpretation": "", "notes": "",
               "vertices": [dict(v) for v in kw["vertices"]]}
        for k, v in (kw.get("attributes") or {}).items():
            rec[k] = v
        self.items[fid] = rec
        return {"id": fid}

    def query(self, variant, stem, channel, all_profiles=False):
        out = []
        for r in self.items.values():
            if not all_profiles and (r["stem"], r["channel"]) != (stem, channel):
                continue
            row = dict(r)
            row.setdefault("profile_stem", r["stem"])
            zs = [v["depth_m"] for v in r["vertices"]]
            row.setdefault("depth_from", min(zs))
            row.setdefault("depth_to", max(zs))
            row.setdefault("metrics", {})
            out.append(row)
        return out

    def update(self, trace_id, vertices=None, epsg=None, attributes=None, metrics=None):
        if trace_id not in self.items:
            raise KeyError(trace_id)
        self.updates.append({"id": trace_id, "vertices": vertices, "attributes": attributes,
                             "metrics": metrics})
        if metrics is not None:
            self.items[trace_id]["metrics"] = dict(metrics)
        if vertices is not None:
            self.items[trace_id]["vertices"] = [dict(v) for v in vertices]
        for k, v in (attributes or {}).items():
            if k in self.EDITABLE:
                self.items[trace_id][k] = v

    def delete(self, trace_id):
        self.deletes.append(trace_id)
        return self.items.pop(trace_id, None) is not None


@unittest.skipUnless(HAS_ENV, "QGIS/matplotlib non disponibili")
class AnnotationTest(_DockBase):
    """Classi di segnale, tempo doppio, selezione e modifica sul radargramma."""

    def setUp(self):
        super().setUp()
        self.fake = _FakeAnnotationLayer()
        d = self.dock
        d._feature_sink = self.fake.sink
        d._feature_query = self.fake.query
        d._feature_update = self.fake.update
        d._feature_delete = self.fake.delete

    def _key(self, key):
        self.dock._on_key(types.SimpleNamespace(key=key))

    def _press(self, x, z, **kw):
        self.dock._on_mouse_press(self._event("button_press_event", x, z, **kw))

    def _move(self, x, z):
        self.dock._on_mouse_move(self._event("motion_notify_event", x, z))

    def _release(self, x, z):
        self.dock._on_mouse_release(self._event("button_release_event", x, z))

    def _draw_line(self, p0=(1.0, 0.05), p1=(2.0, 0.10), key="2"):
        self._key(key)                              # riflettore: arma lo strumento linea
        self._press(*p0)
        self._press(*p1)
        self._key("enter")
        return next(iter(self.fake.items.values()))

    def _edit_mode_with_nothing_selected(self):
        d = self.dock
        d._btn_edit.setChecked(True)
        d._select_feature(None)

    # -- classi ---------------------------------------------------------------------

    def test_classes_are_listed_with_colours_and_default_to_the_first(self):
        d = self.dock
        self.assertGreaterEqual(d._cb_class.count(), 3)
        self.assertEqual(d._cb_class.itemData(0), "iperbole")
        self.assertTrue(d._cb_class.itemText(0).startswith("1 · "))
        self.assertFalse(d._cb_class.itemIcon(0).isNull())
        self.assertEqual(d._cur_class, "iperbole")

    def test_number_keys_choose_the_class_and_arm_its_geometry(self):
        d = self.dock
        self._key("3")                              # attenuazione -> poligono
        self.assertEqual(d._cur_class, "attenuazione")
        self.assertEqual(d._draw_mode, "polygon")
        self.assertTrue(d._btn_draw_polygon.isChecked())
        self._key("2")                              # riflettore -> linea
        self.assertEqual(d._draw_mode, "line")
        self.assertFalse(d._btn_draw_polygon.isChecked())
        self._key("1")                              # iperbole -> punto
        self.assertEqual(d._draw_mode, "point")
        self._key("9")                              # non esiste: nessun effetto
        self.assertEqual(d._cur_class, "iperbole")

    def test_changing_class_in_the_middle_of_a_drawing_keeps_the_geometry(self):
        d = self.dock
        self._key("2")
        self._press(1.0, 0.05)
        self._key("3")                              # a meta' di una linea: cambia solo la classe
        self.assertEqual(d._draw_mode, "line")
        self.assertEqual(len(d._draw_vertices), 1)
        self.assertEqual(d._cur_class, "attenuazione")

    def test_class_menu_arms_the_geometry_too(self):
        d = self.dock
        d._cb_class.setCurrentIndex(2)
        d._on_class_activated(2)
        self.assertEqual(d._draw_mode, "polygon")

    def test_the_last_class_is_remembered(self):
        from qgis.PyQt.QtCore import QSettings
        self._key("2")
        self.assertEqual(str(QSettings().value("GeoSurveyStudio/radargram_dock/class")), "riflettore")

    def test_classes_come_from_the_project_file_when_there_is_one(self):
        AC = importlib.import_module("rasterlinker_feature_interpolation.gpr_annotation_classes")
        AC.save_classes(self.root, [{"code": "vuoto", "label": "Vuoto", "geometry": "polygon", "color": "#123456"}])
        self.dock._project_root_provider = lambda: self.root
        self.dock._reload_classes()
        self.assertEqual(self.dock._cb_class.count(), 1)
        self.assertEqual(self.dock._cur_class, "vuoto")

    # -- disegno con classe e tempo -----------------------------------------------------

    def test_drawn_feature_carries_class_confidence_and_two_way_time(self):
        d = self.dock
        d._cb_conf.setCurrentIndex(d._cb_conf.findData("certo"))
        d._on_conf_activated(d._cb_conf.currentIndex())
        rec = self._draw_line()
        self.assertEqual(rec["class_code"], "riflettore")
        self.assertEqual(rec["class_label"], "Riflettore / interfaccia")
        self.assertEqual(rec["confidence"], "certo")
        dt = d._store.profiles[d._pi].dt_ns
        for v in rec["vertices"]:
            self.assertAlmostEqual(v["twt_ns"], v["depth_m"] / d._cur.dz_m * dt, places=6)
        self.assertLess(rec["vertices"][0]["twt_ns"], rec["vertices"][1]["twt_ns"])

    def test_new_feature_is_selected_shown_in_the_card_and_drawn_in_its_class_colour(self):
        d = self.dock
        rec = self._draw_line()
        self.assertEqual(d._sel_id, rec["id"])
        self.assertFalse(d._card.isHidden())
        self.assertEqual(d._cb_card_class.currentData(), "riflettore")
        self.assertEqual(d._cb_card_conf.currentData(), "probabile")
        self.assertTrue(d._edit_line.get_visible())
        self.assertEqual(len(d._edit_handles.get_xdata()), 2)
        # una volta deselezionata sparisce la scheda e compare l'overlay col colore della classe
        d._select_feature(None)
        self.assertTrue(d._card.isHidden())
        self.assertEqual(len(d._persisted_artists), 1)
        AC = importlib.import_module("rasterlinker_feature_interpolation.gpr_annotation_classes")
        want = AC.color_for(d._classes, "riflettore")
        from matplotlib.colors import to_hex
        self.assertEqual(to_hex(d._persisted_artists[0].get_color()), want)

    def test_starting_a_new_drawing_closes_the_card_of_the_previous_one(self):
        d = self.dock
        self._draw_line()
        self.assertFalse(d._card.isHidden())
        self._press(3.0, 0.05)
        self.assertIsNone(d._sel_id)
        self.assertTrue(d._card.isHidden())

    def test_annotations_stay_on_the_signal_when_the_velocity_changes(self):
        d = self.dock
        rec = self._draw_line()
        d._select_feature(None)
        z_before = [p["depth_m"] for p in d._features[0]["pts"]]
        saved_t = [v["twt_ns"] for v in rec["vertices"]]
        d._spin_vel_mns.setValue(0.05)              # velocita' diversa: cambia la scala di profondita'
        dt = d._store.profiles[d._pi].dt_ns
        z_after = [p["depth_m"] for p in d._features[0]["pts"]]
        self.assertNotAlmostEqual(z_before[1], z_after[1], places=3)
        for t, z in zip(saved_t, z_after):
            self.assertAlmostEqual(z, t / dt * d._cur.dz_m, places=6)   # stesso tempo, nuova profondita'

    def test_annotations_without_time_stay_at_their_saved_depth(self):
        d = self.dock
        ch = d._cur
        p0, p1 = ch.position(ch.trace_at_distance(1.0)), ch.position(ch.trace_at_distance(2.0))
        self.fake.items["old"] = {
            "id": "old", "kind": "line", "stem": d._store.profiles[d._pi].stem, "channel": d._ci,
            "class_code": "", "class_label": "", "confidence": "", "anomaly_id": "", "interpretation": "",
            "notes": "", "vertices": [{"east": p0[0], "north": p0[1], "depth_m": 0.05, "twt_ns": None},
                                      {"east": p1[0], "north": p1[1], "depth_m": 0.09, "twt_ns": None}]}
        d._refresh_persisted_features()
        self.assertEqual([p["depth_m"] for p in d._features[0]["pts"]], [0.05, 0.09])
        from matplotlib.colors import to_hex
        self.assertEqual(to_hex(d._persisted_artists[0].get_color()), "#00e5ff")   # senza classe: colore neutro

    # -- modifica -------------------------------------------------------------------

    def test_edit_tool_is_exclusive_with_the_other_tools(self):
        d = self.dock
        d._btn_draw_line.setChecked(True)
        d._btn_edit.setChecked(True)
        self.assertIsNone(d._draw_mode)
        self.assertFalse(d._btn_draw_line.isChecked())
        d._btn_vel.setChecked(True)
        self.assertFalse(d._btn_edit.isChecked())
        d._btn_edit.setChecked(True)
        self.assertFalse(d._btn_vel.isChecked())
        d._btn_draw_point.setChecked(True)
        self.assertFalse(d._btn_edit.isChecked())

    def test_click_selects_an_annotation_and_a_click_elsewhere_deselects(self):
        d = self.dock
        rec = self._draw_line()
        self._edit_mode_with_nothing_selected()
        self._press(1.5, 0.075)                     # a meta' del segmento
        self.assertEqual(d._sel_id, rec["id"])
        self.assertFalse(d._card.isHidden())
        self._release(1.5, 0.075)
        self._press(4.0, 0.2)                       # lontano da tutto
        self.assertIsNone(d._sel_id)
        self.assertTrue(d._card.isHidden())

    def test_click_inside_a_polygon_selects_it_and_the_smaller_one_wins(self):
        d = self.dock
        self._key("3")
        for x, z in ((1.0, 0.04), (3.0, 0.04), (3.0, 0.14), (1.0, 0.14)):
            self._press(x, z)
        self._key("enter")
        big = next(iter(self.fake.items))
        for x, z in ((1.8, 0.07), (2.2, 0.07), (2.2, 0.10), (1.8, 0.10)):
            self._press(x, z)
        self._key("enter")
        small = [k for k in self.fake.items if k != big][0]
        self._edit_mode_with_nothing_selected()
        self._press(2.0, 0.085)                     # dentro entrambi
        self.assertEqual(d._sel_id, small)
        self._release(2.0, 0.085)
        d._select_feature(None)
        self._press(1.2, 0.12)                      # solo dentro il grande
        self.assertEqual(d._sel_id, big)

    def test_features_without_an_identifier_cannot_be_selected(self):
        d = self.dock
        ch = d._cur
        p0, p1 = ch.position(ch.trace_at_distance(1.0)), ch.position(ch.trace_at_distance(2.0))
        d._feature_query = lambda v, s, c: [{"kind": "line", "vertices": [
            {"east": p0[0], "north": p0[1], "depth_m": 0.05}, {"east": p1[0], "north": p1[1], "depth_m": 0.09}]}]
        d._refresh_persisted_features()
        d._btn_edit.setChecked(True)
        self._press(1.5, 0.07)
        self.assertIsNone(d._sel_id)

    def test_dragging_a_vertex_moves_it_and_saves_depth_and_time(self):
        d = self.dock
        rec = self._draw_line()
        old_t = rec["vertices"][1]["twt_ns"]
        self._edit_mode_with_nothing_selected()
        self._press(1.5, 0.075)                     # seleziona
        self._release(1.5, 0.075)
        self._press(2.0, 0.10)                      # afferra il secondo vertice
        self.assertIsNotNone(d._edit_drag)
        self._move(2.5, 0.13)
        self.assertEqual(self.fake.updates, [])     # durante il trascinamento non si scrive ancora
        self._release(2.5, 0.13)
        self.assertEqual(len(self.fake.updates), 1)
        v = self.fake.items[rec["id"]]["vertices"]
        self.assertAlmostEqual(v[1]["depth_m"], 0.13, places=2)
        self.assertGreater(v[1]["twt_ns"], old_t)
        ch = d._cur
        self.assertAlmostEqual(v[1]["east"], ch.position(ch.trace_at_distance(2.5))[0], places=3)
        self.assertAlmostEqual(v[0]["depth_m"], 0.05, places=2)     # l'altro vertice non si muove

    def test_a_click_without_moving_does_not_write_anything(self):
        d = self.dock
        self._draw_line()
        self._edit_mode_with_nothing_selected()
        self._press(1.5, 0.075)
        self._release(1.5, 0.075)
        self._press(2.0, 0.10)
        self._release(2.0, 0.10)
        self.assertEqual(self.fake.updates, [])

    def test_undo_and_redo_of_a_vertex_move(self):
        d = self.dock
        rec = self._draw_line()
        before = [dict(v) for v in self.fake.items[rec["id"]]["vertices"]]
        self._edit_mode_with_nothing_selected()
        self._press(1.5, 0.075)
        self._release(1.5, 0.075)
        self._press(2.0, 0.10)
        self._move(2.5, 0.13)
        self._release(2.5, 0.13)
        moved = self.fake.items[rec["id"]]["vertices"][1]["depth_m"]
        self._key("ctrl+z")
        back = self.fake.items[rec["id"]]["vertices"]
        self.assertAlmostEqual(back[1]["depth_m"], before[1]["depth_m"], places=6)
        self.assertAlmostEqual(back[1]["twt_ns"], before[1]["twt_ns"], places=6)
        self._key("ctrl+y")
        self.assertAlmostEqual(self.fake.items[rec["id"]]["vertices"][1]["depth_m"], moved, places=6)

    def test_double_click_on_a_segment_adds_a_vertex(self):
        d = self.dock
        rec = self._draw_line()
        self._edit_mode_with_nothing_selected()
        self._press(1.5, 0.075)
        self._release(1.5, 0.075)
        self._press(1.5, 0.075, dblclick=True)
        self.assertEqual(len(self.fake.items[rec["id"]]["vertices"]), 3)
        self._key("ctrl+z")
        self.assertEqual(len(self.fake.items[rec["id"]]["vertices"]), 2)

    def test_right_click_on_a_vertex_removes_it_but_never_below_the_minimum(self):
        d = self.dock
        self._key("3")
        for x, z in ((1.0, 0.05), (2.0, 0.05), (2.0, 0.12), (1.0, 0.12)):
            self._press(x, z)
        self._key("enter")
        fid = next(iter(self.fake.items))
        self._edit_mode_with_nothing_selected()
        self._press(1.5, 0.085)
        self._release(1.5, 0.085)
        self._press(2.0, 0.12, button=3)
        self.assertEqual(len(self.fake.items[fid]["vertices"]), 3)
        self._press(1.0, 0.12, button=3)
        self.assertEqual(len(self.fake.items[fid]["vertices"]), 3)          # un poligono resta di 3 vertici
        self.assertIn("almeno 3", d._lbl_status.text())

    def test_delete_key_removes_the_annotation_and_undo_brings_it_back_with_the_same_id(self):
        d = self.dock
        rec = self._draw_line()
        fid = rec["id"]
        saved = dict(self.fake.items[fid])
        self._edit_mode_with_nothing_selected()
        self._press(1.5, 0.075)
        self._release(1.5, 0.075)
        self._key("delete")
        self.assertEqual(self.fake.items, {})
        self.assertIsNone(d._sel_id)
        self._key("ctrl+z")
        self.assertIn(fid, self.fake.items)
        back = self.fake.items[fid]
        self.assertEqual(back["class_code"], saved["class_code"])
        self.assertEqual(len(back["vertices"]), 2)
        self.assertAlmostEqual(back["vertices"][1]["twt_ns"], saved["vertices"][1]["twt_ns"], places=6)
        self._key("ctrl+y")
        self.assertEqual(self.fake.items, {})

    def test_undo_of_a_new_annotation_removes_it(self):
        self._draw_line()
        self._key("ctrl+z")
        self.assertEqual(self.fake.items, {})
        self.assertEqual(self.dock._features, [])
        self._key("ctrl+y")
        self.assertEqual(len(self.fake.items), 1)

    def test_number_keys_reclassify_the_selected_annotation_in_edit_mode(self):
        d = self.dock
        rec = self._draw_line()
        self._edit_mode_with_nothing_selected()
        self._press(1.5, 0.075)
        self._release(1.5, 0.075)
        self._key("3")
        self.assertEqual(self.fake.items[rec["id"]]["class_code"], "attenuazione")
        self.assertEqual(self.fake.items[rec["id"]]["class_label"], "Attenuazione / perdita di segnale")
        self.assertEqual(d._cur_class, "riflettore")            # la classe con cui si disegna non cambia
        self.assertEqual(d._cb_card_class.currentData(), "attenuazione")
        self._key("ctrl+z")
        self.assertEqual(self.fake.items[rec["id"]]["class_code"], "riflettore")

    def test_card_fields_are_saved_and_undoable(self):
        d = self.dock
        rec = self._draw_line()
        d._ed_code.setText("M12")
        d._ed_notes.setText("continua verso nord")
        d._on_card_text_finished()
        item = self.fake.items[rec["id"]]
        self.assertEqual((item["anomaly_id"], item["notes"]), ("M12", "continua verso nord"))
        d._cb_card_conf.setCurrentIndex(d._cb_card_conf.findData("certo"))
        d._on_card_conf(d._cb_card_conf.currentIndex())
        self.assertEqual(item["confidence"], "certo")
        d._on_card_text_finished()                  # niente di nuovo: nessuna scrittura in piu'
        self.assertEqual(len(self.fake.updates), 2)
        self._key("ctrl+z")
        self.assertEqual(item["confidence"], "probabile")

    def test_escape_in_edit_mode_deselects(self):
        d = self.dock
        self._draw_line()
        d._btn_edit.setChecked(True)
        d._select_feature(next(iter(self.fake.items)))
        self._key("escape")
        self.assertIsNone(d._sel_id)

    def test_leaving_edit_mode_deselects(self):
        d = self.dock
        self._draw_line()
        d._btn_edit.setChecked(True)
        d._select_feature(next(iter(self.fake.items)))
        d._btn_edit.setChecked(False)
        self.assertIsNone(d._sel_id)
        self.assertTrue(d._card.isHidden())

    def test_changing_profile_drops_the_selection(self):
        d = self.dock
        self._draw_line()
        self.assertIsNotNone(d._sel_id)
        d.select_profile(d._pi + 1)
        self.assertIsNone(d._sel_id)
        self.assertTrue(d._card.isHidden())

    def test_failed_save_reverts_the_vertex_and_says_so(self):
        d = self.dock
        rec = self._draw_line()
        self._edit_mode_with_nothing_selected()
        self._press(1.5, 0.075)
        self._release(1.5, 0.075)

        def boom(*a, **kw):
            raise RuntimeError("layer bloccato")
        d._feature_update = boom
        self._press(2.0, 0.10)
        self._move(2.5, 0.13)
        self._release(2.5, 0.13)
        self.assertIn("layer bloccato", d._lbl_status.text())
        self.assertAlmostEqual(d._selected_rec()["pts"][1]["depth_m"], 0.10, places=2)
        self.assertEqual(d._undo_stack[-1]["label"], "Nuova annotazione")   # niente da annullare in piu'

    def test_without_update_callbacks_edits_say_so_and_change_nothing(self):
        d = self.dock
        self._draw_line()
        d._feature_update = None
        d._feature_delete = None
        d._btn_edit.setChecked(True)
        d._select_feature(next(iter(self.fake.items)))
        self._key("delete")
        self.assertEqual(len(self.fake.items), 1)
        self.assertIn("non disponibile", d._lbl_status.text())

    def test_undo_buttons_follow_the_stacks(self):
        d = self.dock
        self.assertFalse(d._btn_undo.isEnabled())
        self.assertFalse(d._btn_redo.isEnabled())
        self._draw_line()
        self.assertTrue(d._btn_undo.isEnabled())
        self._key("ctrl+z")
        self.assertFalse(d._btn_undo.isEnabled())
        self.assertTrue(d._btn_redo.isEnabled())

    def test_new_annotation_bridge_result_can_be_a_qgs_feature_like_object(self):
        d = self.dock
        fake_feat = types.SimpleNamespace(attribute=lambda name: "tr_x" if name == "trace_id" else None)
        self.assertEqual(d._result_feature_id(fake_feat), "tr_x")
        self.assertEqual(d._result_feature_id({"id": "tr_y"}), "tr_y")
        self.assertEqual(d._result_feature_id(None), "")


def _wavelet(n, center, width=2.0):
    x = np.arange(n, dtype=np.float64)
    return np.exp(-0.5 * ((x - center) / width) ** 2)


@unittest.skipUnless(HAS_ENV, "QGIS/matplotlib non disponibili")
class AnnotationToolsTest(_DockBase):
    """Rettangolo, aggancio al picco, tracciamento (Maiusc+clic), iperbole annotata, filtri, tooltip."""

    def setUp(self):
        super().setUp()
        self.fake = _FakeAnnotationLayer()
        d = self.dock
        d._feature_sink = self.fake.sink
        d._feature_query = self.fake.query
        d._feature_update = self.fake.update
        d._feature_delete = self.fake.delete

    def _ev(self, name, x_m, z_m, button=1, key=None):
        cv = self.dock._canvas
        cv.draw()
        px, py = self.dock._ax.transData.transform((x_m, z_m))
        return MouseEvent(name, cv, px, py, button=button, key=key)

    def _key(self, key):
        self.dock._on_key(types.SimpleNamespace(key=key))

    def _press(self, x, z, **kw):
        self.dock._on_mouse_press(self._ev("button_press_event", x, z, **kw))

    def _release(self, x, z):
        self.dock._on_mouse_release(self._ev("button_release_event", x, z))

    def _move(self, x, z):
        self.dock._on_mouse_move(self._ev("motion_notify_event", x, z))

    def _draw(self, key, pts):
        self._key(key)
        for x, z in pts:
            self._press(x, z)
        self._key("enter")

    def _only(self):
        return next(iter(self.fake.items.values()))

    # -- rettangolo ---------------------------------------------------------------------

    def test_rectangle_takes_two_clicks_and_is_saved_as_a_four_vertex_polygon(self):
        d = self.dock
        self._key("3")                                  # attenuazione
        d._btn_draw_rect.setChecked(True)
        self.assertEqual(d._draw_mode, "rect")
        self._press(2.5, 0.12)
        self.assertEqual(len(d._draw_vertices), 1)
        self._move(1.0, 0.05)
        self.assertTrue(d._draw_preview_fill.get_visible())            # anteprima del rettangolo
        self.assertEqual(len(d._draw_preview_line.get_xdata()), 5)
        self.assertEqual(self.fake.items, {})                           # ancora nulla di salvato
        self._press(1.0, 0.05)
        rec = self._only()
        self.assertEqual(rec["kind"], "polygon")
        self.assertEqual(rec["class_code"], "attenuazione")
        self.assertEqual(len(rec["vertices"]), 4)
        zs = sorted({round(v["depth_m"], 3) for v in rec["vertices"]})
        self.assertEqual(zs, [0.05, 0.12])
        self.assertEqual(d._draw_mode, "rect")                          # pronto per il prossimo
        self.assertEqual(d._draw_vertices, [])

    def test_rectangle_with_no_width_or_height_is_refused(self):
        d = self.dock
        d._btn_draw_rect.setChecked(True)
        self._press(1.0, 0.05)
        self._press(2.0, 0.05)                          # stessa profondita'
        self.assertEqual(self.fake.items, {})
        self.assertEqual(len(d._draw_vertices), 1)
        self.assertIn("nulla", d._lbl_status.text())

    def test_class_key_keeps_rectangle_mode_for_polygon_classes_and_escape_cancels(self):
        d = self.dock
        d._btn_draw_rect.setChecked(True)
        self._key("3")
        self.assertEqual(d._draw_mode, "rect")
        self._key("4")                                  # anomalia di ampiezza: poligono
        self.assertEqual(d._draw_mode, "rect")
        self._press(1.0, 0.05)
        self._key("escape")
        self.assertEqual(d._draw_vertices, [])
        self.assertIsNone(d._draw_mode)

    def test_finishing_a_rectangle_early_says_how_to_close_it(self):
        d = self.dock
        d._btn_draw_rect.setChecked(True)
        self._press(1.0, 0.05)
        self._key("enter")
        self.assertIn("secondo clic", d._lbl_status.text())

    # -- aggancio al picco --------------------------------------------------------------

    def _peak_trace(self, x, row=20):
        ch = self.dock._cur
        t = ch.trace_at_distance(x)
        ch.data[:, t] = _wavelet(ch.data.shape[0], row).astype(ch.data.dtype)
        return t

    def test_snap_is_off_by_default_and_remembered(self):
        from qgis.PyQt.QtCore import QSettings
        d = self.dock
        self.assertFalse(d._chk_snap.isChecked())
        d._chk_snap.setChecked(True)
        self.assertEqual(str(QSettings().value("GeoSurveyStudio/radargram_dock/snap")), "1")

    def test_snap_moves_the_vertex_to_the_nearest_phase_peak(self):
        d = self.dock
        ch = d._cur
        self._peak_trace(1.5)
        clicked = (20 - 3 + 0.5) * ch.dz_m
        self._key("1")
        self._press(1.5, clicked)                       # senza aggancio: resta dove e' caduto
        self.assertAlmostEqual(self._only()["vertices"][0]["depth_m"], clicked, places=6)
        self.fake.items.clear()
        d._chk_snap.setChecked(True)
        self._press(1.5, clicked)
        self.assertAlmostEqual(self._only()["vertices"][0]["depth_m"], (20 + 0.5) * ch.dz_m, places=6)

    def test_snap_leaves_the_vertex_alone_when_there_is_no_peak_nearby(self):
        d = self.dock
        ch = d._cur
        t = ch.trace_at_distance(1.5)
        ch.data[:, t] = np.linspace(0, 1, ch.data.shape[0]).astype(ch.data.dtype)    # monotono
        d._chk_snap.setChecked(True)
        self._key("1")
        self._press(1.5, 0.1)
        self.assertAlmostEqual(self._only()["vertices"][0]["depth_m"], 0.1, places=6)

    def test_snap_applies_while_dragging_a_vertex_in_edit_mode(self):
        d = self.dock
        ch = d._cur
        self._peak_trace(2.5, row=24)
        self._draw("2", [(1.0, 0.05), (2.0, 0.10)])
        rid = self._only()["id"]
        d._btn_edit.setChecked(True)
        d._select_feature(None)
        self._press(1.5, 0.075)
        self._release(1.5, 0.075)
        d._chk_snap.setChecked(True)
        self._press(2.0, 0.10)
        self._move(2.5, (24 - 2 + 0.5) * ch.dz_m)
        self._release(2.5, (24 - 2 + 0.5) * ch.dz_m)
        self.assertAlmostEqual(self.fake.items[rid]["vertices"][1]["depth_m"], (24 + 0.5) * ch.dz_m, places=6)

    # -- Maiusc+clic: tracciamento del riflettore ---------------------------------------

    def test_shift_click_follows_the_reflector_between_two_clicks(self):
        d = self.dock
        ch = d._cur
        t0, t1 = ch.trace_at_distance(1.0), ch.trace_at_distance(3.0)
        centre = lambda t: 15 + 0.2 * (t - t0)
        for t in range(t0, t1 + 1):
            ch.data[:, t] = _wavelet(ch.data.shape[0], centre(t)).astype(ch.data.dtype)
        self._key("2")
        self._press(1.0, (15 + 0.5) * ch.dz_m)
        self._press(3.0, (centre(t1) + 0.5) * ch.dz_m, key="shift")
        self._key("enter")
        verts = self._only()["vertices"]
        self.assertGreater(len(verts), 3)               # i vertici intermedi sono stati aggiunti
        self.assertLessEqual(len(verts), 2 + _DOCK._FOLLOW_MAX_VERTICES)
        depths = [v["depth_m"] for v in verts]
        self.assertAlmostEqual(depths[0], (15 + 0.5) * ch.dz_m, places=6)
        self.assertAlmostEqual(depths[-1], (round(centre(t1)) + 0.5) * ch.dz_m, delta=ch.dz_m)
        self.assertTrue(all(b >= a - 1e-9 for a, b in zip(depths, depths[1:])))   # il riflettore scende

    def test_plain_click_adds_only_the_clicked_vertex(self):
        d = self.dock
        self._draw("2", [(1.0, 0.05), (3.0, 0.10)])
        self.assertEqual(len(self._only()["vertices"]), 2)

    # -- iperbole annotata ------------------------------------------------------------

    def _fit_hyperbola(self, x=2.0, z=0.08, dv=-0.02):
        d = self.dock
        d._btn_hyper.setChecked(True)
        self._press(x, z)
        d._adjust_hyper_velocity(dv)
        return d._hyper_v_mns

    def test_annotate_button_follows_the_apex(self):
        d = self.dock
        d._btn_hyper.setChecked(True)
        self.assertFalse(d._btn_hyper_note.isEnabled())
        self._press(2.0, 0.08)
        self.assertTrue(d._btn_hyper_note.isEnabled())
        d._clear_hyper_apex()
        self.assertFalse(d._btn_hyper_note.isEnabled())

    def test_hyperbola_is_saved_as_an_annotation_with_velocity_and_true_top_depth(self):
        d = self.dock
        v_mns = self._fit_hyperbola()
        v_ref = d._reference_velocity_m_s()
        d._btn_hyper_note.click()
        rec = self._only()
        self.assertEqual(rec["source"], "iperbole_fit")
        self.assertEqual(rec["kind"], "point")
        self.assertEqual(rec["class_code"], "iperbole")
        self.assertAlmostEqual(rec["velocity_m_ns"], v_mns, places=6)
        self.assertGreater(rec["rdp"], 1.0)
        self.assertIn("top a", rec["interpretation"])
        vx = rec["vertices"][0]
        z0 = 0.08                                            # profondita' mostrata dell'apice (clic)
        self.assertAlmostEqual(vx["depth_m"], z0 * (v_mns * 1e9) / v_ref, places=6)
        self.assertAlmostEqual(vx["twt_ns"], d._depth_to_twt(z0), places=6)
        self.assertIsNone(d._hyper_apex)                    # apice pulito: niente duplicati
        self.assertEqual(d._sel_id, rec["id"])

    def test_annotated_hyperbola_redraws_its_curve_and_survives_a_velocity_change(self):
        d = self.dock
        self._fit_hyperbola()
        d._btn_hyper_note.click()
        d._select_feature(None)
        curves = [a for a in d._persisted_extra if a.get_linestyle() == "--"]
        self.assertEqual(len(curves), 1)
        xs, zs = curves[0].get_data()
        self.assertGreater(len(xs), 10)
        self.assertAlmostEqual(min(zs), d._features[0]["pts"][0]["depth_m"], delta=0.005)   # il minimo e' l'apice
        d._spin_vel_mns.setValue(0.05)
        curves = [a for a in d._persisted_extra if a.get_linestyle() == "--"]
        self.assertEqual(len(curves), 1)

    def test_manual_point_has_no_curve(self):
        d = self.dock
        self._draw("1", [(2.0, 0.08)])
        d._select_feature(None)
        self.assertEqual([a for a in d._persisted_extra if a.get_linestyle() == "--"], [])

    def test_moving_a_fitted_hyperbola_keeps_the_true_depth_scale(self):
        d = self.dock
        v_mns = self._fit_hyperbola()
        d._btn_hyper_note.click()
        rid = self._only()["id"]
        scale = (v_mns * 1e9) / d._reference_velocity_m_s()
        d._btn_edit.setChecked(True)
        d._select_feature(rid)
        self._press(2.0, 0.08)
        self._move(2.5, 0.10)
        self._release(2.5, 0.10)
        self.assertAlmostEqual(self.fake.items[rid]["vertices"][0]["depth_m"], 0.10 * scale, places=2)

    def test_deleting_and_restoring_a_fitted_hyperbola_keeps_its_source_and_fit(self):
        d = self.dock
        v_mns = self._fit_hyperbola()
        d._btn_hyper_note.click()
        rid = self._only()["id"]
        before = dict(self._only())
        d._btn_edit.setChecked(True)
        d._select_feature(rid)
        self._key("delete")
        self.assertEqual(self.fake.items, {})
        self._key("ctrl+z")
        back = self.fake.items[rid]
        self.assertEqual(back["source"], "iperbole_fit")
        self.assertAlmostEqual(back["velocity_m_ns"], v_mns, places=6)
        self.assertAlmostEqual(back["vertices"][0]["depth_m"], before["vertices"][0]["depth_m"], places=6)

    def test_annotating_without_an_apex_or_a_sink_says_so(self):
        d = self.dock
        d._save_hyper_fit_as_annotation()
        self.assertIn("apice", d._lbl_status.text())
        d._btn_hyper.setChecked(True)
        self._press(2.0, 0.08)
        d._feature_sink = None
        d._save_hyper_fit_as_annotation()
        self.assertIn("non disponibile", d._lbl_status.text())

    # -- filtri, etichette, tooltip ---------------------------------------------------

    def _two_features(self):
        self._draw("2", [(1.0, 0.05), (2.0, 0.10)])
        line_id = self._only()["id"]
        self._draw("3", [(2.5, 0.14), (3.5, 0.14), (3.5, 0.22), (2.5, 0.22)])
        poly_id = [k for k in self.fake.items if k != line_id][0]
        self._key("escape")                             # fuori dal disegno
        self.dock._select_feature(None)
        return line_id, poly_id

    def test_show_menu_lists_every_class_and_confidence_and_acts_as_a_legend(self):
        d = self.dock
        texts = [a.text() for a in d._btn_show.menu().actions() if not a.isSeparator()]
        self.assertTrue(any("Etichette" in t for t in texts))
        for cls in d._classes:
            self.assertIn(cls["label"], texts)
        self.assertIn("Non classificato", texts)
        self.assertTrue(any("Certo" in t for t in texts))
        self.assertIn("Mostra tutto", texts)
        class_actions = [a for a in d._btn_show.menu().actions() if a.text() == d._classes[0]["label"]]
        self.assertFalse(class_actions[0].icon().isNull())

    def test_hiding_a_class_removes_its_overlay_and_makes_it_unselectable(self):
        d = self.dock
        line_id, poly_id = self._two_features()
        self.assertEqual(len(d._persisted_artists), 2)
        d._on_show_class("riflettore", False)
        self.assertEqual(len(d._persisted_artists), 1)
        px = d._ax.transData.transform((1.5, 0.075))
        self.assertIsNone(d._hit_feature(float(px[0]), float(px[1])))
        px2 = d._ax.transData.transform((3.0, 0.18))
        self.assertEqual(d._hit_feature(float(px2[0]), float(px2[1])), poly_id)
        d._show_all()
        self.assertEqual(len(d._persisted_artists), 2)
        self.assertTrue(all(a.isChecked() for a in d._btn_show.menu().actions()
                            if a.isCheckable() and not a.text().startswith("Mappa")))

    def test_hiding_the_selected_class_deselects_it(self):
        d = self.dock
        line_id, _poly = self._two_features()
        d._btn_edit.setChecked(True)
        d._select_feature(line_id)
        d._on_show_class("riflettore", False)
        self.assertIsNone(d._sel_id)
        self.assertTrue(d._card.isHidden())

    def test_hiding_a_confidence_level_hides_those_annotations(self):
        d = self.dock
        self._two_features()
        d._on_show_conf("probabile", False)               # tutte le annotazioni sono "probabile"
        self.assertEqual(d._persisted_artists, [])
        d._on_show_conf("probabile", True)
        self.assertEqual(len(d._persisted_artists), 2)

    def test_annotations_without_confidence_stay_visible_when_a_level_is_hidden(self):
        d = self.dock
        ch = d._cur
        p0, p1 = ch.position(ch.trace_at_distance(1.0)), ch.position(ch.trace_at_distance(2.0))
        self.fake.items["old"] = {
            "id": "old", "kind": "line", "stem": d._store.profiles[d._pi].stem, "channel": d._ci,
            "class_code": "", "class_label": "", "confidence": "", "anomaly_id": "", "interpretation": "",
            "notes": "", "vertices": [{"east": p0[0], "north": p0[1], "depth_m": 0.05},
                                      {"east": p1[0], "north": p1[1], "depth_m": 0.09}]}
        d._refresh_persisted_features()
        d._on_show_conf("certo", False)
        d._on_show_conf("probabile", False)
        d._on_show_conf("possibile", False)
        self.assertEqual(len(d._persisted_artists), 1)

    def test_labels_show_the_anomaly_code_and_can_be_switched_off(self):
        d = self.dock
        line_id, _poly = self._two_features()
        d._btn_edit.setChecked(True)
        d._select_feature(line_id)
        d._ed_code.setText("M12")
        d._on_card_text_finished()
        d._select_feature(None)
        texts = [a.get_text() for a in d._persisted_extra if hasattr(a, "get_text")]
        self.assertEqual(texts, ["M12"])
        d._on_labels_toggled(False)
        self.assertEqual([a for a in d._persisted_extra if hasattr(a, "get_text")], [])

    def test_tooltip_text_has_class_confidence_code_notes_and_depth(self):
        d = self.dock
        line_id, _poly = self._two_features()
        d._btn_edit.setChecked(True)
        d._select_feature(line_id)
        d._ed_code.setText("M12")
        d._ed_interp.setText("piano d'uso")
        d._ed_notes.setText("continua a nord")
        d._on_card_text_finished()
        text = d._feature_tooltip(d._selected_rec())
        for part in ("Riflettore / interfaccia", "Probabile", "M12", "piano d'uso", "continua a nord", "prof. "):
            self.assertIn(part, text)

    def test_hover_sets_and_clears_the_hovered_annotation(self):
        d = self.dock
        line_id, _poly = self._two_features()
        self._move(1.5, 0.075)
        self.assertEqual(d._hover_id, line_id)
        self._move(4.5, 0.02)
        self.assertIsNone(d._hover_id)
        d._btn_draw_line.setChecked(True)                 # disegnando non ci sono tooltip
        self._move(1.5, 0.075)
        self.assertIsNone(d._hover_id)

    def test_tooltip_depth_of_a_fitted_hyperbola_is_the_true_top(self):
        d = self.dock
        v_mns = self._fit_hyperbola()
        d._btn_hyper_note.click()
        rec = d._selected_rec()
        scale = (v_mns * 1e9) / d._reference_velocity_m_s()
        want = f"prof. {rec['pts'][0]['depth_m'] * scale:.2f} m"
        self.assertIn(want, d._feature_tooltip(rec))


@unittest.skipUnless(HAS_ENV, "QGIS/matplotlib non disponibili")
class AnnotationAnalysisTest(_DockBase):
    """Metriche, profili vicini, tabella anomalie, fascia time-slice sulla mappa, figura esportata."""

    def setUp(self):
        super().setUp()
        self.fake = _FakeAnnotationLayer()
        self.queries = []
        d = self.dock
        d._feature_sink = self.fake.sink

        def query(variant, stem, channel, all_profiles=False):
            self.queries.append(all_profiles)
            return self.fake.query(variant, stem, channel, all_profiles=all_profiles)
        d._feature_query = query
        d._feature_update = self.fake.update
        d._feature_delete = self.fake.delete

    # -- utilita' -------------------------------------------------------------------------

    def _ev(self, name, x_m, z_m, button=1):
        cv = self.dock._canvas
        cv.draw()
        px, py = self.dock._ax.transData.transform((x_m, z_m))
        return MouseEvent(name, cv, px, py, button=button)

    def _key(self, key):
        self.dock._on_key(types.SimpleNamespace(key=key))

    def _press(self, x, z, **kw):
        self.dock._on_mouse_press(self._ev("button_press_event", x, z, **kw))

    def _release(self, x, z):
        self.dock._on_mouse_release(self._ev("button_release_event", x, z))

    def _move(self, x, z):
        self.dock._on_mouse_move(self._ev("motion_notify_event", x, z))

    def _draw(self, key, pts):
        self._key(key)
        for x, z in pts:
            self._press(x, z)
        self._key("enter")

    def _rect(self, p0, p1, key="3"):
        self._key(key)
        self.dock._btn_draw_rect.setChecked(True)
        self._press(*p0)
        self._press(*p1)

    def _last(self):
        return list(self.fake.items.values())[-1]

    def _add_foreign(self, fid, stem, channel, kind, coords, depth, code="", class_code="riflettore",
                     twt=None, confidence="probabile"):
        """Annotazione di un altro profilo; coords = [(est, nord), ...]."""
        self.fake.items[fid] = {
            "id": fid, "kind": kind, "stem": stem, "channel": channel, "source": "radargramma",
            "class_code": class_code, "class_label": "", "confidence": confidence, "anomaly_id": code,
            "interpretation": "", "notes": "", "velocity_m_ns": None, "rdp": None,
            "vertices": [{"east": e, "north": n, "depth_m": depth, "twt_ns": twt, "amp": None}
                         for e, n in coords]}

    def _cur_geo(self):
        d = self.dock
        ch = d._cur
        return ch.east.min(), ch.east.max(), float(np.nanmean(ch.north))

    # -- metriche -------------------------------------------------------------------------

    def test_a_drawn_line_stores_length_and_width_but_no_area(self):
        self._draw("2", [(1.0, 0.05), (3.0, 0.05), (3.0, 0.15)])
        m = self._last()["metrics"]
        self.assertAlmostEqual(m["length_m"], 2.0 + 0.1, delta=0.06)
        self.assertAlmostEqual(m["width_m"], 2.0, delta=0.06)
        self.assertIsNone(m["area_m2"])
        self.assertIsNone(m["contrast_db"])

    def test_a_point_has_no_metrics(self):
        self._draw("1", [(2.0, 0.08)])
        self.assertTrue(all(v is None for v in self._last()["metrics"].values()))

    def test_an_attenuated_zone_gets_area_rms_and_negative_contrast(self):
        d = self.dock
        ch = d._cur
        rng = np.random.default_rng(5)
        ch.data[:] = rng.normal(0, 1.0, ch.data.shape).astype(ch.data.dtype)
        i0, i1 = ch.trace_at_distance(2.0), ch.trace_at_distance(3.0)
        ch.data[8:28, i0:i1 + 1] *= 0.2                      # zona "spenta"
        self._rect((2.0, 8 * ch.dz_m), (3.0, 28 * ch.dz_m))
        m = self._last()["metrics"]
        self.assertAlmostEqual(m["area_m2"], 1.0 * 20 * ch.dz_m, delta=0.02)
        self.assertLess(m["contrast_db"], -8.0)
        self.assertGreater(m["amp_rms"], 0.0)
        self.assertLess(m["amp_rms"], 0.5)

    def test_moving_a_vertex_rewrites_the_metrics_and_undo_restores_them(self):
        d = self.dock
        self._draw("2", [(1.0, 0.05), (2.0, 0.05)])
        rid = self._last()["id"]
        before = dict(self._last()["metrics"])
        d._btn_edit.setChecked(True)
        d._select_feature(None)
        self._press(1.5, 0.05)
        self._release(1.5, 0.05)
        self._press(2.0, 0.05)
        self._move(3.0, 0.05)
        self._release(3.0, 0.05)
        after = self.fake.items[rid]["metrics"]
        self.assertGreater(after["length_m"], before["length_m"] + 0.5)
        self._key("ctrl+z")
        self.assertAlmostEqual(self.fake.items[rid]["metrics"]["length_m"], before["length_m"], places=6)

    def test_metrics_appear_in_card_and_tooltip(self):
        d = self.dock
        self._rect((2.0, 0.05), (3.0, 0.15))
        self.assertIn("area", d._lbl_card_info.text())
        self.assertIn("m²", d._lbl_card_info.text())
        self.assertIn("contrasto", d._lbl_card_info.text())
        self.assertIn("contrasto", d._feature_tooltip(d._selected_rec()))

    def test_deleted_polygon_comes_back_with_its_metrics(self):
        d = self.dock
        self._rect((2.0, 0.05), (3.0, 0.15))
        rid = self._last()["id"]
        want = dict(self._last()["metrics"])
        d._btn_edit.setChecked(True)
        d._select_feature(rid)
        self._key("delete")
        self._key("ctrl+z")
        self.assertEqual(self.fake.items[rid]["metrics"], want)

    # -- profili vicini -------------------------------------------------------------------

    def _crossing_line(self, fid="nb1", code="M1", **kw):
        e0, e1, n0 = self._cur_geo()
        east = e0 + 2.0
        self._add_foreign(fid, "Nord9", 0, "line", [(east, n0 - 5.0), (east, n0 + 5.0)], 0.10, code=code, **kw)

    def test_neighbour_annotations_are_hidden_by_default_and_shown_within_tolerance(self):
        d = self.dock
        self._crossing_line()
        d._refresh_persisted_features()
        self.assertEqual([a for a in d._persisted_extra if getattr(a, "_gss_ghost", False)], [])
        d._set_neighbour_tol(1.0)
        ghosts = [a for a in d._persisted_extra if getattr(a, "_gss_ghost", False)]
        self.assertEqual(len(ghosts), 1)
        offs = np.asarray(ghosts[0].get_offsets())
        self.assertGreaterEqual(len(offs), 3)
        self.assertTrue(np.allclose(offs[:, 0], 2.0, atol=0.06))       # dove il muro attraversa il profilo
        self.assertTrue(np.allclose(offs[:, 1], 0.10, atol=1e-6))
        d._set_neighbour_tol(None)
        self.assertEqual([a for a in d._persisted_extra if getattr(a, "_gss_ghost", False)], [])

    def test_the_tolerance_decides_whether_a_parallel_neighbour_is_shown(self):
        d = self.dock
        e0, e1, n0 = self._cur_geo()
        self._add_foreign("par", "Nord9", 0, "line", [(e0 + 1.0, n0 + 0.8), (e0 + 3.0, n0 + 0.8)], 0.1)
        d._refresh_persisted_features()
        ghosts = lambda: [a for a in d._persisted_extra if getattr(a, "_gss_ghost", False)]
        d._set_neighbour_tol(0.5)
        self.assertEqual(ghosts(), [])                                    # a 0,8 m: oltre la tolleranza
        d._set_neighbour_tol(1.0)
        self.assertEqual(len(ghosts()), 1)
        xs = np.asarray(ghosts()[0].get_offsets())[:, 0]
        self.assertAlmostEqual(float(xs.min()), 1.0, delta=0.06)          # il tratto parallelo, tutto
        self.assertAlmostEqual(float(xs.max()), 3.0, delta=0.06)

    def test_far_annotations_and_own_profile_are_not_projected(self):
        d = self.dock
        e0, e1, n0 = self._cur_geo()
        self._add_foreign("far", "Nord9", 0, "point", [(e0 + 2.0, n0 + 30.0)], 0.1)
        self._draw("2", [(1.0, 0.05), (2.0, 0.1)])                     # sul profilo mostrato
        d._set_neighbour_tol(2.0)
        self.assertEqual([a for a in d._persisted_extra if getattr(a, "_gss_ghost", False)], [])

    def test_neighbour_depth_follows_the_two_way_time(self):
        d = self.dock
        e0, e1, n0 = self._cur_geo()
        self._add_foreign("p", "Nord9", 0, "point", [(e0 + 2.0, n0 + 0.4)], 0.05, twt=3.0)
        d._refresh_persisted_features()
        d._set_neighbour_tol(1.0)
        ghost = [a for a in d._persisted_extra if getattr(a, "_gss_ghost", False)][0]
        self.assertAlmostEqual(float(np.asarray(ghost.get_offsets())[0][1]), d._twt_to_depth(3.0), places=6)

    def test_neighbours_follow_class_filters_and_other_channels_of_the_same_profile_count(self):
        d = self.dock
        e0, e1, n0 = self._cur_geo()
        stem = d._store.profiles[d._pi].stem
        self._add_foreign("c1", stem, d._ci + 1, "point", [(e0 + 2.0, n0)], 0.1, class_code="iperbole")
        d._refresh_persisted_features()
        d._set_neighbour_tol(1.0)
        self.assertEqual(len([a for a in d._persisted_extra if getattr(a, "_gss_ghost", False)]), 1)
        d._on_show_class("iperbole", False)
        self.assertEqual([a for a in d._persisted_extra if getattr(a, "_gss_ghost", False)], [])

    def test_selecting_an_annotation_highlights_neighbours_with_the_same_code(self):
        d = self.dock
        self._crossing_line(code="M1")
        self._crossing_line(fid="nb2", code="T7")
        self._draw("2", [(1.0, 0.05), (2.0, 0.1)])
        rid = self._last()["id"]
        d._btn_edit.setChecked(True)
        d._select_feature(rid)
        d._ed_code.setText("M1")
        d._on_card_text_finished()
        d._set_neighbour_tol(1.0)
        sizes = sorted(float(np.asarray(a.get_sizes())[0]) for a in d._persisted_extra if getattr(a, "_gss_ghost", False))
        self.assertEqual(len(sizes), 2)
        self.assertLess(sizes[0], sizes[1])                              # quello con lo stesso codice risalta

    def test_neighbour_labels_show_the_code_and_follow_the_labels_switch(self):
        d = self.dock
        self._crossing_line(code="M1")
        d._refresh_persisted_features()
        d._set_neighbour_tol(1.0)
        texts = [a.get_text() for a in d._persisted_extra if hasattr(a, "get_text")]
        self.assertEqual(texts, ["M1"])
        d._on_labels_toggled(False)
        self.assertEqual([a for a in d._persisted_extra if hasattr(a, "get_text")], [])

    def test_neighbour_menu_has_the_four_choices(self):
        d = self.dock
        sub = next(a.menu() for a in d._btn_show.menu().actions() if a.menu() is not None)
        self.assertEqual([a.text() for a in sub.actions()], ["Nascosti", "Entro 0,5 m", "Entro 1 m", "Entro 2 m"])
        self.assertTrue(sub.actions()[0].isChecked())
        sub.actions()[2].trigger()
        self.assertEqual(d._nb_tol, 1.0)

    def test_all_profiles_records_are_cached_until_something_changes(self):
        d = self.dock
        self.queries.clear()
        d._all_feature_records()
        d._all_feature_records()
        self.assertEqual(self.queries.count(True), 1)
        self._draw("2", [(1.0, 0.05), (2.0, 0.1)])                        # una modifica invalida la cache
        d._all_feature_records()
        self.assertGreaterEqual(self.queries.count(True), 2)

    # -- anomalie -------------------------------------------------------------------------

    def _swath10_record(self, fid, code):
        d = self.dock
        pi = next(i for i, p in enumerate(d._store.profiles) if p.stem == "Swath10")
        ch2 = d._store.channel(pi, 0)
        pos = ch2.position(ch2.trace_at_distance(2.0))
        self._add_foreign(fid, "Swath10", 0, "point", [pos], 0.12, code=code, class_code="iperbole")
        return pos

    def test_anomalies_dialog_groups_the_same_code_across_profiles(self):
        d = self.dock
        self._draw("2", [(1.0, 0.05), (2.0, 0.1)])
        rid = self._last()["id"]
        d._btn_edit.setChecked(True)
        d._select_feature(rid)
        d._ed_code.setText("M1")
        d._on_card_text_finished()
        self._swath10_record("s10", "M1")
        self._swath10_record("s10b", "T7")
        d._btn_edit.setChecked(False)                                     # fuori da Modifica: `1` e' una classe
        self._draw("1", [(3.0, 0.1)])                                     # una senza codice
        dlg = d._build_anomalies_dialog()
        self.assertEqual(dlg._table.rowCount(), 2)
        row = [dlg._table.item(0, c).text() for c in range(6)]
        self.assertEqual(row[0], "M1")
        self.assertEqual(row[2:4], ["2", "2"])
        self.assertIn("Riflettore / interfaccia", row[1])
        self.assertIn("Iperbole di diffrazione", row[1])
        self.assertEqual(dlg._table.item(1, 0).text(), "T7")

    def test_anomalies_dialog_goes_to_the_first_annotation(self):
        d = self.dock
        self._swath10_record("s10", "M1")
        dlg = d._build_anomalies_dialog()
        stem_before = d._store.profiles[d._pi].stem
        dlg._go(0)
        self.assertEqual(d._store.profiles[d._pi].stem, "Swath10")
        self.assertNotEqual(stem_before, "Swath10")
        self.assertEqual(d._sel_id, "s10")
        self.assertFalse(d._card.isHidden())

    def test_goto_a_missing_profile_or_annotation_says_so(self):
        d = self.dock
        d._goto_feature("NonEsiste", 0, "x")
        self.assertIn("non presente", d._lbl_status.text())
        stem = d._store.profiles[d._pi].stem
        d._goto_feature(stem, 0, "mancante")
        self.assertIn("non trovata", d._lbl_status.text())

    def test_anomalies_dialog_without_codes_explains_and_exports_nothing(self):
        d = self.dock
        self._draw("1", [(3.0, 0.1)])
        dlg = d._build_anomalies_dialog()
        self.assertEqual(dlg._table.rowCount(), 0)
        self.assertFalse(dlg._btn_csv.isEnabled())

    def test_anomalies_csv_has_one_row_per_code_in_italian_excel_format(self):
        d = self.dock
        self._swath10_record("s10", "M1")
        self._swath10_record("s10b", "M1")
        dlg = d._build_anomalies_dialog()
        path = os.path.join(self.root, "anomalie.csv")
        self.assertEqual(dlg.export_csv_to(path), 1)
        with open(path, "r", encoding="utf-8-sig") as fh:
            lines = fh.read().strip().splitlines()
        self.assertEqual(lines[0].split(";")[:4], ["codice", "classi", "annotazioni", "profili"])
        cells = lines[1].split(";")
        self.assertEqual((cells[0], cells[1], cells[2], cells[3]), ("M1", "Iperbole di diffrazione", "2", "1"))
        self.assertEqual(cells[7], "Swath10")

    def test_card_shows_how_far_the_object_extends_and_suggests_known_codes(self):
        d = self.dock
        self._swath10_record("s10", "M1")
        self._draw("2", [(1.0, 0.05), (2.0, 0.1)])
        rid = self._last()["id"]
        d._btn_edit.setChecked(True)
        d._select_feature(rid)
        d._ed_code.setText("M1")
        d._on_card_text_finished()
        self.assertIn("M1: 2 annot. su 2 profili", d._lbl_card_info.text())
        self.assertIn("M1", d._completer_model.stringList())
        d._select_feature(None)
        d._select_feature(rid)
        self.assertIn("M1: 2 annot.", d._feature_tooltip(d._selected_rec()))

    def test_open_anomalies_shows_a_non_modal_dialog(self):
        d = self.dock
        d._open_anomalies()
        self.assertTrue(d._anomalies_dialog.isVisible())
        self.assertFalse(d._anomalies_dialog.isModal())
        d._anomalies_dialog.close()

    # -- fascia time-slice sulla mappa -------------------------------------------------------

    def test_band_filter_follows_the_selected_timeslice_only_when_switched_on(self):
        d = self.dock
        calls = []
        d._band_filter = calls.append
        d.set_reference_depth(0.1, band=(0.05, 0.15))
        self.assertEqual(calls, [])                                       # spento: la mappa non si tocca
        d._on_band_filter_toggled(True)
        self.assertEqual(calls, [(0.05, 0.15)])
        d.set_reference_depth(0.2, band=(0.15, 0.25))
        self.assertEqual(calls[-1], (0.15, 0.25))
        n = len(calls)
        d.set_reference_depth(0.2, band=(0.15, 0.25))                    # stessa fascia: niente lavoro inutile
        self.assertEqual(len(calls), n)
        d._on_follow_toggled(False)                                       # senza time-slice: nessuna fascia
        self.assertIsNone(calls[-1])
        d._on_band_filter_toggled(False)
        self.assertIsNone(calls[-1])

    def test_band_filter_without_a_timeslice_explains_and_cleanup_removes_it(self):
        d = self.dock
        calls = []
        d._band_filter = calls.append
        d._on_band_filter_toggled(True)
        self.assertEqual(calls, [None])
        self.assertIn("time-slice", d._lbl_status.text())
        d.set_reference_depth(0.1, band=(0.05, 0.15))
        d.cleanup()
        self.assertIsNone(calls[-1])
        self.assertFalse(d._band_filter_on)

    def test_band_filter_errors_do_not_break_the_panel(self):
        d = self.dock

        def boom(_band):
            raise RuntimeError("layer bloccato")
        d._band_filter = boom
        d.set_reference_depth(0.1, band=(0.05, 0.15))
        d._on_band_filter_toggled(True)
        self.assertIn("layer bloccato", d._lbl_status.text())

    def test_new_dock_clears_a_stale_band_filter_and_menu_item_needs_the_bridge(self):
        calls = []
        dock = _DOCK.RadargramDock(self.iface, workspace_provider=lambda: self.root,
                                   parent=self.iface.mainWindow(), band_filter=calls.append)
        try:
            self.assertEqual(calls, [None])
            act = next(a for a in dock._btn_show.menu().actions() if a.text().startswith("Mappa"))
            self.assertTrue(act.isEnabled())
        finally:
            dock.cleanup()
            dock.close()
        act2 = next(a for a in self.dock._btn_show.menu().actions() if a.text().startswith("Mappa"))
        self.assertFalse(act2.isEnabled())                                # nessun ponte: voce spenta

    # -- figura esportata ----------------------------------------------------------------

    def test_export_figure_has_title_legend_and_the_visible_annotations_only(self):
        d = self.dock
        self._draw("2", [(1.0, 0.05), (2.0, 0.1)])
        self._draw("3", [(2.5, 0.14), (3.5, 0.14), (3.5, 0.22)])
        d._on_show_class("attenuazione", False)
        fig = d._build_export_figure()
        ax = fig.axes[0]
        legend = [t.get_text() for t in ax.get_legend().get_texts()]
        self.assertEqual(legend, ["Riflettore / interfaccia"])
        self.assertIn(d._store.profiles[d._pi].stem, ax.get_title())
        self.assertIn(d._store.name, ax.get_title())
        self.assertEqual(ax.get_xlim(), d._ax.get_xlim())
        self.assertEqual(len(ax.lines), 1)
        self.assertEqual(len(ax.patches), 0)

    def test_export_figure_includes_the_selected_annotation_and_the_labels(self):
        d = self.dock
        self._draw("2", [(1.0, 0.05), (2.0, 0.1)])           # appena creata: e' selezionata
        d._ed_code.setText("M1")
        d._on_card_text_finished()
        self.assertIsNotNone(d._selected_rec())
        fig = d._build_export_figure()
        ax = fig.axes[0]
        self.assertEqual(len(ax.lines), 1)                   # la selezionata c'e' anche se e' animata sul pannello
        self.assertEqual([t.get_text() for t in ax.texts], ["M1"])
        d._on_labels_toggled(False)
        self.assertEqual([t.get_text() for t in d._build_export_figure().axes[0].texts], [])

    def test_export_figure_without_annotations_has_no_legend(self):
        fig = self.dock._build_export_figure()
        self.assertIsNone(fig.axes[0].get_legend())

    def test_export_figure_writes_png_pdf_and_svg(self):
        d = self.dock
        self._draw("2", [(1.0, 0.05), (2.0, 0.1)])
        for ext, marker in (("png", b"\x89PNG"), ("pdf", b"%PDF"), ("svg", b"<?xml")):
            path = os.path.join(self.root, f"fig.{ext}")
            self.assertTrue(d._export_figure_to(path, dpi=80))
            with open(path, "rb") as fh:
                head = fh.read(200)
            self.assertTrue(head.startswith(marker) or marker in head, ext)
            self.assertGreater(os.path.getsize(path), 500)

    def test_export_figure_action_uses_the_chosen_path_and_reports(self):
        from unittest import mock
        d = self.dock
        path = os.path.join(self.root, "scelta.png")
        with mock.patch.object(_DOCK.QFileDialog, "getSaveFileName", return_value=(path, "PNG (*.png)")):
            d._export_figure()
        self.assertTrue(os.path.isfile(path))
        self.assertIn("Figura salvata", d._lbl_status.text())
        with mock.patch.object(_DOCK.QFileDialog, "getSaveFileName", return_value=("", "")):
            d._export_figure()                                            # annullato: nessun errore

    def test_export_figure_is_in_the_more_menu_and_nothing_to_export_says_so(self):
        d = self.dock
        self.assertIn("Esporta figura annotata…", [a.text() for a in d._btn_more.menu().actions()])
        d._cur = None
        d._export_figure()
        self.assertIn("Niente da esportare", d._lbl_status.text())


@unittest.skipUnless(HAS_ENV, "QGIS/matplotlib non disponibili")
class ClassesDialogTest(_DockBase):
    def test_dialog_returns_the_same_classes_untouched(self):
        AC = importlib.import_module("rasterlinker_feature_interpolation.gpr_annotation_classes")
        DLG = importlib.import_module("rasterlinker_feature_interpolation.gpr_annotation_classes_dialog")
        classes = AC.default_classes()
        dlg = DLG.AnnotationClassesDialog(classes)
        self.assertEqual(dlg.classes(), classes)

    def test_new_and_renamed_classes_keep_stable_codes(self):
        AC = importlib.import_module("rasterlinker_feature_interpolation.gpr_annotation_classes")
        DLG = importlib.import_module("rasterlinker_feature_interpolation.gpr_annotation_classes_dialog")
        dlg = DLG.AnnotationClassesDialog(AC.default_classes())
        dlg._table.cellWidget(0, 0).setText("Iperbole (rinominata)")
        dlg._add()
        out = dlg.classes()
        self.assertEqual(out[0]["code"], "iperbole")
        self.assertEqual(out[0]["label"], "Iperbole (rinominata)")
        self.assertEqual(out[-1]["label"], "Nuova classe")
        self.assertEqual(out[-1]["code"], "nuova_classe")

    def test_blank_rows_are_dropped_and_the_last_class_cannot_be_removed(self):
        AC = importlib.import_module("rasterlinker_feature_interpolation.gpr_annotation_classes")
        DLG = importlib.import_module("rasterlinker_feature_interpolation.gpr_annotation_classes_dialog")
        dlg = DLG.AnnotationClassesDialog(AC.default_classes()[:2])
        dlg._table.cellWidget(1, 0).setText("   ")
        self.assertEqual(len(dlg.classes()), 1)
        dlg._table.setCurrentCell(0, 0)
        dlg._remove()
        dlg._table.setCurrentCell(0, 0)
        dlg._remove()
        self.assertGreaterEqual(dlg._table.rowCount(), 1)

    def test_add_stops_at_the_maximum_and_move_reorders(self):
        AC = importlib.import_module("rasterlinker_feature_interpolation.gpr_annotation_classes")
        DLG = importlib.import_module("rasterlinker_feature_interpolation.gpr_annotation_classes_dialog")
        dlg = DLG.AnnotationClassesDialog(AC.default_classes())
        for _ in range(AC.MAX_CLASSES + 3):
            dlg._add()
        self.assertEqual(dlg._table.rowCount(), AC.MAX_CLASSES)
        self.assertFalse(dlg._btn_add.isEnabled())
        dlg._table.setCurrentCell(1, 0)
        first_two = [c["code"] for c in dlg.classes()[:2]]
        dlg._move(-1)
        self.assertEqual([c["code"] for c in dlg.classes()[:2]], first_two[::-1])


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
        ctrl = d.widget().layout().itemAt(0).widget()           # riga dei dati
        h_wide = ctrl.layout().heightForWidth(1400)
        h_narrow = ctrl.layout().heightForWidth(400)
        self.assertGreater(h_narrow, h_wide)                              # stretto: piu' righe
        d.setFloating(True)
        d.resize(420, 500)
        self._pump(80)
        cw = ctrl.width()
        for g in ctrl.findChildren(type(ctrl), options=Qt.FindDirectChildrenOnly):
            if g.isHidden():
                continue                                                  # strumenti non attivi, area unica
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


@unittest.skipUnless(HAS_ENV, "QGIS/matplotlib non disponibili")
class ZonesTest(unittest.TestCase):
    """Piu' aree indagate nello stesso progetto: ognuna ha le sue varianti e i suoi profili."""

    @classmethod
    def setUpClass(cls):
        from qgis.PyQt.QtWidgets import QApplication
        cls._qapp = QApplication.instance()

    def setUp(self):
        _DockBase._forget_saved_look()
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = self._tmp.name
        crs = QgsCoordinateReferenceSystem("EPSG:32632")
        QgsProject.instance().setCrs(crs)
        self.iface = _Iface()
        self.iface.mapCanvas().setDestinationCrs(crs)
        self.sud = os.path.join(self.root, "sud")
        self.nord = os.path.join(self.root, "nord")
        self.v_sud, _ = _write_variant(self.sud, "Lavoro", ["Sud1", "Sud2", "Sud3"])
        self.v_nord, _ = _write_variant(self.nord, "Lavoro", ["Nord1", "Nord2"])
        self.v_nord2, _ = _write_variant(self.nord, "v3", ["Nord1", "Nord2"])
        self.active = "Sud"
        self.dock = _DOCK.RadargramDock(
            self.iface, workspace_provider=lambda: os.path.join(self.sud, "Processing"),
            zones_provider=self._zones, geometry_provider=self._geom,
            parent=self.iface.mainWindow())
        self.iface.addDockWidget(Qt.BottomDockWidgetArea, self.dock)
        self.iface.mainWindow().show()
        self.dock.show()

    def tearDown(self):
        self.dock.cleanup()
        self.dock.close()
        self.iface.mainWindow().close()
        gc.collect()
        self._tmp.cleanup()
        _DockBase._forget_saved_look()

    def _zones(self):
        return [
            {"name": "Sud", "base": os.path.join(self.sud, "Processing"), "active": self.active == "Sud"},
            {"name": "Nord", "base": os.path.join(self.nord, "Processing"), "active": self.active == "Nord"},
            {"name": "Vuota", "base": os.path.join(self.root, "vuota"), "active": False},
        ]

    def _geom(self, stem):
        self.geom_asked = getattr(self, "geom_asked", []) + [stem]
        return None

    def test_every_area_with_data_is_selectable_and_empty_ones_are_left_out(self):
        d = self.dock
        self.assertFalse(d._grp_zone.isHidden())
        self.assertEqual([d._cb_zone.itemText(i) for i in range(d._cb_zone.count())], ["Sud", "Nord"])
        self.assertEqual(d._cb_zone.currentText(), "Sud")          # parte dall'area attiva
        self.assertEqual(d._cb_profile.count(), 3)

    def test_choosing_another_area_shows_its_variants_and_profiles(self):
        d = self.dock
        d._cb_zone.setCurrentIndex(d._cb_zone.findData("Nord"))
        self.assertEqual(d._cb_variant.count(), 2)                  # Lavoro + v3
        self.assertEqual(d._store.path, os.path.normpath(self.v_nord))
        self.assertEqual(d._cb_profile.count(), 2)
        self.assertIn("Nord1", d._cb_profile.itemText(0))
        self.assertIn("Nord / Lavoro", d._lbl_status.text())

    def test_open_variant_by_path_in_another_area_switches_area(self):
        d = self.dock
        self.assertTrue(d.open_variant(self.v_nord2, profile_stem="Nord2"))
        self.assertEqual(d._cb_zone.currentText(), "Nord")
        self.assertEqual(d._store.name, "v3")
        self.assertEqual(d._store.profiles[d._pi].stem, "Nord2")
        self.assertTrue(d.open_variant(self.v_sud))
        self.assertEqual(d._cb_zone.currentText(), "Sud")
        self.assertEqual(len(d._store), 3)

    def test_geometry_from_viewer_only_for_the_area_it_holds(self):
        d = self.dock
        self.geom_asked = []
        d._store.invalidate_geometry()
        d._load_current(reset_view=True)
        self.assertTrue(self.geom_asked)                             # area attiva: chiede al viewer
        d._cb_zone.setCurrentIndex(d._cb_zone.findData("Nord"))
        self.geom_asked = []
        d._store.invalidate_geometry()
        d._load_current(reset_view=True)
        self.assertEqual(self.geom_asked, [])                        # altra area: fotografia dei .npz
        self.assertEqual(d._cur.geometry_source, "npz")

    def test_follows_the_area_made_active_in_the_viewer(self):
        d = self.dock
        self.active = "Nord"
        d.refresh_variants()
        self.assertEqual(d._cb_zone.currentText(), "Nord")
        self.assertEqual(d._store.path, os.path.normpath(self.v_nord))
        d._cb_zone.setCurrentIndex(d._cb_zone.findData("Sud"))     # sfoglio un'altra area a mano...
        d.refresh_variants()                                         # ...e un refresh non mi riporta indietro
        self.assertEqual(d._cb_zone.currentText(), "Sud")


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
                fake._radargram_project_root = lambda: GS.GeoSurveyStudioPlugin._radargram_project_root(fake)
                fake._radargram_workspace_root = lambda: GS.GeoSurveyStudioPlugin._radargram_workspace_root(fake)
                fake._live_profile_viewer = lambda: GS.GeoSurveyStudioPlugin._live_profile_viewer(fake)
                fake._radargram_geometry_provider = lambda s: GS.GeoSurveyStudioPlugin._radargram_geometry_provider(fake, s)
                fake._radargram_feature_sink = lambda **kw: None
                fake._radargram_feature_query = lambda *a, **kw: []
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


# ---------------------------------------------------------------------------------------------
# Sezione ERT sotto il radargramma e dialogo d'importazione
# ---------------------------------------------------------------------------------------------
if HAS_ENV:
    _G = importlib.import_module("rasterlinker_feature_interpolation.gpr_ert_georef")
    _S = importlib.import_module("rasterlinker_feature_interpolation.gpr_ert_store")
    _D = importlib.import_module("rasterlinker_feature_interpolation.gpr_ert_import_dialog")


def _ert_csv(path, length=10.0):
    """Celle 1 m x 0.5 m; log10(res) = 1 + x/10 (dipende solo da x: rivela gli spostamenti)."""
    lines = ["x,y,z,Resistivity(ohm.m),Resistivity(log10)"]
    for xi in np.arange(0.5, length, 1.0):
        for zi in np.arange(-0.25, -3.0, -0.5):
            lg = 1.0 + xi / 10.0
            lines.append(f"{xi},0.0,{zi},{10 ** lg},{lg}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


@unittest.skipUnless(HAS_ENV, "QGIS/matplotlib non disponibili")
class ErtViewTest(_DockBase):
    def setUp(self):
        super().setUp()
        self.dock._project_root_provider = lambda: self.root
        csv = os.path.join(self.root, "export.csv")
        _ert_csv(csv)
        # Il profilo 0 (Swath2) sta a nord = 10 m, da est 500000 a 500004.95. La linea ERT corre lungo
        # lo stesso nord da 499998 a 500008: la traccia GPR a x=0 cade a 2 m lungo la linea ERT.
        georef = _G.ErtGeoref.from_endpoints((499998.0, 10.0), (500008.0, 10.0), 10.0, "EPSG:32632")
        _S.save_ert(self.root, "T", csv, georef)

    def _show(self):
        self.dock._act_ert_show.setChecked(True)
        self._pump(30)
        return self.dock._ert_panel

    def test_hidden_until_requested(self):
        self.assertFalse(self.dock._ert_panel.isVisible())
        self.assertFalse(self.dock._act_ert_show.isChecked())

    def test_listed_selected_and_drawn(self):
        p = self._show()
        self.assertTrue(p.isVisible())
        self.assertEqual(p.current_name(), "T")
        self.assertIsNotNone(p._grid)

    def test_section_is_shifted_onto_the_gpr_distance(self):
        p = self._show()
        # x GPR = 3 m  ->  3 m + 2 m = 5 m lungo l'ERT  ->  log10 = 1.5
        v = p.value_at(3.0, 1.0)
        self.assertIsNotNone(v)
        self.assertAlmostEqual(v, 1.5, delta=0.06)
        # x GPR = 0.5 m -> 2.5 m lungo l'ERT
        self.assertAlmostEqual(p.value_at(0.5, 1.0), 1.25, delta=0.06)

    def test_only_the_gpr_stretch_is_drawn(self):
        p = self._show()
        xs = p._grid[0]
        self.assertLessEqual(float(xs[0]), 0.2)
        self.assertGreaterEqual(float(xs[-1]), self.dock._cur.length_m - 0.2)
        self.assertLessEqual(float(xs[-1]), self.dock._cur.length_m + 1e-6)

    def test_zoom_and_pan_follow_the_radargram(self):
        p = self._show()
        self.dock._set_view(1.0, 3.0, 0.2, 0.0)           # la profondita' cresce verso il basso
        self._pump(10)
        self.assertEqual(tuple(round(v, 6) for v in p._ax.get_xlim()), (1.0, 3.0))
        self.assertAlmostEqual(p._ax.get_ylim()[0], 0.2)
        self.assertAlmostEqual(p._ax.get_ylim()[1], 0.0)

    def test_depth_follow_can_be_switched_off(self):
        p = self._show()
        p._chk_depth.setChecked(False)
        self._pump(10)
        self.dock._set_view(1.0, 3.0, 0.2, 0.0)
        self.assertGreater(p._ax.get_ylim()[0], 0.2)

    def test_changing_profile_redraws_with_the_new_offset(self):
        p = self._show()
        self.dock.select_profile(1)             # nord 0: a 10 m dalla linea ERT
        self._pump(30)
        self.assertIn("10.0 m", p._lbl.text())
        self.dock.select_profile(0)
        self._pump(30)
        self.assertIn("0.0 m", p._lbl.text())

    def test_label_declares_the_flat_mode(self):
        p = self._show()
        self.assertIn("piana", p._lbl.text())

    def test_hiding_leaves_the_radargram_alone(self):
        self._show()
        self.dock._act_ert_show.setChecked(False)
        self.assertFalse(self.dock._ert_panel.isVisible())
        self.dock._set_view(1.0, 3.0, 0.0, 0.2)       # non deve sollevare
        self.assertEqual(tuple(round(v, 6) for v in self.dock._ax.get_xlim()), (1.0, 3.0))

    def test_no_sections_in_the_project(self):
        import shutil
        shutil.rmtree(os.path.join(self.root, "ERT"))
        p = self._show()
        self.assertEqual(p.current_name(), "")
        self.assertIsNone(p._grid)

    def test_profile_far_from_the_line_says_so(self):
        p = self._show()
        # sposta la linea ERT 500 m a nord: nessuna sovrapposizione utile
        csv = os.path.join(self.root, "export.csv")
        far = _G.ErtGeoref.from_endpoints((499998.0, 510.0), (500008.0, 510.0), 10.0, "EPSG:32632")
        _S.save_ert(self.root, "T", csv, far)
        self.dock._ert_cache.clear()
        self.dock._ert_refresh()
        self._pump(30)
        self.assertIn("500.0 m", p._lbl.text())


_KML = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
<Placemark><name>P9</name><LineString><coordinates>
2.455219060577314,36.59265013784221,0 2.455281505640994,36.59328796836483,0</coordinates></LineString></Placemark>
<Placemark><name>P8</name><LineString><coordinates>
2.455374535147292,36.59284739722405,0 2.454614928681824,36.5930129875788,0</coordinates></LineString></Placemark>
</Document></kml>"""


@unittest.skipUnless(HAS_ENV, "QGIS/matplotlib non disponibili")
class ImportDialogTest(_DockBase):
    def setUp(self):
        super().setUp()
        self.csv = os.path.join(self.root, "P9_inverted1.csv")
        _ert_csv(self.csv, length=71.0)
        self.kml = os.path.join(self.root, "doc.kml")
        with open(self.kml, "w", encoding="utf-8") as fh:
            fh.write(_KML)

    def _dialog(self):
        dlg = _D.ErtImportDialog("EPSG:32631", self.iface.mainWindow(), start_dir=self.root)
        dlg._ed_csv.setText(self.csv)
        dlg._on_csv()
        self._pump(10)
        return dlg

    def test_finds_doc_kml_next_to_the_file_and_guesses_the_profile(self):
        dlg = self._dialog()
        self.assertEqual(dlg._ed_kml.text(), self.kml)
        self.assertEqual([dlg._cb_profile.itemText(i) for i in range(dlg._cb_profile.count())], ["P9", "P8"])
        self.assertEqual(dlg._cb_profile.currentText(), "P9")      # il nome del file contiene "P9"
        self.assertEqual(dlg._ed_name.text(), "P9_inverted1")

    def test_builds_a_two_point_georef_in_the_project_crs(self):
        dlg = self._dialog()
        g = dlg.result_data()["georef"]
        self.assertIsNotNone(g)
        self.assertEqual(g.crs_authid, "EPSG:32631")
        self.assertEqual(len(g.points), 2)
        self.assertAlmostEqual(g.map_length(), 71.4, delta=0.6)
        self.assertAlmostEqual(dlg._spin_len.value(), 71.0, delta=0.6)
        self.assertEqual(g.check(), [])
        self.assertTrue(dlg._buttons.button(dlg._buttons.Ok).isEnabled())

    def test_invert_swaps_the_ends(self):
        dlg = self._dialog()
        a = dlg.result_data()["georef"].points
        dlg._chk_invert.setChecked(True)
        b = dlg.result_data()["georef"].points
        self.assertAlmostEqual(a[0].x, b[1].x, places=3)
        self.assertAlmostEqual(a[0].y, b[1].y, places=3)

    def test_wrong_length_warns_but_does_not_block(self):
        dlg = self._dialog()
        dlg._spin_len.setValue(50.0)
        self.assertIn("⚠", dlg._lbl_info.text())
        self.assertTrue(dlg._buttons.button(dlg._buttons.Ok).isEnabled())

    def test_needs_a_name(self):
        dlg = self._dialog()
        dlg._ed_name.setText("")
        dlg._update()
        self.assertFalse(dlg._buttons.button(dlg._buttons.Ok).isEnabled())

    def test_bad_file_disables_ok(self):
        dlg = _D.ErtImportDialog("EPSG:32631", self.iface.mainWindow())
        bad = os.path.join(self.root, "bad.csv")
        with open(bad, "w") as fh:
            fh.write("a,b\n1,2\n")
        dlg._ed_csv.setText(bad)
        dlg._on_csv()
        self.assertFalse(dlg._buttons.button(dlg._buttons.Ok).isEnabled())
        self.assertIn("non valido", dlg._lbl_info.text())


@unittest.skipUnless(HAS_ENV, "QGIS/matplotlib non disponibili")
class ImportDialogManualTest(_DockBase):
    """Georiferimento a mano: due punti (scritti o cliccati) e punti da un layer."""

    def setUp(self):
        super().setUp()
        from qgis.core import QgsFeature, QgsGeometry, QgsVectorLayer
        self._QgsFeature, self._QgsGeometry, self._QgsVectorLayer = QgsFeature, QgsGeometry, QgsVectorLayer
        self.csv = os.path.join(self.root, "P9_inverted1.csv")
        _ert_csv(self.csv, length=71.0)
        self._mem_layers = []

    def tearDown(self):
        if self._mem_layers:
            QgsProject.instance().removeMapLayers([lyr.id() for lyr in self._mem_layers])
        super().tearDown()

    def _dialog(self, crs="EPSG:32631", canvas=None, mode=None):
        dlg = _D.ErtImportDialog(crs, self.iface.mainWindow(), start_dir=self.root, canvas=canvas)
        dlg._ed_csv.setText(self.csv)
        dlg._on_csv()
        if mode is not None:
            dlg._cb_mode.setCurrentIndex(mode)
        self._pump(10)
        return dlg

    def _points_layer(self, rows, crs="EPSG:32631", name="elettrodi"):
        lyr = self._QgsVectorLayer(f"Point?crs={crs}&field=dist:double&field=etichetta:string", name, "memory")
        feats = []
        for dist, x, y in rows:
            f = self._QgsFeature(lyr.fields())
            f.setGeometry(self._QgsGeometry.fromPointXY(QgsPointXY(x, y)))
            f["dist"], f["etichetta"] = dist, "e"
            feats.append(f)
        lyr.dataProvider().addFeatures(feats)
        QgsProject.instance().addMapLayer(lyr)
        self._mem_layers.append(lyr)
        return lyr

    def _ok(self, dlg):
        return dlg._buttons.button(dlg._buttons.Ok).isEnabled()

    # -- due punti sulla mappa ----------------------------------------------------------------
    def test_two_typed_points_make_a_line(self):
        dlg = self._dialog(mode=1)
        dlg._ed_p[0].setText("100; 200")
        dlg._ed_p[1].setText("100,0; 271,0")
        dlg._update()
        g = dlg.result_data()["georef"]
        self.assertEqual(len(g.points), 2)
        self.assertAlmostEqual(g.map_length(), 71.0)
        self.assertEqual(g.check(), [])
        self.assertTrue(self._ok(dlg))

    def test_missing_point_asks_for_it(self):
        dlg = self._dialog(mode=1)
        dlg._ed_p[0].setText("100; 200")
        dlg._update()
        self.assertFalse(self._ok(dlg))
        self.assertIn("due punti", dlg._lbl_info.text())

    def test_wrong_distance_between_points_warns(self):
        dlg = self._dialog(mode=1)
        dlg._ed_p[0].setText("0; 0")
        dlg._ed_p[1].setText("0; 90")
        dlg._update()
        self.assertIn("⚠", dlg._lbl_info.text())

    def test_map_click_fills_the_field_in_the_project_crs(self):
        canvas = self.iface.mapCanvas()                       # EPSG:32632
        dlg = self._dialog(crs="EPSG:32631", canvas=canvas, mode=1)
        dlg._on_map_click(0, QgsPointXY(500000.0, 4000000.0))
        from qgis.core import QgsCoordinateReferenceSystem, QgsCoordinateTransform
        want = QgsCoordinateTransform(QgsCoordinateReferenceSystem("EPSG:32632"),
                                      QgsCoordinateReferenceSystem("EPSG:32631"),
                                      QgsProject.instance().transformContext()).transform(
            QgsPointXY(500000.0, 4000000.0))
        got = _G.parse_xy(dlg._ed_p[0].text())
        self.assertAlmostEqual(got[0], want.x(), places=2)
        self.assertAlmostEqual(got[1], want.y(), places=2)

    def test_pick_hides_the_dialog_and_restores_tool_and_dialog(self):
        canvas = self.iface.mapCanvas()
        before = canvas.mapTool()
        dlg = self._dialog(crs="EPSG:32632", canvas=canvas, mode=1)
        dlg.show()
        dlg._btn_pick[1].click()
        self._pump(10)
        self.assertFalse(dlg.isVisible())
        self.assertIs(canvas.mapTool(), dlg._tool)
        dlg._tool.canvasClicked.emit(QgsPointXY(500010.0, 20.0), Qt.LeftButton)
        self._pump(10)
        self.assertTrue(dlg.isVisible())
        self.assertIs(canvas.mapTool(), before)
        self.assertEqual(_G.parse_xy(dlg._ed_p[1].text()), (500010.0, 20.0))
        dlg.close()

    def test_changing_tool_while_picking_brings_the_dialog_back(self):
        canvas = self.iface.mapCanvas()
        dlg = self._dialog(crs="EPSG:32632", canvas=canvas, mode=1)
        dlg.show()
        dlg._btn_pick[0].click()
        self._pump(10)
        self.assertFalse(dlg.isVisible())
        canvas.unsetMapTool(dlg._tool)
        self._pump(10)
        self.assertTrue(dlg.isVisible())
        dlg.close()

    def test_pick_buttons_disabled_without_canvas(self):
        dlg = self._dialog(mode=1)
        self.assertFalse(dlg._btn_pick[0].isEnabled())

    # -- layer ------------------------------------------------------------------------------------
    def test_layer_with_distance_field_anchors_each_point(self):
        self._points_layer([(0.0, 100.0, 200.0), (10.0, 110.0, 200.0), (25.0, 125.0, 200.0),
                            (71.0, 171.0, 200.0)])
        dlg = self._dialog(mode=2)
        self.assertEqual(dlg._cb_layer.count(), 1)
        i = dlg._cb_dist.findData("dist")
        self.assertGreaterEqual(i, 0)
        self.assertEqual(dlg._cb_dist.findData("etichetta"), -1)     # i campi di testo non si offrono
        dlg._cb_dist.setCurrentIndex(i)
        g = dlg.result_data()["georef"]
        self.assertEqual([round(p.distance, 3) for p in g.points], [0.0, 10.0, 25.0, 71.0])
        self.assertEqual(g.check(), [])
        self.assertTrue(self._ok(dlg))
        self.assertFalse(dlg._chk_invert.isEnabled())

    def test_layer_without_field_uses_proportional_distance(self):
        self._points_layer([(0, 100.0, 200.0), (0, 130.0, 200.0), (0, 130.0, 240.0)])
        dlg = self._dialog(mode=2)
        g = dlg.result_data()["georef"]
        length = dlg._spin_len.value()                  # proposta dalla sezione sintetica (70.5 m)
        self.assertAlmostEqual(g.points[-1].distance, length)
        self.assertAlmostEqual(g.points[1].distance, 30.0 * length / 70.0)
        self.assertTrue(dlg._chk_invert.isEnabled())

    def test_layer_invert_reverses_the_order(self):
        self._points_layer([(0, 100.0, 200.0), (0, 130.0, 200.0), (0, 130.0, 240.0)])
        dlg = self._dialog(mode=2)
        dlg._chk_invert.setChecked(True)
        g = dlg.result_data()["georef"]
        self.assertEqual((g.points[0].x, g.points[0].y), (130.0, 240.0))

    def test_only_selected_features(self):
        lyr = self._points_layer([(0.0, 0.0, 0.0), (10.0, 10.0, 0.0), (20.0, 20.0, 0.0), (30.0, 30.0, 0.0)])
        ids = [f.id() for f in lyr.getFeatures()]
        lyr.selectByIds(ids[1:3])
        dlg = self._dialog(mode=2)
        dlg._cb_dist.setCurrentIndex(dlg._cb_dist.findData("dist"))
        dlg._chk_selected.setChecked(True)
        g = dlg.result_data()["georef"]
        self.assertEqual([round(p.distance, 3) for p in g.points], [10.0, 20.0])

    def test_layer_in_another_crs_is_reprojected(self):
        from qgis.core import QgsCoordinateReferenceSystem, QgsCoordinateTransform
        self._points_layer([(0, 500000.0, 4000000.0), (0, 500070.0, 4000000.0)], crs="EPSG:32632")
        dlg = self._dialog(crs="EPSG:32631", mode=2)
        g = dlg.result_data()["georef"]
        want = QgsCoordinateTransform(QgsCoordinateReferenceSystem("EPSG:32632"),
                                      QgsCoordinateReferenceSystem("EPSG:32631"),
                                      QgsProject.instance().transformContext()).transform(
            QgsPointXY(500000.0, 4000000.0))
        self.assertAlmostEqual(g.points[0].x, want.x(), places=2)
        self.assertAlmostEqual(g.points[0].y, want.y(), places=2)

    def test_layer_with_a_single_point_is_refused(self):
        self._points_layer([(0, 0.0, 0.0)])
        dlg = self._dialog(mode=2)
        self.assertFalse(self._ok(dlg))
        self.assertIn("almeno 2", dlg._lbl_info.text())

    def test_no_point_layers_in_the_project(self):
        dlg = self._dialog(mode=2)
        self.assertFalse(self._ok(dlg))
        self.assertIn("layer di punti", dlg._lbl_info.text())

    def test_duplicate_distances_are_reported_not_raised(self):
        self._points_layer([(5.0, 0.0, 0.0), (5.0, 10.0, 0.0)])
        dlg = self._dialog(mode=2)
        dlg._cb_dist.setCurrentIndex(dlg._cb_dist.findData("dist"))
        self.assertFalse(self._ok(dlg))
        self.assertIn("stessa distanza", dlg._lbl_info.text())


@unittest.skipUnless(HAS_ENV, "QGIS/matplotlib non disponibili")
class FenceDockTest(_DockBase):
    """Vista 3D dei profili: GPR sintetici (nord 0 e 10, est 500000..500004.95) incrociati da una linea ERT lungo y."""

    def setUp(self):
        super().setUp()
        self.dock._project_root_provider = lambda: self.root
        csv = os.path.join(self.root, "export.csv")
        _ert_csv(csv, length=40.0)
        # linea ERT ortogonale ai profili GPR: x = 500002, da y = -20 a y = +20
        g = _G.ErtGeoref.from_endpoints((500002.0, -20.0), (500002.0, 20.0), 40.0, "EPSG:32632")
        _S.save_ert(self.root, "T", csv, g)
        try:
            import pyvista as pv
            from qgis.PyQt.QtWidgets import QLabel
        except ImportError:
            self.skipTest("pyvista non disponibile in questo interprete")
        # niente QtInteractor: sotto la piattaforma Qt offscreen non esiste un contesto OpenGL e VTK va in crash
        self.dock._fence_plotter_factory = lambda parent: (pv.Plotter(off_screen=True, window_size=(600, 400)),
                                                          QLabel("3D", parent))
        self.dock._act_fence.trigger()
        self._pump(50)
        self.fence = self.dock._fence
        if self.fence is None or self.fence._plotter is None:
            self.skipTest("pyvista/pyvistaqt non disponibili in questo interprete")

    def tearDown(self):
        super().tearDown()

    def _build(self, gpr=(), ert=()):
        self.fence.set_checked("gpr", gpr)
        self.fence.set_checked("ert", ert)
        self.fence._timer.stop()
        self.fence._rebuild()

    def test_lists_gpr_profiles_and_ert_sections(self):
        names_gpr = [t for t, _ in self.fence.checked("gpr")]
        self.assertEqual(names_gpr, [])
        top = self.fence._tree.topLevelItem(0)
        self.assertEqual(top.childCount(), 2)
        self.assertEqual(self.fence._tree.topLevelItem(1).childCount(), 1)
        self.assertEqual(self.fence._tree.topLevelItem(1).child(0).text(0), "T")

    def test_checked_profiles_become_curtains(self):
        self._build(gpr=["Swath2", "Swath10"], ert=["T"])
        shown = self.fence._shown
        self.assertIn("gpr::Swath2", shown)
        self.assertIn("gpr::Swath10", shown)
        self.assertIn("ert::T", shown)
        self.assertEqual(sum(1 for v in shown.values() if v == "x"), 3)       # 2 incroci + etichette

    def test_crossings_are_listed_with_both_distances(self):
        self._build(gpr=["Swath10"], ert=["T"])
        text = self.fence._txt_cross.toPlainText()
        self.assertIn("T × Swath10", text)
        self.assertIn("ERT a 20.0 m", text)             # y = 0 sta a 20 m dall'inizio della linea ERT
        self.assertIn("GPR a 2.0 m", text)              # x = 500002 sta a 2 m dall'inizio del profilo

    def test_parallel_profiles_do_not_cross(self):
        self._build(gpr=["Swath2"], ert=[])
        self.assertEqual(self.fence._txt_cross.toPlainText(), "")

    def test_unchecking_removes_the_actors(self):
        self._build(gpr=["Swath2"], ert=["T"])
        self._build(gpr=[], ert=[])
        self.assertEqual(self.fence._shown, {})

    def test_depth_limit_and_labels_toggle(self):
        self.fence._spin_depth.setValue(0.1)
        self.fence._chk_labels.setChecked(False)
        self._build(gpr=["Swath2"], ert=["T"])
        self.assertNotIn("labels", self.fence._shown)
        self.assertIn("ert::T", self.fence._shown)

    def test_ert_alone_and_gpr_alone(self):
        self._build(ert=["T"])
        self.assertIn("ert::T", self.fence._shown)
        self._build(gpr=["Swath10"])
        self.assertNotIn("ert::T", self.fence._shown)

    def test_importing_an_ert_refreshes_the_list(self):
        csv = os.path.join(self.root, "export.csv")
        g = _G.ErtGeoref.from_endpoints((500003.0, -20.0), (500003.0, 20.0), 40.0, "EPSG:32632")
        _S.save_ert(self.root, "T2", csv, g)
        self.dock._fence_refresh()
        self.assertEqual(self.fence._tree.topLevelItem(1).childCount(), 2)

    def test_checks_survive_a_refresh(self):
        self.fence.set_checked("gpr", ["Swath2"])
        self.fence.refresh_lists()
        self.assertEqual([t for t, _ in self.fence.checked("gpr")], ["Swath2"])

    def test_dock_cleanup_closes_the_fence(self):
        self.dock.cleanup()
        self.assertIsNone(self.dock._fence)
        self.assertIsNone(self.fence._plotter)

    def test_screenshot_is_not_blank(self):
        self._build(gpr=["Swath2", "Swath10"], ert=["T"])
        img = self.fence._plotter.screenshot(return_img=True)
        self.assertGreater(int(np.asarray(img).std()), 5)


if __name__ == "__main__":
    unittest.main()
