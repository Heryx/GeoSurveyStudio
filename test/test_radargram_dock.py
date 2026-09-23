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

    def setActiveLayer(self, layer):
        self._active = layer

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


def _write_sliceset(root, variant, slices):
    """Piu' time-slice nello stesso gruppo/sliceset.json: slices = [(grid, z0, z1), ...]."""
    folder = os.path.join(root, "GPR_Processing", "timeslices", variant)
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

    # -- velocita' per materiale ---------------------------------------------------------

    def test_material_preset_combo_writes_override_with_preset_source(self):
        d = self.dock
        VS = importlib.import_module("rasterlinker_feature_interpolation.gpr_velocity_store")
        MP = importlib.import_module("rasterlinker_feature_interpolation.gpr_material_presets")
        stem = d._store.profiles[d._pi].stem
        idx = d._cb_vel_material.findData("acqua_dolce")
        self.assertGreaterEqual(idx, 0)
        d._cb_vel_material.setCurrentIndex(idx)
        got = VS.resolve_velocity_m_ns(d._store.path, stem)
        self.assertEqual(got["source"], "preset:acqua_dolce")
        preset = MP.material_preset("acqua_dolce")
        self.assertAlmostEqual(got["velocity_m_ns"], preset["velocity_m_ns"], places=6)
        self.assertAlmostEqual(d._spin_vel_mns.value(), preset["velocity_m_ns"], places=4)

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


if __name__ == "__main__":
    unittest.main()
