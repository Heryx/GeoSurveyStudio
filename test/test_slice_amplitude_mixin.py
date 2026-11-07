# -*- coding: utf-8 -*-
"""Scheda Ampiezze del pannello principale: applica/salva/ripristina su layer
QGIS veri e ripristino alla riapertura. Serve il Python di QGIS (offscreen)."""

import hashlib
import importlib
import os
import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_PLUGIN_PARENT = _HERE.parents[1].parent
for _p in (str(_PLUGIN_PARENT), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PKG = "rasterlinker_feature_interpolation"
try:
    from qgis.core import QgsApplication, QgsProject, QgsRasterLayer
    from qgis.PyQt.QtCore import Qt
    from qgis.PyQt.QtWidgets import QApplication, QListWidget, QListWidgetItem, QWidget
    from osgeo import gdal
    HAS_QGIS = True
except Exception:  # pragma: no cover
    HAS_QGIS = False

if HAS_QGIS:
    amp = importlib.import_module(PKG + ".slice_amplitude")
    mixin_mod = importlib.import_module(PKG + ".slice_amplitude_mixin")
    from test_slice_amplitude import _noisy, _write_tif

    _app = QgsApplication.instance() or QgsApplication([], True)
    if QgsApplication.instance() is _app and not getattr(_app, "_rl_inited", False):
        _app.initQgis()
        _app._rl_inited = True


class _Msg:
    def pushInfo(self, *a):
        pass


class _Host(mixin_mod.SliceAmplitudeMixin if HAS_QGIS else object):
    def __init__(self, root):
        self._root = root
        self.dlg = SimpleNamespace(rasterListWidget=QListWidget())
        self.iface = SimpleNamespace(messageBar=lambda: _Msg())
        self.limits = []

    def _active_project_root(self):
        return self._root

    def _apply_value_range_to_layer(self, layer, mn, mx, **kw):
        self.limits.append((layer.name(), mn, mx))
        return True

    def add_item(self, path):
        it = QListWidgetItem(os.path.basename(path))
        it.setData(Qt.UserRole, {"project_path": path})
        self.dlg.rasterListWidget.addItem(it)
        return it


@unittest.skipUnless(HAS_QGIS, "QGIS non disponibile")
class MixinTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.src = os.path.join(self.root, "s_0.50.tif")
        _write_tif(self.src, _noisy())
        with open(self.src, "rb") as fh:
            self.md5 = hashlib.md5(fh.read()).hexdigest()
        QgsProject.instance().clear()
        self.layer = QgsRasterLayer(self.src, "s_0.50")
        self.assertTrue(self.layer.isValid())
        QgsProject.instance().addMapLayer(self.layer)
        self.host = _Host(self.root)
        self.parent_widget = QWidget()          # tiene vivi i figli (altrimenti Qt li distrugge)
        self.tab = self.host._build_amplitude_tab(self.parent_widget)
        self.host.add_item(self.src)
        self.host.dlg.rasterListWidget.setCurrentRow(0)

    def tearDown(self):
        QgsProject.instance().clear()
        self.tmp.cleanup()

    def _cur(self):
        lyrs = [l for l in QgsProject.instance().mapLayers().values() if l.name() == "s_0.50"]
        self.assertEqual(len(lyrs), 1)          # mai duplicati
        return lyrs[0]

    def _src(self):
        return os.path.normcase(os.path.normpath(self._cur().source().split("|")[0]))

    def test_apply_saves_and_swaps_without_touching_original(self):
        h = self.host
        h._amp_cb_despeckle.setCurrentIndex(h._amp_cb_despeckle.findData(3))
        h._amp_on_apply()
        saved = amp.get_settings(self.root, self.src)
        self.assertEqual(saved.despeckle, 3)
        self.assertEqual(self._src(), os.path.normcase(os.path.normpath(amp.derived_path(self.root, self.src))))
        self.assertEqual(self._cur().customProperty("rl_amp_src"), self.src)
        with open(self.src, "rb") as fh:
            self.assertEqual(hashlib.md5(fh.read()).hexdigest(), self.md5)
        self.assertTrue(os.path.exists(amp.store_path(self.root)))
        r = self._cur().renderer()
        self.assertTrue(hasattr(r, 'classificationMin'))
        self.assertGreater(r.classificationMax(), r.classificationMin())

    def test_reset_restores_original_and_clears_entry(self):
        h = self.host
        h._amp_cb_despeckle.setCurrentIndex(h._amp_cb_despeckle.findData(3))
        h._amp_on_apply()
        h._amp_on_reset()
        self.assertEqual(self._src(), os.path.normcase(os.path.normpath(self.src)))
        self.assertIsNone(amp.get_settings(self.root, self.src))
        self.assertFalse(os.path.exists(amp.derived_path(self.root, self.src)))

    def test_reopen_regenerates_missing_derived(self):
        h = self.host
        h._amp_cb_lowpass.setCurrentIndex(h._amp_cb_lowpass.findData(5))
        h._amp_on_apply()
        # "riapertura": il derivato sparisce e il layer torna all'originale
        h._amp_show_source(self._cur(), self.src, self.src)
        self._cur().setCustomProperty("rl_amp_src", None)
        os.remove(amp.derived_path(self.root, self.src))
        n = h._amp_reconcile_project_layers()
        self.assertEqual(n, 1)
        self.assertTrue(os.path.exists(amp.derived_path(self.root, self.src)))
        self.assertEqual(self._src(), os.path.normcase(os.path.normpath(amp.derived_path(self.root, self.src))))
        self.assertEqual(self._cur().name(), "s_0.50")

    def test_selection_loads_saved_settings_into_controls(self):
        h = self.host
        amp.set_settings(self.root, self.src, amp.AmplitudeSettings(
            despike_pct=98.0, amp_scale="sqrt", limits_mode="fixed", vmin=0.5, vmax=2.5))
        h._amp_do_selection()
        self.assertTrue(h._amp_chk_despike.isChecked())
        self.assertAlmostEqual(h._amp_sp_despike.value(), 98.0)
        self.assertEqual(h._amp_cb_scale.currentData(), "sqrt")
        self.assertEqual(h._amp_cb_limits.currentData(), "fixed")
        self.assertAlmostEqual(h._amp_sp_vmax.value(), 2.5)

    def test_apply_to_all_selected(self):
        h = self.host
        src2 = os.path.join(self.root, "s_1.00.tif")
        _write_tif(src2, _noisy(1))
        QgsProject.instance().addMapLayer(QgsRasterLayer(src2, "s_1.00"))
        h.add_item(src2)
        for i in range(2):
            h.dlg.rasterListWidget.item(i).setSelected(True)
        h._amp_chk_all.setChecked(True)
        h._amp_cb_despeckle.setCurrentIndex(h._amp_cb_despeckle.findData(3))
        h._amp_on_apply()
        self.assertEqual(len(amp.load_store(self.root)), 2)

    def test_histogram_refresh_does_not_crash_without_selection(self):
        self.host.dlg.rasterListWidget.clear()
        self.host._amp_do_selection()
        self.host._amp_refresh_histogram()


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAS_QGIS, "QGIS non disponibile")
class DragTests(MixinTests):
    def test_histogram_drag_defers_and_sets_limits(self):
        h = self.host
        h._amp_on_hist_limits(0.5, 1.5)
        self.assertEqual(h._amp_cb_limits.currentData(), "fixed")
        before = self._cur().renderer().clone()
        h._amp_on_hist_limits(0.6, 1.4)
        h._amp_on_hist_limits(0.7, 1.3)       # il layer non e' ancora cambiato
        self.assertTrue(h._amp_limits_timer.isActive())
        h._amp_apply_pending_limits()
        r = self._cur().renderer()
        self.assertAlmostEqual(r.classificationMin(), 0.7)
        self.assertAlmostEqual(r.classificationMax(), 1.3)
