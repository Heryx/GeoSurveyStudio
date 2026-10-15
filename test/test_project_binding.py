# -*- coding: utf-8 -*-
"""project_binding: project_root legata al progetto QGIS e raster del canvas
adottati nel catalogo. Girano sotto il Python di QGIS (offscreen):
``python-qgis.bat -m unittest test.test_project_binding``."""

import importlib
import json
import os
import pathlib
import sys
import tempfile
import unittest

try:
    from qgis.core import (
        QgsApplication,
        QgsProject,
        QgsRasterLayer,
    )
    HAS_QGIS = True
except Exception:  # pragma: no cover
    HAS_QGIS = False

_PLUGIN_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(_PLUGIN_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR.parent))

pb = importlib.import_module("rasterlinker_feature_interpolation.project_binding")


def _make_project(base, name):
    root = os.path.join(base, name)
    os.makedirs(os.path.join(root, "metadata"))
    with open(os.path.join(root, "metadata", "project_catalog.json"), "w", encoding="utf-8") as fh:
        json.dump({}, fh)
    return root


class PureLogicTest(unittest.TestCase):
    def test_find_root_above_walks_up_and_ignores_siblings(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = _make_project(tmp, "GPR_A")
            deep = os.path.join(a, "timeslices_2d", "set1")
            os.makedirs(deep)
            f = os.path.join(deep, "s.tif")
            open(f, "wb").close()
            self.assertEqual(pb.find_root_above(f), os.path.normpath(a))
            other = os.path.join(tmp, "elsewhere")
            os.makedirs(other)
            self.assertEqual(pb.find_root_above(other), "")

    def test_most_common_root_votes(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = _make_project(tmp, "GPR_A")
            b = _make_project(tmp, "GPR_B")
            fa = [os.path.join(a, "x%d.tif" % i) for i in range(3)]
            fb = os.path.join(b, "y.tif")
            self.assertEqual(pb.most_common_root(fa + [fb + "|layerid=0"]), os.path.normpath(a))
            self.assertEqual(pb.most_common_root([]), "")


@unittest.skipUnless(HAS_QGIS, "serve QGIS")
class QgisBindingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QgsApplication.instance() or QgsApplication([], False)
        if not QgsApplication.instance() or QgsApplication.instance() is not cls._app:
            pass
        cls._app.initQgis()

    def setUp(self):
        QgsProject.instance().clear()

    def _tif(self, path, value=1.0):
        from osgeo import gdal, osr
        drv = gdal.GetDriverByName("GTiff")
        ds = drv.Create(path, 4, 4, 1, gdal.GDT_Float32)
        ds.SetGeoTransform((500000.0, 1.0, 0, 4500000.0, 0, -1.0))
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32633)
        ds.SetProjection(srs.ExportToWkt())
        ds.GetRasterBand(1).Fill(value)
        ds.FlushCache()
        ds = None

    def test_remember_and_recall_root(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = _make_project(tmp, "GPR_A")
            self.assertEqual(pb.recall_root(), "")
            self.assertTrue(pb.remember_root(root))
            self.assertEqual(pb.recall_root(), os.path.normpath(root))
            self.assertFalse(pb.remember_root(root))   # invariato: nessuna scrittura

    def test_recall_ignores_missing_folder(self):
        QgsProject.instance().writeEntry(pb.SCOPE, pb.KEY_ROOT, os.path.join(tempfile.gettempdir(), "nope_xyz"))
        self.assertEqual(pb.recall_root(), "")

    def test_resolve_root_from_layer_tree_and_adopt(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = _make_project(tmp, "GPR_A")
            ts = os.path.join(root, "timeslices_2d")
            os.makedirs(ts)
            paths = [os.path.join(ts, "slice_0.00-0.10_m.tif"), os.path.join(ts, "slice_0.10-0.20_m.tif")]
            for p in paths:
                self._tif(p)

            tree = QgsProject.instance().layerTreeRoot()
            plugin_root = tree.addGroup("GeoSurvey Studio")
            grp = plugin_root.addGroup("Zona1")
            for p in paths:
                lyr = QgsRasterLayer(p, os.path.basename(p))
                self.assertTrue(lyr.isValid())
                QgsProject.instance().addMapLayer(lyr, False)
                grp.addLayer(lyr)
            # riferimento cross-progetto: non va mai nel catalogo
            xref = plugin_root.addGroup("Imported \u2014 altro").addGroup("G")
            other = os.path.join(tmp, "altro.tif")
            self._tif(other)
            lyr = QgsRasterLayer(other, "altro.tif")
            QgsProject.instance().addMapLayer(lyr, False)
            xref.addLayer(lyr)

            self.assertEqual(pb.resolve_root(plugin_root), os.path.normpath(root))

            res = pb.adopt_untracked_timeslices(root, plugin_root)
            self.assertEqual(res["registered"], 2)
            self.assertEqual(res["assigned"], 2)

            from rasterlinker_feature_interpolation.project_catalog import load_catalog
            cat = load_catalog(root)
            names = {g["name"]: g for g in cat["raster_groups"]}
            self.assertIn("Zona1", names)
            self.assertEqual(len(names["Zona1"]["timeslice_ids"]), 2)
            self.assertEqual(len(cat["timeslices"]), 2)
            self.assertNotIn("Imported \u2014 altro", names)
            rec = cat["timeslices"][0]
            self.assertAlmostEqual(rec["depth_to"] - rec["depth_from"], 0.1, places=3)

            # idempotente
            again = pb.adopt_untracked_timeslices(root, plugin_root)
            self.assertEqual(again, {"registered": 0, "assigned": 0, "groups": 0})

    def test_adopt_assigns_known_timeslice_missing_from_group(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = _make_project(tmp, "GPR_A")
            p = os.path.join(root, "t.tif")
            self._tif(p)
            from rasterlinker_feature_interpolation.project_catalog import load_catalog, register_timeslice
            register_timeslice(root, {"id": "ts1", "name": "t", "normalized_name": "t.tif",
                                      "project_path": p, "source_path": p})
            plugin_root = QgsProject.instance().layerTreeRoot().addGroup("GeoSurvey Studio")
            grp = plugin_root.addGroup("Z")
            lyr = QgsRasterLayer(p, "t.tif")
            QgsProject.instance().addMapLayer(lyr, False)
            grp.addLayer(lyr)
            res = pb.adopt_untracked_timeslices(root, plugin_root)
            self.assertEqual(res["registered"], 0)
            self.assertEqual(res["assigned"], 1)
            cat = load_catalog(root)
            z = next(g for g in cat["raster_groups"] if g["name"] == "Z")
            self.assertEqual(z["timeslice_ids"], ["ts1"])


if __name__ == "__main__":
    unittest.main()
