# -*- coding: utf-8 -*-
import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qgis.PyQt.QtWidgets import QApplication

from .. import group_ordering as go

_app = QApplication.instance() or QApplication([])


class ListOrderTest(unittest.TestCase):
    def test_move_items(self):
        self.assertEqual(go.move_items(list("abcd"), ["c"], -1), list("acbd"))
        self.assertEqual(go.move_items(list("abcd"), ["a"], -1), list("abcd"))
        self.assertEqual(go.move_items(list("abcd"), ["b", "c"], 1), list("adbc"))
        self.assertEqual(go.move_items(list("abcd"), ["d"], 1), list("abcd"))

    def test_sort_by_depth(self):
        recs = {
            "a": {"id": "a", "depth_from": 0.2, "depth_to": 0.3},
            "b": {"id": "b", "depth_from": 0.0, "depth_to": 0.1},
            "c": {"id": "c"},
            "d": {"id": "d", "depth_from": 0.1, "depth_to": 0.2},
        }
        self.assertEqual(go.sort_ids(list("abcd"), recs, "depth"), list("bdac"))
        self.assertEqual(go.sort_ids(list("abcd"), recs, "depth_desc"), list("adbc"))
        self.assertEqual(go.sort_ids(list("abcdz"), recs, "name"), list("abcdz"))


class QgisTreeOrderTest(unittest.TestCase):
    def test_reorder_groups_and_layers(self):
        from qgis.core import QgsApplication, QgsProject, QgsRasterLayer, QgsLayerTreeLayer
        from osgeo import gdal
        from qgis.core import QgsProviderRegistry
        if not QgsProviderRegistry.instance().providerList():
            QgsApplication.initQgis()
        prj = QgsProject.instance()
        prj.removeAllMapLayers()
        root = prj.layerTreeRoot()
        top = root.addGroup("GeoSurvey Studio")
        for n in ("G1", "Other", "G2", "G3"):
            top.addGroup(n)
        with tempfile.TemporaryDirectory() as d:
            paths = []
            g1 = next(g for g in top.children() if g.name() == "G1")
            for i in range(3):
                p = os.path.join(d, f"s{i}.tif")
                ds = gdal.GetDriverByName("GTiff").Create(p, 2, 2, 1)
                ds.SetGeoTransform((0, 1, 0, 2, 0, -1))
                ds = None
                lyr = QgsRasterLayer(p, f"s{i}")
                prj.addMapLayer(lyr, False)
                g1.addLayer(lyr)
                paths.append(p)
            # gruppi: G3, G1, G2 negli slot dei tre gruppi; "Other" resta dov'e'
            self.assertTrue(go.apply_group_order_to_qgis(["G3", "G1", "G2"]))
            self.assertEqual([g.name() for g in top.children()], ["G3", "Other", "G1", "G2"])
            g1 = next(g for g in top.children() if g.name() == "G1")
            self.assertEqual(len([c for c in g1.children() if isinstance(c, QgsLayerTreeLayer)]), 3)
            # immagini: ordine inverso
            self.assertTrue(go.apply_slice_order_to_qgis("G1", list(reversed(paths))))
            g1 = next(g for g in top.children() if g.name() == "G1")
            self.assertEqual([c.name() for c in g1.children()], ["s2", "s1", "s0"])
            prj.removeAllMapLayers()  # sblocca i file prima di cancellare la cartella
        root.removeAllChildren()


class NestedAndDialogTest(unittest.TestCase):
    def test_nested_subgroup_layers_reordered(self):
        from qgis.core import QgsProject, QgsRasterLayer, QgsProviderRegistry
        from osgeo import gdal
        if not QgsProviderRegistry.instance().providerList():
            from qgis.core import QgsApplication
            QgsApplication.initQgis()
        prj = QgsProject.instance()
        root = prj.layerTreeRoot()
        sub = root.addGroup("Altro nome").addGroup("Zone 9")
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for i in range(3):
                p = os.path.join(d, f"n{i}.tif")
                ds = gdal.GetDriverByName("GTiff").Create(p, 2, 2, 1)
                ds.SetGeoTransform((0, 1, 0, 2, 0, -1))
                ds = None
                lyr = QgsRasterLayer(p, f"n{i}")
                prj.addMapLayer(lyr, False)
                sub.addLayer(lyr)
                paths.append(p)
            self.assertTrue(go.apply_slice_order_to_qgis("Zone 9", paths[::-1]))
            sub = root.findGroups(True)[-1]
            self.assertEqual([c.name() for c in sub.children()], ["n2", "n1", "n0"])
            prj.removeAllMapLayers()
        root.removeAllChildren()

    def test_dialog_sort_by_depth(self):
        from qgis.core import QgsProviderRegistry
        from ..project_catalog import (
            load_catalog, save_catalog, create_raster_group, register_timeslices_batch,
            assign_timeslices_to_group)
        from ..timeslice_group_manager_dialog import TimesliceGroupManagerDialog
        with tempfile.TemporaryDirectory() as d:
            recs = []
            for i, (a, b) in enumerate([(0.2, 0.3), (0.0, 0.1), (0.1, 0.2)]):
                recs.append({"id": f"t{i}", "name": f"s{i}", "normalized_name": f"s{i}.tif",
                             "project_path": os.path.join(d, f"s{i}.tif"),
                             "depth_from": a, "depth_to": b, "unit": "m"})
            register_timeslices_batch(d, recs)
            g, _ = create_raster_group(d, "Z")
            assign_timeslices_to_group(d, g["id"], ["t0", "t1", "t2"])
            dlg = TimesliceGroupManagerDialog(d)
            dlg.filter_group_combo.setCurrentIndex(dlg.filter_group_combo.findData(g["id"]))
            order = lambda: [dlg.ts_model.row_payload(i)["id"] for i in range(dlg.ts_model.rowCount())]
            self.assertEqual(order(), ["t0", "t1", "t2"])
            dlg._sort_shown_group("depth")
            self.assertEqual(order(), ["t1", "t2", "t0"])
            self.assertEqual(
                next(x for x in load_catalog(d)["raster_groups"] if x["id"] == g["id"])["timeslice_ids"],
                ["t1", "t2", "t0"])
            dlg._sort_shown_group("depth_desc")
            self.assertEqual(order(), ["t0", "t2", "t1"])
            # con 'All': ordine del gruppo, e il gruppo si deduce dalla selezione
            dlg.filter_group_combo.setCurrentIndex(0)
            self.assertEqual(order(), ["t0", "t2", "t1"])


if __name__ == "__main__":
    unittest.main()
