# -*- coding: utf-8 -*-
import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qgis.PyQt.QtWidgets import QApplication

from ..timeslice_batch_rename import (
    compute_new_names, BatchRenameDialog, MODE_REPLACE, MODE_AFFIX, MODE_SEQUENCE,
)

_app = QApplication.instance() or QApplication([])


class ComputeNewNamesTest(unittest.TestCase):
    def test_replace_part_keeps_rest_and_extension(self):
        names, err = compute_new_names(
            ["a/slice_0000_0.00-0.10m.tif", "a/slice_0001_0.10-0.20m.tif"],
            MODE_REPLACE, {"find": "slice_", "replace": "swath3_", "case": True})
        self.assertIsNone(err)
        self.assertEqual(names, ["swath3_0000_0.00-0.10m.tif", "swath3_0001_0.10-0.20m.tif"])

    def test_replace_is_literal_without_regex(self):
        names, _ = compute_new_names(["x/a.b.tif"], MODE_REPLACE,
                                     {"find": ".b", "replace": r"\1"})
        # il "." e' letterale (non "qualunque carattere") e "\1" non e' un backreference
        self.assertNotIn("a.b", names[0])
        self.assertTrue(names[0].endswith(".tif"))

    def test_regex_error_reported(self):
        _, err = compute_new_names(["x/a.tif"], MODE_REPLACE, {"find": "(", "regex": True})
        self.assertIsNotNone(err)

    def test_affix_and_sequence(self):
        names, _ = compute_new_names(["x/a.tif"], MODE_AFFIX, {"prefix": "P_", "suffix": "_S"})
        self.assertEqual(names, ["P_a_S.tif"])
        names, _ = compute_new_names(["x/b.TIF", "x/a.tif"], MODE_SEQUENCE,
                                     {"prefix": "t_", "start": 5, "padding": 2})
        self.assertEqual(names, ["t_05.tif", "t_06.tif"])


class DialogPlanTest(unittest.TestCase):
    def test_partial_rename_only_selected_and_conflicts(self):
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for n in ("old_1.tif", "old_2.tif", "new_2.tif"):
                p = os.path.join(d, n)
                open(p, "w").close()
                paths.append(p)
            dlg = BatchRenameDialog([("1", paths[0]), ("2", paths[1])])
            dlg.find_edit.setText("old_")
            dlg.repl_edit.setText("new_")
            # old_2 -> new_2 esiste gia' e non e' tra i selezionati: conflitto
            self.assertEqual([p[0] for p in dlg.plan], ["1"])
            self.assertFalse(dlg.buttons.button(dlg.buttons.Ok).isEnabled())
            dlg.repl_edit.setText("ok_")
            self.assertEqual(sorted(p[0] for p in dlg.plan), ["1", "2"])
            self.assertTrue(dlg.buttons.button(dlg.buttons.Ok).isEnabled())


class ManagerSmokeTest(unittest.TestCase):
    def test_manager_builds(self):
        from ..timeslice_group_manager_dialog import TimesliceGroupManagerDialog
        with tempfile.TemporaryDirectory() as d:
            dlg = TimesliceGroupManagerDialog(d)
            self.assertTrue(dlg.set_depth_btn.isEnabled())
            self.assertIn("selected of", dlg.ts_count_label.text())


class LoadedLayerRenameTest(unittest.TestCase):
    def test_rename_file_open_in_qgis(self):
        from qgis.core import QgsApplication, QgsProject, QgsRasterLayer
        from osgeo import gdal
        from ..timeslice_group_manager_dialog import TimesliceGroupManagerDialog as D
        if QgsApplication.instance() is None or not QgsApplication.providerRegistry().providerList():
            QgsApplication.initQgis()
        with tempfile.TemporaryDirectory() as d:
            old = os.path.join(d, "old_a.tif")
            ds = gdal.GetDriverByName("GTiff").Create(old, 4, 4, 1)
            ds.SetGeoTransform((0, 1, 0, 4, 0, -1))
            ds = None
            lyr = QgsRasterLayer(old, "old_a")
            self.assertTrue(lyr.isValid())
            prj = QgsProject.instance()
            prj.addMapLayer(lyr)
            new = os.path.join(d, "new_a.tif")
            saved = D._detach_layers_for_rename([old])
            self.assertEqual(len(saved), 1)
            os.rename(old, new)
            D._reattach_layers_after_rename(saved, [(old, new)])
            layers = list(prj.mapLayers().values())
            self.assertEqual(len(layers), 1)
            self.assertEqual(layers[0].name(), "new_a")
            self.assertTrue(layers[0].source().endswith("new_a.tif"))
            prj.removeAllMapLayers()

    def test_delete_file_open_in_qgis(self):
        from qgis.core import QgsProject, QgsRasterLayer
        from osgeo import gdal
        from ..timeslice_group_manager_dialog import TimesliceGroupManagerDialog as D
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.tif")
            ds = gdal.GetDriverByName("GTiff").Create(p, 4, 4, 1)
            ds.SetGeoTransform((0, 1, 0, 4, 0, -1))
            ds = None
            prj = QgsProject.instance()
            prj.addMapLayer(QgsRasterLayer(p, "x"))
            saved = D._detach_layers_for_rename([p])
            os.remove(p)
            D._reattach_layers_after_rename([s for s in saved if os.path.exists(s["path"])], [])
            self.assertEqual(len(prj.mapLayers()), 0)


if __name__ == "__main__":
    unittest.main()
