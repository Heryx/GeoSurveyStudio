# -*- coding: utf-8 -*-
"""Le slice di un gruppo vivono in timeslices_2d/<gruppo>/ e le slice sciolte
si raggruppano per inizio del nome.
Sotto il Python di QGIS: ``python-qgis.bat -m unittest test.test_group_folders``."""

import importlib
import os
import pathlib
import sys
import tempfile
import unittest

_PLUGIN_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(_PLUGIN_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR.parent))

PKG = "rasterlinker_feature_interpolation"
pc = importlib.import_module(PKG + ".project_catalog")


class PrefixTest(unittest.TestCase):
    def test_depth_references_are_dropped(self):
        f = pc.timeslice_group_prefix
        self.assertEqual(f("Area_Nord_slice_0003_0.30-0.40m.tif"), "Area_Nord")
        self.assertEqual(f("Zona1_slice_0001_z0_5000.tif"), "Zona1")
        self.assertEqual(f("Lavoro_z0.30-0.40.tif"), "Lavoro")
        self.assertEqual(f("slice_0003_0.30-0.40m.tif"), "")

    def test_folder_name_is_filesystem_safe(self):
        self.assertEqual(pc.group_folder_name("Area Nord - Lavoro"), "Area Nord - Lavoro")
        self.assertEqual(pc.group_folder_name('a/b:c'), "a_b_c")
        self.assertEqual(pc.group_folder_name("  "), "group")


class DetectAndMoveTest(unittest.TestCase):
    def test_detect_then_move_into_group_folder(self):
        with tempfile.TemporaryDirectory() as root:
            ts_dir = os.path.join(root, "timeslices_2d")
            os.makedirs(ts_dir)
            recs = []
            for pre, n in (("AreaA", 3), ("AreaB", 2), ("Solo", 1)):
                for i in range(n):
                    name = f"{pre}_slice_{i:04d}_0.{i}0-0.{i + 1}0m.tif"
                    p = os.path.join(ts_dir, name)
                    open(p, "wb").close()
                    open(os.path.join(ts_dir, name[:-4] + ".qml"), "w").close()
                    recs.append({"id": f"{pre}{i}", "name": name[:-4], "normalized_name": name,
                                 "project_path": p})
            pc.register_timeslices_batch(root, recs)

            found = pc.detect_timeslice_prefix_groups(root)
            self.assertEqual([e["prefix"] for e in found], ["AreaA", "AreaB"])
            self.assertEqual([len(e["records"]) for e in found], [3, 2])

            res = pc.move_timeslices_into_group_folder(
                root, "AreaA", [r["id"] for r in found[0]["records"]])
            self.assertEqual(len(res["moved"]), 3)
            self.assertEqual(res["failed"], [])
            folder = os.path.join(ts_dir, "AreaA")
            self.assertEqual(sorted(f for f in os.listdir(folder) if f.endswith(".tif")),
                             sorted(r["normalized_name"] for r in recs if r["id"].startswith("AreaA")))
            self.assertTrue(any(f.endswith(".qml") for f in os.listdir(folder)))
            cat = pc.load_catalog(root)
            moved = [t for t in cat["timeslices"] if t["id"].startswith("AreaA")]
            self.assertTrue(all(os.path.dirname(t["project_path"]) == folder for t in moved))
            # spostate: non vengono piu' proposte
            self.assertEqual([e["prefix"] for e in pc.detect_timeslice_prefix_groups(root)], ["AreaB"])

    def test_import_task_writes_into_group_subfolder(self):
        bt = importlib.import_module(PKG + ".background_tasks")
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "slice_0003_0.30-0.40m.tif")
            open(src, "wb").close()
            root = os.path.join(tmp, "proj")
            task = bt.TimesliceImportTask(root, [{"source_path": src, "meta": {}}], name_prefix="Zona 1")
            self.assertTrue(task.run())
            path = task.imported_records[0]["project_path"]
            self.assertEqual(os.path.basename(os.path.dirname(path)), "Zona 1")
            self.assertEqual(os.path.basename(os.path.dirname(os.path.dirname(path))), "timeslices_2d")


if __name__ == "__main__":
    unittest.main()
