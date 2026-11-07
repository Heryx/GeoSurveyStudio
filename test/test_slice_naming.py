# -*- coding: utf-8 -*-
"""Le time-slice portano nel nome la finestra di profondita' e il nome della
griglia (gruppo) con cui vengono importate nel progetto/canvas.
Sotto il Python di QGIS: ``python-qgis.bat -m unittest test.test_slice_naming``."""

import importlib
import json
import os
import pathlib
import sys
import tempfile
import unittest

import numpy as np

_PLUGIN_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(_PLUGIN_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR.parent))

PKG = "rasterlinker_feature_interpolation"
sm = importlib.import_module(PKG + ".slice_manifest")
pc = importlib.import_module(PKG + ".project_catalog")

try:
    from osgeo import gdal  # noqa: F401
    HAS_GDAL = True
except Exception:  # pragma: no cover
    HAS_GDAL = False


def _tmp():
    try:
        return tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    except TypeError:
        return tempfile.TemporaryDirectory()


class FileNameTest(unittest.TestCase):
    def test_name_carries_the_depth_window(self):
        self.assertEqual(sm.slice_file_name(3, 0.30, 0.40), "slice_0003_0.30-0.40m.tif")
        self.assertEqual(sm.slice_file_name(0, 0.0, 0.125), "slice_0000_0.00-0.125m.tif")
        self.assertEqual(sm.slice_file_name(12, 1.0, 1.1), "slice_0012_1.00-1.10m.tif")

    def test_name_is_read_back_by_the_catalog_parser(self):
        name = sm.slice_file_name(7, 0.45, 0.55)
        d = pc.parse_depth_from_filename(name)
        self.assertAlmostEqual(d["depth_from"], 0.45)
        self.assertAlmostEqual(d["depth_to"], 0.55)
        self.assertEqual(d["unit"], "m")

    def test_names_sort_by_index(self):
        names = [sm.slice_file_name(i, i * 0.1, (i + 1) * 0.1) for i in (10, 2, 1)]
        self.assertEqual(sorted(names), [names[2], names[1], names[0]])


class CopyIntoProjectTest(unittest.TestCase):
    def test_group_name_prefixes_the_file_and_is_not_doubled(self):
        with _tmp() as tmp:
            src = os.path.join(tmp, "slice_0003_0.30-0.40m.tif")
            open(src, "wb").close()
            root = os.path.join(tmp, "proj")
            p1, n1 = pc.normalize_copy_into_project(root, "timeslices_2d", src, name_prefix="Area Nord - Lavoro")
            self.assertEqual(n1, "Area_Nord_-_Lavoro_slice_0003_0.30-0.40m.tif")
            self.assertTrue(os.path.isfile(p1))
            # sorgente gia' col prefisso: non si raddoppia
            src2 = os.path.join(tmp, n1)
            open(src2, "wb").close()
            _p2, n2 = pc.normalize_copy_into_project(root, "timeslices_2d", src2, name_prefix="Area Nord - Lavoro")
            self.assertFalse(n2.startswith("Area_Nord_-_Lavoro_Area_Nord"))
            # senza prefisso: comportamento di sempre
            _p3, n3 = pc.normalize_copy_into_project(root, "timeslices_2d", src)
            self.assertEqual(n3, "slice_0003_0.30-0.40m.tif")

    def test_two_sets_with_the_same_slice_names_do_not_collide(self):
        with _tmp() as tmp:
            src = os.path.join(tmp, "slice_0000_0.00-0.10m.tif")
            open(src, "wb").close()
            root = os.path.join(tmp, "proj")
            _a, na = pc.normalize_copy_into_project(root, "timeslices_2d", src, name_prefix="ZonaA")
            _b, nb = pc.normalize_copy_into_project(root, "timeslices_2d", src, name_prefix="ZonaB")
            self.assertNotEqual(na, nb)
            self.assertNotIn("_001", na + nb)


class DepthWindowTest(unittest.TestCase):
    def test_sliceset_next_to_the_file_wins(self):
        with _tmp() as tmp:
            tif = os.path.join(tmp, "slice_0001_z0_5000.tif")        # nome vecchio: solo il centro
            open(tif, "wb").close()
            with open(os.path.join(tmp, "sliceset.json"), "w", encoding="utf-8") as fh:
                json.dump({"slices": [{"grid": "slice_0001_z0_5000.tif", "z_from": 0.45, "z_to": 0.55}]}, fh)
            self.assertEqual(sm.depth_window_for_file(tif), (0.45, 0.55))

    def test_falls_back_to_the_file_name(self):
        with _tmp() as tmp:
            tif = os.path.join(tmp, "slice_0002_0.10-0.20m.tif")
            open(tif, "wb").close()
            w = sm.depth_window_for_file(tif)
            self.assertAlmostEqual(w[0], 0.10)
            self.assertAlmostEqual(w[1], 0.20)

    def test_unknown_when_nothing_says_the_depth(self):
        with _tmp() as tmp:
            tif = os.path.join(tmp, "mappa.tif")
            open(tif, "wb").close()
            self.assertIsNone(sm.depth_window_for_file(tif))


@unittest.skipUnless(HAS_GDAL, "GDAL richiesto")
class EndToEndTest(unittest.TestCase):
    def test_written_slice_has_window_in_name_and_import_records_the_depth(self):
        slicer = importlib.import_module(PKG + ".gpr_ogpr_slicer")
        bt = importlib.import_module(PKG + ".background_tasks")
        grids = [{"z_lev": 0.35, "index": 3, "z_from": 0.30, "z_to": 0.40,
                  "grid": np.arange(20, dtype=np.float32).reshape(4, 5)}]
        meta = {"x_min": 500000.0, "y_min": 4500000.0, "y_max": 4500000.3, "resolution": 0.1,
                "n_x": 5, "n_y": 4, "x_max": 500000.4}
        with _tmp() as tmp:
            res = slicer.write_grids_to_tifs(grids, meta, os.path.join(tmp, "Lavoro"), 32633)
            self.assertEqual(os.path.basename(res[0]["path"]), "slice_0003_0.30-0.40m.tif")

            root = os.path.join(tmp, "proj")
            task = bt.TimesliceImportTask(
                root, [{"source_path": res[0]["path"], "meta": {}}], name_prefix="Zona1")
            self.assertTrue(task.run())
            rec = task.imported_records[0]
            self.assertEqual(rec["normalized_name"], "Zona1_slice_0003_0.30-0.40m.tif")
            self.assertAlmostEqual(rec["depth_from"], 0.30)
            self.assertAlmostEqual(rec["depth_to"], 0.40)
            self.assertEqual(rec["unit"], "m")

            # il tag GeoTIFF permette di ritrovare la finestra anche col nome vecchio
            legacy = os.path.join(tmp, "slice_0003_z0_3500.tif")
            os.replace(res[0]["path"], legacy)
            os.remove(os.path.join(tmp, "Lavoro", "slice_0003_0.30-0.40m.qml")) if os.path.exists(
                os.path.join(tmp, "Lavoro", "slice_0003_0.30-0.40m.qml")) else None
            w = sm.depth_window_for_file(legacy)
            self.assertAlmostEqual(w[0], 0.30)
            self.assertAlmostEqual(w[1], 0.40)


if __name__ == "__main__":
    unittest.main()
