# -*- coding: utf-8 -*-
"""gpr_ert_store: archivio ERT nel progetto e allineamento al canale GPR (senza Qt)."""

import importlib
import os
import pathlib
import shutil
import sys
import tempfile
import unittest

import numpy as np

_PLUGIN_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(_PLUGIN_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR.parent))

_S = importlib.import_module("rasterlinker_feature_interpolation.gpr_ert_store")
_G = importlib.import_module("rasterlinker_feature_interpolation.gpr_ert_georef")
_I = importlib.import_module("rasterlinker_feature_interpolation.gpr_ert_import")

_CSV = ("x,y,z,Resistivity(ohm.m)\n"
        "1,0,-0.5,30\n3,0,-0.5,40\n5,0,-0.5,50\n2,0,-1.5,60\n4,0,-1.5,70\n3,0,-2.5,80\n")


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.csv = os.path.join(self.root, "export.csv")
        with open(self.csv, "w", encoding="utf-8") as fh:
            fh.write(_CSV)
        self.georef = _G.ErtGeoref.from_endpoints((100.0, 200.0), (106.0, 200.0), 6.0, "EPSG:32631")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_save_list_load_roundtrip(self):
        _S.save_ert(self.root, "P9", self.csv, self.georef)
        self.assertEqual(_S.list_ert(self.root), ["P9"])
        e = _S.load_ert(self.root, "P9")
        self.assertEqual(len(e.section.x), 6)
        self.assertEqual(e.georef.crs_authid, "EPSG:32631")
        self.assertEqual(e.georef.points, self.georef.points)
        self.assertEqual(e.source, "export.csv")

    def test_copy_survives_the_original_being_moved(self):
        _S.save_ert(self.root, "P9", self.csv, self.georef)
        os.remove(self.csv)
        self.assertEqual(len(_S.load_ert(self.root, "P9").section.x), 6)

    def test_bad_file_does_not_touch_the_project(self):
        bad = os.path.join(self.root, "bad.csv")
        with open(bad, "w") as fh:
            fh.write("a,b\n1,2\n")
        with self.assertRaises(_I.ErtImportError):
            _S.save_ert(self.root, "X", bad, self.georef)
        self.assertEqual(_S.list_ert(self.root), [])

    def test_same_name_replaces_and_names_are_sanitised(self):
        _S.save_ert(self.root, "P9/a:b", self.csv, self.georef)
        _S.save_ert(self.root, "P9/a:b", self.csv, self.georef)
        self.assertEqual(_S.list_ert(self.root), ["P9_a_b"])

    def test_delete(self):
        _S.save_ert(self.root, "P9", self.csv, self.georef)
        self.assertTrue(_S.delete_ert(self.root, "P9"))
        self.assertFalse(_S.delete_ert(self.root, "P9"))
        self.assertEqual(_S.list_ert(self.root), [])

    def test_list_without_a_project(self):
        self.assertEqual(_S.list_ert(""), [])
        self.assertEqual(_S.list_ert(os.path.join(self.root, "nope")), [])


class AlignmentTest(unittest.TestCase):
    def setUp(self):
        # ERT lungo x da 100 a 170 m (70 m); GPR parallelo, spostato di 2 m, da x=110 a 150
        self.georef = _G.ErtGeoref.from_endpoints((100.0, 0.0), (170.0, 0.0), 70.0)
        self.gx = np.linspace(0.0, 40.0, 81)
        self.east = 110.0 + self.gx
        self.north = np.full_like(self.gx, 2.0)

    def test_parallel_profile_maps_with_the_right_shift(self):
        al = _S.align_to_channel(self.georef, self.east, self.north, self.gx)
        self.assertAlmostEqual(al.median_offset_m, 2.0)
        self.assertAlmostEqual(float(al.gpr_to_ert(0.0)), 10.0)
        self.assertAlmostEqual(float(al.ert_to_gpr(30.0)), 20.0)
        self.assertAlmostEqual(al.overlap_m, 40.0)

    def test_ert_outside_the_gpr_profile_is_nan(self):
        al = _S.align_to_channel(self.georef, self.east, self.north, self.gx)
        out = al.ert_to_gpr(np.array([5.0, 30.0, 65.0]))
        self.assertTrue(np.isnan(out[0]) and np.isnan(out[2]))
        self.assertAlmostEqual(float(out[1]), 20.0)

    def test_opposite_direction(self):
        al = _S.align_to_channel(self.georef, self.east[::-1], self.north, self.gx)
        # la traccia a distanza GPR 0 sta ora a x=150, cioe' a 50 m lungo l'ERT
        self.assertAlmostEqual(float(al.gpr_to_ert(0.0)), 50.0)
        self.assertAlmostEqual(float(al.ert_to_gpr(30.0)), 20.0)

    def test_partial_overlap(self):
        east = 150.0 + self.gx           # da 150 a 190: solo 20 m dentro i 170 dell'ERT
        al = _S.align_to_channel(self.georef, east, self.north, self.gx)
        self.assertAlmostEqual(al.overlap_m, 20.0, delta=0.6)

    def test_no_valid_coordinates(self):
        nan = np.full(5, np.nan)
        self.assertIsNone(_S.align_to_channel(self.georef, nan, nan, np.arange(5.0)))


if __name__ == "__main__":
    unittest.main()
