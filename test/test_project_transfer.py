# -*- coding: utf-8 -*-
"""Trasferimento del progetto in un'altra cartella con aggiornamento metadati.
Sotto il Python di QGIS: ``python-qgis.bat -m unittest test.test_project_transfer``."""

import importlib
import json
import os
import pathlib
import sys
import tempfile
import unittest
import zipfile

_PLUGIN_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(_PLUGIN_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR.parent))

PKG = "rasterlinker_feature_interpolation"
pt = importlib.import_module(PKG + ".project_transfer")
pc = importlib.import_module(PKG + ".project_catalog")


def _make_project(base):
    root = os.path.join(base, "Old Project")
    pc.ensure_project_structure(root)
    tif = os.path.join(root, "timeslices_2d", "a.tif")
    with open(tif, "wb") as fh:
        fh.write(b"x")
    outside = os.path.join(base, "elsewhere.tif")
    with open(outside, "wb") as fh:
        fh.write(b"y")
    cat = pc.load_catalog(root)
    cat["timeslices"] = [
        {"id": "t1", "name": "a", "project_path": tif, "source_path": tif},
        {"id": "t2", "name": "b", "project_path": outside, "source_path": outside},
    ]
    pc.save_catalog(root, cat)
    with open(os.path.join(root, "metadata", "project.json"), "w", encoding="utf-8") as fh:
        json.dump({"last_processing": root + "\\processed",
                   "other": root + "pippo"}, fh)
    with open(os.path.join(root, "timeslices_2d", "sliceset.json"), "w", encoding="utf-8") as fh:
        json.dump({"dir": root.replace("\\", "/") + "/timeslices_2d"}, fh)
    qgs = '<qgis><Option value="%s"/></qgis>' % root
    with zipfile.ZipFile(os.path.join(root, "p.qgz"), "w") as z:
        z.writestr("p.qgs", qgs)
        z.writestr("p.qgd", "binary")
    return root, outside


class TransferTest(unittest.TestCase):
    def test_copy_rewrites_metadata(self):
        with tempfile.TemporaryDirectory() as base:
            root, outside = _make_project(base)
            new = os.path.join(base, "sub", "New Project")
            rep = pt.transfer_project(root, new)
            self.assertTrue(os.path.isdir(root))
            self.assertTrue(os.path.isfile(os.path.join(new, "timeslices_2d", "a.tif")))

            cat = pc.load_catalog(new, create_backup_on_migrate=False)
            self.assertEqual(os.path.normcase(cat["project_root"]), os.path.normcase(new))
            paths = {r["id"]: r["project_path"] for r in cat["timeslices"]}
            self.assertEqual(os.path.normcase(paths["t1"]),
                             os.path.normcase(os.path.join(new, "timeslices_2d", "a.tif")))
            self.assertEqual(paths["t2"], outside)          # fuori dal progetto: invariato
            self.assertEqual([os.path.normcase(p) for p in rep["external"]],
                             [os.path.normcase(outside)])

            with open(os.path.join(new, "metadata", "project.json"), encoding="utf-8") as fh:
                pj = json.load(fh)
            self.assertTrue(pj["last_processing"].startswith(new))
            self.assertTrue(pj["other"].startswith(root))   # "Old Projectpippo" non va toccato

            with open(os.path.join(new, "timeslices_2d", "sliceset.json"), encoding="utf-8") as fh:
                self.assertIn(new.replace("\\", "/"), json.load(fh)["dir"])

            with zipfile.ZipFile(os.path.join(new, "p.qgz")) as z:
                self.assertIn(new, z.read("p.qgs").decode())
                self.assertEqual(z.read("p.qgd"), b"binary")

    def test_move_removes_original(self):
        with tempfile.TemporaryDirectory() as base:
            root, _ = _make_project(base)
            new = os.path.join(base, "New")
            rep = pt.transfer_project(root, new, move=True)
            self.assertTrue(rep["moved"])
            self.assertFalse(os.path.exists(root))

    def test_validation(self):
        with tempfile.TemporaryDirectory() as base:
            root, _ = _make_project(base)
            self.assertTrue(pt.validate_transfer(root, root))
            self.assertTrue(pt.validate_transfer(root, os.path.join(root, "in")))
            self.assertTrue(pt.validate_transfer(root, base))
            self.assertEqual(pt.validate_transfer(root, os.path.join(base, "ok")), "")

    def test_cancel_leaves_no_partial_copy(self):
        with tempfile.TemporaryDirectory() as base:
            root, _ = _make_project(base)
            new = os.path.join(base, "New")
            with self.assertRaises(pt.TransferError):
                pt.transfer_project(root, new, should_cancel=lambda: True)
            self.assertFalse(os.path.exists(new))
            self.assertTrue(os.path.isdir(root))


if __name__ == "__main__":
    unittest.main()
