# -*- coding: utf-8 -*-
"""Cancellazione a cascata di una variante (variant_cascade.py)."""

import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import project_catalog as pc
import variant_cascade as vc
import variant_provenance as vp


class ListDescendantsTest(unittest.TestCase):
    def test_no_children(self):
        entries = {"a": {"parent": None}}
        self.assertEqual(vc.list_descendants(entries, "a"), [])

    def test_multi_level(self):
        entries = {
            "a": {"parent": None},
            "b": {"parent": "a"},
            "c": {"parent": "b"},
            "d": {"parent": "a"},
            "e": {"parent": "x"},  # non discendente di 'a'
        }
        self.assertEqual(sorted(vc.list_descendants(entries, "a")), ["b", "c", "d"])

    def test_cycle_does_not_loop_forever(self):
        entries = {
            "a": {"parent": "b"},
            "b": {"parent": "a"},
        }
        # 'a' non e' discendente di se stesso; robusto a ciclo malformato
        self.assertEqual(vc.list_descendants(entries, "a"), ["b"])


def _make_project(tmp):
    pc.ensure_project_structure(tmp)
    cat = pc._default_catalog(tmp)
    pc._write_catalog_file(pc.catalog_path(tmp), cat)


def _touch_npz(variant_dir):
    os.makedirs(variant_dir, exist_ok=True)
    with open(os.path.join(variant_dir, "ch0.npz"), "wb") as fh:
        fh.write(b"\x00")


class PlanCascadeDeleteTest(unittest.TestCase):
    def test_plan_includes_children_and_catalog_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_project(tmp)
            processed = os.path.join(tmp, "Processing", "processed")
            timeslices = os.path.join(tmp, "Processing", "timeslices")

            _touch_npz(os.path.join(processed, "bp"))
            _touch_npz(os.path.join(processed, "bp_bg"))
            vp.set_entry(processed, "bp", parent=None)
            vp.set_entry(processed, "bp_bg", parent="bp")

            ts_dir = os.path.join(timeslices, "bp")
            os.makedirs(ts_dir, exist_ok=True)
            ts_file = os.path.join(ts_dir, "slice_000.tif")
            with open(ts_file, "wb") as fh:
                fh.write(b"\x00")

            data = pc.load_catalog(tmp)
            data["timeslices"].append(pc.normalize_timeslice_record({
                "id": "ts_1", "name": "slice_000", "project_path": ts_file,
            }))
            pc.save_catalog(tmp, data)

            plan = vc.plan_cascade_delete(processed, timeslices, tmp, "bp")
            self.assertEqual(plan["names"], ["bp", "bp_bg"])
            self.assertTrue(plan["variant_dirs"]["bp"])
            self.assertTrue(plan["variant_dirs"]["bp_bg"])
            self.assertTrue(plan["timeslice_dirs"]["bp"])
            self.assertIsNone(plan["timeslice_dirs"]["bp_bg"])
            self.assertEqual(len(plan["matched_ts_records"]["bp"]), 1)
            self.assertEqual(plan["layer_paths"], [ts_file])

    def test_describe_plan_mentions_children_count(self):
        plan = {
            "variant_name": "bp",
            "names": ["bp", "bp_bg"],
            "timeslice_dirs": {"bp": "/x/timeslices/bp", "bp_bg": None},
            "matched_ts_records": {"bp": [{"id": "1"}], "bp_bg": []},
            "layer_paths": ["/x/timeslices/bp/slice_000.tif"],
        }
        text = vc.describe_plan(plan)
        self.assertIn("bp_bg", text)
        self.assertIn("1", text)


class ExecuteCascadeDeleteTest(unittest.TestCase):
    def test_full_success_removes_folders_catalog_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_project(tmp)
            processed = os.path.join(tmp, "Processing", "processed")
            timeslices = os.path.join(tmp, "Processing", "timeslices")

            _touch_npz(os.path.join(processed, "bp"))
            vp.set_entry(processed, "bp", parent=None)

            ts_dir = os.path.join(timeslices, "bp")
            os.makedirs(ts_dir, exist_ok=True)
            ts_file = os.path.join(ts_dir, "slice_000.tif")
            with open(ts_file, "wb") as fh:
                fh.write(b"\x00")

            data = pc.load_catalog(tmp)
            data["timeslices"].append(pc.normalize_timeslice_record({
                "id": "ts_1", "name": "slice_000", "project_path": ts_file,
            }))
            pc.save_catalog(tmp, data)

            plan = vc.plan_cascade_delete(processed, timeslices, tmp, "bp")
            result = vc.execute_cascade_delete(processed, timeslices, tmp, plan)

            self.assertEqual(result["deleted_variants"], ["bp"])
            self.assertEqual(result["failed_variants"], [])
            self.assertFalse(os.path.isdir(os.path.join(processed, "bp")))
            self.assertFalse(os.path.isdir(ts_dir))
            self.assertNotIn("bp", vp.load(processed))

            after = pc.load_catalog(tmp)
            self.assertEqual(after["timeslices"], [])

    def test_failed_rmtree_moves_records_to_unassigned_instead_of_losing_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_project(tmp)
            processed = os.path.join(tmp, "Processing", "processed")
            timeslices = os.path.join(tmp, "Processing", "timeslices")

            _touch_npz(os.path.join(processed, "bp"))
            vp.set_entry(processed, "bp", parent=None)

            ts_dir = os.path.join(timeslices, "bp")
            os.makedirs(ts_dir, exist_ok=True)
            ts_file = os.path.join(ts_dir, "slice_000.tif")
            with open(ts_file, "wb") as fh:
                fh.write(b"\x00")

            data = pc.load_catalog(tmp)
            data["timeslices"].append(pc.normalize_timeslice_record({
                "id": "ts_1", "name": "slice_000", "project_path": ts_file,
            }))
            pc.save_catalog(tmp, data)

            plan = vc.plan_cascade_delete(processed, timeslices, tmp, "bp")

            real_rmtree = __import__("shutil").rmtree

            def flaky_rmtree(path, *a, **kw):
                if os.path.normcase(os.path.abspath(path)) == os.path.normcase(os.path.abspath(ts_dir)):
                    raise OSError("simulated: file in use")
                return real_rmtree(path, *a, **kw)

            with mock.patch("variant_cascade.shutil.rmtree", side_effect=flaky_rmtree):
                result = vc.execute_cascade_delete(processed, timeslices, tmp, plan)

            self.assertTrue(result["warnings"])
            # la cartella processed/bp e' comunque stata rimossa -> variante "deleted"
            self.assertEqual(result["deleted_variants"], ["bp"])
            self.assertFalse(os.path.isdir(os.path.join(processed, "bp")))
            # la cartella timeslices/bp e' rimasta -> il record va salvato, non perso
            self.assertTrue(os.path.isdir(ts_dir))

            after = pc.load_catalog(tmp)
            ids = [r.get("id") for r in after["timeslices"]]
            self.assertIn("ts_1", ids)
            unassigned = next(g for g in after["raster_groups"] if g.get("id") == "grp_imported")
            self.assertIn("ts_1", unassigned.get("timeslice_ids") or [])


if __name__ == "__main__":
    unittest.main()
