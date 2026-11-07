# -*- coding: utf-8 -*-
"""Passo 10: provenienza delle varianti (variant_graph.json)."""

import tempfile
import unittest

import variant_provenance as vp


class VariantProvenanceTest(unittest.TestCase):
    def test_load_missing_is_empty(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(vp.load(d), {})

    def test_set_and_reload(self):
        with tempfile.TemporaryDirectory() as d:
            vp.set_entry(d, "bp", parent=None, chain=["bandpass"], step="bandpass", n_profiles=3)
            vp.set_entry(d, "bp_bg", parent="bp", chain=["bandpass", "bg_removal"],
                         step="bg_removal", n_profiles=3)
            e = vp.load(d)
            self.assertIsNone(e["bp"]["parent"])
            self.assertEqual(e["bp_bg"]["parent"], "bp")
            self.assertEqual(e["bp_bg"]["step"], "bg_removal")
            self.assertEqual(e["bp_bg"]["chain"], ["bandpass", "bg_removal"])
            self.assertEqual(e["bp"]["n_profiles"], 3)

    def test_self_parent_neutralised(self):
        with tempfile.TemporaryDirectory() as d:
            vp.set_entry(d, "x", parent="x")
            self.assertIsNone(vp.load(d)["x"]["parent"])

    def test_cycle_broken_on_load_via_build_forest(self):
        with tempfile.TemporaryDirectory() as d:
            vp.set_entry(d, "a", parent=None)
            vp.set_entry(d, "b", parent="a")
            # forza un ciclo a->b->a scrivendo a mano
            e = vp.load(d)
            e["a"]["parent"] = "b"
            vp.save(d, e)
            vp.set_entry(d, "b", parent="a")     # set_entry richiama _break_cycles
            e2 = vp.load(d)
            # almeno uno dei due torna radice: nessun ciclo
            self.assertTrue(e2["a"]["parent"] is None or e2["b"]["parent"] is None)

    def test_build_forest_nests_and_drops_absent(self):
        entries = {
            "root": {"parent": None},
            "child": {"parent": "root"},
            "grand": {"parent": "child"},
            "orphan": {"parent": "ghost"},   # ghost non presente -> orphan diventa radice
        }
        forest = vp.build_forest(entries, ["root", "child", "grand", "orphan", "no_entry"])
        names = sorted(n["name"] for n in forest)
        self.assertEqual(names, ["no_entry", "orphan", "root"])
        root = next(n for n in forest if n["name"] == "root")
        self.assertEqual(root["children"][0]["name"], "child")
        self.assertEqual(root["children"][0]["children"][0]["name"], "grand")

    def test_remove_entry_reparents_children(self):
        with tempfile.TemporaryDirectory() as d:
            vp.set_entry(d, "a", parent=None)
            vp.set_entry(d, "b", parent="a")
            vp.set_entry(d, "c", parent="b")
            vp.remove_entry(d, "b")
            e = vp.load(d)
            self.assertNotIn("b", e)
            self.assertEqual(e["c"]["parent"], "a")   # ri-agganciato al nonno


if __name__ == "__main__":
    unittest.main()
