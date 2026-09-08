import os
import tempfile
import unittest

import numpy as np

from variant_graph import (
    VariantGraph,
    VariantGraphError,
    ROOT_ID,
    GRAPH_FILENAME,
)
from gpr_processing import apply_pipeline, DEFAULT_PIPELINE


def _synth(n_s=192, n_t=32, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n_s, n_t)) * 3.0).astype(np.float32)


class TestVariantGraphModel(unittest.TestCase):

    def test_root_exists(self):
        g = VariantGraph()
        self.assertIn(ROOT_ID, g)
        self.assertEqual(g.node(ROOT_ID).parent_id, None)
        self.assertEqual(g.node(ROOT_ID).step, None)
        self.assertEqual(g.leaves(), [ROOT_ID])

    def test_chain_and_topology(self):
        g = VariantGraph()
        a = g.add_node(ROOT_ID, "dewow", {"dewow_win": 20})
        b = g.add_node(a.id, "bandpass", {"bp_low_mhz": 120.0, "bp_high_mhz": 900.0})
        self.assertEqual([n.id for n in g.ancestors(b.id)], [ROOT_ID, a.id, b.id])
        self.assertEqual([n.id for n in g.children(ROOT_ID)], [a.id])
        self.assertEqual(g.leaves(), [b.id])
        self.assertEqual(g.path_labels(b.id), ["raw", "dewow", "bandpass"])

    def test_add_node_validation(self):
        g = VariantGraph()
        with self.assertRaises(VariantGraphError):
            g.add_node(ROOT_ID, "not_a_step")
        with self.assertRaises(VariantGraphError):
            g.add_node("ghost", "dewow")
        a = g.add_node(ROOT_ID, "dewow")
        with self.assertRaises(VariantGraphError):
            g.add_node(ROOT_ID, "dewow", node_id=a.id)
        with self.assertRaises(VariantGraphError):
            g.add_node(ROOT_ID, "dewow", scope="bogus")

    def test_fresh_id_disambiguates(self):
        g = VariantGraph()
        n1 = g.add_node(ROOT_ID, "notch")
        n2 = g.add_node(n1.id, "notch")
        self.assertEqual(n1.id, "notch")
        self.assertEqual(n2.id, "notch_2")

    def test_resolve_merges_defaults_and_enables_steps(self):
        g = VariantGraph()
        a = g.add_node(ROOT_ID, "dewow", {"dewow_win": 24})
        b = g.add_node(a.id, "bandpass", {"bp_low_mhz": 100.0, "bp_high_mhz": 800.0})
        merged, steps = g.resolve(b.id)
        self.assertEqual(steps, ["dewow", "bandpass"])
        self.assertEqual(merged["chain_order"], ["dewow", "bandpass"])
        self.assertTrue(merged["dewow"])
        self.assertTrue(merged["bandpass"])
        self.assertEqual(merged["dewow_win"], 24)
        self.assertEqual(merged["bp_low_mhz"], 100.0)
        # untouched steps keep DEFAULT_PIPELINE value
        self.assertEqual(merged["migration"], DEFAULT_PIPELINE["migration"])

    def test_content_key_stability(self):
        g = VariantGraph()
        a = g.add_node(ROOT_ID, "bandpass", {"bp_low_mhz": 100.0, "bp_high_mhz": 800.0})
        k1 = g.content_key(a.id)
        self.assertEqual(k1, g.content_key(a.id))
        g.set_params(a.id, {"bp_low_mhz": 150.0, "bp_high_mhz": 800.0})
        self.assertNotEqual(k1, g.content_key(a.id))

    def test_remove_subtree(self):
        g = VariantGraph()
        a = g.add_node(ROOT_ID, "dewow")
        b = g.add_node(a.id, "bandpass")
        c = g.add_node(b.id, "agc")
        dropped = g.remove_subtree(b.id)
        self.assertCountEqual(dropped, [b.id, c.id])
        self.assertNotIn(b.id, g)
        self.assertNotIn(c.id, g)
        self.assertIn(a.id, g)
        with self.assertRaises(VariantGraphError):
            g.remove_subtree(ROOT_ID)


class TestVariantGraphCompute(unittest.TestCase):

    def test_materialize_root_is_copy(self):
        g = VariantGraph()
        raw = [_synth()]
        out = g.materialize_arrays(ROOT_ID, raw, dt_ns=0.3)
        np.testing.assert_array_equal(out[0], raw[0])
        self.assertIsNot(out[0], raw[0])

    def test_materialize_matches_direct_apply_pipeline(self):
        g = VariantGraph()
        a = g.add_node(ROOT_ID, "dewow", {"dewow_win": 16})
        b = g.add_node(a.id, "bandpass", {"bp_low_mhz": 120.0, "bp_high_mhz": 900.0,
                                          "bp_taper_frac": 0.2})
        raw = _synth(seed=3)
        got = g.materialize_arrays(b.id, [raw], dt_ns=0.4)[0]

        merged, steps = g.resolve(b.id)
        expect = apply_pipeline(raw, merged, dt_ns=0.4, normalize_output=False,
                                chain_order=steps)
        np.testing.assert_allclose(got, expect, rtol=1e-6, atol=1e-6)

    def test_materialize_cache_reuse(self):
        g = VariantGraph()
        a = g.add_node(ROOT_ID, "dewow", {"dewow_win": 16})
        b = g.add_node(a.id, "agc", {"agc_win": 64})
        raw = _synth(seed=5)
        cache = {}
        g.materialize_arrays(b.id, [raw], dt_ns=0.2, cache=cache)
        step0_first = cache[0][0][1]           # (fingerprint, array) del 1° step
        # cambia solo il parametro dell'ultimo step: il 1° step deve venire riusato
        g.set_params(b.id, {"agc_win": 128})
        merged2, _ = g.resolve(b.id)
        g.materialize_arrays(b.id, [raw], dt_ns=0.2, cache=cache)
        step0_after = cache[0][0][1]
        self.assertIs(step0_first, step0_after)

    def test_branching_independent(self):
        g = VariantGraph()
        base = g.add_node(ROOT_ID, "dewow", {"dewow_win": 16})
        left = g.add_node(base.id, "bandpass", {"bp_low_mhz": 100.0, "bp_high_mhz": 500.0})
        right = g.add_node(base.id, "bandpass", {"bp_low_mhz": 300.0, "bp_high_mhz": 1200.0})
        raw = _synth(seed=7)
        ol = g.materialize_arrays(left.id, [raw], dt_ns=0.4)[0]
        orr = g.materialize_arrays(right.id, [raw], dt_ns=0.4)[0]
        self.assertFalse(np.allclose(ol, orr))


class TestVariantGraphPersistence(unittest.TestCase):

    def test_save_load_roundtrip(self):
        g = VariantGraph()
        a = g.add_node(ROOT_ID, "dewow", {"dewow_win": 22}, label="dw")
        b = g.add_node(a.id, "notch", {"notch_freq_mhz": 150.0}, scope="survey")
        g.add_node(a.id, "agc", {"agc_win": 96})
        with tempfile.TemporaryDirectory() as d:
            path = g.save(d)
            self.assertTrue(os.path.isfile(path))
            self.assertTrue(path.endswith(GRAPH_FILENAME))
            g2 = VariantGraph.load(d)
        self.assertCountEqual(g2.ids(), g.ids())
        self.assertEqual(g2.node(b.id).scope, "survey")
        self.assertEqual(g2.node(a.id).label, "dw")
        m1, s1 = g.resolve(b.id)
        m2, s2 = g2.resolve(b.id)
        self.assertEqual(s1, s2)
        self.assertEqual(m1["notch_freq_mhz"], m2["notch_freq_mhz"])

    def test_load_missing_returns_fresh(self):
        with tempfile.TemporaryDirectory() as d:
            g = VariantGraph.load(d)
        self.assertEqual(g.ids(), [ROOT_ID])

    def test_from_json_topological_insert(self):
        payload = {
            "format": "rasterlinker_variant_graph_v1",
            "root": ROOT_ID,
            "nodes": [
                {"id": "raw", "parent_id": None, "step": None, "params": {}},
                # figlio prima del genitore: from_json deve riordinare
                {"id": "c", "parent_id": "b", "step": "agc", "params": {}},
                {"id": "b", "parent_id": "raw", "step": "bandpass", "params": {}},
            ],
        }
        g = VariantGraph.from_json(payload)
        self.assertEqual([n.id for n in g.ancestors("c")], ["raw", "b", "c"])

    def test_from_json_orphan_raises(self):
        payload = {"nodes": [
            {"id": "raw", "parent_id": None, "step": None},
            {"id": "x", "parent_id": "does_not_exist", "step": "agc"},
        ]}
        with self.assertRaises(VariantGraphError):
            VariantGraph.from_json(payload)

    def test_from_legacy_manifest_linearizes_active_steps(self):
        manifest = {
            "format": "rasterlinker_gpr_variant_v1",
            "name": "bandpass+bg",
            "pipeline_params": {
                "dewow": True, "dewow_win": 16,
                "bg_removal": True, "bg_mode": "line_by_line",
                "bandpass": True, "bp_low_mhz": 120.0, "bp_high_mhz": 900.0,
                "agc": False,
            },
            "chain_order": ["dewow", "bg_removal", "bandpass", "agc"],
        }
        g = VariantGraph.from_legacy_manifest(manifest)
        leaf = g.leaves()[0]
        _, steps = g.resolve(leaf)
        self.assertEqual(steps, ["dewow", "bg_removal", "bandpass"])  # agc OFF -> escluso


if __name__ == "__main__":
    unittest.main()
