# -*- coding: utf-8 -*-
"""project_zones: aree indagate dentro uno stesso progetto (logica pura)."""

import importlib
import json
import os
import pathlib
import sys
import tempfile
import unittest

_PLUGIN_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(_PLUGIN_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR.parent))

pz = importlib.import_module("rasterlinker_feature_interpolation.project_zones")


def _tmp():
    try:
        return tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    except TypeError:  # python < 3.10
        return tempfile.TemporaryDirectory()


class ZonesTest(unittest.TestCase):
    def test_empty_project_has_only_default_zone(self):
        with _tmp() as root:
            self.assertEqual(pz.list_zones(root), [pz.DEFAULT_ZONE])
            self.assertEqual(pz.active_zone(root), pz.DEFAULT_ZONE)
            self.assertEqual(pz.zone_base(root), os.path.normpath(os.path.join(root, "Processing")))

    def test_legacy_flat_project_json_becomes_the_default_zone(self):
        with _tmp() as root:
            os.makedirs(os.path.join(root, "metadata"))
            legacy = {"version": "1.0", "created": "2026-01-01T00:00:00",
                      "profiles": [{"filename": "a.ogpr", "path": "raw/a.ogpr"}],
                      "last_processing": "processed/Lavoro", "ui_state": {"ve": 3.0}}
            with open(pz.project_json_path(root), "w", encoding="utf-8") as fh:
                json.dump(legacy, fh)
            self.assertEqual(pz.list_zones(root), [pz.DEFAULT_ZONE])
            p = pz.get_payload(root)
            self.assertEqual(p["profiles"][0]["filename"], "a.ogpr")
            self.assertEqual(p["ui_state"]["ve"], 3.0)
            self.assertEqual(pz.zone_base(root), os.path.normpath(os.path.join(root, "Processing")))

    def test_add_zone_creates_folders_and_keeps_other_zone_untouched(self):
        with _tmp() as root:
            pz.set_payload(root, {"profiles": [{"filename": "a.ogpr"}]})
            name = pz.add_zone(root, "Area Nord")
            self.assertEqual(name, "Area Nord")
            self.assertEqual(pz.active_zone(root), "Area Nord")
            base = pz.zone_base(root)
            for sub in ("raw", "processed", "timeslices"):
                self.assertTrue(os.path.isdir(os.path.join(base, sub)))
            self.assertNotEqual(base, pz.zone_base(root, pz.DEFAULT_ZONE))
            pz.set_payload(root, {"profiles": [{"filename": "b.ogpr"}]})
            self.assertEqual(pz.get_payload(root, pz.DEFAULT_ZONE)["profiles"][0]["filename"], "a.ogpr")
            self.assertEqual(pz.get_payload(root, "Area Nord")["profiles"][0]["filename"], "b.ogpr")

    def test_unique_names_and_switching(self):
        with _tmp() as root:
            a = pz.add_zone(root, "Zona")
            b = pz.add_zone(root, "zona")           # stesso nome (maiuscole) -> suffisso
            self.assertEqual(a, "Zona")
            self.assertNotEqual(b.casefold(), a.casefold())
            self.assertTrue(pz.set_active_zone(root, pz.DEFAULT_ZONE))
            self.assertEqual(pz.active_zone(root), pz.DEFAULT_ZONE)
            self.assertFalse(pz.set_active_zone(root, "inesistente"))

    def test_zone_name_cannot_escape_the_project(self):
        with _tmp() as root:
            name = pz.add_zone(root, "../../fuori")
            base = pz.zone_base(root, name)
            self.assertTrue(base.startswith(os.path.normpath(os.path.join(root, "Processing"))))


if __name__ == "__main__":
    unittest.main()
