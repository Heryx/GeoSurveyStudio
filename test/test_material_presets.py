# -*- coding: utf-8 -*-
"""gpr_material_presets: valori di velocita' tipici per materiale/condizione -
nessuna dipendenza Qt/QGIS, gira col Python di sistema."""

import importlib
import pathlib
import sys
import unittest

_PLUGIN_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(_PLUGIN_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR.parent))

_MP = importlib.import_module("rasterlinker_feature_interpolation.gpr_material_presets")


class MaterialPresetsTest(unittest.TestCase):
    def test_list_is_not_empty_and_keys_are_unique(self):
        presets = _MP.MATERIAL_PRESETS
        self.assertGreater(len(presets), 0)
        keys = [p["key"] for p in presets]
        self.assertEqual(len(keys), len(set(keys)))

    def test_values_are_physically_sensible(self):
        c = 299792458.0
        for preset in _MP.MATERIAL_PRESETS:
            v = preset["velocity_m_ns"] * 1e9
            self.assertGreater(v, 0.0)
            self.assertLessEqual(v, c)
            self.assertGreaterEqual(preset["rdp"], 1.0)

    def test_material_preset_lookup(self):
        found = _MP.material_preset("aria")
        self.assertIsNotNone(found)
        self.assertEqual(found["key"], "aria")
        self.assertIsNone(_MP.material_preset("chiave_inesistente"))
        self.assertIsNone(_MP.material_preset(""))

    def test_material_label_falls_back_to_key(self):
        self.assertEqual(_MP.material_label("chiave_inesistente"), "chiave_inesistente")
        label = _MP.material_label("aria")
        self.assertEqual(label, _MP.material_preset("aria")["label"])


if __name__ == "__main__":
    unittest.main()
