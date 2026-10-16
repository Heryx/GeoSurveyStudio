# -*- coding: utf-8 -*-
"""gpr_annotation_classes: classi di segnale per le annotazioni (senza Qt)."""

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

_AC = importlib.import_module("rasterlinker_feature_interpolation.gpr_annotation_classes")


class AnnotationClassesTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_defaults_cover_the_three_geometries_and_fit_the_number_keys(self):
        classes = _AC.default_classes()
        self.assertLessEqual(len(classes), _AC.MAX_CLASSES)
        self.assertEqual({c["geometry"] for c in classes}, set(_AC.GEOMETRIES))
        self.assertEqual(len({c["code"] for c in classes}), len(classes))
        by_code = {c["code"]: c for c in classes}
        self.assertEqual(by_code["iperbole"]["geometry"], "point")
        self.assertEqual(by_code["riflettore"]["geometry"], "line")
        self.assertEqual(by_code["attenuazione"]["geometry"], "polygon")

    def test_without_a_file_the_defaults_are_used(self):
        self.assertEqual(_AC.load_classes(self.root), _AC.default_classes())
        self.assertEqual(_AC.load_classes(None), _AC.default_classes())

    def test_save_and_load_round_trip(self):
        saved = _AC.save_classes(self.root, [
            {"code": "vuoto", "label": "Vuoto / cavita'", "geometry": "polygon", "color": "#112233"},
            {"label": "Muro", "geometry": "line", "color": "#abc"},
        ])
        self.assertEqual([c["code"] for c in saved], ["vuoto", "muro"])
        self.assertEqual(saved[1]["color"], "#aabbcc")
        self.assertEqual(_AC.load_classes(self.root), saved)
        self.assertTrue(os.path.isfile(_AC.classes_path(self.root)))

    def test_broken_file_falls_back_to_defaults(self):
        os.makedirs(os.path.dirname(_AC.classes_path(self.root)), exist_ok=True)
        with open(_AC.classes_path(self.root), "w", encoding="utf-8") as fh:
            fh.write("{non e' json")
        self.assertEqual(_AC.load_classes(self.root), _AC.default_classes())

    def test_cleaning_drops_duplicates_reserved_code_and_extra_entries(self):
        raw = [{"code": "a", "label": "A"}, {"code": "a", "label": "A bis"},
               {"code": _AC.UNCLASSIFIED_CODE, "label": "x"}, "non un dict", {"label": ""}]
        raw += [{"code": f"c{i}", "label": f"C{i}"} for i in range(20)]
        cleaned = _AC.clean_classes(raw)
        self.assertEqual(len(cleaned), _AC.MAX_CLASSES)
        self.assertEqual(cleaned[0]["code"], "a")
        self.assertEqual(len({c["code"] for c in cleaned}), len(cleaned))
        self.assertTrue(all(c["geometry"] in _AC.GEOMETRIES for c in cleaned))

    def test_unknown_class_gets_the_neutral_color_and_the_code_as_label(self):
        classes = _AC.default_classes()
        self.assertEqual(_AC.color_for(classes, "boh"), _AC.UNCLASSIFIED_COLOR)
        self.assertEqual(_AC.color_for(classes, ""), _AC.UNCLASSIFIED_COLOR)
        self.assertEqual(_AC.label_for(classes, "boh"), "boh")
        self.assertEqual(_AC.label_for(classes, ""), "Non classificato")
        self.assertEqual(_AC.label_for(classes, "iperbole"), "Iperbole di diffrazione")

    def test_slug_and_unique_code(self):
        self.assertEqual(_AC.slugify("Perdita di segnale (zona d'ombra)"), "perdita_di_segnale_zona_d_ombra")
        self.assertEqual(_AC.slugify("Cavità"), "cavita")
        self.assertEqual(_AC.slugify("!!!"), "classe")
        classes = [{"code": "muro", "label": "Muro", "geometry": "line", "color": "#fff"}]
        self.assertEqual(_AC.unique_code(classes, "Muro"), "muro_2")
        self.assertEqual(_AC.unique_code(classes, "Fossa"), "fossa")

    def test_renaming_keeps_the_code(self):
        saved = _AC.save_classes(self.root, [{"code": "muro", "label": "Muro", "geometry": "line"}])
        saved[0]["label"] = "Muro in laterizio"
        _AC.save_classes(self.root, saved)
        reloaded = _AC.load_classes(self.root)
        self.assertEqual(reloaded[0]["code"], "muro")
        self.assertEqual(reloaded[0]["label"], "Muro in laterizio")

    def test_file_is_plain_json_that_a_person_can_edit(self):
        _AC.save_classes(self.root, _AC.default_classes())
        with open(_AC.classes_path(self.root), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self.assertEqual(data["version"], 1)
        self.assertEqual(len(data["classes"]), len(_AC.default_classes()))

    def test_confidence_labels(self):
        self.assertEqual(_AC.confidence_label("certo"), "Certo")
        self.assertEqual(_AC.confidence_label("boh"), "")
        self.assertIn(_AC.DEFAULT_CONFIDENCE, [c for c, _l in _AC.CONFIDENCE_LEVELS])


if __name__ == "__main__":
    unittest.main()
