# -*- coding: utf-8 -*-
"""capture_radargram_feature: punti/linee/poligoni disegnati sul radargramma
diventano feature reali (coordinate + profondita' per vertice) nel layer
"Radargram Points/Lines/Polygons"; Build 3D "vertex_depths" segue il disegno.

Si lancia come pacchetto con il Python di QGIS:
    python-qgis.bat -m unittest test.test_radargram_capture
"""

import importlib
import json
import os
import pathlib
import sys
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_PLUGIN_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(_PLUGIN_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR.parent))

try:
    from qgis.PyQt.QtCore import QSettings
    from qgis.core import (
        QgsApplication,
        QgsCoordinateReferenceSystem,
        QgsProject,
        QgsVectorLayer,
        QgsWkbTypes,
    )
    _CTM = importlib.import_module("rasterlinker_feature_interpolation.catalog_tools_mixin")
    _TCM = importlib.import_module("rasterlinker_feature_interpolation.trace_capture_mixin")
    _B3D = importlib.import_module("rasterlinker_feature_interpolation.trace_build3d_mixin")
    HAS_QGIS = True
except Exception:
    HAS_QGIS = False

_QGIS_APP = None


def _ensure_qgis_app():
    global _QGIS_APP
    if not HAS_QGIS:
        return None
    app = QgsApplication.instance()
    if app is None:
        _QGIS_APP = QgsApplication([], False)
        _QGIS_APP.initQgis()
        app = _QGIS_APP
    return app


if HAS_QGIS:
    class _Harness(_CTM.CatalogToolsMixin, _TCM.TraceCaptureMixin, _B3D.TraceBuild3DMixin):
        """Stesso mix di TraceCaptureMixin/TraceBuild3DMixin/CatalogToolsMixin usato dal
        plugin vero (GeoSurveyStudioPlugin): capture_radargram_feature vive nel primo,
        Build 3D nel secondo, la gestione dei gruppi layer nel terzo."""

        def __init__(self):
            self.iface = None
            self.settings = QSettings()
            self.plugin_layer_root_name = "GeoSurvey Studio"
            self.project_manager_dialog = None
            self.settings_key_active_project = "GeoSurveyStudio/active_project_root"
            self.settings_key_default_import_crs = "GeoSurveyStudio/default_import_crs_authid"
            self.trace_line_layer_id = None
            self.trace_connected_layer_ids = set()
            self.trace_z_grid_cache = {}
            self.dlg = None
            self._project_root = ""    # impostata dal test (cartella temporanea), vedi setUp

        def _ui_parent(self):
            return None

        def _active_project_root(self):
            # Evita QSettings reali (persisterebbero fra i test) e il popup modale
            # di _require_project_root quando non c'e' un progetto: il test imposta
            # una cartella temporanea reale prima di chiamare capture_radargram_feature.
            return self._project_root


@unittest.skipUnless(HAS_QGIS, "QGIS runtime is not available")
class CaptureRadargramFeatureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._qgis_app = _ensure_qgis_app()

    def setUp(self):
        QgsProject.instance().removeAllMapLayers()
        QgsProject.instance().setCrs(QgsCoordinateReferenceSystem("EPSG:32632"))
        self._proj_tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.harness = _Harness()
        self.harness._project_root = self._proj_tmp.name

    def tearDown(self):
        QgsProject.instance().removeAllMapLayers()
        self._proj_tmp.cleanup()

    def test_capture_line_creates_layer_with_per_vertex_depths(self):
        h = self.harness
        feat = h.capture_radargram_feature(
            kind="line", variant_name="Lavoro", profile_stem="Swath2", channel=0, epsg=32632,
            vertices=[
                {"east": 500000.0, "north": 10.0, "depth_m": 0.05},
                {"east": 500001.0, "north": 10.0, "depth_m": 0.10},
                {"east": 500002.0, "north": 10.0, "depth_m": 0.07},
            ],
        )
        self.assertIsNotNone(feat)
        layers = QgsProject.instance().mapLayersByName("Radargram Lines")
        self.assertEqual(len(layers), 1)
        layer = layers[0]
        self.assertEqual(layer.geometryType(), QgsWkbTypes.LineGeometry)
        self.assertEqual(layer.featureCount(), 1)

        stored = next(layer.getFeatures())
        self.assertEqual(stored.attribute("profile_stem"), "Swath2")
        self.assertEqual(stored.attribute("channel"), 0)
        self.assertEqual(stored.attribute("group_name"), "Lavoro")
        self.assertEqual(stored.attribute("z_source"), "radargram")
        self.assertEqual(stored.attribute("z_mode"), "profile_pick")
        self.assertAlmostEqual(stored.attribute("depth_from"), 0.05, places=6)
        self.assertAlmostEqual(stored.attribute("depth_to"), 0.10, places=6)
        self.assertTrue(str(stored.attribute("trace_id") or "").strip())

        vd = json.loads(stored.attribute("vertex_depths"))
        self.assertEqual([row["d"] for row in vd], [0.05, 0.10, 0.07])
        self.assertTrue(all(row["s"] == "hit" for row in vd))

        pts = stored.geometry().asPolyline()
        self.assertEqual(len(pts), 3)
        self.assertAlmostEqual(pts[1].x(), 500001.0, places=6)
        self.assertAlmostEqual(pts[1].y(), 10.0, places=6)

        # una seconda geometria riusa lo stesso layer, non ne crea un altro
        h.capture_radargram_feature(
            kind="line", variant_name="Lavoro", profile_stem="Swath2", channel=0, epsg=32632,
            vertices=[{"east": 500003.0, "north": 10.0, "depth_m": 0.02},
                      {"east": 500004.0, "north": 10.0, "depth_m": 0.03}],
        )
        self.assertEqual(len(QgsProject.instance().mapLayersByName("Radargram Lines")), 1)
        self.assertEqual(layer.featureCount(), 2)

    def test_capture_point_and_polygon_use_their_own_layers(self):
        h = self.harness
        h.capture_radargram_feature(
            kind="point", variant_name="Lavoro", profile_stem="Swath2", channel=1, epsg=32632,
            vertices=[{"east": 500000.5, "north": 10.0, "depth_m": 0.08}],
        )
        h.capture_radargram_feature(
            kind="polygon", variant_name="Lavoro", profile_stem="Swath2", channel=1, epsg=32632,
            vertices=[
                {"east": 500000.0, "north": 10.0, "depth_m": 0.05},
                {"east": 500001.0, "north": 10.0, "depth_m": 0.05},
                {"east": 500001.0, "north": 10.5, "depth_m": 0.10},
            ],
        )
        points = QgsProject.instance().mapLayersByName("Radargram Points")
        polygons = QgsProject.instance().mapLayersByName("Radargram Polygons")
        self.assertEqual(len(points), 1)
        self.assertEqual(len(polygons), 1)
        self.assertEqual(points[0].geometryType(), QgsWkbTypes.PointGeometry)
        self.assertEqual(polygons[0].geometryType(), QgsWkbTypes.PolygonGeometry)
        self.assertEqual(points[0].featureCount(), 1)
        self.assertEqual(polygons[0].featureCount(), 1)
        # nessuna interferenza con il layer delle linee
        self.assertEqual(QgsProject.instance().mapLayersByName("Radargram Lines"), [])

    def test_capture_persists_to_a_real_gpkg_file_not_a_memory_layer(self):
        h = self.harness
        h.capture_radargram_feature(
            kind="line", variant_name="Lavoro", profile_stem="Swath2", channel=0, epsg=32632,
            vertices=[{"east": 500000.0, "north": 10.0, "depth_m": 0.05},
                      {"east": 500001.0, "north": 10.0, "depth_m": 0.10}],
        )
        gpkg_path = os.path.join(self._proj_tmp.name, "vector_layers", "radargram_traces.gpkg")
        self.assertTrue(os.path.isfile(gpkg_path))
        layer = QgsProject.instance().mapLayersByName("Radargram Lines")[0]
        self.assertEqual(layer.dataProvider().name(), "ogr")   # non "memory": sopravvive alla chiusura

        # riaprendo il layer direttamente dal file (come dopo un riavvio di QGIS) la
        # feature c'e' ancora - prova reale di persistenza, non solo in-memory.
        reopened = QgsVectorLayer(f"{gpkg_path}|layername=Radargram Lines", "reopened", "ogr")
        self.assertTrue(reopened.isValid())
        self.assertEqual(reopened.featureCount(), 1)

    def test_capture_does_not_overwrite_other_layers_already_in_the_gpkg(self):
        h = self.harness
        h.capture_radargram_feature(
            kind="point", variant_name="Lavoro", profile_stem="Swath2", channel=0, epsg=32632,
            vertices=[{"east": 500000.0, "north": 10.0, "depth_m": 0.05}],
        )
        h.capture_radargram_feature(
            kind="line", variant_name="Lavoro", profile_stem="Swath2", channel=0, epsg=32632,
            vertices=[{"east": 500000.0, "north": 10.0, "depth_m": 0.05},
                      {"east": 500001.0, "north": 10.0, "depth_m": 0.10}],
        )
        gpkg_path = os.path.join(self._proj_tmp.name, "vector_layers", "radargram_traces.gpkg")
        points = QgsVectorLayer(f"{gpkg_path}|layername=Radargram Points", "p", "ogr")
        lines = QgsVectorLayer(f"{gpkg_path}|layername=Radargram Lines", "l", "ogr")
        self.assertTrue(points.isValid())
        self.assertTrue(lines.isValid())
        self.assertEqual(points.featureCount(), 1)
        self.assertEqual(lines.featureCount(), 1)

    def test_capture_auto_fills_amplitude_and_preset_velocity_fields(self):
        h = self.harness
        h.capture_radargram_feature(
            kind="line", variant_name="Lavoro", profile_stem="Swath2", channel=0, epsg=32632,
            vertices=[
                {"east": 500000.0, "north": 10.0, "depth_m": 0.05, "amp": 0.4},
                {"east": 500001.0, "north": 10.0, "depth_m": 0.10, "amp": 0.8},
            ],
            velocity_m_ns=0.09, rdp=11.1,
        )
        stored = next(QgsProject.instance().mapLayersByName("Radargram Lines")[0].getFeatures())
        self.assertAlmostEqual(stored.attribute("amplitude"), 0.6, places=6)
        self.assertAlmostEqual(stored.attribute("velocity_m_ns"), 0.09, places=6)
        self.assertAlmostEqual(stored.attribute("rdp"), 11.1, places=6)
        vd = json.loads(stored.attribute("vertex_depths"))
        self.assertEqual([row["amp"] for row in vd], [0.4, 0.8])

    def test_query_radargram_features_filters_by_profile_and_channel(self):
        h = self.harness
        h.capture_radargram_feature(
            kind="line", variant_name="Lavoro", profile_stem="Swath2", channel=0, epsg=32632,
            vertices=[{"east": 500000.0, "north": 10.0, "depth_m": 0.05},
                      {"east": 500001.0, "north": 10.0, "depth_m": 0.10}],
        )
        h.capture_radargram_feature(
            kind="point", variant_name="Lavoro", profile_stem="Swath2", channel=1, epsg=32632,
            vertices=[{"east": 500000.5, "north": 10.0, "depth_m": 0.08}],
        )
        same = h.query_radargram_features("Lavoro", "Swath2", 0)
        self.assertEqual(len(same), 1)
        self.assertEqual(same[0]["kind"], "line")
        self.assertEqual(len(same[0]["vertices"]), 2)
        self.assertAlmostEqual(same[0]["vertices"][0]["east"], 500000.0, places=6)
        self.assertAlmostEqual(same[0]["vertices"][0]["depth_m"], 0.05, places=6)

        other_channel = h.query_radargram_features("Lavoro", "Swath2", 1)
        self.assertEqual(len(other_channel), 1)
        self.assertEqual(other_channel[0]["kind"], "point")

        other_profile = h.query_radargram_features("Lavoro", "SwathNone", 0)
        self.assertEqual(other_profile, [])

    def test_capture_rejects_too_few_vertices(self):
        h = self.harness
        with self.assertRaises(ValueError):
            h.capture_radargram_feature(
                kind="polygon", variant_name="Lavoro", profile_stem="Swath2", channel=0, epsg=32632,
                vertices=[
                    {"east": 500000.0, "north": 10.0, "depth_m": 0.05},
                    {"east": 500001.0, "north": 10.0, "depth_m": 0.05},
                ],
            )

    def test_build3d_vertex_depths_mode_follows_the_drawn_profile(self):
        h = self.harness
        h.capture_radargram_feature(
            kind="line", variant_name="Lavoro", profile_stem="Swath2", channel=0, epsg=32632,
            vertices=[
                {"east": 500000.0, "north": 10.0, "depth_m": 0.05},
                {"east": 500001.0, "north": 10.0, "depth_m": 0.20},
            ],
        )
        source_layer = QgsProject.instance().mapLayersByName("Radargram Lines")[0]
        out_layer = h._create_3d_output_layer(source_layer, "Radargram_Lines_3D")
        self.assertIsNotNone(out_layer)

        created, skipped, _reasons = h._build_3d_with_mode(source_layer, out_layer, "vertex_depths")
        self.assertEqual((created, skipped), (1, 0))

        out_feat = next(out_layer.getFeatures())
        self.assertTrue(QgsWkbTypes.hasZ(out_layer.wkbType()))
        z0 = out_feat.geometry().vertexAt(0).z()
        z1 = out_feat.geometry().vertexAt(1).z()
        # la Z segue la profondita' disegnata punto per punto, non un valore costante
        self.assertAlmostEqual(z0, 0.05, places=6)
        self.assertAlmostEqual(z1, 0.20, places=6)
        self.assertEqual(out_feat.attribute("z_mode"), "build3d_vertex_depths")
        self.assertEqual(out_feat.attribute("z_source"), "vertex_depths")

    # -- round 3: geometrie "alla QGIS" (codice libero, sorgente, stile) --------------

    def test_capture_fills_source_and_leaves_anomaly_id_free(self):
        h = self.harness
        h.capture_radargram_feature(
            kind="line", variant_name="Lavoro", profile_stem="Swath2", channel=0, epsg=32632,
            vertices=[{"east": 500000.0, "north": 10.0, "depth_m": 0.05},
                      {"east": 500001.0, "north": 10.0, "depth_m": 0.10}],
        )
        stored = next(QgsProject.instance().mapLayersByName("Radargram Lines")[0].getFeatures())
        self.assertEqual(stored.attribute("source"), "radargramma")
        self.assertEqual(stored.attribute("anomaly_id"), "")   # libero: nessuna numerazione imposta

    def test_capture_does_not_prompt_for_anomaly_id_by_default(self):
        # Il flag e' spento di default sull'harness di test (come sarebbe per
        # qualunque chiamante che non lo imposta esplicitamente): niente
        # dialogo modale, altrimenti il test resterebbe bloccato in attesa
        # di un utente che non c'e'.
        h = self.harness
        calls = []
        h._prompt_trace_interpretation_fields = lambda layer, fid: calls.append((layer, fid))
        h.capture_radargram_feature(
            kind="point", variant_name="Lavoro", profile_stem="Swath2", channel=0, epsg=32632,
            vertices=[{"east": 500000.0, "north": 10.0, "depth_m": 0.05}],
        )
        self.assertEqual(calls, [])

    def test_capture_prompts_for_anomaly_id_when_enabled(self):
        h = self.harness
        h.trace_radargram_prompt_after_draw = True
        calls = []
        h._prompt_trace_interpretation_fields = lambda layer, fid: calls.append((layer.name(), fid))
        h.capture_radargram_feature(
            kind="point", variant_name="Lavoro", profile_stem="Swath2", channel=0, epsg=32632,
            vertices=[{"east": 500000.0, "north": 10.0, "depth_m": 0.05}],
        )
        self.assertEqual(len(calls), 1)
        layer_name, fid = calls[0]
        self.assertEqual(layer_name, "Radargram Points")
        layer = QgsProject.instance().mapLayersByName("Radargram Points")[0]
        feature = layer.getFeature(fid)
        self.assertTrue(feature.isValid())
        # il codice libero deve comparire per primo tra i campi del dialogo
        fields = h._trace_interpretation_fields_for_layer(layer)
        self.assertEqual(fields[0][0], "anomaly_id")

    def test_radargram_layers_get_a_distinct_style_per_geometry_kind(self):
        from qgis.core import QgsSingleSymbolRenderer
        h = self.harness
        h.capture_radargram_feature(
            kind="point", variant_name="Lavoro", profile_stem="Swath2", channel=0, epsg=32632,
            vertices=[{"east": 500000.0, "north": 10.0, "depth_m": 0.05}],
        )
        h.capture_radargram_feature(
            kind="line", variant_name="Lavoro", profile_stem="Swath2", channel=0, epsg=32632,
            vertices=[{"east": 500000.0, "north": 10.0, "depth_m": 0.05},
                      {"east": 500001.0, "north": 10.0, "depth_m": 0.10}],
        )
        points = QgsProject.instance().mapLayersByName("Radargram Points")[0]
        lines = QgsProject.instance().mapLayersByName("Radargram Lines")[0]
        self.assertIsInstance(points.renderer(), QgsSingleSymbolRenderer)
        self.assertIsInstance(lines.renderer(), QgsSingleSymbolRenderer)
        points_color = points.renderer().symbol().color().name()
        lines_color = lines.renderer().symbol().color().name()
        self.assertNotEqual(points_color, lines_color)   # colori diversi per tipo di geometria

    # -- classi di segnale, tempo doppio, modifica e cancellazione ------------------------

    def _two_vertices(self, **extra):
        return dict(
            kind="line", variant_name="Lavoro", profile_stem="Swath2", channel=0, epsg=32632,
            vertices=[{"east": 500000.0, "north": 10.0, "depth_m": 0.05, "twt_ns": 1.0},
                      {"east": 500001.0, "north": 10.0, "depth_m": 0.10, "twt_ns": 2.0}],
            **extra)

    def test_capture_stores_class_confidence_and_two_way_time_per_vertex(self):
        h = self.harness
        h.capture_radargram_feature(**self._two_vertices(
            class_code="riflettore", class_label="Riflettore / interfaccia", confidence="certo",
            attributes={"anomaly_id": "A7", "notes": "continua sul profilo vicino"}))
        stored = next(QgsProject.instance().mapLayersByName("Radargram Lines")[0].getFeatures())
        self.assertEqual(stored.attribute("class_code"), "riflettore")
        self.assertEqual(stored.attribute("class_label"), "Riflettore / interfaccia")
        self.assertEqual(stored.attribute("confidence"), "certo")
        self.assertEqual(stored.attribute("anomaly_id"), "A7")
        self.assertEqual(stored.attribute("notes"), "continua sul profilo vicino")
        vd = json.loads(stored.attribute("vertex_depths"))
        self.assertEqual([row["t"] for row in vd], [1.0, 2.0])
        self.assertEqual([row["d"] for row in vd], [0.05, 0.10])

    def test_vertices_without_time_still_work(self):
        h = self.harness
        h.capture_radargram_feature(
            kind="point", variant_name="Lavoro", profile_stem="Swath2", channel=0, epsg=32632,
            vertices=[{"east": 500000.0, "north": 10.0, "depth_m": 0.05}])
        stored = next(QgsProject.instance().mapLayersByName("Radargram Points")[0].getFeatures())
        self.assertNotIn("t", json.loads(stored.attribute("vertex_depths"))[0])
        found = h.query_radargram_features("Lavoro", "Swath2", 0)
        self.assertIsNone(found[0]["vertices"][0]["twt_ns"])

    def test_trace_ids_are_unique_even_within_the_same_second(self):
        h = self.harness
        ids = set()
        for _ in range(5):
            feat = h.capture_radargram_feature(
                kind="point", variant_name="Lavoro", profile_stem="Swath2", channel=0, epsg=32632,
                vertices=[{"east": 500000.0, "north": 10.0, "depth_m": 0.05}])
            ids.add(feat.attribute("trace_id"))
        self.assertEqual(len(ids), 5)

    def test_query_returns_identifier_class_and_texts(self):
        h = self.harness
        h.capture_radargram_feature(**self._two_vertices(
            class_code="riflettore", confidence="possibile", attributes={"interpretation": "piano d'uso"}))
        found = h.query_radargram_features("Lavoro", "Swath2", 0)
        self.assertEqual(len(found), 1)
        feat = found[0]
        self.assertTrue(feat["id"].startswith("tr_"))
        self.assertEqual(feat["class_code"], "riflettore")
        self.assertEqual(feat["confidence"], "possibile")
        self.assertEqual(feat["interpretation"], "piano d'uso")
        self.assertEqual(feat["anomaly_id"], "")
        self.assertEqual([v["twt_ns"] for v in feat["vertices"]], [1.0, 2.0])

    def test_update_moves_vertices_and_recomputes_depth_range(self):
        h = self.harness
        feat = h.capture_radargram_feature(**self._two_vertices(class_code="riflettore"))
        tid = feat.attribute("trace_id")
        h.update_radargram_feature(tid, epsg=32632, vertices=[
            {"east": 500000.0, "north": 10.0, "depth_m": 0.20, "twt_ns": 4.0},
            {"east": 500002.5, "north": 10.0, "depth_m": 0.30, "twt_ns": 6.0},
            {"east": 500004.0, "north": 10.0, "depth_m": 0.25, "twt_ns": 5.0}])
        layer = QgsProject.instance().mapLayersByName("Radargram Lines")[0]
        self.assertEqual(layer.featureCount(), 1)
        stored = next(layer.getFeatures())
        self.assertEqual(stored.attribute("trace_id"), tid)               # stesso identificativo
        self.assertEqual(stored.attribute("class_code"), "riflettore")    # attributi non toccati
        self.assertEqual(len(stored.geometry().asPolyline()), 3)
        self.assertAlmostEqual(stored.attribute("depth_from"), 0.20, places=6)
        self.assertAlmostEqual(stored.attribute("depth_to"), 0.30, places=6)
        self.assertEqual([r["t"] for r in json.loads(stored.attribute("vertex_depths"))], [4.0, 6.0, 5.0])

    def test_update_changes_only_the_allowed_text_attributes(self):
        h = self.harness
        feat = h.capture_radargram_feature(**self._two_vertices(class_code="riflettore"))
        tid = feat.attribute("trace_id")
        h.update_radargram_feature(tid, attributes={
            "class_code": "struttura", "confidence": "certo", "anomaly_id": "M1",
            "notes": "muro", "trace_id": "hack", "profile_stem": "altro"})
        stored = next(QgsProject.instance().mapLayersByName("Radargram Lines")[0].getFeatures())
        self.assertEqual(stored.attribute("class_code"), "struttura")
        self.assertEqual(stored.attribute("anomaly_id"), "M1")
        self.assertEqual(stored.attribute("trace_id"), tid)
        self.assertEqual(stored.attribute("profile_stem"), "Swath2")
        self.assertEqual(len(stored.geometry().asPolyline()), 2)           # geometria intatta

    def test_update_rejects_unknown_feature_and_too_few_vertices(self):
        h = self.harness
        with self.assertRaises(KeyError):
            h.update_radargram_feature("tr_non_esiste", attributes={"notes": "x"})
        feat = h.capture_radargram_feature(**self._two_vertices())
        with self.assertRaises(ValueError):
            h.update_radargram_feature(
                feat.attribute("trace_id"), epsg=32632,
                vertices=[{"east": 500000.0, "north": 10.0, "depth_m": 0.05}])

    def test_delete_removes_the_feature_and_it_can_be_restored_with_the_same_id(self):
        h = self.harness
        feat = h.capture_radargram_feature(**self._two_vertices(class_code="riflettore", confidence="certo"))
        tid = feat.attribute("trace_id")
        layer = QgsProject.instance().mapLayersByName("Radargram Lines")[0]
        snapshot = h.query_radargram_features("Lavoro", "Swath2", 0)[0]
        self.assertTrue(h.delete_radargram_feature(tid))
        self.assertEqual(layer.featureCount(), 0)
        self.assertFalse(h.delete_radargram_feature(tid))                 # gia' sparita
        self.assertEqual(h.query_radargram_features("Lavoro", "Swath2", 0), [])
        # annulla: si ricrea con lo stesso identificativo e gli stessi dati
        h.capture_radargram_feature(**self._two_vertices(
            class_code=snapshot["class_code"], confidence=snapshot["confidence"], trace_id=tid))
        back = h.query_radargram_features("Lavoro", "Swath2", 0)
        self.assertEqual([f["id"] for f in back], [tid])
        self.assertEqual(back[0]["class_code"], "riflettore")

    def test_source_defaults_to_radargramma_and_a_fitted_hyperbola_keeps_its_own(self):
        h = self.harness
        h.capture_radargram_feature(**self._two_vertices())
        h.capture_radargram_feature(
            kind="point", variant_name="Lavoro", profile_stem="Swath2", channel=0, epsg=32632,
            vertices=[{"east": 500000.5, "north": 10.0, "depth_m": 0.12, "twt_ns": 2.4}],
            velocity_m_ns=0.08, rdp=14.06, class_code="iperbole", source="iperbole_fit")
        found = {f["kind"]: f for f in h.query_radargram_features("Lavoro", "Swath2", 0)}
        self.assertEqual(found["line"]["source"], "radargramma")
        self.assertEqual(found["point"]["source"], "iperbole_fit")
        self.assertAlmostEqual(found["point"]["velocity_m_ns"], 0.08, places=6)
        self.assertAlmostEqual(found["point"]["rdp"], 14.06, places=6)
        self.assertEqual(found["point"]["vertices"][0]["twt_ns"], 2.4)

    def test_modal_dialog_is_off_by_default_on_the_plugin(self):
        h = self.harness
        h._init_trace_capture()
        self.assertFalse(h.trace_radargram_prompt_after_draw)


if __name__ == "__main__":
    unittest.main()
