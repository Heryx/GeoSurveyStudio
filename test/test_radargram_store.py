# -*- coding: utf-8 -*-
"""radargram_store: accesso ai radargrammi processati di una variante (senza Qt)."""

import importlib
import os
import pathlib
import sys
import tempfile
import types
import unittest

import numpy as np

_PLUGIN_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(_PLUGIN_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR.parent))

_RS = importlib.import_module("rasterlinker_feature_interpolation.radargram_store")
_IO = importlib.import_module("rasterlinker_feature_interpolation.gpr_npz_io")
_SM = importlib.import_module("rasterlinker_feature_interpolation.slice_manifest")
_SS = importlib.import_module("rasterlinker_feature_interpolation.slice_spec")

DT_NS = 0.125
V_M_S = 1.0e8            # -> 0.00625 m per riga


def _channel(n_t, east, north, n_s=40, seed=0):
    rng = np.random.default_rng(seed)
    east = np.asarray(east, dtype=np.float64)
    return types.SimpleNamespace(
        data=rng.normal(0, 1, (n_s, n_t)).astype(np.float32),
        easting=east,
        northing=np.asarray(north, dtype=np.float64),
        altitude=np.zeros(n_t), heading=np.zeros(n_t),
        distances=np.arange(n_t, dtype=np.float64) * 0.05,
    )


def _profile(name, channels):
    return types.SimpleNamespace(
        path=f"/raw/{name}.ogpr", n_channels=len(channels),
        channel=lambda i, _c=channels: _c[i], channels=channels,
        swath_name="", swath_id="", sampling_step_m=0.05, sampling_time_ns=DT_NS,
        velocity_m_s=V_M_S, frequency_mhz=600.0, epsg=32632, polarization="",
    )


def _write_variant(root, variant, names, north0=0.0, n_t=100, n_ch=2):
    """Variante con un profilo per nome; profilo k a 10 m dal precedente (nord), canali a 4 cm."""
    vdir = os.path.join(root, "Processing", "processed", variant)
    profs = []
    for k, name in enumerate(names):
        chans = [
            _channel(n_t, 500000.0 + np.arange(n_t) * 0.05,
                     np.full(n_t, north0 + 10.0 * k + 0.04 * ci), seed=10 * k + ci)
            for ci in range(n_ch)
        ]
        prof = _profile(name, chans)
        _IO.save_processed_profile_npz(vdir, prof, [c.data for c in chans], {"a": 1}, [],
                                       view_geom=[0, 0, False])
        profs.append(prof)
    return vdir, profs


class DiscoverTest(unittest.TestCase):
    def test_lists_only_variants_with_npz_and_accepts_root_or_processed(self):
        with tempfile.TemporaryDirectory() as root:
            _write_variant(root, "Lavoro", ["p1"])
            _write_variant(root, "v3", ["p1", "p2"])
            os.makedirs(os.path.join(root, "Processing", "processed", "vuota"))
            refs = _RS.discover_variants(root)
            self.assertEqual([(r.name, r.n_files) for r in refs], [("Lavoro", 1), ("v3", 2)])
            self.assertEqual(
                [r.name for r in _RS.discover_variants(os.path.join(root, "Processing"))],
                ["Lavoro", "v3"])
            self.assertEqual([r.name for r in _RS.discover_variants(
                os.path.join(root, "Processing", "processed"))], ["Lavoro", "v3"])
            self.assertEqual(_RS.discover_variants(os.path.join(root, "nulla")), [])
            self.assertEqual(_RS.discover_variants(""), [])


class StoreTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        # p2 prima di p10 (ordine naturale, non lessicale)
        self.vdir, self.profs = _write_variant(self.root, "Lavoro", ["Swath10", "Swath2"])

    def tearDown(self):
        self._tmp.cleanup()

    def test_profiles_are_naturally_sorted_with_metadata(self):
        st = _RS.VariantStore(self.vdir)
        self.assertEqual([p.stem for p in st.profiles], ["Swath2", "Swath10"])
        p = st.profiles[0]
        self.assertEqual((p.n_channels, p.n_traces, p.epsg), (2, (100, 100), 32632))
        self.assertAlmostEqual(p.dz_m, 0.00625)

    def test_channel_data_and_depth_conversion(self):
        st = _RS.VariantStore(self.vdir)
        ch = st.channel(0, 1)
        self.assertEqual(ch.data.shape, (40, 100))
        self.assertAlmostEqual(ch.depth_max_m, 40 * 0.00625)
        self.assertAlmostEqual(ch.length_m, 99 * 0.05)
        self.assertEqual(ch.row_at_depth(0.0), 0)
        self.assertEqual(ch.row_at_depth(0.0125), 2)         # riga i = [i*dz, (i+1)*dz)
        self.assertEqual(ch.row_at_depth(0.0124), 1)
        self.assertEqual(ch.row_at_depth(99.0), 39)          # oltre il fondo: ultima riga
        self.assertEqual(ch.trace_at_distance(1.0), 20)
        self.assertEqual(ch.trace_at_distance(-3.0), 0)
        self.assertEqual(ch.geometry_source, "npz")
        self.assertIs(st.channel(0, 1), ch)                  # cache

    def test_nearest_finds_profile_channel_and_trace(self):
        st = _RS.VariantStore(self.vdir)
        # ordine naturale: indice 0 = Swath2 (nord +10 m), indice 1 = Swath10 (nord 0).
        # Il canale 1 sta 4 cm a nord del canale 0; la traccia 30 e' a x = 500001.5.
        hit = st.nearest(500001.5, 10.0 + 0.04 + 0.01, tol_m=2.0)
        self.assertIsNotNone(hit)
        dist, pi, ci, tr = hit
        self.assertEqual((st.profiles[pi].stem, ci, tr), ("Swath2", 1, 30))
        self.assertLess(dist, 0.02)
        self.assertEqual(st.nearest(500001.5, 0.001, tol_m=2.0)[1:], (1, 0, 30))   # Swath10
        self.assertIsNone(st.nearest(500001.5, 5.0, tol_m=2.0))      # a 5 m da tutto
        self.assertIsNotNone(st.nearest(500001.5, 5.0, tol_m=6.0))   # la tolleranza conta

    def test_decimated_index_still_returns_the_exact_trace(self):
        st = _RS.VariantStore(self.vdir)
        st._MAX_INDEX_POINTS = 40            # 400 punti totali -> indice decimato x10
        hit = st.nearest(500000.0 + 47 * 0.05 + 0.003, 0.0 + 0.001, tol_m=1.0)
        self.assertEqual(hit[1:], (1, 0, 47))        # Swath10; traccia non multipla del passo: serve l'affinamento

    def test_extent(self):
        st = _RS.VariantStore(self.vdir)
        x0, y0, x1, y1 = st.extent()
        self.assertAlmostEqual(x0, 500000.0)
        self.assertAlmostEqual(x1, 500000.0 + 99 * 0.05)
        self.assertAlmostEqual(y0, 0.0)
        self.assertAlmostEqual(y1, 10.04)

    def test_geometry_provider_overrides_saved_snapshot(self):
        moved = _channel(100, 600000.0 + np.arange(100) * 0.05, np.full(100, 7.0))
        moved_prof = _profile("Swath2", [moved, moved])
        provider = lambda stem: moved_prof if stem == "Swath2" else None
        st = _RS.VariantStore(self.vdir, geometry_provider=provider)
        ch = st.channel(0, 0)                          # Swath2: riposizionato
        self.assertEqual(ch.geometry_source, "viewer")
        np.testing.assert_allclose(ch.east[:3], 600000.0 + np.arange(3) * 0.05)
        self.assertEqual(st.channel(1, 0).geometry_source, "npz")   # Swath10: il viewer non lo ha
        # anche il clic sulla mappa usa la geometria nuova
        hit = st.nearest(600000.0 + 5 * 0.05, 7.0, tol_m=1.0)
        self.assertEqual(hit[1:], (0, 0, 5))
        # e con il posizionamento cambiato si invalida
        st.invalidate_geometry()
        self.assertEqual(st.channel(0, 0).geometry_source, "viewer")

    def test_provider_with_mismatching_length_falls_back_to_snapshot(self):
        wrong = _channel(37, np.arange(37.0), np.zeros(37))
        st = _RS.VariantStore(self.vdir, geometry_provider=lambda stem: _profile(stem, [wrong]))
        ch = st.channel(0, 0)
        self.assertEqual(ch.geometry_source, "npz")
        self.assertAlmostEqual(float(ch.east[0]), 500000.0)

    def test_broken_provider_does_not_break_reading(self):
        def boom(_stem):
            raise RuntimeError("viewer chiuso a meta'")
        st = _RS.VariantStore(self.vdir, geometry_provider=boom)
        self.assertEqual(st.channel(0, 0).geometry_source, "npz")

    def test_distance_axis_is_repaired(self):
        d = np.array([0.0, 0.1, np.nan, 0.05, 0.4])       # NaN e ritorno indietro
        out = _RS._clean_distance_axis(d, 5, 0.05)
        self.assertTrue(np.all(np.diff(out) >= 0))
        self.assertEqual(out[0], 0.0)
        flat = _RS._clean_distance_axis(np.zeros(5), 5, 0.05)     # inservibile -> equispaziato
        np.testing.assert_allclose(flat, np.arange(5) * 0.05)

    def test_store_notices_when_the_variant_is_rewritten_on_disk(self):
        st = _RS.VariantStore(self.vdir)
        self.assertFalse(st.is_stale())
        target = os.path.join(self.vdir, "Swath2.npz")
        later = os.path.getmtime(target) + 5
        os.utime(target, (later, later))                       # riscritto (data di modifica nuova)
        self.assertTrue(st.is_stale())
        self.assertFalse(_RS.VariantStore(self.vdir).is_stale())   # una nuova apertura e' allineata
        os.remove(target)                                      # oppure un file sparisce
        self.assertTrue(st.is_stale())
        self.assertFalse(_RS.VariantStore(self.vdir).is_stale())

    def test_unreadable_files_are_skipped(self):
        with open(os.path.join(self.vdir, "rotto.npz"), "wb") as fh:
            fh.write(b"non e' uno zip")
        st = _RS.VariantStore(self.vdir)
        self.assertEqual(len(st), 2)


class SliceInfoTest(unittest.TestCase):
    def _write_set(self, folder, variant="Lavoro"):
        _SM.write_sliceset(
            folder, _SS.SliceSpec(),
            [{"index": 0, "z_from": 0.30, "z_to": 0.40, "z_center": 0.35, "grid": "s0.tif"},
             {"index": 1, "z_from": 0.60, "z_to": 0.50, "grid": "s1.tif"}],   # finestra invertita
            {"kind": "ogpr", "variant_id": variant, "n_profiles": 2},
            crs_epsg=32632)

    def test_reads_depth_window_and_variant(self):
        with tempfile.TemporaryDirectory() as root:
            folder = os.path.join(root, "Processing", "timeslices", "Lavoro")
            self._write_set(folder)
            info = _RS.slice_info_for_path(os.path.join(folder, "s0.tif"))
            self.assertEqual((info["z_from"], info["z_to"], info["z_center"]), (0.30, 0.40, 0.35))
            self.assertEqual((info["variant"], info["epsg"]), ("Lavoro", 32632))
            inv = _RS.slice_info_for_path(os.path.join(folder, "s1.tif") + "|layername=x")
            self.assertEqual((inv["z_from"], inv["z_to"]), (0.50, 0.60))
            self.assertAlmostEqual(inv["z_center"], 0.55)

    def test_unknown_raster_or_missing_manifest_gives_none(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertIsNone(_RS.slice_info_for_path(os.path.join(root, "x.tif")))
            self._write_set(root)
            self.assertIsNone(_RS.slice_info_for_path(os.path.join(root, "altro.tif")))
            self.assertIsNone(_RS.slice_info_for_path(""))

    def test_variant_dir_is_found_from_slice_dir(self):
        with tempfile.TemporaryDirectory() as root:
            vdir, _ = _write_variant(root, "v3", ["p1"])
            sdir = os.path.join(root, "Processing", "timeslices", "v3")
            os.makedirs(sdir)
            self.assertEqual(os.path.normcase(_RS.variant_dir_from_slice_dir(sdir, "v3")),
                             os.path.normcase(vdir))
            self.assertEqual(_RS.variant_dir_from_slice_dir(sdir, "raw"), "")
            self.assertEqual(_RS.variant_dir_from_slice_dir(sdir, "inesistente"), "")


class AlignedGeometryTest(unittest.TestCase):
    def test_trim_and_flip_match_the_saved_data(self):
        e = np.arange(10, dtype=np.float64) + 1000.0
        src = _channel(10, e, np.zeros(10))
        cols = _IO.aligned_geometry_columns(src, 5, [2, 3, True])      # indici 6,5,4,3,2
        np.testing.assert_allclose(cols["easting"], e[[6, 5, 4, 3, 2]])
        self.assertEqual(float(cols["distances"][0]), 0.0)

    def test_none_when_length_cannot_be_reconstructed(self):
        src = _channel(10, np.arange(10.0), np.zeros(10))
        self.assertIsNone(_IO.aligned_geometry_columns(src, 7, None))
        self.assertIsNone(_IO.aligned_geometry_columns(_channel(0, [], []), 0, None))


if __name__ == "__main__":
    unittest.main()
