# -*- coding: utf-8 -*-
"""La geometria dei profili caricati da .npz deve venire dal profilo corrente
in memoria (pagina di posizionamento), non dalle coordinate salvate nel file."""

import importlib
import pathlib
import sys
import tempfile
import types
import unittest

import numpy as np

_PLUGIN_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(_PLUGIN_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR.parent))

_IO = importlib.import_module("rasterlinker_feature_interpolation.gpr_npz_io")


def _channel(n_t, e, n, n_s=8):
    return types.SimpleNamespace(
        data=np.zeros((n_s, n_t), dtype=np.float32),
        easting=np.asarray(e, dtype=np.float64),
        northing=np.asarray(n, dtype=np.float64),
        altitude=np.zeros(len(e), dtype=np.float64),
        heading=np.zeros(len(e), dtype=np.float64),
        distances=np.arange(len(e), dtype=np.float64) * 0.5,
    )


def _profile(ch, path="/x/p1.ogpr"):
    return types.SimpleNamespace(
        path=path, n_channels=1, channel=lambda i: ch, channels=[ch],
        swath_name="", swath_id="", sampling_step_m=0.5, sampling_time_ns=1.0,
        velocity_m_s=1e8, frequency_mhz=600.0, epsg=32631, polarization="",
    )


class TestApplyGeometry(unittest.TestCase):

    def test_geometry_comes_from_current_profile_not_npz(self):
        src_ch = _channel(10, np.linspace(500.0, 509.0, 10), np.full(10, 4_000_000.0))
        npz_ch = _channel(10, np.zeros(10), np.zeros(10))   # coordinate "vecchie"
        npz_prof = types.SimpleNamespace(path="/x/p1.ogpr", channels=[npz_ch], view_geom=None)
        ok = _IO.apply_geometry_from_profile(npz_prof, _profile(src_ch))
        self.assertTrue(ok)
        np.testing.assert_allclose(npz_ch.easting, src_ch.easting)
        np.testing.assert_allclose(npz_ch.northing, src_ch.northing)

    def test_trim_and_flip_are_reproduced(self):
        e = np.arange(10, dtype=np.float64) + 1000.0
        src_ch = _channel(10, e, np.zeros(10))
        # trim 2 all'inizio, 3 alla fine, flip attivo -> indici 6,5,4,3,2
        npz_ch = _channel(5, np.zeros(5), np.zeros(5))
        npz_prof = types.SimpleNamespace(path="p", channels=[npz_ch], view_geom=[2, 3, True])
        self.assertTrue(_IO.apply_geometry_from_profile(npz_prof, _profile(src_ch)))
        np.testing.assert_allclose(npz_ch.easting, e[[6, 5, 4, 3, 2]])
        self.assertAlmostEqual(float(npz_ch.distances[0]), 0.0)
        self.assertTrue(np.all(npz_ch.distances >= 0.0))

    def test_unreconstructable_length_keeps_saved_geometry(self):
        src_ch = _channel(10, np.arange(10.0), np.zeros(10))
        npz_ch = _channel(7, np.full(7, 5.0), np.zeros(7))
        npz_prof = types.SimpleNamespace(path="p", channels=[npz_ch], view_geom=None)
        self.assertFalse(_IO.apply_geometry_from_profile(npz_prof, _profile(src_ch)))
        np.testing.assert_allclose(npz_ch.easting, np.full(7, 5.0))

    def test_view_geom_roundtrip_through_npz_file(self):
        ch = _channel(6, np.arange(6.0) + 1000.0, np.zeros(6))
        prof = _profile(ch)
        with tempfile.TemporaryDirectory() as d:
            fp = _IO.save_processed_profile_npz(
                d, prof, [ch.data], {}, [], view_geom=[1, 2, False])
            loaded = _IO.load_processed_profile_npz(fp)
        self.assertEqual(list(loaded.view_geom), [1, 2, False])


if __name__ == "__main__":
    unittest.main()
