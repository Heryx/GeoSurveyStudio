# -*- coding: utf-8 -*-
"""gpr_fence_slices: gruppi di time-slice e lettura dei raster per la vista 3D.

    python-qgis.bat -m unittest test.test_fence_slices      (serve GDAL)
"""

import importlib
import os
import pathlib
import sys
import tempfile
import unittest

import numpy as np

_PLUGIN_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(_PLUGIN_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR.parent))

try:
    _FS = importlib.import_module("rasterlinker_feature_interpolation.gpr_fence_slices")
    _GEO = importlib.import_module("rasterlinker_feature_interpolation.gpr_fence_geometry")
    _RIO = importlib.import_module("rasterlinker_feature_interpolation.raster_io")
    _SM = importlib.import_module("rasterlinker_feature_interpolation.slice_manifest")
    _SS = importlib.import_module("rasterlinker_feature_interpolation.slice_spec")
    from osgeo import gdal  # noqa: F401
    HAS_GDAL = True
except Exception:  # pragma: no cover - ambiente senza GDAL
    HAS_GDAL = False


def setUpModule():
    # un modulo precedente (es. test_radargram_dock) chiude QGIS a fine esecuzione e con esso i driver
    # GDAL: qui si registrano di nuovo, cosi' il test non dipende dall'ordine di esecuzione
    if HAS_GDAL and gdal.GetDriverByName("GTiff") is None:
        gdal.AllRegister()


def _write_group(root, group, variant, slices, grid_fn=None, epsg=32632):
    folder = os.path.join(root, "Processing", "timeslices", group)
    os.makedirs(folder, exist_ok=True)
    rows = []
    for i, (name, z0, z1) in enumerate(slices):
        grid = grid_fn(i) if grid_fn else np.arange(12, dtype=np.float32).reshape(3, 4) + i
        _RIO.write_tif_singleband(grid, os.path.join(folder, name), 500000.0, 0.0, 2.0, 2.0, epsg)
        rows.append({"index": i, "z_from": z0, "z_to": z1, "grid": name})
    _SM.write_sliceset(folder, _SS.SliceSpec(), rows, {"kind": "ogpr", "variant_id": variant}, crs_epsg=epsg)
    return folder


@unittest.skipUnless(HAS_GDAL, "GDAL non disponibile in questo interprete")
class DiscoverTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_finds_groups_with_their_variant_and_sorted_slices(self):
        _write_group(self.root, "gpr_slices", "Lavoro", [("b.tif", 0.4, 0.6), ("a.tif", 0.0, 0.2)])
        _write_group(self.root, "altro", "v3", [("c.tif", 0.0, 0.1)])
        groups = _FS.discover_slice_groups(self.root)
        self.assertEqual([g["name"] for g in groups], ["altro", "gpr_slices"])
        g = groups[1]
        self.assertEqual(g["variant"], "Lavoro")
        self.assertEqual(g["epsg"], 32632)
        self.assertEqual([round(r["z_center"], 2) for r in g["slices"]], [0.1, 0.5])

    def test_accepts_root_processing_or_timeslices_folder(self):
        _write_group(self.root, "g", "Lavoro", [("a.tif", 0.0, 0.2)])
        for base in (self.root, os.path.join(self.root, "Processing"),
                     os.path.join(self.root, "Processing", "timeslices")):
            self.assertEqual(len(_FS.discover_slice_groups(base)), 1, base)

    def test_ignores_folders_without_manifest_or_rasters(self):
        _write_group(self.root, "ok", "Lavoro", [("a.tif", 0.0, 0.2)])
        os.makedirs(os.path.join(self.root, "Processing", "timeslices", "vuota"))
        folder = _write_group(self.root, "senza_tif", "Lavoro", [("x.tif", 0.0, 0.2)])
        os.remove(os.path.join(folder, "x.tif"))
        self.assertEqual([g["name"] for g in _FS.discover_slice_groups(self.root)], ["ok"])

    def test_no_project(self):
        self.assertEqual(_FS.discover_slice_groups(""), [])
        self.assertEqual(_FS.discover_slice_groups(os.path.join(self.root, "nope")), [])


@unittest.skipUnless(HAS_GDAL, "GDAL non disponibile in questo interprete")
class LoadSliceTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_corners_are_the_outer_edges_with_north_on_top(self):
        # griglia 3 righe x 4 colonne, passo 2 m, primo centro-pixel a x=500000, y da 0 a 2 (y_min=0, y_max=2... 3 righe)
        folder = _write_group(self.root, "g", "Lavoro", [("a.tif", 0.0, 0.2)])
        s = _FS.load_slice(os.path.join(folder, "a.tif"))
        (xl, yt), (xr, yt2), (xr2, yb), (xl2, yb2) = s["corners"]
        self.assertEqual((xl, xl2), (500000.0 - 1.0, 500000.0 - 1.0))
        self.assertAlmostEqual(xr, 500000.0 + 3 * 2.0 + 1.0)
        self.assertGreater(yt, yb)
        self.assertEqual(yt, yt2)
        self.assertEqual(s["epsg"], 32632)
        self.assertEqual(s["data"].shape, (3, 4))

    def test_first_row_of_data_is_the_northern_one(self):
        grid = np.zeros((3, 4), dtype=np.float32)
        grid[2, :] = 9.0                      # riga y-crescente piu' a nord (scrittura: grid[0] e' la piu' a sud)
        folder = _write_group(self.root, "g", "Lavoro", [("a.tif", 0.0, 0.2)], grid_fn=lambda i: grid)
        s = _FS.load_slice(os.path.join(folder, "a.tif"))
        self.assertTrue(np.allclose(s["data"][0], 9.0))
        self.assertTrue(np.allclose(s["data"][2], 0.0))

    def test_limits_default_to_percentile_stretch(self):
        folder = _write_group(self.root, "g", "Lavoro", [("a.tif", 0.0, 0.2)])
        s = _FS.load_slice(os.path.join(folder, "a.tif"))
        self.assertLess(s["vmin"], s["vmax"])
        self.assertEqual(s["colors"], [])

    def test_embedded_style_wins(self):
        folder = _write_group(self.root, "g", "Lavoro", [("a.tif", 0.0, 0.2)])
        path = os.path.join(folder, "a.tif")
        self.assertTrue(_RIO.embed_style_in_tif(path, -3.0, 7.0, ["#000000", "#ffffff"]))
        s = _FS.load_slice(path)
        self.assertEqual((s["vmin"], s["vmax"]), (-3.0, 7.0))
        cm = _FS.colormap_from_hex(s["colors"])
        self.assertIsNotNone(cm)

    def test_large_rasters_are_downsampled(self):
        folder = _write_group(self.root, "g", "Lavoro", [("a.tif", 0.0, 0.2)],
                              grid_fn=lambda i: np.ones((300, 500), dtype=np.float32))
        s = _FS.load_slice(os.path.join(folder, "a.tif"), max_px=100)
        self.assertLessEqual(max(s["data"].shape), 100)


class PlaneAndColormapTest(unittest.TestCase):
    def test_plane_quad_and_texture_orientation(self):
        corners = [(0.0, 10.0), (20.0, 10.0), (20.0, 0.0), (0.0, 0.0)]
        p = _GEO.plane(corners, depth=0.5, origin=(0.0, 0.0)) if HAS_GDAL else None
        if p is None:
            self.skipTest("moduli non importabili senza GDAL")
        np.testing.assert_allclose(p["points"][:, 2], -0.5)
        self.assertEqual(p["faces"].tolist(), [4, 0, 1, 2, 3])
        np.testing.assert_allclose(p["tcoords"][0], [0.0, 1.0])        # alto-sinistra
        np.testing.assert_allclose(p["tcoords"][2], [1.0, 0.0])        # basso-destra

    def test_colormap_from_hex_needs_two_colours(self):
        if not HAS_GDAL:
            self.skipTest("moduli non importabili senza GDAL")
        self.assertIsNone(_FS.colormap_from_hex([]))
        self.assertIsNone(_FS.colormap_from_hex(["#fff"]))
        self.assertIsNotNone(_FS.colormap_from_hex(["#000000", "#ff0000", "#ffffff"]))


if __name__ == "__main__":
    unittest.main()
