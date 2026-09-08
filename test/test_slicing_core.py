# -*- coding: utf-8 -*-
"""Fase A parity tests for the extracted slice core (slicing.py + raster_io.py).

Golden arrays in ``test/data/slicing_golden.npz`` were produced from the
pre-extraction ``gpr_ogpr_slicer`` implementation. These tests assert the
moved helpers and the end-to-end ``compute_ogpr_slice_grids`` still produce
bit-for-bit equivalent results, and that the slicers now delegate to the
shared modules instead of carrying their own copies.
"""

import importlib
import pathlib
import sys
import unittest

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_PLUGIN_DIR = _HERE.parents[1]
_PLUGIN_PARENT = _PLUGIN_DIR.parent
_TEST_DIR = _HERE.parent
for _p in (str(_PLUGIN_PARENT), str(_TEST_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_GOLDEN_PATH = _PLUGIN_DIR / "test" / "data" / "slicing_golden.npz"

slicing = importlib.import_module("rasterlinker_feature_interpolation.slicing")
raster_io = importlib.import_module("rasterlinker_feature_interpolation.raster_io")
ogpr = importlib.import_module("rasterlinker_feature_interpolation.gpr_ogpr_slicer")
las = importlib.import_module("rasterlinker_feature_interpolation.gpr_las_slicer")
_ogpr_test = importlib.import_module("test_gpr_ogpr_slicer")


try:
    import scipy  # noqa: F401

    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


def _point_cloud():
    """Same deterministic cloud used to build the golden file."""
    rng = np.random.default_rng(7)
    n = 400
    xp = rng.uniform(0.0, 20.0, size=n)
    yp = rng.uniform(0.0, 12.0, size=n)
    vp = np.sin(xp * 0.5) + np.cos(yp * 0.4) + rng.normal(0.0, 0.05, size=n)
    return xp, yp, vp


def _grid_equal(a, b, rtol=1e-6, atol=1e-6):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return False
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    if not np.array_equal(nan_a, nan_b):
        return False
    return np.allclose(a[~nan_a], b[~nan_b], rtol=rtol, atol=atol)


class WiringTest(unittest.TestCase):
    """The slicers must reference the shared implementations, not copies."""

    def test_ogpr_delegates_to_slicing(self):
        for name in (
            "_envelope", "_depth_to_sample_range", "_depth_adaptive_radius",
            "_fill_nodata_grid", "_smooth_grid_gaussian",
            "_blanking_mask_from_distance", "_bin_with_idw",
            "_bin_with_idw_anisotropic", "_bin_with_idw_ball",
            "_bin_with_idw_anisotropic_ball",
        ):
            self.assertIs(getattr(ogpr, name), getattr(slicing, name), name)

    def test_ogpr_qml_writer_is_shared(self):
        self.assertIs(ogpr._write_qml_singleband, raster_io.write_qml_singleband)
        self.assertIs(ogpr._write_tif_singleband, raster_io.write_tif_singleband)
        self.assertFalse(hasattr(ogpr, "_QML_SINGLEBAND"))

    def test_las_raster_writers_are_shared(self):
        self.assertIs(las._make_geotransform, raster_io.make_geotransform)
        self.assertIs(las._set_projection, raster_io.set_projection)
        self.assertIs(las._write_tif_singleband, raster_io.write_tif_singleband)
        self.assertIs(las._write_tif_rgb, raster_io.write_tif_rgb)
        self.assertIs(las._write_qml_multiband, raster_io.write_qml_multiband)


class RasterIoTest(unittest.TestCase):
    def test_geotransform_pixel_centre_alignment(self):
        gt = raster_io.make_geotransform(100.0, 200.0, 0.5)
        self.assertEqual(gt, (99.75, 0.5, 0.0, 200.25, 0.0, -0.5))

    def test_writers_are_callable(self):
        for name in ("write_tif_singleband", "write_tif_rgb",
                     "write_qml_singleband", "write_qml_multiband",
                     "set_projection"):
            self.assertTrue(callable(getattr(raster_io, name)), name)


@unittest.skipUnless(HAS_SCIPY, "SciPy required for interpolation parity tests")
class HelperParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = np.load(_GOLDEN_PATH)
        cls.xp, cls.yp, cls.vp = _point_cloud()
        cls.args = (0.0, 0.0, 40, 24, 0.5, 2.0, 2.0, 1)  # x_min,y_min,nx,ny,res,radius,power,minpts

    def test_bin_with_idw_fast(self):
        out = slicing._bin_with_idw(self.xp, self.yp, self.vp, *self.args)
        self.assertTrue(_grid_equal(out, self.g["idw_fast"]))

    def test_bin_with_idw_anisotropic(self):
        out = slicing._bin_with_idw_anisotropic(
            self.xp, self.yp, self.vp, *self.args,
            anisotropy_ratio=2.5, anisotropy_angle=30.0,
        )
        self.assertTrue(_grid_equal(out, self.g["idw_aniso"]))

    def test_bin_with_idw_ball(self):
        out = slicing._bin_with_idw_ball(self.xp, self.yp, self.vp, *self.args)
        self.assertTrue(_grid_equal(out, self.g["idw_ball"]))

    def test_bin_with_idw_anisotropic_ball(self):
        out = slicing._bin_with_idw_anisotropic_ball(
            self.xp, self.yp, self.vp, *self.args,
            anisotropy_ratio=2.5, anisotropy_angle=30.0,
        )
        self.assertTrue(_grid_equal(out, self.g["idw_aniso_ball"]))

    def test_fill_nodata_grid(self):
        out = slicing._fill_nodata_grid(self.g["idw_fast"].copy(), max_distance=5)
        self.assertTrue(_grid_equal(out, self.g["fill_nodata"]))

    def test_smooth_grid_gaussian(self):
        out = slicing._smooth_grid_gaussian(self.g["idw_fast"], sigma=1.5)
        self.assertTrue(_grid_equal(out, self.g["smooth"]))

    def test_blanking_mask_from_distance(self):
        out = slicing._blanking_mask_from_distance(
            self.xp, self.yp, 0.0, 0.0, 40, 24, 0.5, 1.5,
        )
        self.assertIsNotNone(out)
        self.assertTrue(np.array_equal(out.astype(np.uint8), self.g["blank_mask"]))

    def test_envelope(self):
        sig = np.stack([
            np.sin(np.linspace(0, 20, 128)),
            np.cos(np.linspace(0, 14, 128)),
        ], axis=1).astype(np.float32)
        self.assertTrue(_grid_equal(slicing._envelope(sig), self.g["envelope"]))

    def test_depth_to_sample_range(self):
        cases = [(0.0, 0.5, 2.0, 100), (0.4, 0.42, 2.0, 100),
                 (1.9, 2.5, 2.0, 100), (0.0, 0.0, 0.0, 100)]
        got = np.asarray([slicing._depth_to_sample_range(*c) for c in cases], dtype=np.int64)
        self.assertTrue(np.array_equal(got, self.g["depth_sample_range"]))

    def test_depth_adaptive_radius(self):
        got = np.asarray(
            [slicing._depth_adaptive_radius(2.0, float(zc), 2.0, 0.6)
             for zc in np.linspace(0.0, 2.0, 9)],
            dtype=np.float64,
        )
        self.assertTrue(np.allclose(got, self.g["depth_adaptive_radius"], rtol=1e-9))


@unittest.skipUnless(HAS_SCIPY, "SciPy required for end-to-end parity test")
class EndToEndParityTest(unittest.TestCase):
    def setUp(self):
        self.g = np.load(_GOLDEN_PATH)
        self.profiles = _ogpr_test._make_profiles()

    def _run(self, mode):
        grids, _meta = ogpr.compute_ogpr_slice_grids(
            profiles=self.profiles, channel=-1, combine_method="mean",
            resolution=0.20, z_step=0.30, z_min=0.0, z_max=1.2, radius=0.40,
            extraction_mode="las_like", use_processing=False, idw_mode=mode,
            auto_radius=False, min_points=1, fill_nodata=False, smooth_sigma=0.0,
            emit_diagnostics=False, parallel_profiles=False,
        )
        return np.stack([np.asarray(g["grid"], dtype=np.float32) for g in grids], axis=0)

    def test_e2e_fast(self):
        self.assertTrue(_grid_equal(self._run("fast"), self.g["e2e_fast"]))

    def test_e2e_quality(self):
        self.assertTrue(_grid_equal(self._run("quality"), self.g["e2e_quality"]))


if __name__ == "__main__":
    unittest.main()
