import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from gpr_ogpr_reader import OgprReadError, _check_radar_volume_content


class CheckRadarVolumeContentTest(unittest.TestCase):
    def test_all_zero_rejected(self):
        with self.assertRaisesRegex(OgprReadError, "interamente a zero"):
            _check_radar_volume_content(np.zeros((10, 3, 32), np.float32))

    def test_repeated_trace_rejected(self):
        one = np.random.default_rng(0).normal(size=(1, 3, 32)).astype(np.float32)
        with self.assertRaisesRegex(OgprReadError, "identiche"):
            _check_radar_volume_content(np.repeat(one, 10, axis=0))

    def test_real_variation_accepted(self):
        vol = np.random.default_rng(1).normal(size=(10, 3, 32)).astype(np.float32)
        _check_radar_volume_content(vol)

    def test_single_slice_accepted(self):
        vol = np.random.default_rng(2).normal(size=(1, 3, 32)).astype(np.float32)
        _check_radar_volume_content(vol)


if __name__ == "__main__":
    unittest.main()
