import pathlib
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from gpr_instrument_presets import (
    INSTRUMENT_IDS,
    PRESETS,
    detect_instrument,
    preset_by_key,
)


def _prof(path="Swath001_Array02.ogpr", freq=600.0, step=0.0397, dt=0.125, n=744):
    return SimpleNamespace(path=path, frequency_mhz=freq, sampling_step_m=step,
                           sampling_time_ns=dt, n_slices=n, n_channels=11)


class DetectInstrumentTest(unittest.TestCase):
    def test_ogpr_and_dt_are_ids(self):
        self.assertEqual(detect_instrument(_prof("a/b/x.OGPR")), INSTRUMENT_IDS)
        self.assertEqual(detect_instrument(_prof("LA010001.DT")), INSTRUMENT_IDS)

    def test_other_formats_are_unknown(self):
        self.assertIsNone(detect_instrument(_prof("profilo.dzt")))
        self.assertIsNone(detect_instrument(_prof("")))


class IdsPresetTest(unittest.TestCase):
    def setUp(self):
        self.preset = preset_by_key("ids")

    def test_registered_and_tied_to_ids(self):
        self.assertIn(self.preset, PRESETS)
        self.assertEqual(self.preset.instrument, INSTRUMENT_IDS)
        self.assertEqual(self.preset.button_label, "Preset IDS")

    def test_600mhz_values(self):
        s = self.preset.build(_prof())
        self.assertEqual(s["steps"], ["bandpass", "timezero", "bg_removal"])
        p = s["params"]
        self.assertEqual((p["bp_low_mhz"], p["bp_high_mhz"]), (150.0, 850.0))
        self.assertEqual((p["tz_method"], p["tz_mode"]), ("coppens", "line_by_line"))
        self.assertEqual(p["bg_method"], "median")
        self.assertEqual(p["bg_window"], round(10.0 / 0.0397))
        self.assertEqual(p["bg_sample_start"], 0)
        self.assertFalse(s["bg_auto"])

    def test_band_scales_with_antenna_frequency(self):
        p = self.preset.build(_prof(freq=300.0))["params"]
        self.assertEqual((p["bp_low_mhz"], p["bp_high_mhz"]), (75.0, 425.0))

    def test_band_capped_below_nyquist(self):
        p = self.preset.build(_prof(freq=600.0, dt=0.5))["params"]  # Nyquist 1000 MHz
        self.assertLessEqual(p["bp_high_mhz"], 900.0)
        self.assertGreater(p["bp_high_mhz"], p["bp_low_mhz"])

    def test_short_profile_uses_whole_profile(self):
        s = self.preset.build(_prof(n=100))  # 100 tracce < 252 di finestra
        self.assertEqual(s["params"]["bg_window"], 0)
        self.assertTrue(s["bg_auto"])

    def test_unknown_spacing_uses_whole_profile(self):
        s = self.preset.build(SimpleNamespace(path="a.ogpr", frequency_mhz=600.0,
                                              sampling_time_ns=0.125, n_slices=500))
        self.assertEqual(s["params"]["bg_window"], 0)


if __name__ == "__main__":
    unittest.main()
