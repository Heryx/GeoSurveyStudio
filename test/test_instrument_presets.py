import pathlib
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from gpr_instrument_presets import (
    INSTRUMENT_IDS_OPERA_DUO,
    INSTRUMENT_IDS_STREAM_C,
    PRESETS,
    detect_instrument,
    preset_by_key,
)


def _prof(path="Swath001_Array02.ogpr", freq=600.0, step=0.0397, dt=0.125, n=744,
          n_channels=11):
    return SimpleNamespace(path=path, frequency_mhz=freq, sampling_step_m=step,
                           sampling_time_ns=dt, n_slices=n, n_channels=n_channels)


class DetectInstrumentTest(unittest.TestCase):
    def test_multichannel_ogpr_and_dt_are_stream_c(self):
        self.assertEqual(detect_instrument(_prof("a/b/x.OGPR", n_channels=19)), INSTRUMENT_IDS_STREAM_C)
        self.assertEqual(detect_instrument(_prof("LA010001.DT", n_channels=2)), INSTRUMENT_IDS_STREAM_C)

    def test_single_channel_ogpr_is_opera_duo(self):
        self.assertEqual(detect_instrument(_prof("Swath001_Array01_0m.ogpr", n_channels=1)),
                          INSTRUMENT_IDS_OPERA_DUO)
        self.assertEqual(detect_instrument(_prof("x.ogpr", n_channels=0)), INSTRUMENT_IDS_OPERA_DUO)

    def test_other_formats_are_unknown(self):
        self.assertIsNone(detect_instrument(_prof("profilo.dzt")))
        self.assertIsNone(detect_instrument(_prof("")))


class StreamCPresetTest(unittest.TestCase):
    def setUp(self):
        self.preset = preset_by_key("stream_c")

    def test_registered_and_tied_to_stream_c(self):
        self.assertIn(self.preset, PRESETS)
        self.assertEqual(self.preset.instrument, INSTRUMENT_IDS_STREAM_C)
        self.assertEqual(self.preset.button_label, "Preset Stream C")

    def test_600mhz_values(self):
        s = self.preset.build(_prof())
        self.assertEqual(s["steps"], ["bandpass", "timezero", "bg_removal", "pre_agc_gain"])
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
                                              sampling_time_ns=0.125, n_slices=500,
                                              n_channels=11))
        self.assertEqual(s["params"]["bg_window"], 0)

    def test_gain_curve_present(self):
        p = self.preset.build(_prof())["params"]
        self.assertAlmostEqual(p["pre_agc_surface_gain"], 47.75)
        self.assertAlmostEqual(p["pre_agc_deep_gain"], 12956.81)
        bkpts = p["pre_agc_breakpoints"]
        self.assertEqual(bkpts[0], [0.0, 47.75])
        self.assertEqual(bkpts[-1], [1.0, 12956.81])
        self.assertGreaterEqual(len(bkpts), 2)


class OperaDuoPresetTest(unittest.TestCase):
    def setUp(self):
        self.preset = preset_by_key("opera_duo")

    def test_registered_and_tied_to_opera_duo(self):
        self.assertIn(self.preset, PRESETS)
        self.assertEqual(self.preset.instrument, INSTRUMENT_IDS_OPERA_DUO)
        self.assertEqual(self.preset.button_label, "Preset Opera Duo")

    def test_600mhz_values(self):
        prof = _prof(step=0.0324, n=2000, n_channels=1)
        s = self.preset.build(prof)
        self.assertEqual(s["steps"], ["dewow", "bandpass", "timezero", "bg_removal", "pre_agc_gain"])
        p = s["params"]
        self.assertEqual((p["bp_low_mhz"], p["bp_high_mhz"]), (150.0, 1308.5))
        self.assertEqual((p["tz_method"], p["tz_mode"]), ("surface", "line_by_line"))
        self.assertAlmostEqual(p["tz_threshold"], 0.4)
        self.assertEqual(p["tz_backup_nsamp"], 4)
        self.assertEqual(p["dewow_win"], 16)
        self.assertEqual(p["bg_method"], "median")
        self.assertEqual(p["bg_window"], round(16.0 / 0.0324))
        self.assertFalse(s["bg_auto"])

    def test_band_wider_than_stream_c_at_same_frequency(self):
        stream_c = preset_by_key("stream_c").build(_prof(step=0.0324, n=2000, n_channels=1))["params"]
        opera_duo = self.preset.build(_prof(step=0.0324, n=2000, n_channels=1))["params"]
        self.assertGreater(opera_duo["bp_high_mhz"], stream_c["bp_high_mhz"])

    def test_gain_curve_much_steeper_than_stream_c(self):
        p = self.preset.build(_prof(step=0.0324, n=2000, n_channels=1))["params"]
        self.assertAlmostEqual(p["pre_agc_surface_gain"], 804.53)
        self.assertAlmostEqual(p["pre_agc_deep_gain"], 926436.07)
        ratio_opera_duo = p["pre_agc_deep_gain"] / p["pre_agc_surface_gain"]
        stream_c = preset_by_key("stream_c").build(_prof())["params"]
        ratio_stream_c = stream_c["pre_agc_deep_gain"] / stream_c["pre_agc_surface_gain"]
        self.assertGreater(ratio_opera_duo, ratio_stream_c)


if __name__ == "__main__":
    unittest.main()
