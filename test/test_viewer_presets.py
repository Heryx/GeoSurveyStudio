# -*- coding: utf-8 -*-
"""Preset di processing per strumento nel profile viewer (richiede Qt offscreen e QGIS)."""

import unittest

from .test_profile_viewer_interaction import (
    HAS_MPL,
    HAS_VIEWER,
    _ViewerTestBase,
    _make_profile,
)


@unittest.skipUnless(HAS_MPL and HAS_VIEWER, "matplotlib/viewer non disponibili")
class IdsPresetViewerTest(_ViewerTestBase):
    PROFILE = {"n_t": 800}

    def _quick_btn(self):
        return self.v._preset_quick_buttons["ids"]

    def test_button_shown_for_ids_data(self):
        self.assertFalse(self._quick_btn().isHidden())
        self.assertEqual(self.v._cb_preset.currentData(), "ids")
        self.assertIn("IDS", self._quick_btn().toolTip())

    def test_button_hidden_for_other_instruments_but_menu_still_offers_ids(self):
        self.v._profiles = [_make_profile(n_t=800)]
        self.v._profiles[0].path = "profilo.dzt"
        self.v._load_current_profile()
        self._pump(100)
        self.assertTrue(self._quick_btn().isHidden())
        self.assertGreaterEqual(self.v._cb_preset.findData("ids"), 0)

    def test_apply_from_menu_works_on_any_data(self):
        self.v._profiles[0].path = "profilo.dzt"
        self.v._load_current_profile()
        self._pump(100)
        self.v._cb_preset.setCurrentIndex(self.v._cb_preset.findData("ids"))
        self.v._on_apply_selected_preset()
        self.assertTrue(self.v._chk_bp.isChecked())

    def test_menu_placeholder_applies_nothing(self):
        self.v._cb_preset.setCurrentIndex(0)
        self.v._on_apply_selected_preset()
        self.assertFalse(self.v._chk_bp.isChecked())
        self.assertFalse(self.v._chk_timezero.isChecked())

    def test_apply_sets_chain_and_editable_controls(self):
        v = self.v
        v._chk_dewow.setChecked(True)
        v._chk_agc.setChecked(True)
        self._quick_btn().click()
        self._pump(300)

        self.assertTrue(v._chk_bp.isChecked())
        self.assertTrue(v._chk_timezero.isChecked())
        self.assertTrue(v._chk_bg.isChecked())
        self.assertFalse(v._chk_dewow.isChecked())
        self.assertFalse(v._chk_agc.isChecked())
        self.assertEqual(v._current_chain_order()[:3], ["bandpass", "timezero", "bg_removal"])

        # profilo sintetico: 400 MHz, passo 0.02 m -> banda scalata, finestra 10 m = 500 tracce
        self.assertAlmostEqual(v._spin_bp_lo.value(), 100.0, places=1)
        self.assertAlmostEqual(v._spin_bp_hi.value(), 566.7, places=1)
        self.assertEqual(v._cb_tz_method.currentText(), "coppens")
        self.assertEqual(v._cb_tz_mode.currentText(), "line_by_line")
        self.assertEqual(v._cb_bg_method.currentData(), "median")
        self.assertFalse(v._chk_bg_auto.isChecked())
        self.assertEqual(v._spin_bg_window.value(), 500)
        self.assertTrue(v._spin_bg_window.isEnabled())
        self.assertEqual(v._spin_bg_sample_start.value(), 0)

        # il processo e' stato eseguito con questi parametri e si puo' modificare
        self.assertAlmostEqual(v._last_pipeline_params["bp_low_mhz"], 100.0, places=1)
        v._spin_bp_lo.setValue(120.0)
        self.assertAlmostEqual(v._current_processing_params()["bp_low_mhz"], 120.0, places=1)
        self.assertFalse(v._lbl_preset_info.isHidden())
        self.assertIn("Preset IDS applicato", v._lbl_preset_info.text())

    def test_short_profile_uses_whole_profile_for_background(self):
        self.v._profiles = [_make_profile(n_t=300)]  # < 500 tracce
        self.v._load_current_profile()
        self._pump(100)
        self._quick_btn().click()
        self._pump(300)
        self.assertTrue(self.v._chk_bg_auto.isChecked())
        self.assertFalse(self.v._spin_bg_window.isEnabled())


if __name__ == "__main__":
    unittest.main()
