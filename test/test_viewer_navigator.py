# -*- coding: utf-8 -*-
"""Navigazione profili/canali, puntatore inverso time-slice -> radargramma."""

import time
import unittest

import numpy as np

from .test_profile_viewer_interaction import (
    HAS_MPL,
    HAS_VIEWER,
    _ViewerTestBase,
    _make_profile,
)


@unittest.skipUnless(HAS_MPL and HAS_VIEWER, "matplotlib/viewer non disponibili")
class NavigatorTest(_ViewerTestBase):
    def setUp(self):
        super().setUp()
        v = self.v
        # tre profili paralleli, 10 m di distanza tra loro (nord crescente)
        for k in (1, 2):
            p = _make_profile()
            ch = p.channel(0)
            ch.northing = np.asarray(ch.northing) + 10.0 * k
            v._profiles.append(p)
        v._sync_navigator()
        self._idle()

    def _idle(self):
        v = self.v
        t0 = time.time()
        while (v._proc_thread_running() or v._proc_busy) and time.time() - t0 < 30:
            self._pump(10)
        self._pump(100)

    def test_profile_change_reloads_once(self):
        v = self.v
        calls = []
        orig = type(v)._reload_data
        type(v)._reload_data = lambda s, *a, **k: (calls.append(1), orig(s, *a, **k))[1]
        try:
            v._next_profile()
            self._idle()
        finally:
            type(v)._reload_data = orig
        self.assertEqual(v._prof_idx, 1)
        self.assertEqual(len(calls), 1)   # prima: 3 (la combo canali riemetteva i segnali)

    def test_combo_and_sidebar_follow_the_selection(self):
        v = self.v
        self.assertEqual(v._cb_profile.count(), 3)
        self.assertEqual(v._lst_profiles.count(), 3)
        v._select_profile(2)
        self._idle()
        self.assertEqual(v._cb_profile.currentIndex(), 2)
        self.assertEqual(v._lst_profiles.currentRow(), 2)
        self.assertEqual(v._lst_channels.count(), v._profiles[2].n_channels)

    def test_combo_pick_changes_profile(self):
        v = self.v
        v._cb_profile.setCurrentIndex(1)
        self._idle()
        self.assertEqual(v._prof_idx, 1)
        self.assertEqual(v._lst_profiles.currentRow(), 1)

    def test_sidebar_can_be_hidden(self):
        v = self.v
        v._nav_dock.hide()
        self._pump(50)
        self.assertFalse(v._nav_dock.isVisible())
        v._nav_toggle_action().trigger()
        self._pump(50)
        self.assertTrue(v._nav_dock.isVisible())

    def test_click_on_slice_opens_nearest_profile_with_cursor(self):
        v = self.v
        ch = v._profiles[2].channel(0)
        east = float(np.asarray(ch.easting)[150])
        north = float(np.asarray(ch.northing)[150])
        self.assertTrue(v._goto_world_point(east, north + 0.1))
        self._idle()
        self.assertEqual(v._prof_idx, 2)
        self.assertEqual(v._ch_idx, 0)
        m = v._cursor_metrics()
        self.assertIsNotNone(m)
        self.assertEqual(m["idx_trace"], 150)
        self.assertAlmostEqual(m["east"], east, places=3)
        self.assertTrue(v._vline.get_visible())   # il cursore sopravvive al ridisegno
        self.assertAlmostEqual(float(v._vline.get_xdata()[0]), v._cursor_x, places=6)

    def test_click_far_from_any_profile_does_nothing(self):
        v = self.v
        before = v._prof_idx
        self.assertFalse(v._goto_world_point(500000.0, 4500500.0))
        self.assertEqual(v._prof_idx, before)

    def test_click_respects_the_checkbox(self):
        v = self.v
        v._chk_nav_slice_click.setChecked(False)
        ch = v._profiles[1].channel(0)
        v._on_slice_click(float(np.asarray(ch.easting)[10]), float(np.asarray(ch.northing)[10]))
        self._idle()
        self.assertEqual(v._prof_idx, 0)

    def test_click_takes_the_slice_depth_for_the_cursor(self):
        v = self.v
        v._slice_catalog = [{"z_center": 0.2}, {"z_center": 0.4}]
        v._slice_current_idx = 1
        ch = v._profiles[1].channel(0)
        v._goto_world_point(float(np.asarray(ch.easting)[50]), float(np.asarray(ch.northing)[50]))
        self._idle()
        self.assertAlmostEqual(v._cursor_z, 0.4, places=6)

    def test_slice_tracks_drawn_and_current_highlighted(self):
        v = self.v
        v._slice_im = v._ax_slice.imshow(np.zeros((2, 2)), extent=(500000, 500010, 4500000, 4500030))
        v._nav_track_key = None
        self.assertTrue(v._update_slice_track_overlay())
        # 2 profili "altri" + traccia corrente + marcatore di inizio
        self.assertEqual(len(v._nav_track_artists), 4)
        self.assertFalse(v._update_slice_track_overlay())   # invariato: nessun rifacimento
        v._chk_nav_slice_tracks.setChecked(False)
        self.assertEqual(len(v._nav_track_artists), 0)

    def test_forced_channel_is_kept(self):
        v = self.v
        v._select_profile(1, channel=0)
        self._idle()
        self.assertEqual(v._ch_idx, 0)


if __name__ == "__main__":
    unittest.main()
