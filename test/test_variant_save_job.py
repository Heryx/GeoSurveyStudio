# -*- coding: utf-8 -*-
"""Salvataggio della variante in background: `_save_processed_variant` non blocca la GUI,
mostra l'avanzamento, si puo' annullare, e scrive gli stessi dati di `apply_pipeline`.

Richiede Qt offscreen + QGIS (come test_profile_viewer_interaction); si salta altrimenti.
Lanciare come pacchetto: python-qgis.bat -m unittest test.test_variant_save_job
"""

import importlib
import os
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

from .test_profile_viewer_interaction import HAS_MPL, HAS_VIEWER, _GV, _ViewerTestBase, _make_profile

if HAS_VIEWER:
    from qgis.PyQt.QtCore import QTimer

    _RUNNER = importlib.import_module("rasterlinker_feature_interpolation.gpr_variant_runner")
    _VM = importlib.import_module("rasterlinker_feature_interpolation.gpr_viewer_variants")
    _PROC = importlib.import_module("rasterlinker_feature_interpolation.gpr_processing")
    _RD = importlib.import_module("rasterlinker_feature_interpolation.gpr_ogpr_reader")

N_CH = 2
N_T = (60, 61, 60)   # il profilo 1 e' distinguibile dalla larghezza (per far fallire solo lui)


def _multichannel_profile(idx, n_t, n_ch=N_CH):
    prof = _make_profile(n_s=96, n_t=n_t)
    base = prof.channels[0]
    chans = [base]
    for ci in range(1, n_ch):
        rng = np.random.default_rng(100 * idx + ci)
        chans.append(_RD.OgprChannel(
            ci, (base.data + rng.normal(0, 0.02, base.data.shape)).astype(np.float32),
            base.easting + 0.04 * ci, base.northing, base.altitude, base.heading, base.distances,
        ))
    prof.channels = chans
    prof.n_channels = n_ch
    prof.path = f"synthetic_{idx}.ogpr"
    return prof


@unittest.skipUnless(HAS_MPL and HAS_VIEWER, "matplotlib/viewer non disponibili")
class VariantSaveJobTest(_ViewerTestBase):
    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        v = self.v
        v._profiles = [_multichannel_profile(i, n) for i, n in enumerate(N_T)]
        v._prof_idx = 0
        v._gpr_workspace_override = self._tmp.name
        v._load_current_profile()
        self._pump(100)
        v._chk_dewow.setChecked(True)
        v._chk_bg.setChecked(True)
        self._pump(50)

    # -- helpers --------------------------------------------------------
    def _variant_dir(self, name="t"):
        return os.path.join(self._tmp.name, "Processing", "processed", name)

    def _saved(self, name="t"):
        d = self._variant_dir(name)
        return sorted(f for f in os.listdir(d) if f.endswith(".npz")) if os.path.isdir(d) else []

    def _expected(self, prof, ci, params, tidx):
        raw = np.asarray(prof.channel(ci).data, dtype=np.float32)[:, tidx]
        return _PROC.apply_pipeline(
            raw, params, dt_ns=prof.dt_ns, normalize_output=False,
            chain_order=params.get("chain_order"),
        )

    def _save(self, workers, **kw):
        with mock.patch.dict(os.environ, {_RUNNER.WORKERS_ENV: str(workers)}):
            return self.v._save_processed_variant("t", quiet=True, overwrite=True, **kw)

    # -- tests ----------------------------------------------------------
    def test_writes_every_profile_equal_to_apply_pipeline_with_trim_and_flip(self):
        v = self.v
        v._spin_trim_start.setValue(4)
        v._spin_trim_end.setValue(3)
        v._chk_flip_profile.setChecked(True)
        self._pump(100)
        params = v._current_processing_params()
        for workers in (1, 2):
            with self.subTest(workers=workers):
                import shutil
                shutil.rmtree(self._variant_dir(), ignore_errors=True)
                self.assertEqual(self._save(workers), "t")
                self.assertEqual(self._saved(), [f"synthetic_{i}.npz" for i in range(3)])
                self.assertIsNone(v._variant_job)
                for pi, prof in enumerate(v._profiles):
                    tidx = v._trace_indices_for(int(prof.channel(0).data.shape[1]))
                    with np.load(os.path.join(self._variant_dir(), f"synthetic_{pi}.npz")) as z:
                        for ci in range(N_CH):
                            self.assertTrue(
                                np.array_equal(z[f"data_{ci}"], self._expected(prof, ci, params, tidx)),
                                f"profilo {pi} canale {ci} (workers={workers})",
                            )
                            self.assertTrue(np.array_equal(
                                z[f"east_{ci}"], np.asarray(prof.channel(ci).easting)[tidx]))

    def test_gui_stays_responsive_while_saving(self):
        ticks = []
        timer = QTimer()
        timer.setInterval(20)
        timer.timeout.connect(lambda: ticks.append(time.perf_counter()))
        real = _RUNNER._apply_channel

        def slow(payload):
            time.sleep(0.15)
            return real(payload)

        timer.start()
        with mock.patch.object(_RUNNER, "_apply_channel", slow):
            t0 = time.perf_counter()
            self._save(1)
            took = time.perf_counter() - t0
        timer.stop()
        self.assertGreater(took, 0.8)   # 6 canali x 0.15 s
        # la vecchia versione sincrona non ritornava mai all'event loop: ~0 tick
        self.assertGreater(len(ticks), int(took / 0.02 * 0.4), f"{len(ticks)} tick in {took:.2f}s")

    def test_progress_dialog_reports_and_closes(self):
        seen = []
        real = _RUNNER._apply_channel

        def slow(payload):
            time.sleep(0.05)
            return real(payload)

        def snap():
            job = self.v._variant_job
            if job:
                seen.append((job["dlg"].isVisible(), job["dlg"]._bar.value(), job["dlg"]._bar.maximum(),
                             job["dlg"]._lbl.text()))

        timer = QTimer()
        timer.setInterval(40)
        timer.timeout.connect(snap)
        timer.start()
        with mock.patch.object(_RUNNER, "_apply_channel", slow):
            self._save(1)
        timer.stop()
        self.assertTrue(seen, "la finestra di avanzamento non e' mai stata visibile")
        self.assertTrue(all(vis for vis, *_ in seen))
        self.assertTrue(all(mx == 6 for _v, _b, mx, _t in seen))       # 3 profili x 2 canali
        self.assertGreater(max(b for _v, b, _m, _t in seen), 0)
        self.assertIn("Canali elaborati", seen[-1][3])
        self.assertIsNone(self.v._variant_job)

    def test_cancel_stops_between_channels_and_keeps_written_profiles(self):
        real = _RUNNER._apply_channel
        calls = []

        def slow(payload):
            calls.append(1)
            time.sleep(0.1)
            return real(payload)

        def click_when_third_channel_started():
            job = self.v._variant_job
            if job and len(calls) >= 3:
                poll.stop()
                job["dlg"]._btn.click()

        # Non a tempo fisso: si annulla quando il worker ha INIZIATO il 3o canale
        # (il profilo 0 = canali 1-2 e' gia' calcolato, il 3o e' in corso).
        poll = QTimer()
        poll.setInterval(10)
        poll.timeout.connect(click_when_third_channel_started)
        poll.start()
        with mock.patch.object(_RUNNER, "_apply_channel", slow):
            name = self._save(1)
        poll.stop()
        self.assertEqual(len(calls), 3, "l'annullamento non ha fermato i canali successivi")
        self.assertEqual(name, "t")                       # almeno un profilo scritto: variante valida
        self.assertEqual(self._saved(), ["synthetic_0.npz"])
        self.assertIsNone(self.v._variant_job)

    def test_error_in_one_profile_does_not_stop_the_others(self):
        real = _RUNNER._apply_channel

        def picky(payload):
            if np.asarray(payload[0]).shape[1] == N_T[1]:
                raise ValueError("canale rotto")
            return real(payload)

        with mock.patch.object(_RUNNER, "_apply_channel", picky), \
                mock.patch.object(_VM.QMessageBox, "warning") as warn:
            name = self._save(1)
        self.assertEqual(name, "t")
        self.assertEqual(self._saved(), ["synthetic_0.npz", "synthetic_2.npz"])
        self.assertTrue(warn.called)
        self.assertIn("synthetic_1.ogpr", warn.call_args[0][2])
        self.assertIn("canale rotto", warn.call_args[0][2])

    def test_second_save_while_running_is_refused(self):
        real = _RUNNER._apply_channel
        answers = []

        def slow(payload):
            time.sleep(0.1)
            return real(payload)

        def again():
            answers.append(self.v._save_processed_variant("t2", quiet=True, overwrite=True))

        with mock.patch.object(_RUNNER, "_apply_channel", slow), \
                mock.patch.object(_VM.QMessageBox, "information") as info:
            QTimer.singleShot(150, again)
            self._save(1)
        self.assertEqual(answers, [None])
        self.assertTrue(info.called)
        self.assertEqual(self._saved("t2"), [])
        self.assertEqual(self._saved(), [f"synthetic_{i}.npz" for i in range(3)])

    def test_pool_that_cannot_start_falls_back_and_still_writes_everything(self):
        with mock.patch.object(_RUNNER, "_child_python", return_value=None):
            name = self._save(2)
        self.assertEqual(name, "t")
        self.assertEqual(self._saved(), [f"synthetic_{i}.npz" for i in range(3)])
        self.assertIn("in serie", self.v._lbl_status.text())


if __name__ == "__main__":
    unittest.main()
