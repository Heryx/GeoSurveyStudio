# -*- coding: utf-8 -*-
"""Esecuzione parallela e annullabile della pipeline sui canali (salvataggio varianti).

Il parallelismo non deve cambiare i numeri: ogni canale passa da `apply_pipeline`
con gli stessi dati e parametri, in serie o in un processo figlio."""

import importlib
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_PLUGIN_PARENT = _HERE.parents[1].parent
for _p in (str(_PLUGIN_PARENT), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PKG = "rasterlinker_feature_interpolation"
runner = importlib.import_module(PKG + ".gpr_variant_runner")
gp = importlib.import_module(PKG + ".gpr_processing")

PARAMS = {"dewow": True, "dewow_win": 8, "bg_removal": True, "agc": True, "agc_win": 16}
DT = 0.125


def _raw(pi, ci, n_s=96, n_t=48):
    rng = np.random.default_rng(1000 * pi + ci)
    return (rng.standard_normal((n_s, n_t)) * 50 + 5.0 * pi + ci).astype(np.float32)


def _expected(pi, ci):
    return gp.apply_pipeline(_raw(pi, ci), PARAMS, dt_ns=DT, normalize_output=False)


class _Job:
    """Registra cio' che l'esecutore consegna."""

    def __init__(self, counts, fail_payload=(), fail_done=()):
        self.counts = counts
        self.fail_payload = set(fail_payload)
        self.fail_done = set(fail_done)
        self.done = {}
        self.progress = []

    def make_payload(self, pi, ci):
        if pi in self.fail_payload:
            raise ValueError("payload non costruibile")
        return (_raw(pi, ci), PARAMS, DT, None, None)

    def on_done(self, pi, arrays):
        if pi in self.fail_done:
            raise OSError("disco pieno")
        self.done[pi] = arrays

    def on_progress(self, done, total, saved, pi):
        self.progress.append((done, total, saved, pi))

    def run(self, **kw):
        return runner.run_channels(
            self.counts, self.make_payload, self.on_done,
            on_progress=self.on_progress, **kw,
        )


class TestSerial(unittest.TestCase):
    def test_serial_equals_apply_pipeline(self):
        job = _Job([3, 2])
        res = job.run(workers=1)
        self.assertEqual(res["mode"], "seriale")
        self.assertFalse(res["cancelled"])
        self.assertEqual(res["errors"], {})
        self.assertEqual(sorted(job.done), [0, 1])
        for pi, n in enumerate(job.counts):
            self.assertEqual(len(job.done[pi]), n)
            for ci in range(n):
                self.assertTrue(np.array_equal(job.done[pi][ci], _expected(pi, ci)))

    def test_progress_counts_every_channel_and_ends_at_total(self):
        job = _Job([2, 2, 1])
        res = job.run(workers=1)
        self.assertEqual([p[0] for p in job.progress], [1, 2, 3, 4, 5])
        self.assertEqual(job.progress[-1][:3], (5, 5, 3))
        self.assertEqual(res["done"], 5)
        self.assertEqual(res["total"], 5)

    def test_payload_error_isolates_the_profile(self):
        job = _Job([2, 2, 2], fail_payload={1})
        res = job.run(workers=1)
        self.assertEqual(sorted(job.done), [0, 2])
        self.assertEqual(list(res["errors"]), [1])
        # la barra arriva comunque in fondo: i canali saltati contano
        self.assertEqual(res["done"], 6)

    def test_pipeline_error_isolates_the_profile(self):
        job = _Job([2, 2])
        orig = job.make_payload

        def bad(pi, ci):
            if pi == 0 and ci == 1:
                return (np.zeros(5, dtype=np.float32), PARAMS, DT, None, None)  # non 2D -> errore o vuoto
            return orig(pi, ci)

        job.make_payload = bad
        with mock.patch.object(runner, "_apply_channel", side_effect=_raise_on_1d(runner._apply_channel)):
            res = job.run(workers=1)
        self.assertEqual(list(res["errors"]), [0])
        self.assertEqual(sorted(job.done), [1])

    def test_save_error_is_recorded_and_others_continue(self):
        job = _Job([2, 2], fail_done={0})
        res = job.run(workers=1)
        self.assertEqual(sorted(job.done), [1])
        self.assertIn("disco pieno", res["errors"][0])

    def test_profile_without_channels_is_not_an_error(self):
        job = _Job([0, 2])
        res = job.run(workers=1)
        self.assertEqual(res["errors"], {})
        self.assertEqual(sorted(job.done), [1])

    def test_cancel_stops_before_next_channel(self):
        job = _Job([3, 3])
        res = job.run(workers=1, is_cancelled=lambda: len(job.progress) >= 2)
        self.assertTrue(res["cancelled"])
        self.assertEqual(res["done"], 2)
        self.assertEqual(job.done, {})


def _raise_on_1d(real):
    def wrapped(payload):
        if np.asarray(payload[0]).ndim != 2:
            raise ValueError("attesa matrice 2D")
        return real(payload)
    return wrapped


class TestPool(unittest.TestCase):
    def test_pool_equals_serial(self):
        job = _Job([3, 3, 2])
        res = job.run(workers=2)
        self.assertEqual(res["mode"], "processi", res["note"])
        self.assertEqual(res["workers"], 2)
        self.assertEqual(res["errors"], {})
        self.assertEqual(sorted(job.done), [0, 1, 2])
        for pi, n in enumerate(job.counts):
            for ci in range(n):
                self.assertTrue(np.array_equal(job.done[pi][ci], _expected(pi, ci)),
                                f"profilo {pi} canale {ci} diverso dal seriale")

    def test_pool_progress_reaches_total(self):
        job = _Job([2, 2])
        res = job.run(workers=2)
        self.assertEqual(res["mode"], "processi", res["note"])
        self.assertEqual(job.progress[-1][:3], (4, 4, 2))

    def test_pool_payload_error_isolates_the_profile(self):
        job = _Job([2, 2, 2], fail_payload={1})
        res = job.run(workers=2)
        self.assertEqual(sorted(job.done), [0, 2])
        self.assertEqual(list(res["errors"]), [1])

    def test_pool_cancel_terminates_and_reports(self):
        job = _Job([6, 6])
        res = job.run(workers=2, is_cancelled=lambda: len(job.progress) >= 2)
        self.assertTrue(res["cancelled"])
        self.assertLess(res["done"], 12)


class TestFallback(unittest.TestCase):
    def _check_fell_back(self, job, res):
        self.assertEqual(res["mode"], "seriale")
        self.assertTrue(res["note"])
        self.assertEqual(res["errors"], {})
        self.assertEqual(sorted(job.done), [0, 1])
        for pi, n in enumerate(job.counts):
            for ci in range(n):
                self.assertTrue(np.array_equal(job.done[pi][ci], _expected(pi, ci)))

    def test_no_child_interpreter_falls_back_to_serial(self):
        job = _Job([2, 2])
        with mock.patch.object(runner, "_child_python", return_value=None):
            res = job.run(workers=2)
        self._check_fell_back(job, res)

    def test_unlaunchable_child_falls_back_to_serial(self):
        job = _Job([2, 2])
        bad = os.path.join(tempfile.gettempdir(), "non_esiste", "python_finto.exe")
        with mock.patch.object(runner, "_child_python", return_value=bad):
            res = job.run(workers=2)
        self._check_fell_back(job, res)


class TestGlobalStateRestored(unittest.TestCase):
    """`multiprocessing.spawn.set_executable` e' globale: non va lasciato cambiato."""

    def _run_and_compare(self, patch_exe):
        from multiprocessing import spawn as mp_spawn
        before = mp_spawn.get_executable()
        job = _Job([2, 2])
        if patch_exe is None:
            job.run(workers=2)
        else:
            with mock.patch.object(runner, "_child_python", return_value=patch_exe):
                job.run(workers=2)
        self.assertEqual(mp_spawn.get_executable(), before)

    def test_restored_after_success(self):
        self._run_and_compare(None)

    def test_restored_after_launch_failure(self):
        self._run_and_compare(os.path.join(tempfile.gettempdir(), "non_esiste", "python_finto.exe"))

    def test_env_threads_restored(self):
        keys = runner._THREAD_ENV
        before = {k: os.environ.get(k) for k in keys}
        _Job([2, 2]).run(workers=2)
        self.assertEqual({k: os.environ.get(k) for k in keys}, before)


class TestParentScriptNotReexecuted(unittest.TestCase):
    """I figli 'spawn' non devono rieseguire lo script del padre.

    Caso reale visto in un banco di prova: uno script SENZA la guardia
    `if __name__ == "__main__"` lancia il salvataggio; ogni figlio reimportava lo
    script, rifaceva tutto il lavoro in serie e il pool non serviva a nulla."""

    def test_hide_main_module_hides_and_restores(self):
        main = sys.modules["__main__"]
        before = {a: getattr(main, a, "<assente>") for a in ("__file__", "__spec__")}
        with runner._hide_main_module():
            self.assertFalse(hasattr(main, "__file__"))
            self.assertIsNone(main.__spec__)
        after = {a: getattr(main, a, "<assente>") for a in ("__file__", "__spec__")}
        self.assertEqual(after, before)

    def test_unguarded_parent_script_runs_once(self):
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            marker = os.path.join(tmp, "marker.txt")
            script = os.path.join(tmp, "unguarded.py")
            with open(script, "w", encoding="utf-8") as fh:
                fh.write(
                    "import os, sys\n"
                    f"open({marker!r}, 'a').write('run pid=%d\\n' % os.getpid())\n"   # senza guardia
                    f"sys.path.insert(0, {str(_PLUGIN_PARENT)!r})\n"
                    "import numpy as np\n"
                    f"from {PKG} import gpr_variant_runner as vr\n"
                    "res = vr.run_channels([2, 2], "
                    "lambda pi, ci: (np.ones((64, 32), dtype=np.float32) * (pi + ci + 1), "
                    "{'dewow': True, 'dewow_win': 8}, 0.125, None, None), "
                    "lambda pi, arrs: None, workers=2)\n"
                    "print('MODE=' + res['mode'] + ' NOTE=' + res['note'])\n"
                )
            proc = subprocess.run([sys.executable, script], capture_output=True, text=True, timeout=180)
            out = proc.stdout + proc.stderr
            with open(marker, encoding="utf-8") as fh:
                runs = fh.read().splitlines()
        self.assertEqual(len(runs), 1, f"lo script del padre e' stato rieseguito {len(runs)} volte:\n{out}")
        self.assertIn("MODE=processi", out, out)


class TestWorkerCount(unittest.TestCase):
    def test_env_override_and_bounds(self):
        with mock.patch.dict(os.environ, {runner.WORKERS_ENV: "1"}):
            self.assertEqual(runner.default_worker_count(50), 1)
        with mock.patch.dict(os.environ, {runner.WORKERS_ENV: "99"}):
            self.assertEqual(runner.default_worker_count(4), 4)
        with mock.patch.dict(os.environ, {runner.WORKERS_ENV: "boh"}):
            self.assertGreaterEqual(runner.default_worker_count(50), 1)

    def test_never_more_than_tasks_or_cap(self):
        env = {k: v for k, v in os.environ.items() if k != runner.WORKERS_ENV}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(runner.default_worker_count(1), 1)
            self.assertLessEqual(runner.default_worker_count(10_000), runner._MAX_WORKERS)
            self.assertGreaterEqual(runner.default_worker_count(10_000), 1)


class TestChildPython(unittest.TestCase):
    def test_python_executable_is_used_as_is(self):
        with mock.patch.object(sys, "executable", os.path.join("x", "python.exe")):
            self.assertEqual(runner._child_python(), os.path.join("x", "python.exe"))

    def test_qgis_bin_is_replaced_by_python_next_to_it(self):
        with tempfile.TemporaryDirectory() as prefix:
            names = ("pythonw.exe", "python.exe") if os.name == "nt" else ("bin/python3",)
            for n in names:
                p = os.path.join(prefix, n)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                open(p, "w").close()
            with mock.patch.object(sys, "executable", os.path.join(prefix, "qgis-bin.exe")), \
                    mock.patch.object(sys, "exec_prefix", prefix), \
                    mock.patch.object(sys, "base_exec_prefix", prefix):
                got = runner._child_python()
            self.assertEqual(os.path.normcase(got), os.path.normcase(os.path.join(prefix, names[0])))

    def test_no_python_found_returns_none(self):
        with tempfile.TemporaryDirectory() as prefix, \
                mock.patch.object(sys, "executable", os.path.join(prefix, "qgis-bin.exe")), \
                mock.patch.object(sys, "exec_prefix", prefix), \
                mock.patch.object(sys, "base_exec_prefix", prefix):
            self.assertIsNone(runner._child_python())


if __name__ == "__main__":
    unittest.main()
