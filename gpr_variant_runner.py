# -*- coding: utf-8 -*-
"""Esecuzione della pipeline GPR sui canali di un rilievo, in parallelo e annullabile.

Serve al salvataggio delle varianti: ogni canale di ogni profilo e' un'unita' di
lavoro indipendente (`apply_pipeline` su una matrice campioni x tracce).

Perche' PROCESSI e non thread: la pipeline (Kirchhoff, coppens, ...) e' fatta di
cicli Python su array piccoli. Misurato su 4 canali reali: seriale 88 s, 4 thread
324 s (il GIL li rallenta di 3,7 volte), risultati identici. I processi non
condividono il GIL.

Nessuna dipendenza da Qt: il modulo si importa anche nei processi figli e nei
test. Il risultato per canale e' quello di `apply_pipeline`, identico al
percorso seriale; il parallelismo cambia solo l'ordine di esecuzione.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import spawn as mp_spawn
from concurrent.futures.process import BrokenProcessPool

from .gpr_processing import apply_pipeline, log_exception

# Core lasciati alla GUI e al resto del sistema; tetto per non saturare la memoria.
_RESERVED_CORES = 2
_MAX_WORKERS = 16
# Sovrascrive il numero di processi (1 = forza il percorso seriale).
WORKERS_ENV = "RASTERLINKER_GPR_WORKERS"

# Un processo per canale: le librerie numeriche non devono aprire altri thread.
_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def _apply_channel(payload):
    """Unita' di lavoro (eseguita nel processo figlio o in-process): un canale."""
    raw, params, dt_ns, bg_ref, chain = payload
    return apply_pipeline(
        raw, params, dt_ns=dt_ns, bg_reference_trace=bg_ref,
        normalize_output=False, chain_order=chain,
    )


def default_worker_count(n_tasks: int) -> int:
    """Processi da usare per `n_tasks` canali (>= 1)."""
    forced = os.environ.get(WORKERS_ENV, "").strip()
    if forced:
        try:
            return max(1, min(int(forced), max(1, int(n_tasks))))
        except ValueError:
            pass
    cpu = os.cpu_count() or 1
    return max(1, min(int(n_tasks), cpu - _RESERVED_CORES, _MAX_WORKERS))


def _child_python():
    """Interprete Python con cui avviare i processi figli, o None.

    In QGIS `sys.executable` e' `qgis-bin.exe`: avviarlo come figlio aprirebbe un
    altro QGIS. Si usa il Python che sta accanto (su Windows `pythonw.exe`, per
    non far comparire finestre di console)."""
    exe = sys.executable or ""
    if os.path.basename(exe).lower().startswith("python"):
        return exe
    prefixes = []
    for p in (sys.exec_prefix, sys.base_exec_prefix):
        if p and p not in prefixes:
            prefixes.append(p)
    names = (("pythonw.exe", "python.exe") if os.name == "nt"
             else (os.path.join("bin", "python3"), os.path.join("bin", "python")))
    for prefix in prefixes:
        for name in names:
            cand = os.path.join(prefix, name)
            if os.path.isfile(cand):
                return cand
    return None


@contextlib.contextmanager
def _spawn_context():
    """Contesto 'spawn' che avvia l'interprete giusto.

    `set_executable` e' globale in `multiprocessing`: all'uscita si ripristina il
    valore precedente, cosi' altro codice nella stessa sessione non lo eredita."""
    exe = _child_python()
    if exe is None:
        raise RuntimeError("interprete Python per i processi figli non trovato")
    previous = mp_spawn.get_executable()
    ctx = multiprocessing.get_context("spawn")
    ctx.set_executable(exe)
    try:
        yield ctx
    finally:
        mp_spawn.set_executable(previous)


@contextlib.contextmanager
def _hide_main_module():
    """Impedisce ai figli 'spawn' di rieseguire lo script del processo padre.

    Ogni figlio reimporta il `__main__` del padre (se ha `__file__`/`__spec__`).
    Se il padre e' uno script senza la guardia `if __name__ == "__main__"` (o QGIS
    avviato con --code), ogni figlio rifarebbe tutto il lavoro e proverebbe a
    lanciare a sua volta un pool: visto in un banco di prova, 16 figli che
    rieseguivano l'intero salvataggio in serie. I figli qui servono solo al
    pacchetto, non al `__main__`: lo si nasconde finche' il pool e' attivo."""
    main = sys.modules.get("__main__")
    saved = {}
    if main is not None:
        for attr in ("__file__", "__spec__"):
            if attr in vars(main):
                saved[attr] = vars(main)[attr]
        vars(main).pop("__file__", None)
        if "__spec__" in saved:
            main.__spec__ = None
    try:
        yield
    finally:
        if main is not None:
            for attr, value in saved.items():
                setattr(main, attr, value)


@contextlib.contextmanager
def _single_thread_env():
    """Imposta (se non lo e' gia') 1 thread per libreria numerica nei figli."""
    added = [k for k in _THREAD_ENV if k not in os.environ]
    for k in added:
        os.environ[k] = "1"
    try:
        yield
    finally:
        for k in added:
            os.environ.pop(k, None)


class _Collector:
    """Raccoglie i risultati per profilo e li consegna quando sono completi.

    `consumed` tiene i canali gia' trattati (riusciti, falliti o saltati): dopo un
    guasto del pool si riparte in serie da quelli che mancano."""

    def __init__(self, counts, on_profile_done, on_progress):
        self.counts = [int(n) for n in counts]
        self.total = int(sum(self.counts))
        self.on_profile_done = on_profile_done
        self.on_progress = on_progress
        self.errors = {}
        self.partial = {}
        self.consumed = set()
        self.saved = 0

    def failed(self, pi):
        return pi in self.errors

    def _tick(self, pi, ci):
        self.consumed.add((pi, ci))
        if self.on_progress is not None:
            self.on_progress(len(self.consumed), self.total, self.saved, pi)

    def add(self, pi, ci, arr):
        if pi not in self.errors:
            got = self.partial.setdefault(pi, {})
            got[ci] = arr
            if len(got) == self.counts[pi]:
                ordered = [got[c] for c in range(self.counts[pi])]
                del self.partial[pi]
                try:
                    self.on_profile_done(pi, ordered)
                    self.saved += 1
                except Exception as exc:  # scrittura del profilo fallita
                    log_exception("salvataggio profilo della variante", exc)
                    self.errors.setdefault(pi, str(exc))
        self._tick(pi, ci)

    def fail(self, pi, ci, exc):
        log_exception(f"pipeline profilo {pi} canale {ci}", exc)
        self.errors.setdefault(pi, f"canale {ci}: {exc}")
        self.partial.pop(pi, None)
        self._tick(pi, ci)

    def skip(self, pi, ci):
        self._tick(pi, ci)


def _run_serial(keys, make_payload, collector, is_cancelled):
    """Esegue i canali uno dopo l'altro nel thread corrente. Ritorna True se annullato."""
    for pi, ci in keys:
        if is_cancelled():
            return True
        if collector.failed(pi):
            collector.skip(pi, ci)
            continue
        try:
            out = _apply_channel(make_payload(pi, ci))
        except Exception as exc:
            collector.fail(pi, ci, exc)
            continue
        collector.add(pi, ci, out)
    return False


def _kill_pool(executor, pending):
    """Interrompe subito il pool (annullamento o guasto).

    Si terminano prima i figli e poi si chiude con wait=True: a quel punto sono
    morti, la chiusura e' immediata e il thread di gestione dell'executor esce in
    modo ordinato. Con shutdown(wait=False) prima, su Python 3.8 quel thread
    scriveva su handle gia' chiusi ('Exception in thread QueueManagerThread')."""
    procs = list((getattr(executor, "_processes", None) or {}).values())
    for fut in list(pending):
        fut.cancel()
    for p in procs:
        try:
            p.terminate()
        except Exception:
            pass
    for p in procs:
        try:
            p.join(5)
        except Exception:
            pass
    try:
        executor.shutdown(wait=True)
    except Exception:
        pass


def _run_pool(keys, make_payload, collector, is_cancelled, workers):
    """Esegue i canali in un pool di processi.

    Ritorna (annullato, guasto). Con guasto=True (i figli non partono o muoiono)
    i canali non ancora consumati vanno rifatti in serie."""
    with _spawn_context() as ctx, _hide_main_module():
        executor = ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
        pending = {}  # future -> chiave
        it = iter(keys)
        window = max(2, workers * 2)
        cancelled = False
        broken = False
        clean = False
        try:
            while True:
                # finestra limitata: i payload sono copie dei dati, non si accodano tutti
                while len(pending) < window and not cancelled and not broken:
                    key = next(it, None)
                    if key is None:
                        break
                    pi, ci = key
                    if collector.failed(pi):
                        collector.skip(pi, ci)
                        continue
                    try:
                        payload = make_payload(pi, ci)
                    except Exception as exc:
                        collector.fail(pi, ci, exc)
                        continue
                    try:
                        pending[executor.submit(_apply_channel, payload)] = key
                    except BrokenProcessPool:
                        broken = True
                if broken or not pending:
                    break
                done, _ = wait(list(pending), timeout=0.2, return_when=FIRST_COMPLETED)
                if is_cancelled():
                    cancelled = True
                    break
                for fut in done:
                    pi, ci = pending.pop(fut)
                    try:
                        out = fut.result()
                    except BrokenProcessPool:
                        broken = True
                        break
                    except Exception as exc:
                        collector.fail(pi, ci, exc)
                        continue
                    collector.add(pi, ci, out)
                if broken:
                    break
            clean = not cancelled and not broken
        finally:
            if clean:
                executor.shutdown(wait=True)
            else:
                _kill_pool(executor, pending)
        return cancelled, broken


def run_channels(
    counts,
    make_payload,
    on_profile_done,
    on_progress=None,
    is_cancelled=None,
    workers=None,
):
    """Elabora tutti i canali di tutti i profili.

    counts           : numero di canali per profilo (indice = profilo).
    make_payload     : (pi, ci) -> (raw, params, dt_ns, bg_ref, chain). Chiamata
                       nel thread che orchestra, non nei figli.
    on_profile_done  : (pi, [dato_processato per canale]) quando un profilo e'
                       completo; puo' sollevare (il profilo va in `errors`).
    on_progress      : (canali_fatti, canali_totali, profili_salvati, pi).
    is_cancelled     : () -> bool, controllato prima di ogni canale.
    workers          : processi (default: `default_worker_count`).

    Ritorna un dict: cancelled, errors {pi: messaggio}, mode ('processi' |
    'seriale'), workers, note (motivo del ripiego), elapsed, done, total.
    """
    t0 = time.perf_counter()
    is_cancelled = is_cancelled or (lambda: False)
    keys = [(pi, ci) for pi, n in enumerate(counts) for ci in range(int(n))]
    collector = _Collector(counts, on_profile_done, on_progress)
    n_workers = default_worker_count(len(keys)) if workers is None else max(1, int(workers))
    result = {"cancelled": False, "errors": collector.errors, "mode": "seriale",
              "workers": 1, "note": "", "elapsed": 0.0, "done": 0, "total": collector.total}

    todo = keys
    if n_workers > 1 and len(keys) > 1:
        try:
            with _single_thread_env():
                cancelled, broken = _run_pool(keys, make_payload, collector, is_cancelled, n_workers)
            if cancelled:
                result["cancelled"] = True
                todo = []
            elif broken:
                result["note"] = "il calcolo in parallelo non e' partito: elaborazione in serie"
            else:
                result["mode"], result["workers"] = "processi", n_workers
                todo = []
        except Exception as exc:
            result["note"] = f"calcolo in parallelo non disponibile ({exc}): elaborazione in serie"
            log_exception("avvio dei processi paralleli della variante", exc)
        todo = [k for k in todo if k not in collector.consumed]
    if todo and _run_serial(todo, make_payload, collector, is_cancelled):
        result["cancelled"] = True
    result["elapsed"] = time.perf_counter() - t0
    result["done"] = len(collector.consumed)
    return result
