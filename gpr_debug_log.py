"""Log di debug per la lettura/applicazione delle coordinate dei profili GPR.

Buffer circolare in memoria (consultabile dal viewer, con pulsante "Pulisci")
+ stampa in console Python + pannello Log Messages di QGIS (tab "GPR coord").
Nessuna dipendenza da Qt: importabile anche dai test.
"""
from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

_MAX_LINES = 3000
_lock = threading.Lock()
_buffer: deque[str] = deque(maxlen=_MAX_LINES)
ENABLED = True


def log(tag: str, msg: str) -> None:
    if not ENABLED:
        return
    line = f"{time.strftime('%H:%M:%S')} [{tag}] {msg}"
    with _lock:
        _buffer.append(line)
    print(f"[GPR-COORD] {line}")
    try:
        from qgis.core import QgsMessageLog
        QgsMessageLog.logMessage(f"[{tag}] {msg}", "GPR coord")
    except Exception:
        pass


_seen: set = set()


def log_once(key, tag: str, msg: str) -> None:
    """Come log(), ma una sola volta per chiave (evita spam nei ridisegni)."""
    with _lock:
        if key in _seen:
            return
        _seen.add(key)
    log(tag, msg)


def get_text() -> str:
    with _lock:
        return "\n".join(_buffer)


def clear() -> None:
    with _lock:
        _buffer.clear()
        _seen.clear()


def describe_track(easting, northing) -> str:
    """Riepilogo compatto di una traccia E/N: n, non-finiti, range, primo/ultimo, lunghezza."""
    try:
        e = np.asarray(easting, dtype=np.float64).ravel()
        n = np.asarray(northing, dtype=np.float64).ravel()
        if e.size == 0 or n.size == 0:
            return "vuota"
        ok = np.isfinite(e) & np.isfinite(n)
        bad = int(e.size - ok.sum())
        if not ok.any():
            return f"n={e.size} tutte non finite"
        ee, nn = e[ok], n[ok]
        length = float(np.sum(np.hypot(np.diff(ee), np.diff(nn)))) if ee.size > 1 else 0.0
        return (
            f"n={e.size} nonfinite={bad} "
            f"E[{ee.min():.3f}..{ee.max():.3f}] N[{nn.min():.3f}..{nn.max():.3f}] "
            f"first=({ee[0]:.3f},{nn[0]:.3f}) last=({ee[-1]:.3f},{nn[-1]:.3f}) "
            f"lung={length:.3f}m"
        )
    except Exception as exc:  # il log non deve mai rompere il flusso
        return f"errore descrizione: {exc}"
