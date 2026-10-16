# -*- coding: utf-8 -*-
"""Aiuti al disegno sul radargramma: aggancio al picco e tracciamento di un riflettore.

Modulo puro (solo numpy), senza Qt: lavora su un radargramma ``data`` di forma
(campioni, tracce) e restituisce indici di riga (campione).  Un riflettore e' una
fase (picco positivo o ventre negativo) che si segue di traccia in traccia: si aggancia
l'estremo locale piu' vicino con lo stesso segno del punto di partenza.
"""

from __future__ import annotations

import math

import numpy as np


def local_extremum_row(trace, row: int, half_window: int, sign: int | None = None) -> int:
    """Riga del picco (o ventre) piu' vicino a ``row`` entro ``+-half_window`` campioni.

    ``sign`` = +1 cerca i picchi positivi, -1 i ventri; se manca si usa il segno del
    campione di partenza.  Se nella finestra non c'e' nessun estremo locale (segnale
    monotono, traccia piatta) si restituisce ``row`` invariato: meglio non agganciare
    che inventare un punto.
    """
    tr = np.asarray(trace, dtype=np.float64)
    n = tr.size
    if n == 0:
        return 0
    row = int(np.clip(row, 0, n - 1))
    w = max(0, int(half_window))
    if w == 0 or n < 3:
        return row
    if sign is None:
        sign = 1 if tr[row] >= 0 else -1
    lo, hi = max(0, row - w), min(n - 1, row + w)
    sig = tr * float(sign)
    best = None            # (distanza, -ampiezza, riga)
    for i in range(max(lo, 1), min(hi, n - 2) + 1):
        if sig[i] > 0 and sig[i] >= sig[i - 1] and sig[i] >= sig[i + 1]:
            cand = (abs(i - row), -sig[i], i)
            if best is None or cand < best:
                best = cand
    return int(best[2]) if best is not None else row


def trace_between(data, trace0: int, row0: int, trace1: int, row1: int,
                  half_window: int, step: int = 1, max_drift: int | None = None) -> list:
    """Segue la fase del riflettore da (trace0,row0) a (trace1,row1).

    Ritorna ``[(traccia, riga), ...]`` con gli estremi inclusi, una voce ogni ``step``
    tracce (l'ultima e' sempre ``trace1``).  Traccia per traccia si agganciano l'estremo
    locale vicino alla riga precedente (continuita' di fase), tenendo il percorso entro
    ``max_drift`` campioni dalla retta tra i due punti: cosi' il tracciamento non
    "salta" su un riflettore vicino e arriva dove ha cliccato l'utente.
    """
    arr = np.asarray(data)
    n_rows, n_traces = arr.shape
    t0 = int(np.clip(trace0, 0, n_traces - 1))
    t1 = int(np.clip(trace1, 0, n_traces - 1))
    r0 = int(np.clip(row0, 0, n_rows - 1))
    r1 = int(np.clip(row1, 0, n_rows - 1))
    if t0 == t1:
        return [(t0, r0)] if r0 == r1 else [(t0, r0), (t1, r1)]
    w = max(1, int(half_window))
    drift = int(max_drift) if max_drift is not None else 3 * w
    step = max(1, int(step))
    direction = 1 if t1 > t0 else -1
    sign = 1 if arr[r0, t0] >= 0 else -1
    path = [(t0, r0)]
    prev = r0
    count = abs(t1 - t0)
    for k in range(1, count):
        t = t0 + direction * k
        expected = r0 + (r1 - r0) * (k / count)
        row = local_extremum_row(arr[:, t], int(round(prev)), w, sign)
        row = int(np.clip(row, math.floor(expected - drift), math.ceil(expected + drift)))
        row = int(np.clip(row, 0, n_rows - 1))
        prev = row
        if k % step == 0:
            path.append((t, row))
    path.append((t1, r1))
    return path
