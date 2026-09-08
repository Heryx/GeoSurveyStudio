# -*- coding: utf-8 -*-
"""Navigazione e posizionamento dei profili — Fase 5b del blueprint processing.

Modello puro (dataclass + funzioni numpy, nessun Qt/QGIS): assegna una posizione
a ogni scan di un radargramma a partire da *marker* noti, con dead-reckoning
(interpolazione lineare) fra un marker e il successivo — lo stesso meccanismo
dei software di time-slicing GPR.

Quattro modi di assegnazione:
  * ``artificial_markers`` — N marker equidistanti sul totale scan (ruota
    odometrica con slittamento);
  * ``field_markers``      — da tag inseriti in campo (indici di scan noti);
  * ``interval_markers``   — passo fisso di scan (ruota "ideale");
  * ``gps_trace_markers``  — un marker per lettura GPS con scan sincronizzato.

Più: ``reverse_along`` (linee invertite / zig-zag), ``stagger_shift``
(correzione staggering), ``line_plan_coords`` (ascissa lungo linea → E/N, per
qualunque angolo, incluse linee ANG a lunghezza frazionaria).

L'integrazione con `gpr_manual_positioning_dialog` è separata (fase UI).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MARKER_KINDS = ("field", "artificial", "interval", "gps")


@dataclass
class Marker:
    trace_idx: int
    pos_m: float | None = None                 # ascissa nota lungo il profilo [m]
    coord: tuple | None = None                 # (E, N, Z) per navigazione GPS
    kind: str = "field"

    def __post_init__(self):
        self.trace_idx = int(self.trace_idx)
        if self.kind not in MARKER_KINDS:
            raise ValueError(f"kind marker non valido: {self.kind!r}")


# ---------------------------------------------------------------------------
# generazione marker (i 4 modi)
# ---------------------------------------------------------------------------

def artificial_markers(
    n_traces: int,
    length_m: float,
    n_marks: int | None = None,
) -> list[Marker]:
    """N marker equidistanti da 0 a ``length_m`` sul totale di ``n_traces`` scan.

    Se ``n_marks`` è None → ``round(length_m) + 1`` (marker a ogni unità di
    range, come per una ruota metrica). Gli scan fra due marker vengono poi
    posizionati per dead-reckoning.
    """
    n = int(n_traces)
    L = float(length_m)
    if n < 2 or not np.isfinite(L) or L <= 0.0:
        return []
    if n_marks is None:
        n_marks = int(round(L)) + 1
    n_marks = max(2, int(n_marks))
    idx = np.round(np.linspace(0, n - 1, n_marks)).astype(int)
    pos = np.linspace(0.0, L, n_marks)
    return [Marker(int(i), float(p), kind="artificial") for i, p in zip(idx, pos)]


def interval_markers(
    n_traces: int,
    scans_per_mark: int,
    unit_m: float = 1.0,
) -> tuple[list[Marker], int]:
    """Marker a passo fisso di scan (nessuno slittamento).

    Ritorna ``(markers, leftover)`` dove ``leftover`` = scan oltre l'ultimo
    marker (informativo: quella coda resta fuori da ogni elaborazione).
    """
    n = int(n_traces)
    step = max(1, int(scans_per_mark))
    if n < 2:
        return [], 0
    idxs = list(range(0, n, step))
    markers = [Marker(i, float(k * unit_m), kind="interval") for k, i in enumerate(idxs)]
    leftover = int(n - 1 - idxs[-1])
    return markers, leftover


def field_markers(
    flag_trace_indices,
    length_m: float | None = None,
    unit_m: float = 1.0,
) -> list[Marker]:
    """Marker dai tag di campo: indici di scan noti.

    Se ``length_m`` è dato, le ascisse vengono distribuite equidistanti da 0 a
    ``length_m`` sui marker trovati (utile quando i tag segnano unità di range
    regolari); altrimenti ``pos_m = k * unit_m``.
    """
    idxs = sorted(int(i) for i in (flag_trace_indices or []))
    if not idxs:
        return []
    if length_m is not None and len(idxs) >= 2:
        pos = np.linspace(0.0, float(length_m), len(idxs))
    else:
        pos = np.arange(len(idxs), dtype=float) * float(unit_m)
    return [Marker(i, float(p), kind="field") for i, p in zip(idxs, pos)]


def gps_trace_markers(rows) -> list[Marker]:
    """Un marker per lettura GPS. ``rows`` = iterabile di ``(scan_idx, E, N, Z)``
    (Z opzionale → 0)."""
    out: list[Marker] = []
    for r in rows or []:
        r = list(r)
        si = int(r[0])
        e = float(r[1]); nth = float(r[2])
        z = float(r[3]) if len(r) > 3 else 0.0
        out.append(Marker(si, coord=(e, nth, z), kind="gps"))
    return out


# ---------------------------------------------------------------------------
# dead-reckoning
# ---------------------------------------------------------------------------

def dead_reckon_along(markers: list[Marker], n_traces: int) -> np.ndarray:
    """Ascissa lungo il profilo [m] per ogni scan, interpolando fra i marker
    (estrapolazione costante oltre il primo/ultimo)."""
    n = int(n_traces)
    ms = sorted((m for m in markers if m.pos_m is not None), key=lambda m: m.trace_idx)
    if n < 1:
        return np.zeros(0, dtype=np.float64)
    if len(ms) == 0:
        return np.full(n, np.nan)
    if len(ms) == 1:
        return np.full(n, float(ms[0].pos_m))
    xi = np.array([m.trace_idx for m in ms], dtype=np.float64)
    yi = np.array([m.pos_m for m in ms], dtype=np.float64)
    return np.interp(np.arange(n, dtype=np.float64), xi, yi)


def dead_reckon_coords(markers: list[Marker], n_traces: int) -> np.ndarray:
    """(n_traces, 3) E/N/Z per ogni scan da marker GPS (coord != None)."""
    n = int(n_traces)
    ms = sorted((m for m in markers if m.coord is not None), key=lambda m: m.trace_idx)
    out = np.full((max(n, 0), 3), np.nan, dtype=np.float64)
    if n < 1 or len(ms) == 0:
        return out
    xi = np.array([m.trace_idx for m in ms], dtype=np.float64)
    C = np.array([list(m.coord)[:3] + [0.0] * (3 - len(m.coord)) for m in ms], dtype=np.float64)
    q = np.arange(n, dtype=np.float64)
    if len(ms) == 1:
        out[:] = C[0]
        return out
    for k in range(3):
        out[:, k] = np.interp(q, xi, C[:, k])
    return out


# ---------------------------------------------------------------------------
# reverse / staggering / geometria di linea
# ---------------------------------------------------------------------------

def reverse_along(along: np.ndarray, length_m: float) -> np.ndarray:
    """Inverte l'ascissa di una linea acquisita in direzione opposta
    (``L - along``). Per linee zig-zag alternare l'applicazione."""
    a = np.asarray(along, dtype=np.float64)
    return float(length_m) - a


def stagger_shift(along: np.ndarray, stagger_m: float) -> np.ndarray:
    """Correzione staggering: trasla l'ascissa di una linea (tipicamente le
    linee reverse) di ``stagger_m`` prima del gridding."""
    return np.asarray(along, dtype=np.float64) + float(stagger_m)


def line_plan_coords(
    x0: float, y0: float, x1: float, y1: float, along: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Mappa l'ascissa ``along`` lungo il segmento (x0,y0)->(x1,y1) in coordinate
    di piano (E, N). Funziona per qualunque angolo (survey X, Y, XY, ANG); se la
    lunghezza nominale del segmento differisce dal max(along), l'ascissa viene
    proiettata proporzionalmente sul segmento."""
    a = np.asarray(along, dtype=np.float64)
    dx = float(x1) - float(x0)
    dy = float(y1) - float(y0)
    seg_len = float(np.hypot(dx, dy))
    if seg_len < 1e-12:
        return np.full_like(a, float(x0)), np.full_like(a, float(y0))
    ux, uy = dx / seg_len, dy / seg_len
    amax = float(np.nanmax(a)) if a.size and np.isfinite(a).any() else 0.0
    scale = (seg_len / amax) if amax > 1e-12 else 1.0
    d = a * scale
    return float(x0) + d * ux, float(y0) + d * uy
