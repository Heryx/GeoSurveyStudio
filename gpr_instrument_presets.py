# -*- coding: utf-8 -*-
"""Riconoscimento dello strumento di acquisizione e preset di processing per strumento.

Modulo senza dipendenze Qt: i preset sono descritti come dati e calcolati sul
profilo corrente (finestre in metri -> tracce, banda in funzione della
frequenza dell'antenna). Il viewer li carica nei controlli normali, quindi ogni
valore resta modificabile dall'utente.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import numpy as np

INSTRUMENT_IDS = "ids"
INSTRUMENT_LABELS = {INSTRUMENT_IDS: "IDS GeoRadar"}

# Estensioni dei raw prodotti dagli strumenti IDS GeoRadar: OGPR (Stream, GeoSurvey
# Studio / IQMaps) e DT.
_IDS_EXTENSIONS = frozenset({".ogpr", ".dt"})

# Il preset IDS e' tarato su un'antenna a 600 MHz (banda 150-850 MHz); per altre
# frequenze la banda viene scalata proporzionalmente.
IDS_REF_FREQ_MHZ = 600.0
IDS_BAND_MHZ_AT_REF = (150.0, 850.0)
IDS_BG_WINDOW_M = 10.0


def detect_instrument(profile) -> str | None:
    """Chiave dello strumento che ha acquisito il profilo, o None se non riconosciuto."""
    path = str(getattr(profile, "path", "") or "")
    if os.path.splitext(path)[1].lower() in _IDS_EXTENSIONS:
        return INSTRUMENT_IDS
    return None


def describe_profile_instrument(profile) -> str:
    """Testo breve per tooltip/stato: strumento rilevato e dati di acquisizione."""
    key = detect_instrument(profile)
    if key is None:
        return "Strumento non riconosciuto"
    parts = [INSTRUMENT_LABELS.get(key, key)]
    freq = float(getattr(profile, "frequency_mhz", 0.0) or 0.0)
    if freq > 0:
        parts.append(f"{freq:.0f} MHz")
    n_ch = int(getattr(profile, "n_channels", 0) or 0)
    if n_ch > 1:
        parts.append(f"{n_ch} canali")
    return ", ".join(parts)


def trace_spacing_m(profile) -> float:
    """Passo medio tra le tracce (m); 0 se non ricavabile."""
    step = float(getattr(profile, "sampling_step_m", 0.0) or 0.0)
    if np.isfinite(step) and step > 0:
        return step
    try:
        d = np.diff(np.asarray(profile.channel(0).distances, dtype=np.float64))
        d = d[np.isfinite(d) & (d > 0)]
        return float(np.median(d)) if d.size else 0.0
    except Exception:
        return 0.0


@dataclass(frozen=True)
class Preset:
    key: str
    label: str
    description: str
    build: Callable[[object], dict]
    # Strumento a cui il preset e' dedicato: se il profilo corrente e' di
    # quello strumento il viewer mostra anche un pulsante rapido.
    instrument: str | None = None
    button_label: str = ""


def _build_ids(profile) -> dict:
    """Processo IDS: passa-banda -> tempo zero -> rimozione del fondo (mediana su 10 m).

    Corrisponde alla catena FIR / Start Time Shifter / Background Removal del
    software IDS. STC Smoothed Gain e Multi-Channel Aligner non sono implementati.
    """
    notes: list[str] = []

    freq = float(getattr(profile, "frequency_mhz", 0.0) or 0.0)
    scale = freq / IDS_REF_FREQ_MHZ if freq > 0 else 1.0
    lo = IDS_BAND_MHZ_AT_REF[0] * scale
    hi = IDS_BAND_MHZ_AT_REF[1] * scale
    dt_ns = float(getattr(profile, "sampling_time_ns", 0.0) or 0.0)
    if dt_ns > 0:
        nyquist_mhz = 1000.0 / (2.0 * dt_ns)
        hi = min(hi, 0.9 * nyquist_mhz)
    if hi <= lo:
        hi = lo + 1.0
    if freq > 0 and abs(scale - 1.0) > 1e-6:
        notes.append(f"banda scalata per antenna a {freq:.0f} MHz")

    step = trace_spacing_m(profile)
    n_traces = int(getattr(profile, "n_slices", 0) or 0)
    bg_window = 0
    if step > 0:
        bg_window = max(8, int(round(IDS_BG_WINDOW_M / step)))
        if n_traces > 0 and bg_window >= n_traces:
            bg_window = 0  # profilo piu' corto della finestra: tutto il profilo
    if bg_window == 0:
        notes.append("finestra di fondo = intero profilo")

    params = {
        "bp_low_mhz": round(lo, 1),
        "bp_high_mhz": round(hi, 1),
        "tz_method": "coppens",
        "tz_mode": "line_by_line",
        "bg_mode": "line_by_line",
        "bg_method": "median",
        "bg_window": int(bg_window),
        "bg_sample_start": 0,
        "bg_sample_end": 0,
    }
    win_txt = (f"mediana su {IDS_BG_WINDOW_M:g} m (~{bg_window} tracce)"
               if bg_window else "mediana sull'intero profilo")
    summary = (
        f"passa-banda {lo:.0f}-{hi:.0f} MHz, tempo zero (coppens), "
        f"rimozione fondo {win_txt}"
    )
    if notes:
        summary += " [" + "; ".join(notes) + "]"
    return {
        "steps": ["bandpass", "timezero", "bg_removal"],
        "params": params,
        "bg_auto": bg_window == 0,
        "summary": summary,
    }


PRESETS: tuple[Preset, ...] = (
    Preset(
        key="ids",
        label="IDS",
        instrument=INSTRUMENT_IDS,
        button_label="Preset IDS",
        description=(
            "Processo dei dati IDS GeoRadar: passa-banda 150-850 MHz (antenna a 600 MHz, "
            "scalato per le altre frequenze), tempo zero (Coppens, un solo t0 per profilo) "
            "e rimozione del fondo con mediana su 10 m. Disattiva gli altri passi.\n"
            "Non inclusi: STC Smoothed Gain e Multi-Channel Aligner (non implementati). "
            "Tutti i valori restano modificabili."
        ),
        build=_build_ids,
    ),
)


def preset_by_key(key: str) -> Preset | None:
    for p in PRESETS:
        if p.key == key:
            return p
    return None
