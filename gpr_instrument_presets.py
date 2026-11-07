# -*- coding: utf-8 -*-
"""Riconoscimento dello strumento di acquisizione e preset di processing per strumento.

Modulo senza dipendenze Qt: i preset sono oggetti (una sottoclasse di
``InstrumentPreset`` per strumento, ognuna nel proprio file) che calcolano i
parametri sul profilo corrente (finestre in metri -> tracce, banda in funzione
della frequenza dell'antenna, curva di range-gain di base). Il viewer li carica
nei controlli normali, quindi ogni valore resta modificabile dall'utente.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import numpy as np

# Estensioni dei raw prodotti dagli strumenti IDS GeoRadar: OGPR (Stream, GeoSurvey
# Studio / IQMaps) e DT.
_IDS_EXTENSIONS = frozenset({".ogpr", ".dt"})

# Chiavi/etichette degli strumenti riconosciuti. Il file OGPR non porta un campo
# "modello strumento": la distinzione si fa sul numero di canali del profilo
# (Stream C = array multicanale, Opera Duo = antenna monocanale, un file per
# frequenza). Se in futuro serve una distinzione piu' fine, e' qui che va.
INSTRUMENT_IDS_STREAM_C = "ids_stream_c"
INSTRUMENT_IDS_OPERA_DUO = "ids_opera_duo"
INSTRUMENT_LABELS = {
    INSTRUMENT_IDS_STREAM_C: "IDS Stream C",
    INSTRUMENT_IDS_OPERA_DUO: "IDS Opera Duo",
}


def detect_instrument(profile) -> str | None:
    """Chiave dello strumento che ha acquisito il profilo, o None se non riconosciuto."""
    path = str(getattr(profile, "path", "") or "")
    if os.path.splitext(path)[1].lower() not in _IDS_EXTENSIONS:
        return None
    n_ch = int(getattr(profile, "n_channels", 0) or 0)
    return INSTRUMENT_IDS_STREAM_C if n_ch > 1 else INSTRUMENT_IDS_OPERA_DUO


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
class GainCurve:
    """Curva di range-gain (TVG) di base per uno strumento.

    ``breakpoints`` sono coppie (frazione 0..1 della finestra campioni, guadagno
    lineare), nello stesso formato salvato/letto dal pannello "Range gain (TVG)".
    E' il punto di partenza proposto dal preset: resta modificabile dall'utente
    con "Edit curve...".
    """

    gain_min: float
    gain_max: float
    breakpoints: tuple[tuple[float, float], ...]

    def to_params(self) -> dict:
        return {
            "pre_agc_surface_gain": float(self.gain_min),
            "pre_agc_deep_gain": float(self.gain_max),
            "pre_agc_breakpoints": [[float(x), float(y)] for x, y in self.breakpoints],
        }


class InstrumentPreset:
    """Base per il preset di processing di uno strumento.

    Ogni strumento e' una sottoclasse (in un proprio file) che imposta gli
    attributi di classe qui sotto; la logica di calcolo (scaling della banda,
    finestra di fondo in tracce, assemblaggio steps/params) e' condivisa qui
    cosi' i preset restano solo dati, non ricette duplicate.
    """

    # Identita' (obbligatori nelle sottoclassi)
    key: str = ""
    label: str = ""
    button_label: str = ""
    description: str = ""
    instrument: str | None = None

    # Passa-banda: banda di riferimento a ``ref_freq_mhz``, scalata linearmente
    # sulla frequenza reale dell'antenna e limitata al 90% del Nyquist.
    ref_freq_mhz: float = 600.0
    band_mhz_at_ref: tuple[float, float] = (150.0, 850.0)

    # Rimozione del fondo: finestra in metri, convertita in tracce dal passo
    # del profilo (minimo 8 tracce; se la finestra supera il profilo, si usa
    # l'intero profilo).
    bg_window_m: float = 10.0
    bg_method: str = "median"

    # Tempo zero.
    tz_method: str = "coppens"
    tz_mode: str = "line_by_line"
    tz_threshold: float | None = None   # None = non forzare (resta il default globale)
    tz_backup_nsamp: int | None = None  # idem

    # Dewow: solo se incluso in ``steps`` sotto (altrimenti non toccato).
    dewow_win: int | None = None

    # Step da abilitare, nell'ordine in cui il preset li mette in testa alla
    # catena (gli altri step restano dietro, nel loro ordine attuale). Non
    # includere "pre_agc_gain" qui: viene aggiunto automaticamente se ``gain``
    # e' impostato.
    steps: tuple[str, ...] = ("bandpass", "timezero", "bg_removal")

    # Curva di range-gain (TVG) di base; None = il preset non tocca il gain.
    gain: GainCurve | None = None

    def _scaled_band(self, profile) -> tuple[float, float, list[str]]:
        notes: list[str] = []
        freq = float(getattr(profile, "frequency_mhz", 0.0) or 0.0)
        scale = freq / self.ref_freq_mhz if freq > 0 else 1.0
        lo = self.band_mhz_at_ref[0] * scale
        hi = self.band_mhz_at_ref[1] * scale
        dt_ns = float(getattr(profile, "sampling_time_ns", 0.0) or 0.0)
        if dt_ns > 0:
            nyquist_mhz = 1000.0 / (2.0 * dt_ns)
            hi = min(hi, 0.9 * nyquist_mhz)
        if hi <= lo:
            hi = lo + 1.0
        if freq > 0 and abs(scale - 1.0) > 1e-6:
            notes.append(f"banda scalata per antenna a {freq:.0f} MHz")
        return lo, hi, notes

    def _bg_window_traces(self, profile) -> tuple[int, list[str]]:
        notes: list[str] = []
        step = trace_spacing_m(profile)
        n_traces = int(getattr(profile, "n_slices", 0) or 0)
        bg_window = 0
        if step > 0:
            bg_window = max(8, int(round(self.bg_window_m / step)))
            if n_traces > 0 and bg_window >= n_traces:
                bg_window = 0  # profilo piu' corto della finestra: tutto il profilo
        if bg_window == 0:
            notes.append("finestra di fondo = intero profilo")
        return bg_window, notes

    def build(self, profile) -> dict:
        """Calcola steps/params/summary per il profilo corrente."""
        notes: list[str] = []
        lo, hi, band_notes = self._scaled_band(profile)
        notes.extend(band_notes)
        bg_window, bg_notes = self._bg_window_traces(profile)
        notes.extend(bg_notes)

        params: dict = {
            "bp_low_mhz": round(lo, 1),
            "bp_high_mhz": round(hi, 1),
            "tz_method": self.tz_method,
            "tz_mode": self.tz_mode,
            "bg_mode": "line_by_line",
            "bg_method": self.bg_method,
            "bg_window": int(bg_window),
            "bg_sample_start": 0,
            "bg_sample_end": 0,
        }
        if self.tz_threshold is not None:
            params["tz_threshold"] = float(self.tz_threshold)
        if self.tz_backup_nsamp is not None:
            params["tz_backup_nsamp"] = int(self.tz_backup_nsamp)
        if self.dewow_win is not None:
            params["dewow_win"] = int(self.dewow_win)

        steps = list(self.steps)
        if self.gain is not None:
            params.update(self.gain.to_params())
            if "pre_agc_gain" not in steps:
                steps.append("pre_agc_gain")

        win_txt = (f"mediana su {self.bg_window_m:g} m (~{bg_window} tracce)"
                   if bg_window else "mediana sull'intero profilo")
        summary_parts = [
            f"passa-banda {lo:.0f}-{hi:.0f} MHz",
            f"tempo zero ({self.tz_method})",
            f"rimozione fondo {win_txt}",
        ]
        if self.gain is not None:
            summary_parts.append(
                f"range gain {self.gain.gain_min:.3g}→{self.gain.gain_max:.3g}"
            )
        summary = ", ".join(summary_parts)
        if notes:
            summary += " [" + "; ".join(notes) + "]"

        return {
            "steps": steps,
            "params": params,
            "bg_auto": bg_window == 0,
            "summary": summary,
        }


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


def _as_preset(instance: InstrumentPreset) -> Preset:
    return Preset(
        key=instance.key,
        label=instance.label,
        description=instance.description,
        build=instance.build,
        instrument=instance.instrument,
        button_label=instance.button_label,
    )


# Import in coda al modulo: ogni strumento e' una classe nel proprio file, che
# importa InstrumentPreset/GainCurve da qui sopra. Cosi' aggiungere uno
# strumento nuovo (es. Stream X) significa aggiungere un file, non toccare
# questo (a parte la riga di registrazione sotto).
try:
    from .gpr_preset_stream_c import StreamCPreset
    from .gpr_preset_opera_duo import OperaDuoPreset
except ImportError:  # esecuzione fuori pacchetto (test/script)
    from gpr_preset_stream_c import StreamCPreset
    from gpr_preset_opera_duo import OperaDuoPreset

PRESETS: tuple[Preset, ...] = (
    _as_preset(StreamCPreset()),
    _as_preset(OperaDuoPreset()),
)


def preset_by_key(key: str) -> Preset | None:
    for p in PRESETS:
        if p.key == key:
            return p
    return None
