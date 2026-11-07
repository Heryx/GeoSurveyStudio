# -*- coding: utf-8 -*-
"""Preset di processing per l'array multicanale IDS Stream C.

Tarato sul rilievo SUASA (19 canali, antenna 600 MHz): banda passa-banda
scalata sulla frequenza, tempo zero Coppens, rimozione del fondo a mediana su
10 m, piu' la curva di range-gain (TVG) di base gia' usata su quel rilievo.
"""

from __future__ import annotations

try:
    from .gpr_instrument_presets import GainCurve, INSTRUMENT_IDS_STREAM_C, InstrumentPreset
except ImportError:  # esecuzione fuori pacchetto (test/script)
    from gpr_instrument_presets import GainCurve, INSTRUMENT_IDS_STREAM_C, InstrumentPreset

# Curva TVG di base usata su SUASA (breakpoints: frazione 0..1 della finestra
# campioni -> guadagno lineare). Resta modificabile dall'utente con
# "Edit curve..." dopo aver applicato il preset.
_SUASA_GAIN_CURVE = GainCurve(
    gain_min=47.75,
    gain_max=12956.81,
    breakpoints=(
        (0.0, 47.75),
        (0.0894, 278.80),
        (0.2427, 1367.85),
        (0.4774, 4500.57),
        (0.7695, 7090.04),
        (1.0, 12956.81),
    ),
)


class StreamCPreset(InstrumentPreset):
    key = "stream_c"
    label = "IDS Stream C"
    button_label = "Preset Stream C"
    instrument = INSTRUMENT_IDS_STREAM_C
    description = (
        "Processo per l'array multicanale IDS Stream C (usato su SUASA): passa-banda "
        "150-850 MHz scalato sulla frequenza dell'antenna (riferimento 600 MHz), tempo "
        "zero Coppens (un solo t0 per profilo), rimozione del fondo a mediana su 10 m e "
        "range gain (TVG) con la curva di base gia' tarata su quel rilievo. Disattiva "
        "gli altri passi. Tutti i valori restano modificabili."
    )

    ref_freq_mhz = 600.0
    band_mhz_at_ref = (150.0, 850.0)
    bg_window_m = 10.0
    bg_method = "median"
    tz_method = "coppens"
    tz_mode = "line_by_line"

    steps = ("bandpass", "timezero", "bg_removal")
    gain = _SUASA_GAIN_CURVE
