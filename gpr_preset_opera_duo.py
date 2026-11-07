# -*- coding: utf-8 -*-
"""Preset di processing per l'antenna monocanale IDS Opera Duo (600/700 MHz).

Tarato sul rilievo TIPASA (canale 600 MHz, confrontato con IQMaps): dewow,
passa-banda piu' largo del preset Stream C (l'Opera Duo ha una risposta in
frequenza diversa dall'array), tempo zero a superficie, rimozione del fondo a
mediana su ~16 m e la curva di range-gain (TVG) di base tarata su quel
rilievo. Opera Duo acquisisce due frequenze (600 e 700 MHz, un file per
canale/array): la banda scala comunque sulla frequenza reale del profilo, cosi'
il preset resta corretto anche sul canale a 700 MHz.
"""

from __future__ import annotations

try:
    from .gpr_instrument_presets import GainCurve, INSTRUMENT_IDS_OPERA_DUO, InstrumentPreset
except ImportError:  # esecuzione fuori pacchetto (test/script)
    from gpr_instrument_presets import GainCurve, INSTRUMENT_IDS_OPERA_DUO, InstrumentPreset

# Curva TVG di base tarata su TIPASA (breakpoints: frazione 0..1 della finestra
# campioni -> guadagno lineare). Molto piu' ripida di quella Stream C: l'Opera
# Duo su questo rilievo perde piu' segnale in profondita'. Resta modificabile
# dall'utente con "Edit curve..." dopo aver applicato il preset.
_TIPASA_GAIN_CURVE = GainCurve(
    gain_min=804.53,
    gain_max=926436.07,
    breakpoints=(
        (0.0, 804.53),
        (0.0176, 8341.85),
        (0.0560, 33719.01),
        (0.0798, 188607.80),
        (0.2044, 327629.57),
        (0.3528, 550933.62),
        (0.5300, 896826.01),
        (0.7743, 691592.23),
        (1.0, 926436.07),
    ),
)


class OperaDuoPreset(InstrumentPreset):
    key = "opera_duo"
    label = "IDS Opera Duo"
    button_label = "Preset Opera Duo"
    instrument = INSTRUMENT_IDS_OPERA_DUO
    description = (
        "Processo per l'antenna monocanale IDS Opera Duo (600/700 MHz, un file per "
        "frequenza/array): dewow, passa-banda 150-1308 MHz scalato sulla frequenza "
        "dell'antenna (riferimento 600 MHz), tempo zero a superficie, rimozione del "
        "fondo a mediana su ~16 m e range gain (TVG) con la curva tarata sul rilievo "
        "TIPASA confrontando con IQMaps. Disattiva gli altri passi. Tutti i valori "
        "restano modificabili."
    )

    ref_freq_mhz = 600.0
    band_mhz_at_ref = (150.0, 1308.53)
    bg_window_m = 16.0
    bg_method = "median"
    tz_method = "surface"
    tz_mode = "line_by_line"
    tz_threshold = 0.4
    tz_backup_nsamp = 4
    dewow_win = 16

    # Dewow in testa (rimuove la deriva DC prima di banda/t0/fondo); range gain
    # in coda (aggiunto automaticamente perche' 'gain' e' impostato sotto).
    steps = ("dewow", "bandpass", "timezero", "bg_removal")
    gain = _TIPASA_GAIN_CURVE
