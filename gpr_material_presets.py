# -*- coding: utf-8 -*-
"""Velocita' tipiche di letteratura GPR per materiale/condizione del terreno.

Valori indicativi di partenza (range tipici da manualistica GPR, es. Conyers
"Ground-Penetrating Radar for Archaeology", tabelle Annan/GSSI): servono solo
come punto di partenza comodo per il campo velocita' del RadargramDock e per la
scheda "Stima velocita'" del Profile Viewer - restano sempre sovrascrivibili a
mano o con un fit sull'iperbole. Nessuna stima automatica del materiale: la
scelta resta dell'utente, in base a cio' che sa del contesto (umidita', tipo di
suolo, muratura, ecc.).
"""

from __future__ import annotations

MATERIAL_PRESETS = [
    {"key": "aria", "label": "Aria", "velocity_m_ns": 0.29979, "rdp": 1.0},
    {"key": "sabbia_asciutta", "label": "Sabbia/suolo asciutto", "velocity_m_ns": 0.150, "rdp": 4.0},
    {"key": "suolo_medio", "label": "Suolo medio (umidita' moderata)", "velocity_m_ns": 0.100, "rdp": 9.0},
    {"key": "argilla_umida", "label": "Argilla/suolo umido", "velocity_m_ns": 0.060, "rdp": 25.0},
    {"key": "muratura_calcestruzzo", "label": "Muratura/calcestruzzo asciutto", "velocity_m_ns": 0.120, "rdp": 6.25},
    {"key": "calcare", "label": "Calcare/roccia compatta", "velocity_m_ns": 0.130, "rdp": 5.3},
    {"key": "acqua_dolce", "label": "Acqua dolce", "velocity_m_ns": 0.033, "rdp": 81.0},
]

_BY_KEY = {p["key"]: p for p in MATERIAL_PRESETS}


def material_preset(key: str) -> dict | None:
    """Ritorna il preset per `key`, o None se non esiste."""
    return _BY_KEY.get(str(key or "").strip())


def material_label(key: str) -> str:
    """Etichetta leggibile per `key` (per i popup/etichette sorgente), o `key` stesso."""
    preset = material_preset(key)
    return preset["label"] if preset else str(key or "")
