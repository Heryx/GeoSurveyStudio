# -*- coding: utf-8 -*-
"""Velocita'/RDP preimpostata per variante+profilo (sidecar JSON, no Qt).

La velocita' che si vede nel Profile Viewer con lo strumento "Hyperbola Fit" non e'
mai stata scritta da nessuna parte (verificato: nessuna assegnazione a
``profile.velocity_m_s`` in tutto il plugin) - resta un overlay diagnostico che
sparisce alla chiusura. Questo modulo introduce un posto persistente e condiviso,
separato dal ``velocity_m_s`` "di fabbrica" salvato nel .npz (header strumento):

- ``<processed>/<variante>/velocity_overrides.json`` con due livelli:
  - ``"default"``: valore di variante, si applica a ogni profilo che non ha un
    override piu' specifico (tipico: un fit fatto una volta, valido per tutta
    l'area indagata).
  - ``"profiles"``: override per singolo profilo (stem del file .npz), ha
    precedenza sul default.

Usato da ``RadargramDock`` (puntatore "Misura velocita'", campo editabile) e dal
Profile Viewer (pulsante "Salva velocita' nella variante", salvataggio automatico
alla scrittura di una variante). Non tocca in alcun modo ``ProfileInfo.velocity_m_s``
ne' il rendering del Profile Viewer.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional

from .gpr_hyperbola_math import DEFAULT_VELOCITY_M_NS, m_ns_to_velocity_m_s, velocity_to_rdp

__all__ = [
    "VELOCITY_OVERRIDES_FILENAME",
    "velocity_overrides_path",
    "read_velocity_overrides",
    "write_velocity_override",
    "resolve_velocity_m_ns",
]

VELOCITY_OVERRIDES_FILENAME = "velocity_overrides.json"


def velocity_overrides_path(variant_dir: str) -> str:
    return os.path.join(str(variant_dir or ""), VELOCITY_OVERRIDES_FILENAME)


def read_velocity_overrides(variant_dir: str) -> dict:
    """Contenuto grezzo del sidecar, ``{"default": {...}|None, "profiles": {...}}``."""
    path = velocity_overrides_path(variant_dir)
    if not os.path.isfile(path):
        return {"default": None, "profiles": {}}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return {"default": None, "profiles": {}}
    if not isinstance(data, dict):
        return {"default": None, "profiles": {}}
    default = data.get("default") if isinstance(data.get("default"), dict) else None
    profiles = data.get("profiles") if isinstance(data.get("profiles"), dict) else {}
    return {"default": default, "profiles": profiles}


def write_velocity_override(variant_dir: str, *, profile_stem: Optional[str] = None,
                             velocity_m_ns: float, rdp: Optional[float] = None,
                             source: str = "manual", updated_at: Optional[str] = None) -> str:
    """Scrive (o aggiorna) l'override per ``profile_stem``, o il default di variante
    se ``profile_stem`` e' None/vuoto. Ritorna il path del sidecar scritto."""
    variant_dir = str(variant_dir or "").strip()
    if not variant_dir:
        raise ValueError("variant_dir mancante.")
    os.makedirs(variant_dir, exist_ok=True)

    v_mns = float(velocity_m_ns)
    entry = {
        "velocity_m_ns": v_mns,
        "rdp": float(rdp) if rdp is not None else velocity_to_rdp(m_ns_to_velocity_m_s(v_mns)),
        "source": str(source or "manual"),
        "updated_at": updated_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    data = read_velocity_overrides(variant_dir)
    stem = str(profile_stem or "").strip()
    if stem:
        data["profiles"][stem] = entry
    else:
        data["default"] = entry

    path = velocity_overrides_path(variant_dir)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    return path


def resolve_velocity_m_ns(variant_dir: str, profile_stem: Optional[str]) -> dict:
    """Valore effettivo per un profilo: override per-profilo, poi default di
    variante, poi la costante condivisa. Ritorna sempre
    ``{"velocity_m_ns", "rdp", "source"}`` (mai None)."""
    data = read_velocity_overrides(variant_dir)
    stem = str(profile_stem or "").strip()
    entry = data["profiles"].get(stem) if stem else None
    if not entry:
        entry = data["default"]
    if entry and entry.get("velocity_m_ns") is not None:
        v_mns = float(entry["velocity_m_ns"])
        rdp = entry.get("rdp")
        return {
            "velocity_m_ns": v_mns,
            "rdp": float(rdp) if rdp is not None else velocity_to_rdp(m_ns_to_velocity_m_s(v_mns)),
            "source": str(entry.get("source") or "manual"),
        }
    return {
        "velocity_m_ns": DEFAULT_VELOCITY_M_NS,
        "rdp": velocity_to_rdp(m_ns_to_velocity_m_s(DEFAULT_VELOCITY_M_NS)),
        "source": "default",
    }
