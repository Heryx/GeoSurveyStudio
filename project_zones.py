# -*- coding: utf-8 -*-
"""Aree indagate dentro uno stesso progetto GPR.

Un progetto puo' contenere piu' aree rilevate in momenti diversi. Ogni area ha
le sue cartelle di lavoro (raw / processed / timeslices) e il suo stato
(profili, pipeline, posizionamento, ultima cartella time-slice), cosi'
importare o processare una zona non tocca le altre.

Layout su disco::

    <progetto>/metadata/project.json          stato di tutte le aree
    <progetto>/Processing/                    area "Principale" (progetti storici)
    <progetto>/Processing/zones/<cartella>/   ogni altra area

``project.json`` v2::

    {"version": "2.0", "created": ..., "updated": ...,
     "active_zone": "Principale",
     "zones": {"Principale": {"folder": "", "profiles": [...], "ui_state": ...,
                              "processing_state": ..., ...},
               "Area Nord":   {"folder": "zones/Area_Nord", ...}}}

I ``project.json`` storici (senza ``zones``) vengono letti come un'unica area
"Principale", con le stesse cartelle di prima: nessuna migrazione dei file.
Modulo senza dipendenze da Qt/QGIS."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime

DEFAULT_ZONE = "Principale"
ZONES_SUBDIR = "zones"
STATE_VERSION = "2.0"

# Chiavi del vecchio project.json "piatto" che appartengono all'area.
_ZONE_KEYS = (
    "profiles", "last_processing", "last_timeslice", "ui_state",
    "grid_positioning", "processing_state",
)


def project_json_path(root: str) -> str:
    return os.path.join(str(root or ""), "metadata", "project.json")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _read_raw(root: str) -> dict:
    path = project_json_path(root)
    if not root or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def read_state(root: str) -> dict:
    """Stato normalizzato (v2) del progetto; mai eccezioni, mai None."""
    raw = _read_raw(root)
    zones = raw.get("zones")
    if isinstance(zones, dict) and zones:
        norm = {}
        for name, payload in zones.items():
            payload = dict(payload) if isinstance(payload, dict) else {}
            payload.setdefault("folder", "" if name == DEFAULT_ZONE else _folder_for(name))
            norm[str(name)] = payload
        active = str(raw.get("active_zone") or "")
        if active not in norm:
            active = DEFAULT_ZONE if DEFAULT_ZONE in norm else next(iter(norm))
        return {
            "version": STATE_VERSION,
            "created": raw.get("created") or _now(),
            "updated": raw.get("updated") or _now(),
            "active_zone": active,
            "zones": norm,
        }
    legacy = {k: raw[k] for k in _ZONE_KEYS if k in raw}
    legacy["folder"] = ""
    return {
        "version": STATE_VERSION,
        "created": raw.get("created") or _now(),
        "updated": raw.get("updated") or _now(),
        "active_zone": DEFAULT_ZONE,
        "zones": {DEFAULT_ZONE: legacy},
    }


def write_state(root: str, state: dict) -> bool:
    path = project_json_path(root)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        out = dict(state)
        out["version"] = STATE_VERSION
        out["updated"] = _now()
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def list_zones(root: str) -> list:
    return list(read_state(root)["zones"].keys())


def active_zone(root: str) -> str:
    return read_state(root)["active_zone"]


def zone_folder(root: str, zone: str) -> str:
    """Sottocartella (relativa a Processing/) dell'area; '' per la principale."""
    payload = read_state(root)["zones"].get(zone)
    if payload is None:
        return "" if zone in ("", DEFAULT_ZONE) else _folder_for(zone)
    return str(payload.get("folder") or "")


def zone_base(root: str, zone: str | None = None) -> str:
    """Cartella di lavoro (quella che contiene raw/processed/timeslices)."""
    state = read_state(root)
    name = zone if zone else state["active_zone"]
    payload = state["zones"].get(name) or {}
    folder = str(payload.get("folder") or "")
    base = os.path.join(str(root or ""), "Processing")
    if folder:
        base = os.path.join(base, *folder.split("/"))
    return os.path.normpath(base)


def _safe(name: str) -> str:
    s = re.sub(r"[^\w\- ]+", "_", str(name or "").strip(), flags=re.UNICODE).strip(" _")
    return s.replace(" ", "_") or "area"


def _folder_for(name: str) -> str:
    return f"{ZONES_SUBDIR}/{_safe(name)}"


def unique_zone_name(root: str, wanted: str) -> str:
    """``wanted`` se libero (anche come cartella), altrimenti ``wanted 2``, ..."""
    state = read_state(root)
    taken = {n.casefold() for n in state["zones"]}
    folders = {str(p.get("folder") or "").casefold() for p in state["zones"].values()}
    base = str(wanted or "").strip() or "Nuova area"
    name, i = base, 1
    while name.casefold() in taken or _folder_for(name).casefold() in folders:
        i += 1
        name = f"{base} {i}"
    return name


def get_payload(root: str, zone: str | None = None) -> dict:
    state = read_state(root)
    name = zone if zone else state["active_zone"]
    return dict(state["zones"].get(name) or {})


def set_payload(root: str, payload: dict, zone: str | None = None, make_active: bool = True) -> bool:
    """Scrive lo stato di ``zone`` (default: attiva) lasciando intatte le altre."""
    state = read_state(root)
    name = zone if zone else state["active_zone"]
    prev = state["zones"].get(name) or {}
    new = {k: v for k, v in dict(payload or {}).items() if k != "created"}
    new["folder"] = str(prev.get("folder", "" if name == DEFAULT_ZONE else _folder_for(name)))
    state["zones"][name] = new
    if make_active:
        state["active_zone"] = name
    return write_state(root, state)


def add_zone(root: str, name: str, make_active: bool = True) -> str:
    """Crea un'area vuota (cartelle incluse) e ritorna il nome effettivo."""
    state = read_state(root)
    final = unique_zone_name(root, name)
    state["zones"][final] = {"folder": _folder_for(final)}
    if make_active:
        state["active_zone"] = final
    write_state(root, state)
    base = zone_base(root, final)
    for sub in ("raw", "processed", "timeslices"):
        os.makedirs(os.path.join(base, sub), exist_ok=True)
    return final


def set_active_zone(root: str, zone: str) -> bool:
    state = read_state(root)
    if zone not in state["zones"]:
        return False
    state["active_zone"] = zone
    return write_state(root, state)


def zone_summary(root: str, zone: str) -> dict:
    """``{"profiles": n, "variants": n}`` per mostrare l'area in un elenco."""
    payload = get_payload(root, zone)
    base = zone_base(root, zone)
    processed = os.path.join(base, "processed")
    variants = 0
    try:
        variants = sum(
            1 for d in os.listdir(processed) if os.path.isdir(os.path.join(processed, d))
        )
    except OSError:
        pass
    return {"profiles": len(payload.get("profiles") or []), "variants": variants}
