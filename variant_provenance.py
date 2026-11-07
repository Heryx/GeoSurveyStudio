# -*- coding: utf-8 -*-
"""Provenienza delle varianti di processing — passo 10 dell'integrazione UI.

Le varianti su disco sono cartelle ``processed/<nome>/`` (una serie di ``.npz``
+ ``variant.json``). Questo modulo aggiunge un file ``variant_graph.json`` nella
cartella ``processed/`` che tiene traccia, per ogni variante, di **da quale
variante è stata derivata** (``parent``), della catena filtri e dell'ultimo
step attivo — così il pannello "Varianti" può mostrarle come albero invece che
come lista piatta.

Puro stdlib, nessuna dipendenza da Qt/QGIS. È volutamente più leggero di
``variant_graph.VariantGraph`` (un nodo = un'intera pipeline materializzata, non
un singolo step): serve solo a ricostruire la gerarchia visiva.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os

FORMAT = "rasterlinker_variant_provenance_v1"
FILENAME = "variant_graph.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def graph_path(processed_root: str) -> str:
    return os.path.join(processed_root, FILENAME)


def load(processed_root: str) -> dict:
    """``{nome: {parent, chain, step, n_profiles, created}}`` (dict vuoto se
    assente/illeggibile)."""
    path = graph_path(processed_root)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    nodes = data.get("nodes")
    if not isinstance(nodes, dict):
        return {}
    out = {}
    for name, meta in nodes.items():
        if not isinstance(meta, dict):
            continue
        out[str(name)] = {
            "parent": (str(meta["parent"]) if meta.get("parent") else None),
            "chain": [str(s) for s in (meta.get("chain") or []) if s],
            "step": (str(meta["step"]) if meta.get("step") else None),
            "n_profiles": int(meta.get("n_profiles") or 0),
            "created": str(meta.get("created") or ""),
        }
    return out


def save(processed_root: str, entries: dict) -> str:
    os.makedirs(processed_root, exist_ok=True)
    payload = {
        "format": FORMAT,
        "updated": _now(),
        "nodes": {
            str(name): {
                "parent": (str(m["parent"]) if m.get("parent") else None),
                "chain": [str(s) for s in (m.get("chain") or []) if s],
                "step": (str(m["step"]) if m.get("step") else None),
                "n_profiles": int(m.get("n_profiles") or 0),
                "created": str(m.get("created") or _now()),
            }
            for name, m in (entries or {}).items()
        },
    }
    path = graph_path(processed_root)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return path


def set_entry(
    processed_root: str,
    name: str,
    *,
    parent: str | None = None,
    chain: list | None = None,
    step: str | None = None,
    n_profiles: int | None = None,
) -> dict:
    """Aggiorna/crea la voce ``name`` e riscrive il file. Un ciclo parent==name
    (o antenato mancante) viene neutralizzato a ``parent=None``. Ritorna il
    dict aggiornato."""
    entries = load(processed_root)
    prev = entries.get(str(name), {})
    p = str(parent) if parent else None
    if p == str(name):
        p = None
    entries[str(name)] = {
        "parent": p,
        "chain": [str(s) for s in (chain if chain is not None else prev.get("chain") or []) if s],
        "step": (str(step) if step else prev.get("step")),
        "n_profiles": int(n_profiles if n_profiles is not None else prev.get("n_profiles") or 0),
        "created": str(prev.get("created") or _now()),
    }
    _break_cycles(entries)
    save(processed_root, entries)
    return entries


def remove_entry(processed_root: str, name: str, *, reparent_children: bool = True) -> dict:
    """Toglie ``name`` dal grafo. I figli vengono ri-agganciati al nonno
    (``reparent_children``) o resi radici."""
    entries = load(processed_root)
    gone = entries.pop(str(name), None)
    if gone is not None:
        new_parent = gone.get("parent") if reparent_children else None
        for m in entries.values():
            if m.get("parent") == str(name):
                m["parent"] = new_parent
        save(processed_root, entries)
    return entries


def _break_cycles(entries: dict) -> None:
    """Se risalendo i ``parent`` da un nodo si ritorna sul nodo stesso, il suo
    ``parent`` viene azzerato."""
    for name in list(entries):
        seen = {name}
        cur = entries[name].get("parent")
        while cur is not None and cur in entries:
            if cur in seen:
                entries[name]["parent"] = None
                break
            seen.add(cur)
            cur = entries[cur].get("parent")


def build_forest(entries: dict, present_names: list) -> list:
    """Costruisce la foresta ``[{name, meta, children:[...]}]`` limitandosi alle
    varianti effettivamente presenti su disco (``present_names``). Le cartelle
    senza voce nel grafo diventano radici. I parent mancanti → radice."""
    present = [str(n) for n in present_names]
    present_set = set(present)
    meta = {n: dict(entries.get(n, {})) for n in present}
    for n in present:
        p = meta[n].get("parent")
        if p not in present_set:
            meta[n]["parent"] = None

    nodes = {n: {"name": n, "meta": meta[n], "children": []} for n in present}
    roots = []
    for n in present:                             # ordine stabile: come present_names
        p = meta[n].get("parent")
        if p and p in nodes:
            nodes[p]["children"].append(nodes[n])
        else:
            roots.append(nodes[n])
    return roots
