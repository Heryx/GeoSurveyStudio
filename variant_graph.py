# -*- coding: utf-8 -*-
"""Grafo di varianti di processing GPR — Fase 3 del blueprint.

Modello non distruttivo: ogni *variante* è un nodo che applica **un solo**
step di filtro sopra il proprio genitore. La radice `raw` è il dato convertito,
mai modificato. Una catena lineare di nodi = una pipeline; l'albero permette di
tenere vive N varianti nominate e di ramificare da un punto qualsiasi.

Questo modulo è puro Python (numpy + stdlib): nessuna dipendenza da Qt o QGIS.
Si appoggia a `gpr_processing.apply_pipeline` / `DEFAULT_PIPELINE` /
`FILTER_REGISTRY` e alla cache per-step (`stage_cache`) già esistente per il
calcolo incrementale.

Persistenza: `<processed_root>/variant_graph.json`
(formato `rasterlinker_variant_graph_v1`). I radargrammi materializzati per
profilo restano gestiti da `gpr_npz_io` (una sottocartella per nodo).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

import numpy as np

try:  # package import (dentro QGIS) o import piatto (test headless)
    from .gpr_processing import DEFAULT_PIPELINE, FILTER_REGISTRY, apply_pipeline
except ImportError:  # pragma: no cover
    from gpr_processing import DEFAULT_PIPELINE, FILTER_REGISTRY, apply_pipeline

GRAPH_FORMAT = "rasterlinker_variant_graph_v1"
GRAPH_FILENAME = "variant_graph.json"
ROOT_ID = "raw"


class VariantGraphError(Exception):
    pass


@dataclass
class VariantNode:
    id: str
    parent_id: str | None            # None solo per la radice `raw`
    step: str | None                 # None per la radice; altrimenti id in FILTER_REGISTRY
    params: dict = field(default_factory=dict)   # parametri del SOLO step di questo nodo
    scope: str = "line"              # "line" | "survey"
    label: str = ""
    created: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["params"] = dict(self.params or {})
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "VariantNode":
        return cls(
            id=str(d["id"]),
            parent_id=(str(d["parent_id"]) if d.get("parent_id") else None),
            step=(str(d["step"]) if d.get("step") else None),
            params=dict(d.get("params") or {}),
            scope=str(d.get("scope", "line") or "line"),
            label=str(d.get("label", "") or ""),
            created=str(d.get("created", "") or ""),
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class VariantGraph:
    """Albero di `VariantNode`. La radice `raw` esiste sempre."""

    def __init__(self):
        root = VariantNode(id=ROOT_ID, parent_id=None, step=None, label="raw",
                           created=_now())
        self._nodes: dict[str, VariantNode] = {ROOT_ID: root}

    # ------------------------------------------------------------------ query
    def __contains__(self, node_id: str) -> bool:
        return str(node_id) in self._nodes

    def node(self, node_id: str) -> VariantNode:
        try:
            return self._nodes[str(node_id)]
        except KeyError:
            raise VariantGraphError(f"Nodo variante inesistente: {node_id!r}")

    def ids(self) -> list[str]:
        return list(self._nodes.keys())

    def children(self, node_id: str) -> list[VariantNode]:
        nid = str(node_id)
        return [n for n in self._nodes.values() if n.parent_id == nid]

    def ancestors(self, node_id: str) -> list[VariantNode]:
        """Radice → nodo (incluso)."""
        chain: list[VariantNode] = []
        cur: str | None = str(node_id)
        seen: set[str] = set()
        while cur is not None:
            if cur in seen:
                raise VariantGraphError(f"Ciclo nel grafo varianti su {cur!r}")
            seen.add(cur)
            n = self.node(cur)
            chain.append(n)
            cur = n.parent_id
        chain.reverse()
        return chain

    def leaves(self) -> list[str]:
        parents = {n.parent_id for n in self._nodes.values() if n.parent_id}
        return [nid for nid in self._nodes if nid not in parents]

    def path_labels(self, node_id: str) -> list[str]:
        return [n.label or n.step or n.id for n in self.ancestors(node_id)]

    # ------------------------------------------------------------------ mutate
    def add_node(
        self,
        parent_id: str,
        step: str,
        params: dict | None = None,
        scope: str = "line",
        label: str = "",
        node_id: str | None = None,
    ) -> VariantNode:
        pid = str(parent_id)
        if pid not in self._nodes:
            raise VariantGraphError(f"Genitore inesistente: {parent_id!r}")
        step = str(step)
        if step not in FILTER_REGISTRY:
            raise VariantGraphError(f"Step non registrato in FILTER_REGISTRY: {step!r}")
        if scope not in ("line", "survey"):
            raise VariantGraphError(f"scope non valido: {scope!r}")
        nid = str(node_id) if node_id else self._fresh_id(step)
        if nid in self._nodes:
            raise VariantGraphError(f"Id nodo già usato: {nid!r}")
        node = VariantNode(
            id=nid, parent_id=pid, step=step, params=dict(params or {}),
            scope=scope, label=(label or step), created=_now(),
        )
        self._nodes[nid] = node
        return node

    def remove_subtree(self, node_id: str) -> list[str]:
        nid = str(node_id)
        if nid == ROOT_ID:
            raise VariantGraphError("La radice `raw` non è rimovibile.")
        if nid not in self._nodes:
            raise VariantGraphError(f"Nodo inesistente: {node_id!r}")
        to_drop = [nid]
        stack = [nid]
        while stack:
            cur = stack.pop()
            for ch in self.children(cur):
                to_drop.append(ch.id)
                stack.append(ch.id)
        for d in to_drop:
            self._nodes.pop(d, None)
        return to_drop

    def rename(self, node_id: str, label: str) -> None:
        self.node(node_id).label = str(label or "")

    def set_params(self, node_id: str, params: dict) -> None:
        n = self.node(node_id)
        if n.step is None:
            raise VariantGraphError("La radice `raw` non ha parametri.")
        n.params = dict(params or {})

    def _fresh_id(self, step: str) -> str:
        base = step
        if base not in self._nodes:
            return base
        i = 2
        while f"{base}_{i}" in self._nodes:
            i += 1
        return f"{base}_{i}"

    # ------------------------------------------------------------------ resolve
    def resolve(self, node_id: str) -> tuple[dict, list[str]]:
        """Restituisce (params_pipeline_completi, [step_id ordinati]).

        `params` parte da DEFAULT_PIPELINE; per ogni antenato con uno step:
        abilita quello step e sovrascrive con `node.params`.
        """
        merged = dict(DEFAULT_PIPELINE)
        steps: list[str] = []
        for n in self.ancestors(node_id):
            if not n.step:
                continue
            merged[n.step] = True
            for k, v in (n.params or {}).items():
                merged[k] = v
            steps.append(n.step)
        merged["chain_order"] = list(steps)
        return merged, steps

    def content_key(self, node_id: str) -> str:
        """Hash stabile del contenuto risolto (per naming cache / staleness)."""
        merged, steps = self.resolve(node_id)
        payload = {
            "steps": steps,
            "params": {k: merged[k] for k in sorted(merged) if k != "chain_order"},
        }
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha1(blob).hexdigest()[:16]

    # ------------------------------------------------------------------ compute
    def materialize_arrays(
        self,
        node_id: str,
        raw_arrays: list[np.ndarray],
        dt_ns: float = 0.117,
        cache: dict | None = None,
    ) -> list[np.ndarray]:
        """Applica la catena risolta del nodo a ogni array di `raw_arrays`.

        `raw_arrays` : lista di matrici (n_samples, n_traces) float32 — il dato
                       convertito (radice `raw`), una per profilo/canale.
        `cache`      : dict opzionale {indice_array: stage_cache} riusato tra
                       chiamate; `apply_pipeline` valida ogni step per
                       fingerprint, quindi il prefisso invariato non viene
                       ricalcolato.
        Ritorna una nuova lista di array (float32, non normalizzati).
        """
        arrays = [np.asarray(a, dtype=np.float32) for a in (raw_arrays or [])]
        if str(node_id) == ROOT_ID:
            return [a.copy() for a in arrays]
        merged, steps = self.resolve(node_id)
        if not steps:
            return [a.copy() for a in arrays]
        out: list[np.ndarray] = []
        for i, arr in enumerate(arrays):
            sc = None
            if isinstance(cache, dict):
                sc = cache.setdefault(i, {})
            res = apply_pipeline(
                arr, merged, dt_ns=float(dt_ns),
                normalize_output=False, chain_order=list(steps), stage_cache=sc,
            )
            out.append(np.asarray(res, dtype=np.float32))
        return out

    # ------------------------------------------------------------------ persist
    def to_json(self) -> dict:
        return {
            "format": GRAPH_FORMAT,
            "root": ROOT_ID,
            "created": _now(),
            "nodes": [self._nodes[nid].to_dict() for nid in self._nodes],
        }

    @classmethod
    def from_json(cls, payload: dict) -> "VariantGraph":
        if not isinstance(payload, dict):
            raise VariantGraphError("Payload grafo non valido.")
        g = cls()
        nodes = payload.get("nodes") or []
        # ricostruisci mantenendo la radice creata dal costruttore
        parsed = {}
        for d in nodes:
            n = VariantNode.from_dict(d)
            parsed[n.id] = n
        if ROOT_ID in parsed:
            g._nodes[ROOT_ID] = parsed.pop(ROOT_ID)
        # inserisci in ordine topologico (genitori prima)
        pending = list(parsed.values())
        guard = len(pending) ** 2 + 1
        while pending and guard > 0:
            guard -= 1
            n = pending.pop(0)
            if n.parent_id in g._nodes:
                g._nodes[n.id] = n
            else:
                pending.append(n)
        if pending:
            raise VariantGraphError(
                "Grafo varianti incoerente: nodi senza genitore "
                f"({[n.id for n in pending]})."
            )
        return g

    def save(self, processed_root: str) -> str:
        os.makedirs(processed_root, exist_ok=True)
        path = os.path.join(processed_root, GRAPH_FILENAME)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_json(), fh, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def load(cls, processed_root: str) -> "VariantGraph":
        path = os.path.join(processed_root, GRAPH_FILENAME)
        if not os.path.isfile(path):
            return cls()
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_json(json.load(fh))

    # ------------------------------------------------------------------ legacy
    @classmethod
    def from_legacy_manifest(cls, manifest: dict) -> "VariantGraph":
        """Converte un vecchio `variant.json` piatto (pipeline intera +
        chain_order) in una catena lineare di nodi, uno per step attivo."""
        g = cls()
        if not isinstance(manifest, dict):
            return g
        params = dict(manifest.get("pipeline_params") or {})
        order = list(manifest.get("chain_order") or [])
        if not order:
            order = [s for s in DEFAULT_PIPELINE.get("chain_order") or [] if s]
        parent = ROOT_ID
        for step in order:
            if step not in FILTER_REGISTRY:
                continue
            if not bool(params.get(step, False)):
                continue
            step_params = {
                k: v for k, v in params.items()
                if k == step or k.startswith(step.split("_")[0] + "_") or _param_belongs(k, step)
            }
            node = g.add_node(parent, step, params=step_params,
                              label=manifest.get("name") and f"{manifest['name']}:{step}" or step)
            parent = node.id
        return g


# Mappa grezza parametro -> step, per estrarre i soli parametri rilevanti da un
# manifest legacy (best effort; i parametri extra non fanno danno in resolve()).
_PARAM_STEP_HINTS = {
    "dewow": ("dewow_win",),
    "timezero": ("tz_method", "tz_mode", "tz_threshold", "tz_backup_nsamp"),
    "bg_removal": ("bg_mode", "bg_method", "bg_window", "bg_sample_start", "bg_sample_end"),
    "bandpass": ("bp_low_mhz", "bp_high_mhz", "bp_taper_frac"),
    "migration": ("mig_velocity_m_s", "mig_dx_m"),
    "pre_agc_gain": ("pre_agc_surface_gain", "pre_agc_deep_gain", "pre_agc_breakpoints"),
    "agc": ("agc_win", "agc_sample_start", "agc_max_gain"),
    "notch": ("notch_freq_mhz", "notch_bw_mhz"),
    "whiten": ("whiten_low_mhz", "whiten_high_mhz", "whiten_gain"),
    "amp_eq": ("amp_eq_window", "amp_eq_gain"),
    "rad_threshold": ("rad_thr_neg", "rad_thr_pos"),
    "power_gain": ("power_gain_mode", "power_gain_exp", "power_gain_t0_ns", "power_gain_scale"),
    "dcshift": ("dcshift_sample_start", "dcshift_sample_end"),
}


def _param_belongs(key: str, step: str) -> bool:
    return key in _PARAM_STEP_HINTS.get(step, ())
