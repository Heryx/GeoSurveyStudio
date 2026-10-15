# -*- coding: utf-8 -*-
"""Export dei radargrammi processati in un formato semplice per una pipeline di
Deep Learning esterna (fuori da QGIS, es. notebook Jupyter in un repo separato).

Per ogni canale scrive una cartella <output_root>/<variante>/<profilo>__ch<N>/:
  radargram.npz  : data (n_samples, n_traces) float32, dist (n_traces,) float64
  geometry.json  : easting/northing/altitude/heading/distances per traccia +
                    geometry_source ("viewer" = posizionamento corrente,
                    "npz" = fotografia salvata al momento del processing, puo'
                    non essere piu' aggiornata)
  meta.json      : dt/velocita'/frequenza/pipeline applicata, per tracciabilita'

La geometria segue lo stesso schema del pannello Radargrammi processati
(radargram_store.VariantStore): se il Profile Viewer e' aperto si legge il
posizionamento corrente tramite il suo geometry_provider, altrimenti si
ripiega sulla fotografia scritta nel .npz al momento del salvataggio.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np

from .gpr_npz_io import aligned_geometry_columns, read_variant_manifest

TRAINING_GEOMETRY_FORMAT = "rasterlinker_gpr_training_geometry_v1"
TRAINING_META_FORMAT = "rasterlinker_gpr_training_meta_v1"

_GEOMETRY_COLS = ("easting", "northing", "altitude", "heading")
_NPZ_KEY = {"easting": "east", "northing": "north", "altitude": "alt", "heading": "head"}


def _full_geometry(store, pi: int, ci: int):
    """(colonne easting/northing/altitude/heading, sorgente) per il canale —
    stessa logica di VariantStore._geometry, ma senza scartare altitude/heading
    (il dock non ne ha bisogno, l'export per training si'')."""
    info = store.profiles[pi]
    n_t = int(info.n_traces[ci]) if ci < len(info.n_traces) else 0
    provider = store.geometry_provider
    if provider is not None:
        try:
            src = provider(info.stem)
            n_src = int(getattr(src, "n_channels", 0) or 0)
            if src is not None and n_src > 0:
                cols = aligned_geometry_columns(
                    src.channel(min(ci, n_src - 1)), n_t, info.view_geom)
                if cols is not None:
                    return {k: cols[k] for k in _GEOMETRY_COLS}, "viewer"
        except Exception:
            pass    # provider guasto: si ripiega sulla fotografia del .npz
    with np.load(info.path, allow_pickle=False) as npz:
        cols = {k: np.asarray(npz[f"{_NPZ_KEY[k]}_{ci}"], dtype=np.float64) for k in _GEOMETRY_COLS}
    return cols, "npz"


def export_channel_for_training(store, pi: int, ci: int, output_root: str):
    """Esporta un canale in <output_root>/<variante>/<profilo>__ch<ci>/.

    Ritorna (cartella_scritta, geometry_source).
    """
    info = store.profiles[pi]
    ch = store.channel(pi, ci)
    cols, source = _full_geometry(store, pi, ci)

    sample_dir = os.path.join(output_root, store.name, f"{info.stem}__ch{ci}")
    os.makedirs(sample_dir, exist_ok=True)

    np.savez_compressed(os.path.join(sample_dir, "radargram.npz"),
                         data=ch.data, dist=ch.dist)

    geometry = {
        "format": TRAINING_GEOMETRY_FORMAT,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "geometry_source": source,
        "epsg": int(info.epsg),
        "easting": cols["easting"].tolist(),
        "northing": cols["northing"].tolist(),
        "altitude": cols["altitude"].tolist(),
        "heading": cols["heading"].tolist(),
        "distances": ch.dist.tolist(),
    }
    with open(os.path.join(sample_dir, "geometry.json"), "w", encoding="utf-8") as fh:
        json.dump(geometry, fh, ensure_ascii=False, indent=2)

    variant_meta = read_variant_manifest(store.path) or {}
    meta = {
        "format": TRAINING_META_FORMAT,
        "variant": store.name,
        "profile_stem": info.stem,
        "channel": ci,
        "n_samples": int(ch.data.shape[0]),
        "n_traces": int(ch.data.shape[1]),
        "dt_ns": info.dt_ns,
        "dz_m": ch.dz_m,
        "velocity_m_s": store.effective_velocity_m_s(pi),
        "frequency_mhz": info.frequency_mhz,
        "sampling_step_m": info.sampling_step_m,
        "pipeline_params": variant_meta.get("pipeline_params", {}),
        "chain_order": variant_meta.get("chain_order", []),
        "source_npz": info.path,
    }
    with open(os.path.join(sample_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    return sample_dir, source


def export_variant_for_training(store, output_root: str) -> dict:
    """Esporta tutti i profili/canali della variante aperta.

    Ritorna {"exported": [cartelle...], "stale_geometry": [cartelle con
    geometry_source == "npz"...], "errors": [(stem, ci, messaggio), ...]}.
    """
    exported, stale, errors = [], [], []
    for pi, info in enumerate(store.profiles):
        for ci in range(info.n_channels):
            try:
                sample_dir, source = export_channel_for_training(store, pi, ci, output_root)
            except Exception as exc:
                errors.append((info.stem, ci, str(exc)))
                continue
            exported.append(sample_dir)
            if source == "npz":
                stale.append(sample_dir)
    return {"exported": exported, "stale_geometry": stale, "errors": errors}
