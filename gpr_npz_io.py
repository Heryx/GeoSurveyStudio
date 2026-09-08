# -*- coding: utf-8 -*-
"""Persistenza dei radargrammi GPR processati in formato NumPy .npz.

Motivazione
-----------
Il Profile Viewer applica la pipeline di processing solo in RAM (per la
visualizzazione). Questo modulo permette di *applicare davvero* le modifiche
ai dati, scrivendo su disco il radargramma processato estratto dall'OGPR in
un file .npz autosufficiente e ri-importabile, cosi' da poter poi generare
le time-slice a partire dai dati gia' filtrati e confrontare varianti di
processing diverse (es. bandpass+background vs solo background).

Layout di un file .npz (una per profilo OGPR)
---------------------------------------------
  meta            : stringa JSON (0-d array unicode) con dt, velocita', EPSG,
                    numero canali/campioni/tracce, pipeline applicata.
  data_<ci>       : (n_samples, n_traces) float32   radargramma processato
  east_<ci>       : (n_traces,) float64             easting per traccia
  north_<ci>      : (n_traces,) float64             northing per traccia
  alt_<ci>        : (n_traces,) float64             quota per traccia
  head_<ci>       : (n_traces,) float64             heading per traccia
  dist_<ci>       : (n_traces,) float64             distanza cumulativa (m)

I canali sono salvati separatamente perche' il time-zero puo' produrre un
numero di campioni diverso tra canali; evitare un array rettangolare unico
previene padding artificiosi.

Nota: nessun oggetto Python viene serializzato (allow_pickle=False sia in
scrittura sia in lettura); 'meta' e' una semplice stringa JSON.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np

PROCESSED_FORMAT = "rasterlinker_gpr_processed_v1"
VARIANT_FORMAT = "rasterlinker_gpr_variant_v1"
VARIANT_MANIFEST = "variant.json"


# ---------------------------------------------------------------------------
# Radargramma processato per profilo
# ---------------------------------------------------------------------------

def save_processed_profile_npz(
    variant_dir: str,
    profile,
    processed_by_channel: list,
    pipeline_params: dict | None = None,
    chain_order: list | None = None,
) -> str:
    """Scrive un profilo processato in <variant_dir>/<basename>.npz.

    Parametri
    ---------
    variant_dir          : cartella della variante (creata se assente).
    profile              : OgprProfile sorgente (per geometria e metadati).
    processed_by_channel : lista di array (n_samples, n_traces) float32,
                           uno per canale, gia' processati.
    pipeline_params      : parametri pipeline applicati (per tracciabilita').
    chain_order          : ordine dei filtri applicati.

    Ritorna il percorso del file .npz scritto.
    """
    os.makedirs(variant_dir, exist_ok=True)
    n_ch = int(len(processed_by_channel))
    if n_ch <= 0:
        raise ValueError("Nessun canale processato da salvare.")

    arrays: dict = {}
    for ci in range(n_ch):
        data = np.asarray(processed_by_channel[ci], dtype=np.float32)
        if data.ndim != 2:
            raise ValueError(
                f"Canale {ci}: attesa matrice 2D (n_samples, n_traces), "
                f"trovata shape {data.shape}."
            )
        ch = profile.channel(ci)
        n_traces = int(data.shape[1])

        def _geo(attr):
            arr = np.asarray(getattr(ch, attr), dtype=np.float64)
            # Allinea la geometria al numero di tracce del dato processato.
            if arr.shape[0] != n_traces:
                out = np.zeros(n_traces, dtype=np.float64)
                m = min(arr.shape[0], n_traces)
                out[:m] = arr[:m]
                return out
            return arr

        arrays[f"data_{ci}"] = data
        arrays[f"east_{ci}"] = _geo("easting")
        arrays[f"north_{ci}"] = _geo("northing")
        arrays[f"alt_{ci}"] = _geo("altitude")
        arrays[f"head_{ci}"] = _geo("heading")
        arrays[f"dist_{ci}"] = _geo("distances")

    meta = {
        "format": PROCESSED_FORMAT,
        "source_path": str(getattr(profile, "path", "") or ""),
        "n_channels": n_ch,
        "n_samples_by_channel": [int(a.shape[0]) for a in processed_by_channel],
        "n_traces_by_channel": [int(a.shape[1]) for a in processed_by_channel],
        "sampling_time_ns": float(
            getattr(profile, "sampling_time_ns", getattr(profile, "dt_ns", 0.0)) or 0.0
        ),
        "sampling_step_m": float(getattr(profile, "sampling_step_m", 0.0) or 0.0),
        "velocity_m_s": float(getattr(profile, "velocity_m_s", 0.0) or 0.0),
        "frequency_mhz": float(getattr(profile, "frequency_mhz", 0.0) or 0.0),
        "epsg": int(getattr(profile, "epsg", 0) or 0),
        "polarization": str(getattr(profile, "polarization", "") or ""),
        "swath_name": str(getattr(profile, "swath_name", "") or ""),
        "swath_id": str(getattr(profile, "swath_id", "") or ""),
        "pipeline_params": dict(pipeline_params or {}),
        "chain_order": list(chain_order or []),
    }
    arrays["meta"] = np.array(json.dumps(meta, ensure_ascii=False))

    base = os.path.splitext(
        os.path.basename(str(getattr(profile, "path", "") or ""))
    )[0] or "profilo"
    out_path = os.path.join(variant_dir, base + ".npz")
    np.savez_compressed(out_path, **arrays)
    return out_path


def load_processed_profile_npz(path: str):
    """Ricostruisce un OgprProfile a partire da un .npz processato.

    Il profilo restituito espone gli stessi attributi/proprieta' usati dallo
    slicer OGPR (channel(i).data/easting/northing/..., dt_ns, depth_max_m,
    n_channels, n_slices, epsg, ...), quindi puo' essere passato direttamente
    a import_ogpr_as_slices con use_processing=False.
    """
    from .gpr_ogpr_reader import OgprProfile, OgprChannel

    with np.load(path, allow_pickle=False) as npz:
        meta_raw = npz["meta"]
        meta = json.loads(str(meta_raw))
        n_ch = int(meta.get("n_channels", 0) or 0)
        channels = []
        for ci in range(n_ch):
            channels.append(OgprChannel(
                channel_idx=ci,
                data=np.asarray(npz[f"data_{ci}"], dtype=np.float32),
                easting=np.asarray(npz[f"east_{ci}"], dtype=np.float64),
                northing=np.asarray(npz[f"north_{ci}"], dtype=np.float64),
                altitude=np.asarray(npz[f"alt_{ci}"], dtype=np.float64),
                heading=np.asarray(npz[f"head_{ci}"], dtype=np.float64),
                distances=np.asarray(npz[f"dist_{ci}"], dtype=np.float64),
            ))

    n_samples = int(channels[0].data.shape[0]) if channels else 0
    n_slices = int(channels[0].data.shape[1]) if channels else 0

    return OgprProfile(
        path=str(meta.get("source_path", path) or path),
        version_major=0,
        version_minor=0,
        swath_name=str(meta.get("swath_name", "") or ""),
        swath_id=str(meta.get("swath_id", "") or ""),
        array_id=0,
        n_samples=n_samples,
        n_channels=n_ch,
        n_slices=n_slices,
        sampling_step_m=float(meta.get("sampling_step_m", 0.0) or 0.0),
        sampling_time_ns=float(meta.get("sampling_time_ns", 0.0) or 0.0),
        velocity_m_s=float(meta.get("velocity_m_s", 0.0) or 0.0),
        frequency_mhz=float(meta.get("frequency_mhz", 0.0) or 0.0),
        polarization=str(meta.get("polarization", "") or ""),
        epsg=int(meta.get("epsg", 0) or 0),
        value_type="float32",
        channels=channels,
        header_raw={},
    )


# ---------------------------------------------------------------------------
# Manifest di variante
# ---------------------------------------------------------------------------

def write_variant_manifest(
    variant_dir: str,
    name: str,
    pipeline_params: dict | None,
    chain_order: list | None,
    files: list,
    extra: dict | None = None,
) -> str:
    """Scrive <variant_dir>/variant.json descrivendo la variante di processing."""
    os.makedirs(variant_dir, exist_ok=True)
    payload = {
        "format": VARIANT_FORMAT,
        "name": str(name or ""),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pipeline_params": dict(pipeline_params or {}),
        "chain_order": list(chain_order or []),
        "files": [os.path.basename(str(f)) for f in (files or [])],
        "n_profiles": int(len(files or [])),
    }
    if extra:
        payload.update(dict(extra))
    path = os.path.join(variant_dir, VARIANT_MANIFEST)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return path


def read_variant_manifest(variant_dir: str) -> dict | None:
    path = os.path.join(variant_dir, VARIANT_MANIFEST)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def list_variants(processed_root: str) -> list:
    """Elenca le sottocartelle-variante presenti in processed_root.

    Una cartella e' considerata variante se contiene un variant.json oppure
    almeno un file .npz.
    """
    if not processed_root or not os.path.isdir(processed_root):
        return []
    out = []
    for name in sorted(os.listdir(processed_root)):
        d = os.path.join(processed_root, name)
        if not os.path.isdir(d):
            continue
        has_manifest = os.path.isfile(os.path.join(d, VARIANT_MANIFEST))
        has_npz = any(f.lower().endswith(".npz") for f in os.listdir(d))
        if has_manifest or has_npz:
            out.append(d)
    return out
