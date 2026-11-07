# -*- coding: utf-8 -*-
"""Diagnostica di un file .DT (IDS raw) quando il radargramma esce come rumore.

Uso:
    python scripts/diagnose_dt.py  percorso/LA010001.DT

Non modifica nulla: stampa l'header V0, la dimensione file, e prova diverse
interpretazioni del blocco dati (n_samples / dtype / header per-traccia)
assegnando a ognuna un punteggio di "coerenza orizzontale": un radargramma
vero ha una riga forte e piatta vicino a t=0 (onda diretta). Incolla
l'output completo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HEADER_BLOCK = 1028
V0_FIELD_W = 16
V0_N_FIELDS = 10


def parse_v0(raw: bytes) -> dict:
    tag = raw[:2]
    blob = raw[8:8 + V0_FIELD_W * V0_N_FIELDS]
    vals = []
    for i in range(V0_N_FIELDS):
        ch = blob[i * V0_FIELD_W:(i + 1) * V0_FIELD_W]
        try:
            vals.append(float(ch.decode("ascii", "replace").strip()))
        except ValueError:
            vals.append(float("nan"))
    return {"tag": tag, "fields": vals}


def coherence_score(arr: np.ndarray, top_rows: int = 60) -> tuple[float, int]:
    """arr: (n_samples, n_traces). Ritorna (score, riga) del piu' forte
    evento orizzontale nelle prime `top_rows` righe. score alto = onda diretta
    piatta e coerente fra le tracce."""
    if arr.ndim != 2 or arr.shape[1] < 8:
        return 0.0, -1
    a = arr[:top_rows].astype(np.float64)
    # rimuovi trend per-traccia (dewow grezzo) per non premiare offset DC
    a = a - a.mean(axis=0, keepdims=True)
    mean_abs = np.abs(a.mean(axis=1))          # ampiezza coerente per riga
    std = a.std(axis=1) + 1e-9                  # rumore per riga
    ratio = mean_abs / std
    r = int(np.argmax(ratio))
    return float(ratio[r]), r


def try_layout(raw: bytes, data_start: int, n_samples: int, dtype: str,
               per_trace_hdr: int = 0):
    itemsize = np.dtype(dtype).itemsize
    trace_bytes = n_samples * itemsize + per_trace_hdr
    if trace_bytes <= 0:
        return None
    avail = len(raw) - data_start
    n_traces = avail // trace_bytes
    if n_traces < 8:
        return None
    if per_trace_hdr == 0:
        flat = np.frombuffer(raw, dtype=dtype, count=n_samples * n_traces,
                             offset=data_start).astype(np.float64)
        arr = flat.reshape(n_traces, n_samples).T
    else:
        rows = []
        off = data_start
        for _ in range(n_traces):
            seg = np.frombuffer(raw, dtype=dtype, count=n_samples,
                                offset=off + per_trace_hdr).astype(np.float64)
            rows.append(seg)
            off += trace_bytes
        arr = np.asarray(rows).T
    score, row = coherence_score(arr)
    return {
        "n_samples": n_samples, "dtype": dtype, "per_trace_hdr": per_trace_hdr,
        "n_traces": int(n_traces), "leftover_bytes": int(avail - n_traces * trace_bytes),
        "score": round(score, 3), "direct_wave_row": row,
        "amp_min": float(np.nanmin(arr)), "amp_max": float(np.nanmax(arr)),
    }


def main(path_str: str) -> None:
    path = Path(path_str)
    raw = path.read_bytes()
    print(f"file            : {path}")
    print(f"file size       : {len(raw):,} byte")

    v0 = parse_v0(raw)
    print(f"V0 tag          : {v0['tag']!r}")
    print(f"V0 fields       : {v0['fields']}")
    hdr_ns = v0["fields"][0]
    hdr_ns = int(round(hdr_ns)) if hdr_ns == hdr_ns else 0
    print(f"header n_samples : {hdr_ns}")
    print(f"header time win  : {v0['fields'][2]} s")
    print(f"first 64 byte    : {raw[:64]!r}")

    # stima header: multipli di 1028
    for nb in (8, 9, 10, 11, 12):
        off = nb * HEADER_BLOCK
        print(f"  se header = {nb:2d} blocchi -> data_start={off:6d}  "
              f"(size-off)={len(raw)-off:,}")

    print("\n--- prova interpretazioni del blocco dati "
          "(score alto = onda diretta coerente) ---")
    candidates = []
    ns_list = sorted({v for v in (hdr_ns, 128, 200, 256, 400, 512, 1024, 2048) if v > 0})
    for ns in ns_list:
        for dtype in ("<i2", ">i2", "<i4", "<f4"):
            for pth in (0, 2, 4, 8, 16):
                # data_start: cerca il migliore fra 8..12 blocchi header per (size-off) divisibile
                for nb in (8, 9, 10, 11, 12):
                    ds = nb * HEADER_BLOCK
                    res = try_layout(raw, ds, ns, dtype, pth)
                    if res is None:
                        continue
                    res["data_start"] = ds
                    candidates.append(res)
    candidates.sort(key=lambda r: r["score"], reverse=True)
    print(f"{'score':>7} {'ns':>5} {'dtype':>5} {'pth':>4} {'ds':>6} "
          f"{'ntr':>5} {'left':>6} {'dwRow':>6}   amp_range")
    for r in candidates[:15]:
        print(f"{r['score']:7.3f} {r['n_samples']:5d} {r['dtype']:>5} "
              f"{r['per_trace_hdr']:4d} {r['data_start']:6d} {r['n_traces']:5d} "
              f"{r['leftover_bytes']:6d} {r['direct_wave_row']:6d}   "
              f"[{r['amp_min']:.1f}, {r['amp_max']:.1f}]")
    print("\nIl candidato giusto ha di solito: score nettamente piu' alto,"
          " leftover=0, dwRow piccolo (5-20), amp_range plausibile per int16.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("uso: python scripts/diagnose_dt.py  percorso/al/file.DT")
        raise SystemExit(2)
    main(sys.argv[1])
