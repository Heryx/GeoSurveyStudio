# -*- coding: utf-8 -*-
"""Misure e raggruppamenti sulle annotazioni del radargramma (senza Qt ne' QGIS).

- metriche di sezione (lunghezza, larghezza, area) e contrasto d'ampiezza di una zona
  rispetto ai lati: serve per le zone di attenuazione (quanto "spengono" il segnale);
- raggruppamento delle annotazioni per codice anomalia: lo stesso codice su piu' profili
  e' un unico oggetto;
- campionamento in pianta di una polilinea/anello, per proiettare le annotazioni dei
  profili vicini sul radargramma mostrato.
"""

from __future__ import annotations

import math
import re

import numpy as np

# Campi numerici delle metriche, nell'ordine in cui compaiono nelle tabelle.
METRIC_FIELDS = ("length_m", "width_m", "area_m2", "amp_rms", "contrast_db")


def section_metrics(kind: str, xs, zs) -> dict:
    """Misure geometriche nella sezione (distanza lungo il profilo x profondita', in m).

    linea: lunghezza e larghezza; poligono: perimetro (``length_m``), larghezza e area.
    Il punto non ha misure.
    """
    xs = [float(v) for v in xs]
    zs = [float(v) for v in zs]
    n = len(xs)
    if kind == "point" or n < 2 or n != len(zs):
        return {}
    width = max(xs) - min(xs)
    closed = kind == "polygon" and n >= 3
    pairs = list(zip(range(n), range(1, n))) + ([(n - 1, 0)] if closed else [])
    length = sum(math.hypot(xs[j] - xs[i], zs[j] - zs[i]) for i, j in pairs)
    out = {"length_m": length, "width_m": width}
    if closed:
        a = np.asarray(xs)
        b = np.asarray(zs)
        out["area_m2"] = 0.5 * abs(float(np.dot(a, np.roll(b, -1)) - np.dot(b, np.roll(a, -1))))
    return out


def points_in_polygon(px, py, xs, zs) -> np.ndarray:
    """Maschera booleana dei punti (px, py) dentro il poligono (xs, zs): ray casting."""
    px = np.asarray(px, dtype=np.float64)
    py = np.asarray(py, dtype=np.float64)
    inside = np.zeros(px.shape, dtype=bool)
    n = len(xs)
    j = n - 1
    for i in range(n):
        xi, zi, xj, zj = xs[i], zs[i], xs[j], zs[j]
        if zi != zj:
            crosses = (zi > py) != (zj > py)
            x_at = (xj - xi) * (py - zi) / (zj - zi) + xi
            inside ^= crosses & (px < x_at)
        j = i
    return inside


def region_amplitude(data, dist, dz_m: float, xs, zs, margin: float = 1.0) -> dict:
    """RMS dell'ampiezza dentro un poligono e contrasto in dB rispetto ai due lati.

    ``data`` (campioni, tracce), ``dist`` distanza di ogni traccia (m), ``dz_m``
    profondita' di un campione.  Il riferimento sono le stesse righe nelle tracce a
    sinistra e a destra del poligono, per una fascia larga ``margin`` volte il
    poligono.  ``contrast_db`` = 20*log10(rms_dentro / rms_fuori): negativo = zona piu'
    "spenta" del suo intorno (attenuazione), positivo = piu' riflettente.
    Ritorna ``{}`` se il poligono non contiene nessun campione.
    """
    arr = np.asarray(data, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64)
    n_rows, n_traces = arr.shape
    if len(xs) < 3 or dz_m <= 0 or n_traces == 0:
        return {}
    i0 = int(np.searchsorted(dist, min(xs), side="left"))
    i1 = int(np.searchsorted(dist, max(xs), side="right")) - 1
    r0 = max(0, int(math.floor(min(zs) / dz_m)))
    r1 = min(n_rows - 1, int(math.ceil(max(zs) / dz_m)))
    if i1 < i0 or r1 < r0:
        return {}
    tr_idx = np.arange(i0, i1 + 1)
    row_idx = np.arange(r0, r1 + 1)
    gx, gz = np.meshgrid(dist[tr_idx], (row_idx + 0.5) * dz_m)
    mask = points_in_polygon(gx, gz, list(xs), list(zs))
    if not mask.any():
        return {}
    block = arr[r0:r1 + 1, i0:i1 + 1]
    rms_in = float(np.sqrt(np.mean(block[mask] ** 2)))
    out = {"amp_rms": rms_in}
    span = max(1, int(round((i1 - i0 + 1) * float(margin))))
    left = np.arange(max(0, i0 - span), i0)
    right = np.arange(i1 + 1, min(n_traces, i1 + 1 + span))
    side = np.concatenate([left, right])
    if side.size:
        rms_out = float(np.sqrt(np.mean(arr[r0:r1 + 1, side] ** 2)))
        if rms_out > 0 and rms_in > 0:
            out["contrast_db"] = 20.0 * math.log10(rms_in / rms_out)
    return out


def _natural_key(text: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(text))]


def group_anomalies(features) -> tuple:
    """Raggruppa le annotazioni per codice anomalia.

    ``features``: dict con almeno ``anomaly_id``, ``profile_stem``, ``channel``, ``id``
    (e, se ci sono, ``class_code``, ``depth_from``, ``depth_to``, ``contrast_db``,
    ``area_m2``, ``length_m``).  Ritorna ``(righe, senza_codice)``: una riga per codice
    (ordine naturale: M2 prima di M10) con numero di annotazioni, profili, classi,
    fascia di profondita', contrasto medio e la prima annotazione (per andarci), piu'
    quante annotazioni non hanno ancora un codice.
    """
    groups: dict = {}
    without = 0
    for f in features:
        code = str(f.get("anomaly_id") or "").strip()
        if not code:
            without += 1
            continue
        groups.setdefault(code, []).append(f)
    rows = []
    for code in sorted(groups, key=_natural_key):
        items = groups[code]
        profiles = {str(i.get("profile_stem") or "") for i in items}
        classes = sorted({str(i.get("class_code") or "") for i in items} - {""})
        d_from = [float(i["depth_from"]) for i in items if _num(i.get("depth_from"))]
        d_to = [float(i["depth_to"]) for i in items if _num(i.get("depth_to"))]
        contrast = [float(i["contrast_db"]) for i in items if _num(i.get("contrast_db"))]
        first = min(items, key=lambda i: (str(i.get("profile_stem") or ""), int(i.get("channel") or 0)))
        rows.append({
            "code": code,
            "n": len(items),
            "profiles": len(profiles),
            "classes": classes,
            "depth_from": min(d_from) if d_from else None,
            "depth_to": max(d_to) if d_to else None,
            "contrast_db": (sum(contrast) / len(contrast)) if contrast else None,
            "first": {"profile_stem": str(first.get("profile_stem") or ""),
                      "channel": int(first.get("channel") or 0), "id": str(first.get("id") or "")},
            "ids": [str(i.get("id") or "") for i in items],
        })
    return rows, without


ANOMALY_CSV_HEADER = ("codice", "classi", "annotazioni", "profili", "profondita_da_m", "profondita_a_m",
                      "contrasto_medio_db", "primo_profilo", "primo_canale")


def anomalies_to_csv(rows, path: str, class_labels: dict | None = None) -> int:
    """Scrive la tabella delle anomalie in CSV (separatore ';', UTF-8 con BOM: si apre
    bene in Excel italiano). ``class_labels`` traduce i codici di classe in nomi.
    Ritorna il numero di righe scritte."""
    import csv

    labels = class_labels or {}

    def fmt(v, nd):
        return "" if v is None else f"{float(v):.{nd}f}"

    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(ANOMALY_CSV_HEADER)
        for r in rows:
            writer.writerow([
                r["code"], ", ".join(labels.get(c, c) for c in r["classes"]), r["n"], r["profiles"],
                fmt(r["depth_from"], 3), fmt(r["depth_to"], 3), fmt(r["contrast_db"], 1),
                r["first"]["profile_stem"], r["first"]["channel"],
            ])
    return len(rows)


def _num(v) -> bool:
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def plan_samples(points, closed: bool, step: float) -> list:
    """Campiona una polilinea (o un anello) in pianta ogni al piu' ``step`` metri.

    ``points``: ``[(est, nord, valore), ...]``; il ``valore`` (es. profondita') si
    interpola linearmente.  Ritorna ``[(est, nord, valore), ...]`` con i vertici originali
    sempre inclusi: serve a trovare dove un'annotazione di un altro profilo passa vicino
    al profilo mostrato, anche se nessun suo vertice ci cade sopra.
    """
    pts = [(float(e), float(n), float(v)) for e, n, v in points]
    if len(pts) <= 1:
        return list(pts)
    step = max(float(step), 1e-6)
    segs = list(zip(pts, pts[1:])) + ([(pts[-1], pts[0])] if closed and len(pts) >= 3 else [])
    out = [pts[0]]
    for a, b in segs:
        dist = math.hypot(b[0] - a[0], b[1] - a[1])
        k = max(1, int(math.ceil(dist / step)))
        for s in range(1, k + 1):
            t = s / k
            out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t))
    if closed and len(pts) >= 3:
        out.pop()           # l'ultimo campione coincide col primo vertice
    return out
