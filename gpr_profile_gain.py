# -*- coding: utf-8 -*-
"""Range gain (TVG) per singolo profilo (no Qt).

Ogni profilo ha la propria curva TVG: un profilo piu' attenuato ha bisogno di
piu' guadagno di uno meno attenuato. Le curve vivono in
``params["pre_agc_by_profile"]`` = ``{chiave_profilo: [[x, guadagno], ...]}``;
i profili senza voce usano la curva "di base" (``pre_agc_breakpoints``).

Chi applica la pipeline a un profilo specifico (salvataggio varianti, time-slice
dal grezzo, vista 3D) passa i parametri da :func:`resolve_profile_params`, che
sostituisce la curva di base con quella del profilo e toglie la mappa.

Il preset TVG e' adattivo: parte dalla forma della curva del preset e ne alza o
abbassa il livello in profondita' in base a quanto il profilo si attenua
rispetto alla mediana degli altri (:func:`adaptive_gain_curves`).
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

import numpy as np

BY_PROFILE_KEY = "pre_agc_by_profile"
RANGE_GAIN_MAX = 1.0e6
MAX_POINTS = 16  # limite dell'editor curva

# Finestre (frazione dei campioni) con cui si misura l'attenuazione: una zona
# "utile" subito sotto l'onda diretta e una profonda. Gli estremi restano fuori
# (burst d'antenna in alto, fondo d'acquisizione in basso).
_SHALLOW = (0.10, 0.30)
_DEEP = (0.70, 0.95)
_X_SHALLOW = 0.5 * sum(_SHALLOW)
_X_DEEP = 0.5 * sum(_DEEP)
# L'eccesso di attenuazione rispetto alla mediana non puo' cambiare il guadagno
# in profondita' piu' di questo fattore (in una direzione o nell'altra).
MAX_ADAPT_FACTOR = 6.0
_MAX_TRACES = 400


def profile_gain_key(profile) -> str:
    """Chiave stabile del profilo: nome del file senza estensione."""
    path = profile if isinstance(profile, str) else getattr(profile, "path", "")
    return os.path.splitext(os.path.basename(str(path or "")))[0]


def _clean_curve(points) -> list[list[float]]:
    out = []
    for p in points or []:
        try:
            x, y = float(p[0]), float(p[1])
        except Exception:
            continue
        if np.isfinite(x) and np.isfinite(y):
            out.append([min(max(x, 0.0), 1.0), min(max(y, 0.0), RANGE_GAIN_MAX)])
    return out


def _band_rms(power: np.ndarray, band: tuple[float, float]) -> float:
    n = power.shape[0]
    a = int(round(band[0] * n))
    b = max(a + 1, int(round(band[1] * n)))
    return float(np.sqrt(np.mean(power[a:b])))


def attenuation_index(profile) -> Optional[float]:
    """ln(inviluppo profondo / inviluppo superficiale) del profilo.

    Piu' e' negativo, piu' il profilo e' attenuato. Si usa il dato grezzo con la
    mediana di ogni campione tolta (il rumore orizzontale costante non conta) e
    la mediana tra i canali. None se il profilo non e' misurabile."""
    vals = []
    for ci in range(int(getattr(profile, "n_channels", 0) or 0)):
        try:
            data = np.asarray(profile.channel(ci).data, dtype=np.float32)
        except Exception:
            continue
        if data.ndim != 2 or data.shape[0] < 20 or data.shape[1] < 2:
            continue
        step = max(1, data.shape[1] // _MAX_TRACES)
        x = np.nan_to_num(data[:, ::step].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        x = x - np.median(x, axis=1, keepdims=True)
        power = np.mean(x * x, axis=1)
        shallow = _band_rms(power, _SHALLOW)
        deep = _band_rms(power, _DEEP)
        if shallow <= 0.0 or not np.isfinite(shallow) or not np.isfinite(deep):
            continue
        vals.append(float(np.log((deep + 1e-12) / shallow)))
    return float(np.median(vals)) if vals else None


def adapt_curve(base_points: Iterable, excess_attenuation: float) -> list[list[float]]:
    """Curva ``base_points`` con in piu' ``exp(eccesso * rampa(x))``.

    ``excess_attenuation`` (in ln) e' quanto il profilo si attenua piu' della
    mediana; la rampa va da 0 (zona superficiale di misura) a 1 (zona profonda)
    e poi resta costante."""
    base = np.asarray(_clean_curve(base_points), dtype=np.float64)
    if base.shape[0] < 2:
        return [[0.0, 1.0], [1.0, 1.0]]
    base = base[np.argsort(base[:, 0], kind="mergesort")]
    cap = float(np.log(MAX_ADAPT_FACTOR))
    e = float(np.clip(excess_attenuation, -cap, cap))
    xs = np.unique(np.concatenate([base[:, 0], [0.0, _X_SHALLOW, _X_DEEP, 1.0]]))
    if xs.size > MAX_POINTS:
        # Tengo gli estremi e i nodi della rampa; il resto, a passi regolari.
        must = np.array([0.0, _X_SHALLOW, _X_DEEP, 1.0])
        rest = np.setdiff1d(xs, must)
        take = MAX_POINTS - must.size
        idx = np.linspace(0, rest.size - 1, take).round().astype(int)
        xs = np.unique(np.concatenate([must, rest[idx]]))
    y0 = np.interp(xs, base[:, 0], base[:, 1])
    ramp = np.clip((xs - _X_SHALLOW) / (_X_DEEP - _X_SHALLOW), 0.0, 1.0)
    y = np.clip(y0 * np.exp(e * ramp), 0.0, RANGE_GAIN_MAX)
    return [[float(a), float(b)] for a, b in zip(xs, y)]


def adaptive_gain_curves(profiles, base_points, progress=None) -> tuple[dict, dict]:
    """Una curva per profilo a partire da ``base_points`` (curva del preset).

    Ritorna ``(curve_per_chiave, info)``; ``info`` ha ``median_index`` e, per
    chiave, l'indice d'attenuazione e il fattore applicato in profondita'.
    ``progress(i, n)`` e' chiamata dopo ogni profilo. Profili non misurabili e
    insiemi di un solo profilo ricevono la curva del preset invariata."""
    profiles = list(profiles or [])
    idx = {}
    for i, prof in enumerate(profiles):
        try:
            idx[profile_gain_key(prof)] = attenuation_index(prof)
        except Exception:
            idx[profile_gain_key(prof)] = None
        if progress is not None:
            progress(i + 1, len(profiles))
    measured = [v for v in idx.values() if v is not None]
    median = float(np.median(measured)) if measured else 0.0
    curves, factors = {}, {}
    cap = float(np.log(MAX_ADAPT_FACTOR))
    for key, v in idx.items():
        excess = 0.0 if v is None else float(median - v)
        curves[key] = adapt_curve(base_points, excess)
        factors[key] = float(np.exp(np.clip(excess, -cap, cap)))
    return curves, {"median_index": median, "index": idx, "factor": factors}


# ---------------------------------------------------------------------------
# Guadagno automatico regolabile
# ---------------------------------------------------------------------------

AUTO_DEFAULTS = {
    "start": None,        # frazione della finestra da cui parte il guadagno (None = onda diretta rilevata)
    "strength": 0.8,      # quota (0..1.5) dell'attenuazione misurata che si compensa
    "max_db": 60.0,       # guadagno massimo (dB)
    "n_points": 12,       # punti della curva (max 16, limite dell'editor)
    "monotonic": True,    # il guadagno non cala mai con la profondita'
}


def _smooth(v: np.ndarray, win: int) -> np.ndarray:
    win = int(max(1, win))
    if win <= 1 or v.size < 3:
        return v.astype(np.float64, copy=True)
    k = np.ones(win) / win
    pad = win // 2
    padded = np.pad(v.astype(np.float64), (pad, win - 1 - pad), mode="edge")
    return np.convolve(padded, k, mode="valid")


def envelope_db(data, smooth_frac: float = 0.03) -> np.ndarray:
    """Inviluppo medio per campione (dB) di una matrice campioni x tracce.

    Si toglie la mediana di ogni campione (rumore orizzontale costante), si fa
    l'RMS sulle tracce e si leviga lungo la profondita'."""
    x = np.asarray(data, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 8 or x.shape[1] < 1:
        return np.zeros(0)
    step = max(1, x.shape[1] // _MAX_TRACES)
    x = np.nan_to_num(x[:, ::step], nan=0.0, posinf=0.0, neginf=0.0)
    x = x - np.median(x, axis=1, keepdims=True)
    rms = np.sqrt(np.mean(x * x, axis=1))
    rms = _smooth(rms, int(round(smooth_frac * rms.size)))
    return 20.0 * np.log10(rms + 1e-12)


def profile_envelope_db(profile) -> np.ndarray:
    """Inviluppo (dB) del profilo: mediana tra i canali, in dB."""
    envs = []
    for ci in range(int(getattr(profile, "n_channels", 0) or 0)):
        try:
            e = envelope_db(profile.channel(ci).data)
        except Exception:
            continue
        if e.size:
            envs.append(e)
    if not envs:
        return np.zeros(0)
    n = min(e.size for e in envs)
    return np.median(np.stack([e[:n] for e in envs]), axis=0)


def ground_wave_end(env_db: np.ndarray, margin: float = 0.02) -> float:
    """Frazione della finestra in cui finisce l'onda diretta (0..0.5).

    E' il punto dopo il massimo, nella prima meta', in cui l'inviluppo resta
    10 dB sotto il picco per qualche campione di fila."""
    env = np.asarray(env_db, dtype=np.float64)
    n = env.size
    if n < 16:
        return 0.0
    half = max(2, n // 2)
    peak = int(np.argmax(env[:half]))
    below = env[peak:] < env[peak] - 10.0
    run = max(3, n // 100)
    end = peak
    count = 0
    for i, b in enumerate(below):
        count = count + 1 if b else 0
        if count >= run:
            end = peak + i - run + 1
            break
    return float(np.clip(end / n + margin, 0.0, 0.5))


def estimate_decay(env_db, start: float) -> tuple[float, float]:
    """(pendenza, fine) dell'attenuazione dell'inviluppo dopo ``start``.

    ``pendenza`` e' in dB per unita' di profondita' (frazione della finestra,
    negativa se il segnale cala); ``fine`` e' la profondita' (frazione) oltre la
    quale l'inviluppo e' sul fondo di rumore. La retta si adatta tra una zona
    ``start + 15%`` (si salta la coda dell'onda diretta) e ``fine``."""
    env = np.asarray(env_db, dtype=np.float64)
    n = env.size
    x = np.linspace(0.0, 1.0, n)
    sm = _smooth(env, max(3, n // 20))
    floor = float(np.median(sm[int(0.85 * n):]))
    above = np.where((x >= start) & (sm > floor + 6.0))[0]
    end = float(x[above[-1]]) if above.size else 1.0
    end = max(end, min(1.0, start + 0.2))
    a = start + 0.15 * (end - start)
    sel = (x >= a) & (x <= end)
    if sel.sum() < 8:
        sel = x >= start
    if sel.sum() < 4:
        return 0.0, 1.0
    slope = float(np.polyfit(x[sel], sm[sel], 1)[0])
    return slope, end


def auto_gain_points(env_db, start=None, strength=0.8, max_db=60.0,
                     n_points=12, **_ignored) -> list[list[float]]:
    """Curva di guadagno (x, guadagno lineare) dall'inviluppo del profilo.

    Sopra ``start`` il guadagno e' 1 (0 dB): l'onda diretta non si tocca. Sotto,
    il guadagno cresce in dB in modo lineare (cioe' esponenziale sul guadagno)
    con la pendenza dell'attenuazione misurata per ``strength``, fino alla
    profondita' in cui il segnale arriva sul rumore: li' si ferma, per non
    amplificare solo rumore. ``max_db`` e' il tetto. I punti sono equidistanti;
    la curva va usata con interpolazione "log" (dB)."""
    env = np.asarray(env_db, dtype=np.float64)
    if env.size < 16:
        return [[0.0, 1.0], [1.0, 1.0]]
    start = ground_wave_end(env) if start is None else float(np.clip(start, 0.0, 0.9))
    slope, end = estimate_decay(env, start)
    rate = max(0.0, -slope) * float(strength)            # dB per unita' di profondita'
    n_points = int(np.clip(n_points, 5, MAX_POINTS))
    n_ramp = max(2, n_points - (2 if start > 0.0 else 1) - (1 if end < 0.999 else 0))
    xs_ramp = np.linspace(start, end, n_ramp + 1)[1:]
    xs = ([0.0] if start > 0.0 else []) + [start] + list(xs_ramp)
    if end < 0.999:
        xs.append(1.0)
    xs = np.asarray(xs)
    db = np.clip(rate * (np.minimum(xs, end) - start), 0.0, float(max_db))
    db[xs <= start] = 0.0
    return [[float(a), float(min(10.0 ** (b / 20.0), RANGE_GAIN_MAX))] for a, b in zip(xs, db)]


def auto_gain_for_profiles(profiles, settings: dict | None = None, progress=None,
                          envelope_fn=None) -> dict:
    """Curva automatica per ogni profilo (chiave = :func:`profile_gain_key`),
    con le stesse impostazioni; l'onda diretta si rileva profilo per profilo
    se ``start`` e' None. ``envelope_fn(profilo)`` sostituisce il calcolo
    dell'inviluppo (es. dato dopo i passi che precedono il guadagno)."""
    cfg = {**AUTO_DEFAULTS, **(settings or {})}
    profiles = list(profiles or [])
    out = {}
    for i, prof in enumerate(profiles):
        env = envelope_fn(prof) if envelope_fn is not None else profile_envelope_db(prof)
        out[profile_gain_key(prof)] = auto_gain_points(
            env, start=cfg["start"], strength=cfg["strength"], max_db=cfg["max_db"],
            n_points=cfg["n_points"], monotonic=cfg["monotonic"],
        )
        if progress is not None:
            progress(i + 1, len(profiles))
    return out


# ---------------------------------------------------------------------------
# AGC per profilo
# ---------------------------------------------------------------------------

AGC_BY_PROFILE_KEY = "agc_by_profile"
AGC_KEYS = ("agc_win", "agc_max_gain", "agc_sample_start", "agc_lateral")
AGC_AUTO_DEFAULTS = {
    "periods": 4.0,        # finestra = tanti periodi del segnale dominante
    "limit_noise": True,   # il guadagno massimo non supera il salto segnale/rumore
}


def dominant_period_samples(data, start: int = 0) -> float:
    """Periodo (in campioni) del segnale dominante sotto ``start``; 0 se non misurabile."""
    x = np.asarray(data, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] - start < 32 or x.shape[1] < 1:
        return 0.0
    step = max(1, x.shape[1] // 100)
    sub = x[start:, ::step]
    sub = sub - sub.mean(axis=0, keepdims=True)
    spec = np.abs(np.fft.rfft(sub * np.hanning(sub.shape[0])[:, None], axis=0)).mean(axis=1)
    if spec.size < 4:
        return 0.0
    spec[:2] = 0.0                      # niente componenti lentissime
    idx = int(np.argmax(spec))
    if idx < 2:
        return 0.0
    return float(sub.shape[0]) / float(idx)


def auto_agc_params(data, periods: float = 4.0, limit_noise: bool = True) -> dict:
    """Parametri AGC suggeriti dal dato che arriva al passo AGC.

    * Da campione: fine dell'onda diretta.
    * Finestra: ``periods`` periodi del segnale dominante (pari, 8..512).
    * Max gain: salto tra livello tipico e fondo di rumore dell'inviluppo (0 se
      ``limit_noise`` e' falso): impedisce di portare il rumore al livello del segnale."""
    x = np.asarray(data, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 16:
        return {"agc_win": 128, "agc_max_gain": 0.0, "agc_sample_start": 0}
    n_s = x.shape[0]
    env = envelope_db(x)
    s0 = int(round(ground_wave_end(env) * n_s))
    period = dominant_period_samples(x, s0)
    win = int(round(float(periods) * period)) if period > 0 else max(8, n_s // 8)
    win = int(np.clip(win, 8, min(512, max(8, n_s // 2))))
    win += win % 2
    max_gain = 0.0
    if limit_noise:
        sm = _smooth(env, max(3, env.size // 20))[s0:]
        if sm.size >= 8:
            dyn = float(np.median(sm) - np.percentile(sm, 5.0))
            max_gain = float(np.clip(10.0 ** (dyn / 20.0), 2.0, 1000.0))
    return {"agc_win": win, "agc_max_gain": round(max_gain, 1), "agc_sample_start": s0}


def resolve_profile_params(params: Optional[dict], key: str) -> dict:
    """Parametri della pipeline per il profilo ``key`` (nuovo dict).

    Se esiste una curva TVG o un AGC propri li mette al posto di quelli comuni
    (gain di superficie/fondo inclusi); in ogni caso rimuove le mappe per-profilo."""
    out = dict(params or {})
    by_profile = out.pop(BY_PROFILE_KEY, None)
    if isinstance(by_profile, dict) and key in by_profile:
        pts = _clean_curve(by_profile[key])
        if len(pts) >= 2:
            pts.sort(key=lambda p: p[0])
            out["pre_agc_breakpoints"] = pts
            out["pre_agc_surface_gain"] = pts[0][1]
            out["pre_agc_deep_gain"] = pts[-1][1]
    agc_map = out.pop(AGC_BY_PROFILE_KEY, None)
    if isinstance(agc_map, dict) and isinstance(agc_map.get(key), dict):
        for k in AGC_KEYS:
            if k in agc_map[key]:
                out[k] = agc_map[key][k]
    return out
