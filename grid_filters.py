# -*- coding: utf-8 -*-
"""Filtri nel dominio griglia / time-slice — Fase 5a del blueprint processing.

Operano su un raster 2D `(n_y, n_x)` float (NaN = nodata), non sui radargrammi.
Puro numpy, nessuna dipendenza da Qt/QGIS/scipy. Sono un secondo motore di
filtraggio, indipendente da `gpr_processing` (che lavora sulle tracce).

Tecniche standard di letteratura GPR/geofisica (nessun codice di terze parti):
- low-pass / hi-pass / Laplacian per ripulire il rumore di gridding;
- background di linea (verticale/orizzontale) con soglie percentili per non
  "spalmare" i target forti;
- 2D-FFT per togliere frequenze spaziali direzionali (es. solchi di aratura);
- blanking;
- correzioni mosaico (0-mean per griglia / per linea, histogram matching).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "lowpass",
    "hipass",
    "laplacian",
    "gaussian_smooth",
    "line_background",
    "fft2d_filter",
    "fft2d_directional_notch",
    "blank",
    "mosaic_zero_mean",
    "mosaic_histogram_match",
]


# ---------------------------------------------------------------------------
# helper NaN-aware
# ---------------------------------------------------------------------------

def _as2d(grid) -> np.ndarray:
    a = np.asarray(grid, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"grid deve essere 2D, ricevuto shape {a.shape}")
    return a


def _box_mean_nan(a: np.ndarray, half_y: int, half_x: int) -> np.ndarray:
    """Media mobile box (2*half+1) NaN-aware via cumsum su valori e maschera."""
    m = np.isfinite(a)
    v = np.where(m, a, 0.0)
    w = m.astype(np.float64)

    def _run_sum(x, half, axis):
        if half <= 0:
            return x
        n = x.shape[axis]
        he = min(half, max(0, n - 1))
        if he <= 0:
            return x
        k = 2 * he + 1
        pad = [(0, 0), (0, 0)]
        pad[axis] = (he, he)
        xp = np.pad(x, pad, mode="constant", constant_values=0.0)
        cs = np.cumsum(xp, axis=axis)
        zshape = list(xp.shape)
        zshape[axis] = 1
        cs = np.concatenate([np.zeros(zshape), cs], axis=axis)
        hi = [slice(None), slice(None)]
        lo = [slice(None), slice(None)]
        hi[axis] = slice(k, k + n)
        lo[axis] = slice(0, n)
        return cs[tuple(hi)] - cs[tuple(lo)]

    vs = _run_sum(_run_sum(v, half_y, 0), half_x, 1)
    ws = _run_sum(_run_sum(w, half_y, 0), half_x, 1)
    out = np.divide(vs, ws, out=np.full_like(a, np.nan), where=ws > 0.0)
    out[~m] = np.nan  # non "inventare" celle originariamente vuote
    return out


# ---------------------------------------------------------------------------
# filtri passa-basso / passa-alto / bordi
# ---------------------------------------------------------------------------

def lowpass(grid, size: int = 3) -> np.ndarray:
    """Boxcar low-pass NxN (NaN-aware). `size` dispari >= 3 (3, 5, 7, 9...)."""
    a = _as2d(grid)
    s = max(3, int(size) | 1)
    half = s // 2
    return _box_mean_nan(a, half, half)


def hipass(grid, size: int = 3) -> np.ndarray:
    """Hi-pass = grid - low-pass(grid, size)."""
    a = _as2d(grid)
    return a - lowpass(a, size)


def laplacian(grid, gain: float = 1.0) -> np.ndarray:
    """Enfasi bordi con kernel Laplaciano 3x3 (NaN-aware, riempimento locale)."""
    a = _as2d(grid)
    filled = _box_mean_nan(a, 1, 1)
    base = np.where(np.isfinite(a), a, filled)
    base = np.nan_to_num(base, nan=0.0)
    k = np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])
    ny, nx = base.shape
    pad = np.pad(base, 1, mode="edge")
    out = np.zeros_like(base)
    for dy in range(3):
        for dx in range(3):
            out += k[dy, dx] * pad[dy:dy + ny, dx:dx + nx]
    out *= float(gain)
    out[~np.isfinite(a)] = np.nan
    return out


def gaussian_smooth(grid, sigma: float = 1.0) -> np.ndarray:
    """Lisciatura gaussiana separabile NaN-aware (normalizzata sulla maschera)."""
    a = _as2d(grid)
    s = float(sigma)
    if s <= 0.0:
        return a.copy()
    rad = max(1, int(round(3.0 * s)))
    x = np.arange(-rad, rad + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / s) ** 2)
    k /= k.sum()
    m = np.isfinite(a)
    v = np.where(m, a, 0.0)
    w = m.astype(np.float64)

    def _conv1(arr, axis):
        return np.apply_along_axis(
            lambda r: np.convolve(r, k, mode="same"), axis, arr
        )

    vs = _conv1(_conv1(v, 0), 1)
    ws = _conv1(_conv1(w, 0), 1)
    out = np.divide(vs, ws, out=np.full_like(a, np.nan), where=ws > 1e-9)
    out[~m] = np.nan
    return out


# ---------------------------------------------------------------------------
# background di linea (verticale / orizzontale) sulla griglia
# ---------------------------------------------------------------------------

def line_background(
    grid,
    axis: str = "x",
    length: int = 0,
    clip_lo_pct: float = 0.0,
    clip_hi_pct: float = 100.0,
    gain: float = 1.0,
) -> np.ndarray:
    """Sottrae la media mobile lungo ogni linea (per togliere striature/mosaico).

    ``axis='x'`` : linee parallele all'asse X (righe)  → filtra bande verticali.
    ``axis='y'`` : linee parallele all'asse Y (colonne) → filtra bande orizzontali.

    ``length`` = lunghezza del filtro in celle (0 = intera linea → sottrae un
    singolo valor medio per linea). ``clip_lo_pct`` / ``clip_hi_pct`` escludono
    dal calcolo della media i valori fuori da quei percentili **della linea**,
    così i target forti non vengono "spalmati". Il risultato è moltiplicato per
    ``gain``.
    """
    a = _as2d(grid)
    ax = 0 if str(axis).lower().startswith("y") else 1  # linea = lungo questo asse
    n = a.shape[ax]
    L = int(length) if int(length) > 0 else n
    L = max(1, min(L, n))
    half = L // 2

    work = np.moveaxis(a, ax, 1)  # (n_lines, n_along)
    out = np.empty_like(work)
    lo = float(clip_lo_pct)
    hi = float(clip_hi_pct)
    for i in range(work.shape[0]):
        line = work[i, :]
        finite = np.isfinite(line)
        keep = finite.copy()
        if finite.any() and (lo > 0.0 or hi < 100.0):
            vlo = np.nanpercentile(line, lo) if lo > 0.0 else -np.inf
            vhi = np.nanpercentile(line, hi) if hi < 100.0 else np.inf
            keep &= (line >= vlo) & (line <= vhi)
        src = np.where(keep, line, np.nan)
        if L >= n:
            ref = np.nanmean(src) if np.isfinite(src).any() else 0.0
            ref_arr = np.full_like(line, ref, dtype=np.float64)
        else:
            v = np.where(np.isfinite(src), src, 0.0)
            w = np.isfinite(src).astype(np.float64)
            vp = np.pad(v, half, mode="reflect")
            wp = np.pad(w, half, mode="reflect")
            cs_v = np.concatenate([[0.0], np.cumsum(vp)])
            cs_w = np.concatenate([[0.0], np.cumsum(wp)])
            k = 2 * half + 1
            idx = np.arange(n)
            sv = cs_v[idx + k] - cs_v[idx]
            sw = cs_w[idx + k] - cs_w[idx]
            ref_arr = np.divide(sv, sw, out=np.zeros(n), where=sw > 0.0)
        res = (line - ref_arr) * float(gain)
        res[~finite] = np.nan
        out[i, :] = res
    return np.moveaxis(out, 1, ax)


# ---------------------------------------------------------------------------
# 2D-FFT
# ---------------------------------------------------------------------------

def fft2d_filter(grid, mask) -> np.ndarray:
    """Applica una maschera nello spazio delle frequenze spaziali e antitrasforma.

    ``mask`` : array reale (n_y, n_x) in [0,1] **già in ordine fftshift**
    (0 = blanca, 1 = passa). I NaN in ingresso sono riempiti con la media
    prima della FFT e ripristinati in uscita.
    """
    a = _as2d(grid)
    m = np.isfinite(a)
    if not m.any():
        return a.copy()
    fill = float(np.nanmean(a))
    base = np.where(m, a, fill)
    msk = np.asarray(mask, dtype=np.float64)
    if msk.shape != a.shape:
        raise ValueError(f"mask {msk.shape} != grid {a.shape}")
    F = np.fft.fftshift(np.fft.fft2(base))
    out = np.real(np.fft.ifft2(np.fft.ifftshift(F * msk)))
    out[~m] = np.nan
    return out


def fft2d_directional_notch(
    grid,
    angle_deg: float,
    halfwidth_deg: float = 10.0,
    mode: str = "reject",
) -> np.ndarray:
    """Rimuove (``reject``) o isola (``pass``) le frequenze spaziali orientate
    lungo ``angle_deg`` ± ``halfwidth_deg``.

    Utile per togliere strutture lineari periodiche a direzione nota (solchi di
    aratura, tracce di mezzi agricoli): la loro firma in FFT è un fascio di
    componenti **normale** alla direzione osservata sul piano, quindi passare
    ``angle_deg`` = direzione dei solchi + 90.
    """
    a = _as2d(grid)
    ny, nx = a.shape
    ky = np.fft.fftshift(np.fft.fftfreq(ny))[:, None]
    kx = np.fft.fftshift(np.fft.fftfreq(nx))[None, :]
    theta = np.degrees(np.arctan2(ky, kx))          # -180..180
    ang = float(angle_deg) % 180.0
    d = np.abs(((np.abs(theta) % 180.0) - ang + 90.0) % 180.0 - 90.0)
    inband = d <= float(halfwidth_deg)
    inband[(ky == 0) & (kx == 0)] = False           # tieni sempre la DC
    keep = (~inband) if str(mode).lower() == "reject" else inband
    keep[(ky == 0) & (kx == 0)] = True
    return fft2d_filter(a, keep.astype(np.float64))


def blank(grid, mask) -> np.ndarray:
    """Mette a NaN le celle dove ``mask`` è vera (blanking manuale/poligono)."""
    a = _as2d(grid).copy()
    mk = np.asarray(mask, dtype=bool)
    if mk.shape != a.shape:
        raise ValueError(f"mask {mk.shape} != grid {a.shape}")
    a[mk] = np.nan
    return a


# ---------------------------------------------------------------------------
# correzioni mosaico
# ---------------------------------------------------------------------------

def mosaic_zero_mean(grids, scope: str = "grid", axis: str = "x") -> list[np.ndarray]:
    """Rimuove il bias medio da un insieme di griglie affiancate.

    ``scope='grid'`` : sottrae da ogni griglia la propria media globale.
    ``scope='line'`` : sottrae da ogni linea (riga se ``axis='x'``, colonna se
                       ``axis='y'``) la propria media — toglie le fasce dovute a
                       linee acquisite in tempi/condizioni diverse.
    """
    out = []
    ax = 0 if str(axis).lower().startswith("y") else 1
    for g in grids:
        a = _as2d(g).copy()
        if str(scope).lower().startswith("line"):
            mean_along = np.nanmean(a, axis=ax, keepdims=True)
            mean_along = np.where(np.isfinite(mean_along), mean_along, 0.0)
            a = a - mean_along
        else:
            mu = float(np.nanmean(a)) if np.isfinite(a).any() else 0.0
            a = a - mu
        out.append(a)
    return out


def mosaic_histogram_match(grid, reference) -> np.ndarray:
    """Rimappa i valori di ``grid`` perché la sua distribuzione combaci con
    quella di ``reference`` (matching di istogramma per giunzioni invisibili)."""
    a = _as2d(grid)
    r = _as2d(reference)
    src = a[np.isfinite(a)]
    ref = r[np.isfinite(r)]
    if src.size < 2 or ref.size < 2:
        return a.copy()
    s_sorted = np.sort(src)
    s_q = np.linspace(0.0, 1.0, s_sorted.size)
    r_sorted = np.sort(ref)
    r_q = np.linspace(0.0, 1.0, r_sorted.size)
    out = np.full_like(a, np.nan)
    finite = np.isfinite(a)
    q = np.interp(a[finite], s_sorted, s_q)
    out[finite] = np.interp(q, r_q, r_sorted)
    return out
