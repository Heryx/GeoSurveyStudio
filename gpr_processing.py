"""Pipeline di processing per radargram GPR — implementazione puro numpy.

Tutte le funzioni operano su matrice (n_samples, n_slices) float32.
Nessuna dipendenza da scipy: Hilbert e bandpass sono implementati
tramite FFT numpy.
"""

from __future__ import annotations

import os
import tempfile
import traceback
from datetime import datetime

import numpy as np

try:
    from qgis.core import Qgis, QgsMessageLog
except Exception:  # pragma: no cover - QGIS runtime not available in headless tests
    Qgis = None
    QgsMessageLog = None


# File di log persistente: utile quando il pannello messaggi di QGIS non e'
# consultabile o il log si perde tra una sessione e l'altra. Ogni riga e'
# timestampata cosi' da poter ricostruire la sequenza auto-tune -> pipeline.
LOG_FILE_PATH = os.path.join(tempfile.gettempdir(), "rasterlinker_gpr_debug.log")


def _write_log_file(msg: str) -> None:
    try:
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass


def _log_gpr(message: str, level=None) -> None:
    """Log to QGIS panel + persistent file, fallback to stdout otherwise."""
    msg = str(message)
    _write_log_file(msg)
    if QgsMessageLog is not None and Qgis is not None:
        try:
            lvl = Qgis.Info if level is None else level
            QgsMessageLog.logMessage(msg, "RasterLinker", lvl)
            return
        except Exception:
            pass
    print(msg)


def log_exception(context: str, exc: Exception) -> None:
    """Log full traceback of an exception both to file and QGIS panel.

    Safe to call outside an active except block (e.g. for manually
    constructed errors) — traceback is included only if one is available.
    """
    tb = traceback.format_exc()
    suffix = f"\n{tb}" if tb and tb.strip() != "NoneType: None" else ""
    _log_gpr(f"[GPR] ERRORE in {context}: {exc}{suffix}", level=(Qgis.Critical if Qgis is not None else None))


# ---------------------------------------------------------------------------
# Primitivi FFT (senza scipy)
# ---------------------------------------------------------------------------

def _bandpass_fft(
    data: np.ndarray,
    dt_ns: float,
    low_mhz: float,
    high_mhz: float,
    taper_frac: float = 0.12,
) -> np.ndarray:
    if data is None:
        return np.zeros((0, 0), dtype=np.float32)
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2 or data.size == 0:
        return data.copy()
    data = np.nan_to_num(data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    N = int(data.shape[0])
    dt_s = float(dt_ns) * 1e-9
    if N <= 1 or (not np.isfinite(dt_s)) or dt_s <= 0.0:
        return np.array(data, dtype=np.float32, copy=True)

    # Reflect padding mitigates circular-convolution edge artifacts that
    # often appear as horizontal striping after FFT filtering.
    pad = int(np.clip(N // 4, 16, 256))
    if pad >= N:
        pad = max(1, N - 1)
    Np = int(N + 2 * pad)
    freqs = np.fft.rfftfreq(Np, d=dt_s)
    f_lo = max(0.0, float(low_mhz) * 1e6)
    f_hi = max(f_lo, float(high_mhz) * 1e6)
    if f_hi <= f_lo:
        return np.array(data, dtype=np.float32, copy=True)
    nyq = 0.5 / dt_s
    if np.isfinite(nyq) and nyq > 0.0:
        f_hi = min(f_hi, nyq * 0.995)
    if f_hi <= f_lo:
        return np.array(data, dtype=np.float32, copy=True)

    # Tukey-like cosine rolloff to reduce Gibbs ringing. `taper_frac` is the
    # fraction of the band width used as cosine roll-off on each edge
    # (0 = boxcar/sharp, 0.5 = full Hann). 0.12 = comportamento storico.
    try:
        tf = float(taper_frac)
    except Exception:
        tf = 0.12
    tf = float(np.clip(tf, 0.0, 0.5))
    bw = max(1e-9, f_hi - f_lo)
    edge = max(1e-9, tf * bw)
    mask = np.zeros(freqs.shape, dtype=np.float64)
    inside = (freqs >= f_lo) & (freqs <= f_hi)
    if np.any(inside):
        mask[inside] = 1.0
        dist_edge = np.minimum(freqs[inside] - f_lo, f_hi - freqs[inside])
        taper = dist_edge < edge
        if np.any(taper):
            x = np.clip(dist_edge[taper] / edge, 0.0, 1.0)
            mask_inside = mask[inside]
            mask_inside[taper] = 0.5 * (1.0 - np.cos(np.pi * x))
            mask[inside] = mask_inside
    if float(np.sum(mask)) < 1.0:
        _log_gpr(
            f"[GPR] bandpass: banda [{low_mhz:.3f},{high_mhz:.3f}] MHz vuota "
            f"(dt_ns={dt_ns:.6g})."
        )
        return np.array(data, dtype=np.float32, copy=True)

    out = np.empty_like(data, dtype=np.float32)
    n_traces = int(data.shape[1])
    n_freq = int(mask.shape[0])
    # Keep memory bounded on very large profile sets.
    target_bytes = 64 * 1024 * 1024
    bytes_per_trace = max(1, 16 * (Np + n_freq))
    cols_per_block = max(1, int(target_bytes // bytes_per_trace))
    cols_per_block = min(cols_per_block, max(1, n_traces))
    for start in range(0, n_traces, cols_per_block):
        stop = min(n_traces, start + cols_per_block)
        # Work on (n_traces_block, n_samples) contiguous chunks for faster FFT.
        block = np.ascontiguousarray(data[:, start:stop].T, dtype=np.float64)
        block = block - np.mean(block, axis=1, keepdims=True)
        block_pad = np.pad(block, ((0, 0), (pad, pad)), mode="reflect")
        Xf = np.fft.rfft(block_pad, axis=1)
        y_pad = np.fft.irfft(Xf * mask.reshape(1, -1), n=block_pad.shape[1], axis=1)
        out[:, start:stop] = y_pad[:, pad:pad + N].T.astype(np.float32, copy=False)
    return out


# ---------------------------------------------------------------------------
# Singole operazioni di processing
# ---------------------------------------------------------------------------

def dewow(data: np.ndarray, window: int = 16) -> np.ndarray:
    """Rimuove componente DC lenta sottraendo la media mobile."""
    if data is None:
        return np.zeros((0, 0), dtype=np.float32)
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2 or data.size == 0:
        return data.copy()
    data = np.nan_to_num(data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    n_s, n_t = data.shape
    try:
        win = int(window)
    except Exception:
        win = 16
    if win <= 1 or n_s <= 2:
        return data.copy()
    win = max(2, min(win, n_s))
    kernel = np.ones(win, dtype=np.float32) / float(win)
    half = win // 2
    out = np.empty_like(data)
    # Reflect padding avoids edge spikes at trace start/end.
    for i in range(n_t):
        tr = data[:, i].astype(np.float32, copy=False)
        pad_left = half
        pad_right = max(0, win - 1 - half)
        tr_pad = np.pad(tr, (pad_left, pad_right), mode="reflect")
        trend = np.convolve(tr_pad, kernel, mode="valid")
        out[:, i] = tr - trend[:n_s]
    return out


def _detect_t0_single(
    trace: np.ndarray,
    method: str,
    threshold: float,
    backup_nsamp: int,
    search_end: int,
) -> int:
    tr = np.asarray(trace, dtype=np.float64)
    win = tr[:search_end]
    abs_win = np.abs(win)
    pk = float(np.max(abs_win))
    if pk < 1e-12:
        return 0
    thr_abs = threshold * pk
    breaches = np.flatnonzero(abs_win >= thr_abs)
    breach_idx = int(breaches[0]) if breaches.size > 0 else 0
    if method == "threshold":
        t0 = breach_idx
    elif method == "peak":
        t0 = breach_idx
        seg = np.abs(tr[breach_idx:search_end])
        if seg.size >= 3:
            peaks = np.where((seg[1:-1] > seg[:-2]) & (seg[1:-1] > seg[2:]))[0]
            if peaks.size > 0:
                t0 = breach_idx + int(peaks[0]) + 1
    elif method == "zero_crossing":
        peak_idx = breach_idx
        seg = np.abs(tr[breach_idx:search_end])
        if seg.size >= 3:
            peaks = np.where((seg[1:-1] > seg[:-2]) & (seg[1:-1] > seg[2:]))[0]
            if peaks.size > 0:
                peak_idx = breach_idx + int(peaks[0]) + 1
        t0 = peak_idx
        seg_cross = tr[peak_idx:search_end]
        if seg_cross.size >= 2:
            crossings = np.where(seg_cross[:-1] * seg_cross[1:] <= 0.0)[0]
            if crossings.size > 0:
                t0 = peak_idx + int(crossings[0])
    else:
        t0 = breach_idx
    return max(0, t0 - backup_nsamp)


def time_zero_correction(
    data: np.ndarray,
    method: str = "peak",
    mode: str = "scan_by_scan",
    threshold: float = 0.2,
    backup_nsamp: int = 4,
    search_fraction: float = 0.25,
) -> np.ndarray:
    if data is None:
        return np.zeros((0, 0), dtype=np.float32)
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2 or data.size == 0:
        return data.copy()
    data = np.nan_to_num(data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    n_s, n_t = data.shape
    if n_s <= 1 or n_t <= 0:
        return data.copy()
    search_end = max(4, int(n_s * search_fraction))
    t0_indices = np.zeros(n_t, dtype=np.int64)
    for i in range(n_t):
        t0_indices[i] = _detect_t0_single(
            data[:, i], method, threshold, backup_nsamp, search_end
        )
    if mode == "line_by_line":
        t0_median = int(np.median(t0_indices))
        t0_indices[:] = t0_median
    if mode == "individual":
        # Come scan_by_scan ma loggato separatamente: ogni traccia è shiftata
        # al proprio t0, l'output è troncato a n_s - t0_max (no zero-padding).
        # Lo zero-padding (vecchio comportamento) creava righe inconsistenti in
        # fondo al radargram che distorcevano bg_removal e la visualizzazione.
        t0_max_ind = int(t0_indices.max())
        _log_gpr(
            f"[GPR] time_zero: method={method} mode={mode}  "
            f"t0_min={t0_indices.min()}  t0_max={t0_max_ind}  "
            f"n_samples_out={n_s - t0_max_ind}"
        )
        if t0_max_ind <= 0:
            return data
        n_out_ind = n_s - t0_max_ind
        out = np.zeros((n_out_ind, n_t), dtype=np.float32)
        for i in range(n_t):
            t0_i = int(t0_indices[i])
            src = data[t0_i:, i]
            n_copy = min(len(src), n_out_ind)
            out[:n_copy, i] = src[:n_copy]
        return out
    t0_max = int(t0_indices.max())
    _log_gpr(
        f"[GPR] time_zero: method={method} mode={mode}  "
        f"t0_min={t0_indices.min()}  t0_max={t0_max}  "
        f"n_samples_out={n_s - t0_max}"
    )
    if t0_max <= 0:
        return data
    n_out = n_s - t0_max
    out   = np.zeros((n_out, n_t), dtype=np.float32)
    for i in range(n_t):
        t0_i   = int(t0_indices[i])
        src    = data[t0_i:, i]
        n_copy = min(len(src), n_out)
        out[:n_copy, i] = src[:n_copy]
    return out


def background_removal(
    data: np.ndarray,
    mode: str = "line_by_line",
    window: int = 0,
    reference_trace: np.ndarray | None = None,
    sample_start: int = 0,
    sample_end: int = 0,
    method: str = "mean",
) -> np.ndarray:
    """
    Background removal: sottrae la traccia orizzontale media o mediana (clutter) da ogni campione.

    Sottrae una traccia di riferimento (il "clutter" orizzontale costante)
    da ogni traccia del radargram.

    Modalita'
    ---------
    'line_by_line'
        window = 0  (AUTO): media o mediana globale sull'intero profilo.
        window > 0        : media o mediana scorrevole di `window` tracce.

    'grid_by_grid'
        Sottrae `reference_trace` pre-calcolata su tutti i radargram.
        Il chiamante deve fornirla via apply_pipeline(bg_reference_trace=...).
        Se None: fallback a line_by_line auto.

    Metodi
    ------
    method : 'mean' | 'median'
        'mean'   : media aritmetica (standard, ottimo per clutter uniforme).
        'median' : mediana (robusta: evita ombre scure sopra/sotto forti iperboli).

    Finestra temporale (sample_start / sample_end)
    -----------------------------------------------
    Permette di limitare la sottrazione a un range di campioni [s_lo, s_hi).
    I campioni FUORI dalla finestra vengono lasciati INVARIATI.

        sample_start = 0, sample_end = 0  ->  intera traccia (default).
        sample_start = 20, sample_end = 0 ->  salta i primi 20 campioni
                                               (ground coupling), applica
                                               BG removal dal campione 20
                                               in poi.

    Parametri
    ----------
    data             : (n_samples, n_traces) float32
    mode             : 'line_by_line' | 'grid_by_grid'
    window           : int  lunghezza filtro in tracce (0 = auto)
    reference_trace  : (n_samples,) float64 | None  solo per grid_by_grid
    sample_start     : int  primo campione incluso nella finestra (0 = inizio)
    sample_end       : int  ultimo campione escluso (0 = auto = n_samples)
    method           : 'mean' | 'median'
    """
    if data is None:
        return np.zeros((0, 0), dtype=np.float32)
    d64 = np.asarray(data, dtype=np.float64)
    if d64.ndim != 2 or d64.size == 0:
        return data.copy() if hasattr(data, "copy") else np.zeros((0, 0), dtype=np.float32)
    d64 = np.nan_to_num(d64, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    n_s, n_t = d64.shape
    if n_s <= 0 or n_t <= 0:
        return d64.astype(np.float32)
    try:
        win_req = int(window)
    except Exception:
        win_req = 0
    win_eff = int(win_req)
    if win_eff > n_t and n_t > 0:
        win_eff = int(n_t)
        _log_gpr(
            f"[GPR] bg_removal: win clampato a n_traces={n_t} "
            f"(richiesto={win_req})."
        )

    row_std = d64.std(axis=1)
    n_zero  = int(np.count_nonzero(row_std == 0.0))
    s_lo    = max(0, int(sample_start))
    s_hi    = min(n_s, int(sample_end)) if sample_end > 0 else n_s
    if sample_end > 0 and s_hi <= s_lo:
        _log_gpr(
            f"[GPR] bg_removal: range campioni non valido "
            f"[{s_lo}:{s_hi}], imposto A=0 (fino alla fine)."
        )
        s_hi = n_s
    _log_gpr(
        f"[GPR] bg_diag: mode={mode} method={method} win={win_eff if win_eff > 0 else 'auto'}  "
        f"samples=[{s_lo}:{s_hi}/{n_s}]  "
        f"mean_inter-trace_std={row_std.mean():.6g}  "
        f"rows_zero_std={n_zero}/{n_s}"
    )
    if (s_hi - s_lo) < 64:
        _log_gpr(
            f"[GPR] bg_removal: ATTENZIONE finestra campioni corta "
            f"({s_hi - s_lo} campioni). Potresti vedere bande residue."
        )

    if s_lo >= s_hi:
        _log_gpr("[GPR] bg_removal: finestra campioni vuota, nessuna modifica.")
        return d64.astype(np.float32)

    sub = d64[s_lo:s_hi, :]   # (n_win, n_t)
    row_abs_max = np.max(np.abs(sub), axis=1)
    valid_rows = row_abs_max > 0.0
    use_median = str(method or "mean").strip().lower() == "median"

    # ---------------------------------------------------------------
    # grid_by_grid
    # ---------------------------------------------------------------
    if mode == "grid_by_grid":
        if reference_trace is not None:
            ref = np.asarray(reference_trace, dtype=np.float64)
            ref_sub = ref[s_lo:min(s_hi, ref.shape[0])]
            if ref_sub.shape[0] != sub.shape[0]:
                target_n = int(sub.shape[0])
                old_n = int(ref_sub.shape[0])
                _log_gpr(
                    f"[GPR] bg_removal grid_by_grid: ref len {old_n} "
                    f"!= window {target_n}; ridimensiono la traccia di riferimento."
                )
                if target_n < old_n:
                    ref_sub = ref_sub[:target_n]
                else:
                    ext = np.zeros(target_n, dtype=np.float64)
                    ext[:old_n] = ref_sub
                    ref_sub = ext
        else:
            _log_gpr("[GPR] bg_removal: grid_by_grid senza reference_trace; fallback a line_by_line auto.")
            ref_sub = np.median(sub, axis=1) if use_median else sub.mean(axis=1)
        out = d64.copy()
        corr = sub - ref_sub.reshape(-1, 1)
        block = out[s_lo:s_hi, :]
        block[valid_rows, :] = corr[valid_rows, :]
        out[s_lo:s_hi, :] = block
        return out.astype(np.float32)

    # ---------------------------------------------------------------
    # line_by_line — auto: media o mediana globale del sub-range
    # ---------------------------------------------------------------
    if win_eff <= 0 or win_eff >= n_t:
        if use_median:
            ref_trace = np.median(sub, axis=1, keepdims=True)
            _log_gpr(f"[GPR] bg_removal: global median subtracted  n_t={n_t}  s=[{s_lo}:{s_hi}]")
        else:
            ref_trace = sub.mean(axis=1, keepdims=True)
            _log_gpr(f"[GPR] bg_removal: global mean subtracted  n_t={n_t}  s=[{s_lo}:{s_hi}]")
        out = d64.copy()
        corr = sub - ref_trace
        block = out[s_lo:s_hi, :]
        block[valid_rows, :] = corr[valid_rows, :]
        out[s_lo:s_hi, :] = block
        return out.astype(np.float32)

    # ---------------------------------------------------------------
    # line_by_line — finestra scorrevole lungo asse tracce
    # ---------------------------------------------------------------
    half   = win_eff // 2
    n_sub  = sub.shape[1]
    if use_median:
        local_ref = np.empty_like(sub)
        for j in range(n_sub):
            j_lo = max(0, j - half)
            j_hi = min(n_sub, j + half + 1)
            local_ref[:, j] = np.median(sub[:, j_lo:j_hi], axis=1)
        _log_gpr(
            f"[GPR] bg_removal: sliding median win={win_eff}  s=[{s_lo}:{s_hi}]"
        )
    else:
        idx    = np.arange(n_sub)
        hi_t   = np.minimum(n_sub - 1, idx + half)
        lo_t   = np.maximum(0,         idx - half)
        cs     = np.cumsum(sub, axis=1)
        sum_hi = cs[:, hi_t]
        lo_prev = np.maximum(0, lo_t - 1)
        sum_lo  = cs[:, lo_prev]
        sum_lo[:, lo_t == 0] = 0.0
        win_len    = (hi_t - lo_t + 1).reshape(1, -1)
        local_ref = (sum_hi - sum_lo) / win_len
        _log_gpr(
            f"[GPR] bg_removal: sliding mean win={win_eff}  s=[{s_lo}:{s_hi}]  "
            f"eff_win=[{(hi_t-lo_t+1).min()},{(hi_t-lo_t+1).max()}]"
        )
    out = d64.copy()
    corr = sub - local_ref
    block = out[s_lo:s_hi, :]
    block[valid_rows, :] = corr[valid_rows, :]
    out[s_lo:s_hi, :] = block
    return out.astype(np.float32)


def agc_gain(
    data: np.ndarray,
    window: int = 128,
    clip_percentile: float | None = None,
    sample_start: int = 0,
    max_gain_factor: float | None = None,
    noise_threshold: float | None = None,
) -> np.ndarray:
    """
    Automatic Gain Control vettorizzato con cumsum.
    O(n_s * n_t) senza loop Python.

    sample_start: i campioni [0:sample_start] vengono copiati invariati
                  (gain=1). L'AGC viene applicato solo da sample_start in poi.
                  Utile per escludere il burst superficiale (coupling antenna)
                  dalla normalizzazione AGC.
    max_gain_factor: limita il guadagno massimo applicabile rispetto al livello medio.
    noise_threshold: soglia minima RMS per non amplificare il rumore di fondo profondo.
    """
    if data is None:
        return np.zeros((0, 0), dtype=np.float32)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return arr.copy()
    arr = np.nan_to_num(arr, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    n_s, n_t = arr.shape
    if n_s <= 1 or n_t <= 0:
        return arr
    s0 = max(0, min(int(sample_start), n_s - 1))
    # Sottoarray su cui applicare l'AGC
    sub = arr[s0:, :]
    n_sub = sub.shape[0]
    if n_sub <= 1:
        return arr
    win = max(8, int(window))
    win = min(win, max(8, n_sub))
    if win % 2 != 0:
        win += 1
    half = max(4, win // 2)
    d64 = sub.astype(np.float64, copy=False)
    sq = d64 ** 2
    # Full window at edges via reflect padding.
    sq_pad = np.pad(sq, ((half, half), (0, 0)), mode="reflect")
    cs = np.vstack([np.zeros((1, n_t), dtype=np.float64), np.cumsum(sq_pad, axis=0)])
    s_idx = np.arange(n_sub, dtype=np.int64)
    sum_win = cs[s_idx + win, :] - cs[s_idx, :]
    rms = np.sqrt(np.maximum(sum_win / float(win), 0.0))
    try:
        valid_rms = rms[rms > 1e-10]
        if valid_rms.size > 0:
            rms_global = float(np.percentile(valid_rms, 95.0))
            if np.isfinite(rms_global) and rms_global > 1e-12:
                rms = np.minimum(rms, rms_global * 10.0)
            if max_gain_factor is not None and float(max_gain_factor) > 0.0:
                rms_med = float(np.median(valid_rms))
                if np.isfinite(rms_med) and rms_med > 1e-12:
                    min_rms = rms_med / float(max_gain_factor)
                    rms = np.maximum(rms, min_rms)
    except Exception:
        pass
    if noise_threshold is not None and float(noise_threshold) > 0.0:
        rms = np.maximum(rms, float(noise_threshold))
    rms = rms + 1e-10
    agc_sub = (d64 / rms).astype(np.float32)
    if clip_percentile is not None and agc_sub.size > 0:
        clip = float(np.percentile(np.abs(agc_sub), clip_percentile))
        if clip > 1e-12:
            agc_sub = np.clip(agc_sub, -clip, clip)
    arr[s0:, :] = agc_sub
    return arr


def stolt_migration_2d(
    data: np.ndarray,
    dt_ns: float,
    dx_m: float = 0.05,
    velocity_m_s: float = 100000000.0,
) -> np.ndarray:
    """Migrazione 2D F-K (Stolt migration) in puro NumPy.

    Collassa le iperboli di diffrazione al loro apice e riposiziona
    i riflettori inclinati nel profilo temporale zero-offset.

    Parametri
    ----------
    data         : (n_samples, n_traces) float32
    dt_ns        : intervallo di campionamento temporale in ns
    dx_m         : spaziatura orizzontale media tra le tracce in metri
    velocity_m_s : velocita' di propagazione nel mezzo in m/s (es. 1e8 = 0.1 m/ns)
    """
    if data is None:
        return np.zeros((0, 0), dtype=np.float32)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return arr.copy()
    arr = np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    n_s, n_t = arr.shape
    if n_s < 4 or n_t < 4:
        return arr.copy()

    dt = float(dt_ns) * 1e-9
    dx = max(float(dx_m), 1e-4)
    v = max(float(velocity_m_s), 1e3)
    vm = v / 2.0  # Exploding reflector model per two-way travel time

    # Pad a potenza di 2 per massima efficienza FFT 2D
    npad_s = int(2 ** np.ceil(np.log2(max(16, n_s * 1.5))))
    npad_t = int(2 ** np.ceil(np.log2(max(16, n_t * 1.5))))

    pad_data = np.zeros((npad_s, npad_t), dtype=np.float64)
    pad_data[:n_s, :n_t] = arr.astype(np.float64, copy=False)

    fk = np.fft.fft2(pad_data)
    f = np.fft.fftfreq(npad_s, d=dt)      # Hz
    kx = np.fft.fftfreq(npad_t, d=dx)    # cycles/m

    kz = f / vm  # cycles/m
    fk_mig = np.zeros_like(fk, dtype=np.complex128)

    half_s = npad_s // 2
    kz_pos = kz[:half_s]
    f_orig_pos = f[:half_s]

    for ik, kx_val in enumerate(kx):
        rad = kz_pos ** 2 + kx_val ** 2
        f_req = vm * np.sqrt(rad)
        denom = np.sqrt(rad)
        jacobian = np.where(denom > 1e-12, kz_pos / np.maximum(denom, 1e-12), 0.0)

        real_interp = np.interp(f_req, f_orig_pos, np.real(fk[:half_s, ik]), left=0.0, right=0.0)
        imag_interp = np.interp(f_req, f_orig_pos, np.imag(fk[:half_s, ik]), left=0.0, right=0.0)

        mig_pos = (real_interp + 1j * imag_interp) * jacobian
        fk_mig[:half_s, ik] = mig_pos
        # Simmetria hermitiana per segnali reali
        fk_mig[half_s + 1:, ik] = np.conj(mig_pos[1:][::-1])

    migrated = np.real(np.fft.ifft2(fk_mig))
    return np.nan_to_num(migrated[:n_s, :n_t].astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)


def bandpass_filter(
    data: np.ndarray,
    dt_ns: float,
    low_mhz: float,
    high_mhz: float,
    order: int = 4,
    taper_frac: float = 0.12,
) -> np.ndarray:
    # `order` mantenuto per compatibilita' di firma: il filtro e' FFT a fase
    # zero con roll-off coseno controllato da `taper_frac`.
    return _bandpass_fft(data, dt_ns, low_mhz, high_mhz, taper_frac=taper_frac)


def hilbert_envelope(data: np.ndarray) -> np.ndarray:
    """Compute Hilbert envelope along the sample axis (axis 0).

    Constructs the analytic signal via full FFT (two-sided spectrum):
      Xf = fft(x)
      h[k]: 1 at DC, 2 for positive freqs, 0 for negative freqs, 1 at Nyquist (even N)
      z = ifft(Xf * h)   →  complex analytic signal
      env = |z| = sqrt(real² + imag²)

    This matches scipy.signal.hilbert exactly and is computed in blocks
    to keep peak RAM bounded on large radargrams.
    """
    if data is None:
        return np.zeros((0, 0), dtype=np.float32)
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        return np.abs(arr).astype(np.float32, copy=True)
    arr = np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    n_s, n_t = arr.shape
    if n_s <= 1 or n_t <= 0:
        return np.abs(arr).astype(np.float32, copy=True)
    out = np.empty((n_s, n_t), dtype=np.float32)
    # Build the two-sided Hilbert multiplier h[k] of length n_s.
    h = np.zeros(n_s, dtype=np.float64)
    if n_s % 2 == 0:
        h[0] = h[n_s // 2] = 1.0
        h[1:n_s // 2] = 2.0
    else:
        h[0] = 1.0
        h[1:(n_s + 1) // 2] = 2.0
    target_bytes = 64 * 1024 * 1024
    # Each trace needs ~3 complex arrays of length n_s (fft, multiply, ifft).
    bytes_per_trace = max(1, 3 * 16 * n_s)
    cols_per_block = max(1, int(target_bytes // bytes_per_trace))
    cols_per_block = min(cols_per_block, max(1, n_t))
    for start in range(0, n_t, cols_per_block):
        stop = min(n_t, start + cols_per_block)
        block = np.ascontiguousarray(arr[:, start:stop].T, dtype=np.float64)
        Xf = np.fft.fft(block, axis=1)
        analytic = np.fft.ifft(Xf * h.reshape(1, -1), axis=1)
        env = np.abs(analytic)
        out[:, start:stop] = env.T.astype(np.float32, copy=False)
    return out


def manual_gain(
    data: np.ndarray,
    gain_type: str = "linear",
    gain_start: float = 1.0,
    gain_end: float = 1.0,
    sample_start: int = 0,
    sample_end: int = 0,
) -> np.ndarray:
    """Applica un gain manuale dipendente dalla profondità (depth-varying gain).

    Gain manuale dipendente dalla profondità: moltiplica ogni campione
    per una curva di amplificazione che cresce dalla superficie
    (gain_start) alla massima profondità (gain_end).

    Parametri
    ----------
    data         : (n_samples, n_traces) float32
    gain_type    : 'linear' | 'exponential'
                     linear      : g(t) interpolato linearmente tra gain_start e gain_end
                     exponential : g(t) = gain_start * exp(log(gain_end/gain_start) * t)
                                   se gain_start > 0, altrimenti fallback a lineare
    gain_start   : amplificazione al campione 0 (superficie)
    gain_end     : amplificazione al campione n_samples-1 (profondità max)
    sample_start : primo campione su cui applicare il gain (0 = dall'inizio)
    sample_end   : ultimo campione escluso (0 = fino alla fine)
    """
    if data is None:
        return np.zeros((0, 0), dtype=np.float32)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return arr.copy()
    arr = np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    n_s, n_t = arr.shape
    if n_s <= 0 or n_t <= 0:
        return arr.copy()

    g0 = max(float(gain_start), 0.0)
    g1 = max(float(gain_end), 0.0)
    s_lo = max(0, int(sample_start))
    s_hi = min(n_s, int(sample_end)) if sample_end > 0 else n_s
    if s_lo >= s_hi:
        return arr.copy()

    n_win = s_hi - s_lo
    t = np.linspace(0.0, 1.0, n_win, dtype=np.float64)

    if gain_type == "exponential" and g0 > 1e-12 and g1 > 1e-12:
        alpha = np.log(g1 / g0)
        g = g0 * np.exp(alpha * t)
    else:
        g = g0 + (g1 - g0) * t

    g = np.clip(g, 0.0, 1e6).astype(np.float32)

    out = arr.copy()
    out[s_lo:s_hi, :] = (arr[s_lo:s_hi, :] * g.reshape(-1, 1)).astype(np.float32)
    _log_gpr(
        f"[GPR] manual_gain: type={gain_type}  g0={g0:.3g}  g1={g1:.3g}  "
        f"s=[{s_lo}:{s_hi}]  shape={arr.shape}"
    )
    return out


# ---------------------------------------------------------------------------
# Filtri aggiuntivi (Fase 1) — funzioni pure numpy, disattivi di default
# ---------------------------------------------------------------------------

def dcshift(
    data: np.ndarray,
    sample_start: int = 0,
    sample_end: int = 0,
) -> np.ndarray:
    """Rimozione DC: sottrae da ogni traccia la sua media su una finestra di campioni.

    A differenza di `dewow` (media *mobile*), qui si sottrae un singolo valore
    costante per traccia — la media dei campioni nell'intervallo
    ``[sample_start:sample_end)`` (default: intera traccia). Utile quando il
    livello di zero varia da traccia a traccia ma non lungo la traccia.
    """
    if data is None:
        return np.zeros((0, 0), dtype=np.float32)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return arr.copy()
    arr = np.nan_to_num(arr, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    n_s, n_t = arr.shape
    if n_s <= 0 or n_t <= 0:
        return arr
    s_lo = max(0, int(sample_start))
    s_hi = min(n_s, int(sample_end)) if int(sample_end) > 0 else n_s
    if s_hi <= s_lo:
        s_lo, s_hi = 0, n_s
    baseline = arr[s_lo:s_hi, :].astype(np.float64).mean(axis=0)
    out = (arr - baseline.reshape(1, -1)).astype(np.float32)
    _log_gpr(f"[GPR] dcshift: baseline su campioni [{s_lo}:{s_hi}]  shape={arr.shape}")
    return out


def boxcar_smooth(
    data: np.ndarray,
    width: int = 3,
    height: int = 3,
) -> np.ndarray:
    """Lisciatura boxcar 2D (media non pesata) separabile.

    ``width``  = semi-ampiezza in *tracce* (0 = nessuna lisciatura orizzontale).
    ``height`` = semi-ampiezza in *campioni* (0 = nessuna lisciatura verticale).
    Con width=3 la finestra copre 3 tracce a sinistra + la traccia + 3 a destra
    (2*3+1 = 7), coerente con la convenzione classica dei software GPR.
    """
    if data is None:
        return np.zeros((0, 0), dtype=np.float32)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return arr.copy()
    arr = np.nan_to_num(arr, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    n_s, n_t = arr.shape
    try:
        w = max(0, int(width))
        h = max(0, int(height))
    except Exception:
        w, h = 3, 3
    if (w == 0 and h == 0) or n_s < 2 or n_t < 2:
        return arr

    def _movmean(x: np.ndarray, half: int, axis: int) -> np.ndarray:
        if half <= 0:
            return x
        k = 2 * half + 1
        n = x.shape[axis]
        half_eff = min(half, max(0, n - 1))
        if half_eff <= 0:
            return x
        k = 2 * half_eff + 1
        pad = [(0, 0), (0, 0)]
        pad[axis] = (half_eff, half_eff)
        xp = np.pad(x.astype(np.float64), pad, mode="reflect")
        cs = np.cumsum(xp, axis=axis)
        zero_shape = list(xp.shape)
        zero_shape[axis] = 1
        cs = np.concatenate([np.zeros(zero_shape), cs], axis=axis)
        sl_hi = [slice(None), slice(None)]
        sl_lo = [slice(None), slice(None)]
        sl_hi[axis] = slice(k, k + n)
        sl_lo[axis] = slice(0, n)
        return (cs[tuple(sl_hi)] - cs[tuple(sl_lo)]) / float(k)

    out = arr.astype(np.float64)
    if h > 0:
        out = _movmean(out, h, axis=0)
    if w > 0:
        out = _movmean(out, w, axis=1)
    _log_gpr(f"[GPR] boxcar_smooth: width={w} height={h}  shape={arr.shape}")
    return out.astype(np.float32)


def _spectral_rfft_blocks(data: np.ndarray, dt_ns: float, transform):
    """Applica `transform(Xf, freqs_hz)` alla rFFT lungo l'asse campioni.

    Gestisce reflect-padding sui bordi, blocchi di colonne per limitare la RAM
    e la de-media per traccia. `transform` riceve lo spettro complesso
    (n_freq, n_block) e le frequenze (Hz) e restituisce lo spettro modificato.
    """
    arr = np.asarray(data, dtype=np.float32)
    arr = np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    n_s, n_t = arr.shape
    dt_s = float(dt_ns) * 1e-9
    if n_s <= 2 or not np.isfinite(dt_s) or dt_s <= 0.0:
        return arr.copy()
    pad = int(np.clip(n_s // 4, 16, 256))
    if pad >= n_s:
        pad = max(1, n_s - 1)
    Np = int(n_s + 2 * pad)
    freqs = np.fft.rfftfreq(Np, d=dt_s)
    out = np.empty_like(arr, dtype=np.float32)
    target_bytes = 64 * 1024 * 1024
    bytes_per_trace = max(1, 32 * (Np + freqs.shape[0]))
    cols = min(max(1, int(target_bytes // bytes_per_trace)), max(1, n_t))
    for start in range(0, n_t, cols):
        stop = min(n_t, start + cols)
        block = np.ascontiguousarray(arr[:, start:stop].T, dtype=np.float64)
        block = block - np.mean(block, axis=1, keepdims=True)
        block_pad = np.pad(block, ((0, 0), (pad, pad)), mode="reflect")
        Xf = np.fft.rfft(block_pad, axis=1)
        Yf = transform(Xf, freqs)
        y_pad = np.fft.irfft(Yf, n=block_pad.shape[1], axis=1)
        out[:, start:stop] = y_pad[:, pad:pad + n_s].T.astype(np.float32, copy=False)
    return out


def notch_filter(
    data: np.ndarray,
    dt_ns: float,
    freq_mhz: float,
    bandwidth_mhz: float = 50.0,
) -> np.ndarray:
    """Filtro notch: attenua una banda stretta gaussiana centrata su `freq_mhz`."""
    if data is None:
        return np.zeros((0, 0), dtype=np.float32)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return arr.copy()
    f0 = float(freq_mhz) * 1e6
    bw = max(1e3, float(bandwidth_mhz) * 1e6)
    if f0 <= 0.0:
        _log_gpr("[GPR] notch_filter: freq_mhz<=0, nessuna modifica.")
        return arr.copy()
    sigma = bw / 2.3548  # FWHM -> sigma

    def _tf(Xf, freqs):
        notch = 1.0 - np.exp(-0.5 * ((freqs - f0) / sigma) ** 2)
        return Xf * notch.reshape(1, -1)

    out = _spectral_rfft_blocks(arr, dt_ns, _tf)
    _log_gpr(f"[GPR] notch_filter: f0={freq_mhz:.1f}MHz bw={bandwidth_mhz:.1f}MHz  shape={arr.shape}")
    return out


def spectral_whitening(
    data: np.ndarray,
    dt_ns: float,
    low_mhz: float = 0.0,
    high_mhz: float = 0.0,
    gain: float = 20.0,
) -> np.ndarray:
    """Whitening spettrale: |X(f)|=1 nella banda [low,high], fase preservata, 0 fuori.

    Il risultato viene diviso per `gain` (tipico 10-100) per riportarlo a una
    scala confrontabile con gli altri filtri. Con low/high <= 0 la banda e'
    l'intero spettro disponibile.
    """
    if data is None:
        return np.zeros((0, 0), dtype=np.float32)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return arr.copy()
    g = float(gain) if float(gain) > 1e-6 else 1.0
    f_lo = max(0.0, float(low_mhz) * 1e6)
    f_hi = float(high_mhz) * 1e6

    def _tf(Xf, freqs):
        mag = np.abs(Xf)
        phase = np.divide(Xf, mag, out=np.zeros_like(Xf), where=mag > 1e-20)
        hi = f_hi if f_hi > f_lo else float(freqs[-1])
        band = (freqs >= f_lo) & (freqs <= hi)
        return (phase * band.reshape(1, -1)) / g

    out = _spectral_rfft_blocks(arr, dt_ns, _tf)
    _log_gpr(
        f"[GPR] spectral_whitening: banda [{low_mhz:.1f},{high_mhz:.1f}]MHz "
        f"gain={g:.3g}  shape={arr.shape}"
    )
    return out


def amplitude_equalization(
    data: np.ndarray,
    window: int = 120,
    gain: float = 1.0,
) -> np.ndarray:
    """Equalizzazione d'ampiezza: gain per campione = scala_ref / RMS(finestra mobile).

    Ogni campione viene diviso per l'RMS calcolato su una finestra di `window`
    campioni centrata su di esso (lungo l'asse tempo) e riscalato per una
    scala di riferimento robusta (mediana degli RMS) cosi' che l'ampiezza
    complessiva resti confrontabile. `gain` e' un moltiplicatore finale.
    """
    if data is None:
        return np.zeros((0, 0), dtype=np.float32)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return arr.copy()
    arr = np.nan_to_num(arr, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    n_s, n_t = arr.shape
    if n_s <= 2 or n_t <= 0:
        return arr
    win = max(4, int(window))
    win = min(win, n_s if n_s % 2 == 0 else n_s - 1)
    if win % 2:
        win += 1
    half = max(2, win // 2)
    d64 = arr.astype(np.float64)
    sq = d64 ** 2
    sq_pad = np.pad(sq, ((half, half), (0, 0)), mode="reflect")
    cs = np.vstack([np.zeros((1, n_t)), np.cumsum(sq_pad, axis=0)])
    idx = np.arange(n_s, dtype=np.int64)
    k = 2 * half + 1
    rms = np.sqrt(np.maximum((cs[idx + k, :] - cs[idx, :]) / float(k), 0.0))
    valid = rms[rms > 1e-12]
    ref = float(np.median(valid)) if valid.size else 1.0
    rms = rms + 1e-12
    out = (d64 / rms) * (ref * float(gain))
    _log_gpr(f"[GPR] amplitude_equalization: win={win} ref={ref:.3g} gain={gain:.3g}")
    return out.astype(np.float32)


def radar_threshold(
    data: np.ndarray,
    neg_fraction: float = 0.0,
    pos_fraction: float = 0.0,
) -> np.ndarray:
    """Azzera i campioni la cui ampiezza sta dentro le soglie.

    Le soglie sono frazioni (0..1) del massimo assoluto del radargramma:
    ogni valore in ``(-neg_fraction*amax, +pos_fraction*amax)`` viene messo a 0.
    """
    if data is None:
        return np.zeros((0, 0), dtype=np.float32)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return arr.copy()
    arr = np.nan_to_num(arr, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    amax = float(np.max(np.abs(arr))) if arr.size else 0.0
    if amax <= 1e-20:
        return arr
    pos = max(0.0, float(pos_fraction)) * amax
    neg = -max(0.0, float(neg_fraction)) * amax
    out = arr.copy()
    out[(arr > neg) & (arr < pos)] = 0.0
    _log_gpr(
        f"[GPR] radar_threshold: neg={neg:.3g} pos={pos:.3g} (amax={amax:.3g})  "
        f"azzerati={int(np.count_nonzero(out == 0.0))}/{arr.size}"
    )
    return out


def power_gain(
    data: np.ndarray,
    dt_ns: float,
    mode: str = "power",
    exponent: float = 1.0,
    t0_ns: float = 0.0,
    scale: float = 1.0,
) -> np.ndarray:
    """Range gain per legge analitica: potenza ``t^a`` o esponenziale ``exp(a*t)``.

    Il tempo e' misurato da `t0_ns`. La curva viene normalizzata a 1 a
    meta' traccia, cosi' che `scale` sia l'unico controllo di ampiezza globale.
    """
    if data is None:
        return np.zeros((0, 0), dtype=np.float32)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return arr.copy()
    arr = np.nan_to_num(arr, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    n_s, n_t = arr.shape
    if n_s <= 1 or n_t <= 0:
        return arr
    dt = float(dt_ns) if float(dt_ns) > 0 else 1.0
    a = float(exponent)
    t = np.maximum(np.arange(n_s, dtype=np.float64) * dt - float(t0_ns), 0.0)
    if str(mode).strip().lower() == "exp":
        g = np.exp(a * (t / max(t[-1], 1e-9)))
    else:
        g = np.power(t / max(t[-1], 1e-9), a, where=(t > 0), out=np.zeros_like(t))
    g_mid = float(g[n_s // 2]) if abs(g[n_s // 2]) > 1e-12 else 1.0
    g = np.clip(g / g_mid * float(scale), 0.0, 1e6)
    out = (arr.astype(np.float64) * g.reshape(-1, 1)).astype(np.float32)
    _log_gpr(f"[GPR] power_gain: mode={mode} a={a:.3g} t0={t0_ns:.3g}ns scale={scale:.3g}")
    return out


# ---------------------------------------------------------------------------
# Filtri avanzati (Fase 4) — eigen / median 2D / FK / Kirchhoff. Puro numpy.
# ---------------------------------------------------------------------------

def eigen_filter(
    data: np.ndarray,
    drop: int = 1,
    keep_last: int = 0,
) -> np.ndarray:
    """Filtro a immagini proprie (SVD): rimuove il clutter orizzontale coerente.

    Decompone il radargramma ``A = U S Vt`` e ricostruisce azzerando le prime
    ``drop`` componenti (le più energetiche = bande orizzontali quasi costanti,
    l'onda diretta, il ringing). È un background removal "morbido" che non
    sottrae una singola traccia media.

    ``keep_last`` > 0 tiene solo le componenti nell'intervallo
    ``[drop:keep_last]`` (utile per isolare strutture a media energia).

    Adattato concettualmente da RGPR (E. Huber, G. Hans), GPL (>= 2),
    ``R/eigenFilter.R`` — reimplementato da zero con ``numpy.linalg.svd``.
    """
    if data is None:
        return np.zeros((0, 0), dtype=np.float32)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return arr.copy()
    arr = np.nan_to_num(arr, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    n_s, n_t = arr.shape
    if n_s < 2 or n_t < 2:
        return arr
    d0 = max(0, int(drop))
    k = int(min(n_s, n_t))
    if d0 >= k:
        _log_gpr(f"[GPR] eigen_filter: drop={d0} >= rango {k}, output nullo.")
        return np.zeros_like(arr)
    try:
        U, S, Vt = np.linalg.svd(arr.astype(np.float64), full_matrices=False)
    except np.linalg.LinAlgError:
        _log_gpr("[GPR] eigen_filter: SVD non convergente, nessuna modifica.")
        return arr
    S2 = S.copy()
    S2[:d0] = 0.0
    hi = int(keep_last)
    if hi > d0:
        S2[hi:] = 0.0
    out = (U * S2) @ Vt
    _log_gpr(
        f"[GPR] eigen_filter: drop={d0} keep_last={hi if hi > d0 else 'all'}  "
        f"rango={k}  energia_residua={float(np.sum(S2 ** 2) / max(np.sum(S ** 2), 1e-20)):.3f}"
    )
    return out.astype(np.float32)


def median_filter2d(
    data: np.ndarray,
    w_trace: int = 1,
    w_sample: int = 1,
) -> np.ndarray:
    """Mediana 2D scorrevole (de-spike). Semi-ampiezze ``w_trace`` (asse tracce)
    e ``w_sample`` (asse campioni). 0/0 = no-op. Bordi in reflect.

    Adattato concettualmente da RGPR ``filter2D(type="median")``, GPL (>= 2) —
    reimplementato con ``numpy.lib.stride_tricks.sliding_window_view``.
    """
    if data is None:
        return np.zeros((0, 0), dtype=np.float32)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return arr.copy()
    arr = np.nan_to_num(arr, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    n_s, n_t = arr.shape
    hs = max(0, int(w_sample))
    ht = max(0, int(w_trace))
    if (hs == 0 and ht == 0) or n_s < 2 or n_t < 2:
        return arr
    hs = min(hs, max(0, n_s - 1))
    ht = min(ht, max(0, n_t - 1))
    ks, kt = 2 * hs + 1, 2 * ht + 1
    padded = np.pad(arr.astype(np.float64), ((hs, hs), (ht, ht)), mode="reflect")
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        win = sliding_window_view(padded, (ks, kt))          # (n_s, n_t, ks, kt)
        out = np.median(win, axis=(-2, -1))
    except Exception:
        # fallback O(n_s*n_t*k^2) senza stride tricks
        out = np.empty((n_s, n_t), dtype=np.float64)
        for i in range(n_s):
            for j in range(n_t):
                out[i, j] = np.median(padded[i:i + ks, j:j + kt])
    _log_gpr(f"[GPR] median_filter2d: w_trace={ht} w_sample={hs}  shape={arr.shape}")
    return out.astype(np.float32)


def _fk_transform(arr64: np.ndarray, dt_s: float, dx_m: float):
    """FFT 2D spazio-tempo. Ritorna (FK, f_hz, kx_cyc_per_m) con fftshift."""
    n_s, n_t = arr64.shape
    FK = np.fft.fftshift(np.fft.fft2(arr64))
    f = np.fft.fftshift(np.fft.fftfreq(n_s, d=dt_s))       # Hz (può essere <0)
    kx = np.fft.fftshift(np.fft.fftfreq(n_t, d=dx_m))      # cicli/m
    return FK, f, kx


def fk_filter(
    data: np.ndarray,
    dt_ns: float,
    dx_m: float,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Applica una maschera nel dominio FK (f, kx) e antitrasforma.

    ``mask`` : array reale (n_s, n_t) in [0,1] **già in ordine fftshift**
    (0 = blanca, 1 = passa). Se None → identità. La UI di painting fornirà
    la maschera; qui è la primitiva pura.
    """
    if data is None:
        return np.zeros((0, 0), dtype=np.float32)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return arr.copy()
    arr = np.nan_to_num(arr, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    if mask is None:
        return arr
    n_s, n_t = arr.shape
    m = np.asarray(mask, dtype=np.float64)
    if m.shape != (n_s, n_t):
        _log_gpr(f"[GPR] fk_filter: mask {m.shape} != dato {arr.shape}, ignorata.")
        return arr
    dt_s = float(dt_ns) * 1e-9
    dx = max(float(dx_m), 1e-6)
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        return arr
    FK = np.fft.fftshift(np.fft.fft2(arr.astype(np.float64)))
    out = np.real(np.fft.ifft2(np.fft.ifftshift(FK * m)))
    _log_gpr(f"[GPR] fk_filter: mask fill={float(np.mean(m)):.3f}  shape={arr.shape}")
    return out.astype(np.float32)


def fk_dip_filter(
    data: np.ndarray,
    dt_ns: float,
    dx_m: float,
    v_min: float = 0.0,
    v_max: float = 0.0,
    mode: str = "reject",
    taper: float = 0.15,
) -> np.ndarray:
    """Filtro a ventaglio in FK per velocità apparente ``v_app = f / kx``.

    Rimuove (``mode='reject'``) o isola (``mode='pass'``) gli eventi la cui
    pendenza apparente cade in ``[v_min, v_max]`` m/s. Tipico per togliere
    onda diretta / ground-roll / rumore inclinato coerente. ``v_min<=0`` →
    0 ; ``v_max<=0`` → +inf. ``taper`` = frazione di transizione morbida.
    """
    if data is None:
        return np.zeros((0, 0), dtype=np.float32)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return arr.copy()
    arr = np.nan_to_num(arr, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    n_s, n_t = arr.shape
    if n_s < 4 or n_t < 4:
        return arr
    dt_s = float(dt_ns) * 1e-9
    dx = max(float(dx_m), 1e-6)
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        return arr
    vlo = max(0.0, float(v_min))
    vhi = float(v_max) if float(v_max) > 0.0 else np.inf

    _, f, kx = _fk_transform(arr.astype(np.float64), dt_s, dx)
    F, KX = np.meshgrid(np.abs(f), np.abs(kx), indexing="ij")
    with np.errstate(divide="ignore", invalid="ignore"):
        v_app = np.where(KX > 1e-12, F / KX, np.inf)   # m/s
    inband = (v_app >= vlo) & (v_app <= vhi)
    keep = (~inband) if str(mode).strip().lower() == "reject" else inband
    mask = keep.astype(np.float64)
    # taper morbido sul bordo della banda in scala log-velocità
    tf = float(np.clip(taper, 0.0, 0.5))
    if tf > 0.0 and np.isfinite(vhi):
        edge = tf * (np.log(max(vhi, 1e-6)) - np.log(max(vlo, 1e-6) + 1e-6))
        if edge > 0:
            dist = np.minimum(
                np.abs(np.log(np.clip(v_app, 1e-6, None)) - np.log(max(vlo, 1e-6) + 1e-6)),
                np.abs(np.log(np.clip(v_app, 1e-6, None)) - np.log(max(vhi, 1e-6))),
            )
            soft = 0.5 * (1.0 - np.cos(np.pi * np.clip(dist / edge, 0.0, 1.0)))
            near = dist < edge
            mask[near] = soft[near] if str(mode).strip().lower() == "reject" else (1.0 - soft[near])

    FK = np.fft.fftshift(np.fft.fft2(arr.astype(np.float64)))
    out = np.real(np.fft.ifft2(np.fft.ifftshift(FK * mask)))
    _log_gpr(
        f"[GPR] fk_dip_filter: v=[{vlo:.3g},{vhi:.3g}] mode={mode} "
        f"pass_frac={float(np.mean(mask)):.3f}  shape={arr.shape}"
    )
    return out.astype(np.float32)


def kirchhoff_migration_2d(
    data: np.ndarray,
    dt_ns: float,
    dx_m: float = 0.05,
    velocity_m_s: float = 1.0e8,
    aperture_m: float = 0.0,
    tr_offset_m: float = 0.0,
) -> np.ndarray:
    """Migrazione Kirchhoff / diffraction-stack 2D a tempi.

    Per ogni campione di uscita ``(i0, j0)`` somma le ampiezze del dato lungo
    l'iperbole di diffrazione entro l'apertura, con pesi di obliquità
    ``cos θ`` e di spreading sferico ``1/sqrt(v t)``. ``velocity_m_s`` può
    essere uno scalare **oppure** un array 1D lungo ``n_samples`` (profilo di
    velocità variabile con la profondità → copre ``migration_v_profile``).
    ``tr_offset_m`` = separazione fissa Tx-Rx (modello bistatico).
    ``aperture_m`` <= 0 → apertura automatica (mezza traccia).

    Adattato concettualmente da RGPR ``migration(type="kirchhoff")``, GPL (>= 2),
    e da GPR standard (diffraction stack) — reimplementato da zero.
    """
    if data is None:
        return np.zeros((0, 0), dtype=np.float32)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return arr.copy()
    arr = np.nan_to_num(arr, copy=True, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
    n_s, n_t = arr.shape
    if n_s < 4 or n_t < 4:
        return arr.astype(np.float32)

    dt = float(dt_ns) * 1e-9
    dx = max(float(dx_m), 1e-4)
    if not np.isfinite(dt) or dt <= 0.0:
        return arr.astype(np.float32)

    v = np.asarray(velocity_m_s, dtype=np.float64)
    if v.ndim == 0:
        v_vec = np.full(n_s, max(float(v), 1e3))
    else:
        v_vec = np.interp(
            np.arange(n_s, dtype=np.float64),
            np.linspace(0.0, n_s - 1, v.size),
            np.clip(v.astype(np.float64), 1e3, None),
        )
    half = float(tr_offset_m) * 0.5
    ap = float(aperture_m) if float(aperture_m) > 0.0 else (0.5 * n_t * dx)
    max_off = max(1, int(round(ap / dx)))

    t_axis = np.arange(n_s, dtype=np.float64) * dt
    out = np.zeros_like(arr)
    jj = np.arange(n_t)
    for i0 in range(n_s):
        t0 = t_axis[i0]
        if t0 <= 0.0:
            out[i0, :] = arr[i0, :]
            continue
        vi = v_vec[i0]
        depth = 0.5 * vi * t0                       # profondità apparente all'apice
        for off in range(-max_off, max_off + 1):
            src_j = jj + off
            m = (src_j >= 0) & (src_j < n_t)
            if not np.any(m):
                continue
            xr = off * dx
            r1 = np.hypot(xr - half, depth)
            r2 = np.hypot(xr + half, depth)
            t_hyp = (r1 + r2) / vi                  # tempo bistatico a due vie
            s = t_hyp / dt
            if s > (n_s - 1) or s < 0.0:
                continue
            s0 = int(np.floor(s))
            fr = s - s0
            s1 = min(s0 + 1, n_s - 1)
            dist_mean = 0.5 * (r1 + r2)
            obliq = depth / dist_mean if dist_mean > 1e-9 else 0.0
            spread = 1.0 / np.sqrt(max(vi * t_hyp, 1e-12))
            w = obliq * spread
            if w <= 0.0:
                continue
            a = arr[s0, src_j[m]] * (1.0 - fr) + arr[s1, src_j[m]] * fr
            out[i0, jj[m]] += w * a
    # riscala per riportare l'ampiezza nell'ordine del dato d'ingresso
    peak_in = float(np.max(np.abs(arr))) or 1.0
    peak_out = float(np.max(np.abs(out))) or 1.0
    if peak_out > 1e-20:
        out *= peak_in / peak_out
    _log_gpr(
        f"[GPR] kirchhoff_migration_2d: v={'array' if v.ndim else f'{float(v):.3e}'} "
        f"dx={dx:.3f} aperture={ap:.2f}m txrx={tr_offset_m:.3f}  shape={arr.shape}"
    )
    return out.astype(np.float32)


def kirchhoff_migration_topo(
    data: np.ndarray,
    dt_ns: float,
    dx_m: float = 0.05,
    velocity_m_s: float = 1.0e8,
    elev_m=None,
    datum_m: float | None = None,
    aperture_m: float = 0.0,
    tr_offset_m: float = 0.0,
) -> np.ndarray:
    """Migrazione Kirchhoff 2D con **correzione topografica**.

    ``elev_m`` : quota della superficie per ogni traccia (array lungo n_traces).
    Ogni traccia ha la sua superficie a profondità ``d_j = datum - elev_j`` sotto
    un piano di riferimento orizzontale (``datum_m``, default = quota massima).
    L'uscita è riferita al **datum**: la riga ``i0`` corrisponde alla profondità
    ``z = i0 * v*dt/2`` sotto il datum (non sotto la superficie locale). Le
    porzioni sopra la superficie di una traccia non contribuiscono.

    Con ``elev_m`` costante o assente collassa in ``kirchhoff_migration_2d``.
    """
    if data is None:
        return np.zeros((0, 0), dtype=np.float32)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return arr.copy()
    arr = np.nan_to_num(arr, copy=True, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
    n_s, n_t = arr.shape

    elev = np.asarray(elev_m, dtype=np.float64).ravel() if elev_m is not None else None
    _fin = elev[np.isfinite(elev)] if elev is not None else np.zeros(0)
    if elev is None or elev.size != n_t or _fin.size < 2 or float(_fin.max() - _fin.min()) < 1e-6:
        # niente topografia utile → migrazione piana standard
        return kirchhoff_migration_2d(arr, dt_ns, dx_m, velocity_m_s, aperture_m, tr_offset_m)

    if n_s < 4 or n_t < 4:
        return arr.astype(np.float32)
    dt = float(dt_ns) * 1e-9
    dx = max(float(dx_m), 1e-4)
    if not np.isfinite(dt) or dt <= 0.0:
        return arr.astype(np.float32)

    v = np.asarray(velocity_m_s, dtype=np.float64)
    v_scalar = float(v) if v.ndim == 0 else float(np.median(np.clip(v, 1e3, None)))
    v_scalar = max(v_scalar, 1e3)

    elev = np.nan_to_num(elev, nan=float(np.nanmedian(elev)))
    datum = float(datum_m) if datum_m is not None else float(np.max(elev))
    d_surf = np.clip(datum - elev, 0.0, None)          # (n_t,) prof. superficie sotto datum
    dz = 0.5 * v_scalar * dt                            # incremento profondità per riga

    half = float(tr_offset_m) * 0.5
    ap = float(aperture_m) if float(aperture_m) > 0.0 else (0.5 * n_t * dx)
    max_off = max(1, int(round(ap / dx)))

    out = np.zeros_like(arr)
    j0 = np.arange(n_t)
    for i0 in range(n_s):
        z0 = i0 * dz                                    # profondità sotto datum
        for off in range(-max_off, max_off + 1):
            src = j0 + off
            m = (src >= 0) & (src < n_t)
            if not np.any(m):
                continue
            src_c = np.clip(src, 0, n_t - 1)
            zloc = z0 - d_surf[src_c]                   # prof. riflettore sotto la superficie di src
            valid = m & (zloc > 0.0)
            if not np.any(valid):
                continue
            xr = off * dx
            r1 = np.hypot(xr - half, zloc)
            r2 = np.hypot(xr + half, zloc)
            t_hyp = (r1 + r2) / v_scalar
            s = t_hyp / dt
            valid &= (s >= 0.0) & (s <= (n_s - 1))
            if not np.any(valid):
                continue
            s0 = np.floor(s).astype(np.int64)
            fr = s - s0
            s0c = np.clip(s0, 0, n_s - 1)
            s1c = np.clip(s0 + 1, 0, n_s - 1)
            a = arr[s0c, src_c] * (1.0 - fr) + arr[s1c, src_c] * fr
            dist_mean = 0.5 * (r1 + r2)
            with np.errstate(divide="ignore", invalid="ignore"):
                obliq = np.where(dist_mean > 1e-9, zloc / np.maximum(dist_mean, 1e-12), 0.0)
                spread = 1.0 / np.sqrt(np.maximum(v_scalar * t_hyp, 1e-12))
            w = np.nan_to_num(obliq * spread, nan=0.0, posinf=0.0, neginf=0.0)
            contrib = np.where(valid, w * a, 0.0)
            out[i0, :] += contrib
    peak_in = float(np.max(np.abs(arr))) or 1.0
    peak_out = float(np.max(np.abs(out))) or 1.0
    if peak_out > 1e-20:
        out *= peak_in / peak_out
    _log_gpr(
        f"[GPR] kirchhoff_migration_topo: v={v_scalar:.3e} dx={dx:.3f} "
        f"datum={datum:.3f} dz_surf=[{d_surf.min():.3f},{d_surf.max():.3f}]m  "
        f"shape={arr.shape}"
    )
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Fase 6 — attributi istantanei, NMO / analisi velocità CMP, blending multi-freq
# ---------------------------------------------------------------------------

def _analytic_signal(arr64: np.ndarray) -> np.ndarray:
    """Segnale analitico complesso lungo l'asse campioni (axis 0), a blocchi."""
    n_s, n_t = arr64.shape
    h = np.zeros(n_s, dtype=np.float64)
    if n_s % 2 == 0:
        h[0] = h[n_s // 2] = 1.0
        h[1:n_s // 2] = 2.0
    else:
        h[0] = 1.0
        h[1:(n_s + 1) // 2] = 2.0
    out = np.empty((n_s, n_t), dtype=np.complex128)
    cols = max(1, min(n_t, int((64 * 1024 * 1024) // max(1, 3 * 16 * n_s))))
    for a in range(0, n_t, cols):
        b = min(n_t, a + cols)
        blk = np.ascontiguousarray(arr64[:, a:b].T, dtype=np.float64)
        Xf = np.fft.fft(blk, axis=1)
        out[:, a:b] = np.fft.ifft(Xf * h.reshape(1, -1), axis=1).T
    return out


def instantaneous_attributes(
    data: np.ndarray,
    dt_ns: float,
    kind: str = "phase",
) -> np.ndarray:
    """Attributi complessi dal segnale analitico (Hilbert).

    ``kind`` :
      * ``envelope``   → |z|  (come `hilbert_envelope`)
      * ``phase``      → fase istantanea avvolta [rad]
      * ``cos_phase``  → cos(fase): normalizza l'ampiezza, evidenzia la
                         continuità dei riflettori
      * ``frequency``  → frequenza istantanea [MHz] da d(fase)/dt (unwrap)

    Adattato concettualmente dagli attributi complessi (Taner et al.) e da RGPR
    ``Hilbert``, GPL (>= 2) — reimplementato con FFT numpy.
    """
    if data is None:
        return np.zeros((0, 0), dtype=np.float32)
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        return arr.astype(np.float32, copy=True)
    arr = np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    n_s, n_t = arr.shape
    if n_s <= 2 or n_t <= 0:
        return arr.astype(np.float32, copy=True)
    k = str(kind).strip().lower()
    z = _analytic_signal(arr)
    if k == "envelope":
        out = np.abs(z)
    elif k == "cos_phase":
        out = np.cos(np.angle(z))
    elif k == "frequency":
        ph = np.unwrap(np.angle(z), axis=0)
        dt_s = float(dt_ns) * 1e-9
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            dt_s = 1.0
        dphi = np.gradient(ph, axis=0)
        out = dphi / (2.0 * np.pi * dt_s) / 1e6           # MHz
        out = np.clip(out, 0.0, None)
    else:  # phase
        out = np.angle(z)
    _log_gpr(f"[GPR] instantaneous_attributes: kind={k}  shape={arr.shape}")
    return out.astype(np.float32)


def nmo_correction(
    data: np.ndarray,
    dt_ns: float,
    offsets_m,
    velocity_m_s: float,
    stretch_mute: float = 1.5,
) -> np.ndarray:
    """Correzione normal-moveout di un CMP/WARR gather.

    ``data`` : (n_samples, n_traces) con tracce a offset Tx-Rx crescente.
    ``offsets_m`` : scalare (incremento per traccia, offset[j] = j*val) oppure
                    array lungo n_traces.
    Per ogni campione di uscita ``t0`` si prende l'ampiezza in ingresso a
    ``t = sqrt(t0^2 + (x/v)^2)`` (interpolazione lineare). ``stretch_mute``
    azzera i campioni con stiramento ``(t - t0)/t0 > stretch_mute``.
    """
    if data is None:
        return np.zeros((0, 0), dtype=np.float32)
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        return arr.astype(np.float32, copy=True)
    arr = np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    n_s, n_t = arr.shape
    dt = float(dt_ns) * 1e-9
    v = max(float(velocity_m_s), 1e3)
    if n_s < 2 or n_t < 1 or not np.isfinite(dt) or dt <= 0.0:
        return arr.astype(np.float32)
    off = np.asarray(offsets_m, dtype=np.float64)
    if off.ndim == 0:
        off = np.arange(n_t, dtype=np.float64) * float(off)
    elif off.size != n_t:
        off = np.resize(off, n_t)
    t0 = np.arange(n_s, dtype=np.float64) * dt
    out = np.zeros_like(arr)
    src_grid = np.arange(n_s, dtype=np.float64)
    for j in range(n_t):
        x = off[j]
        t = np.sqrt(t0 ** 2 + (x / v) ** 2)
        s = t / dt
        vals = np.interp(s, src_grid, arr[:, j], left=0.0, right=0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            stretch = np.where(t0 > 0.0, (t - t0) / t0, 0.0)
        vals[stretch > float(stretch_mute)] = 0.0
        out[:, j] = vals
    _log_gpr(
        f"[GPR] nmo_correction: v={v:.3e} n_traces={n_t} "
        f"off=[{off.min():.3g},{off.max():.3g}]m  shape={arr.shape}"
    )
    return out.astype(np.float32)


def cmp_semblance(
    gather: np.ndarray,
    dt_ns: float,
    offsets_m,
    v_min: float = 3.0e7,
    v_max: float = 1.5e8,
    n_v: int = 60,
    win_samples: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Pannello di semblance per l'analisi di velocità di un CMP gather.

    Ritorna ``(semblance, v_axis)`` con ``semblance`` di shape (n_samples, n_v):
    per ogni velocità candidata si applica la NMO e si misura la coerenza
    (semblance normalizzata) su una finestra di ``win_samples`` campioni.
    Il massimo lungo ``v`` a un dato ``t0`` stima la velocità RMS a quel tempo.
    """
    g = np.asarray(gather, dtype=np.float64)
    if g.ndim != 2 or g.size == 0:
        return np.zeros((0, 0), dtype=np.float32), np.zeros(0, dtype=np.float64)
    g = np.nan_to_num(g, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    n_s, n_t = g.shape
    v_axis = np.linspace(float(v_min), float(v_max), max(2, int(n_v)))
    half = max(1, int(win_samples) // 2)
    kern = np.ones(2 * half + 1, dtype=np.float64)
    sem = np.zeros((n_s, v_axis.size), dtype=np.float64)
    for iv, v in enumerate(v_axis):
        nmo = np.asarray(
            nmo_correction(g, dt_ns, offsets_m, float(v), stretch_mute=1.0),
            dtype=np.float64,
        )
        num = np.convolve(np.sum(nmo, axis=1) ** 2, kern, mode="same")
        den = np.convolve(np.sum(nmo ** 2, axis=1), kern, mode="same") * max(n_t, 1)
        sem[:, iv] = np.divide(num, den, out=np.zeros(n_s), where=den > 1e-20)
    _log_gpr(
        f"[GPR] cmp_semblance: n_v={v_axis.size} v=[{v_min:.2e},{v_max:.2e}] "
        f"win={2 * half + 1}  shape={g.shape}"
    )
    return sem.astype(np.float32), v_axis


def multifreq_blend(
    radargrams: list,
    weights=None,
    dt_ns_list=None,
    t0_shift_samples=None,
    target: int = 0,
) -> np.ndarray:
    """Fonde radargrammi di antenne a frequenza diversa in un'unica sezione.

    Ogni radargramma viene ricampionato sull'asse tempi del ``target`` (indice
    nella lista), shiftato di ``t0_shift_samples[i]`` campioni per allineare il
    tempo 0, quindi sommato con ``weights[i]`` (default: pesi uguali che sommano
    a 1). Le tracce devono avere lo stesso ``n_traces``.
    """
    rgs = [np.asarray(r, dtype=np.float64) for r in (radargrams or [])]
    rgs = [np.nan_to_num(r) for r in rgs if r.ndim == 2 and r.size > 0]
    if not rgs:
        return np.zeros((0, 0), dtype=np.float32)
    ti = int(target) if 0 <= int(target) < len(rgs) else 0
    n_s_t, n_t = rgs[ti].shape
    n_r = len(rgs)
    if weights is None:
        weights = np.full(n_r, 1.0 / n_r)
    w = np.asarray(weights, dtype=np.float64)
    if w.size != n_r:
        w = np.resize(w, n_r)
    dts = (
        np.asarray(dt_ns_list, dtype=np.float64)
        if dt_ns_list is not None else np.full(n_r, 1.0)
    )
    if dts.size != n_r:
        dts = np.resize(dts, n_r)
    shifts = (
        np.asarray(t0_shift_samples, dtype=np.int64)
        if t0_shift_samples is not None else np.zeros(n_r, dtype=np.int64)
    )
    if shifts.size != n_r:
        shifts = np.resize(shifts, n_r)

    dt_target = float(dts[ti]) if float(dts[ti]) > 0 else 1.0
    out = np.zeros((n_s_t, n_t), dtype=np.float64)
    t_target = np.arange(n_s_t, dtype=np.float64) * dt_target
    for i, r in enumerate(rgs):
        n_si = r.shape[0]
        if r.shape[1] != n_t:
            r = r[:, :n_t] if r.shape[1] > n_t else np.pad(r, ((0, 0), (0, n_t - r.shape[1])))
        dti = float(dts[i]) if float(dts[i]) > 0 else 1.0
        src_t = np.arange(n_si, dtype=np.float64) * dti
        res = np.empty((n_s_t, n_t), dtype=np.float64)
        for j in range(n_t):
            res[:, j] = np.interp(t_target, src_t, r[:, j], left=0.0, right=0.0)
        sh = int(shifts[i])
        if sh > 0:
            res = np.vstack([res[sh:, :], np.zeros((sh, n_t))])
        elif sh < 0:
            res = np.vstack([np.zeros((-sh, n_t)), res[:sh, :]])
        out += w[i] * res
    _log_gpr(
        f"[GPR] multifreq_blend: {n_r} radargrammi  weights={np.round(w, 3).tolist()}  "
        f"target={ti}  shape={out.shape}"
    )
    return out.astype(np.float32)


def normalize_display(data: np.ndarray) -> np.ndarray:
    if data is None:
        return np.zeros((0, 0), dtype=np.float32)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return arr.copy()
    arr = np.nan_to_num(arr, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    finite = np.isfinite(arr)
    if finite.any():
        vmax = _normalize_display_vmax(np.abs(arr[finite]))
    else:
        vmax = 0.0
    if vmax < 1e-12:
        if finite.any():
            mn = float(np.nanmin(arr[finite]))
            mx = float(np.nanmax(arr[finite]))
        else:
            mn = 0.0
            mx = 0.0
        _log_gpr(
            "[GPR] ATTENZIONE: dati nulli dopo processing. "
            f"Shape={arr.shape}  dtype={arr.dtype}  "
            f"min={mn:.4g}  max={mx:.4g}",
            level=(Qgis.Warning if Qgis is not None else None),
        )
        return np.zeros_like(arr)
    return np.clip(arr / vmax, -1.0, 1.0).astype(np.float32)


def _normalize_display_vmax(arr_abs_finite: np.ndarray) -> float:
    """Percentile-based vmax: robusto agli spike isolati."""
    if arr_abs_finite.size == 0:
        return 0.0
    p98 = float(np.percentile(arr_abs_finite, 98.0))
    # Se il 98° percentile è quasi zero (dato piatto), usa il massimo assoluto
    # come fallback per non perdere segnali molto deboli ma reali.
    if p98 < 1e-12:
        return float(np.max(arr_abs_finite))
    return p98


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

DEFAULT_PIPELINE = {
    "dewow":                True,
    "dewow_win":            16,
    "timezero":             False,
    "tz_method":            "peak",
    "tz_mode":              "line_by_line",
    "tz_threshold":         0.2,
    "tz_backup_nsamp":      4,
    "bg_removal":           True,
    "bg_mode":              "line_by_line",
    "bg_method":            "mean",       # 'mean' | 'median'
    "bg_window":            0,
    "bg_sample_start":      0,            # 0 = dall'inizio
    "bg_sample_end":        0,            # 0 = auto (intera traccia)
    "bandpass":             False,
    "bp_low_mhz":           100.0,
    "bp_high_mhz":          1200.0,
    "bp_taper_frac":        0.12,         # roll-off coseno per lato (0=netto, 0.5=Hann)
    "migration":            False,
    "mig_velocity_m_s":     100000000.0,  # 0.10 m/ns
    "mig_dx_m":             0.05,
    "pre_agc_gain":         False,
    "pre_agc_surface_gain": 1.0,
    "pre_agc_deep_gain":    1.0,
    "pre_agc_breakpoints":  None,         # [[x(0..1), gain], ...]
    "agc":                  True,
    "agc_win":              128,
    "agc_sample_start":     0,            # 0 = dall'inizio; >0 = lascia invariati i primi N campioni
    "agc_max_gain":         0.0,          # 0 = unlimited
    "envelope":             False,
    # --- Filtri Fase 1 (disattivi di default) -----------------------------
    "dcshift":              False,
    "dcshift_sample_start": 0,
    "dcshift_sample_end":   0,            # 0 = intera traccia
    "boxcar":               False,
    "boxcar_width":         3,            # semi-ampiezza in tracce (0 = off)
    "boxcar_height":        3,            # semi-ampiezza in campioni (0 = off)
    "notch":                False,
    "notch_freq_mhz":       0.0,          # 0 = disattivo
    "notch_bw_mhz":         50.0,
    "whiten":               False,
    "whiten_low_mhz":       0.0,          # 0/0 = intero spettro disponibile
    "whiten_high_mhz":      0.0,
    "whiten_gain":          20.0,
    "amp_eq":               False,
    "amp_eq_window":        120,
    "amp_eq_gain":          1.0,
    "rad_threshold":        False,
    "rad_thr_neg":          0.0,          # frazione (0..1) di max|amp|
    "rad_thr_pos":          0.0,
    "power_gain":           False,
    "power_gain_mode":      "power",      # 'power' | 'exp'
    "power_gain_exp":       1.0,
    "power_gain_t0_ns":     0.0,
    "power_gain_scale":     1.0,
    # --- Filtri Fase 4 (disattivi di default) ----------------------------
    "eigen_filter":         False,
    "eigen_drop":           1,            # n. componenti SVD azzerate
    "eigen_keep_last":      0,            # 0 = tieni tutte oltre `drop`
    "median2d":             False,
    "median2d_w_trace":     1,
    "median2d_w_sample":    1,
    "fk_dip":               False,
    "fk_dip_vmin":          0.0,          # m/s (0 = 0)
    "fk_dip_vmax":          0.0,          # m/s (0 = +inf)
    "fk_dip_mode":          "reject",     # 'reject' | 'pass'
    "fk_dip_taper":         0.15,
    "fk_dip_dx_m":          0.05,         # spaziatura tracce (m)
    "kirchhoff":            False,
    "kirchhoff_velocity_m_s": 1.0e8,
    "kirchhoff_dx_m":       0.05,
    "kirchhoff_aperture_m": 0.0,          # 0 = auto (mezza traccia)
    "kirchhoff_txrx_m":     0.0,
    "kirchhoff_topo":       False,        # correzione topografica (serve kirchhoff_elev_m)
    "kirchhoff_elev_m":     None,         # array quota per traccia (iniettato dal viewer)
    # --- Filtri Fase 6 (disattivi di default) ---------------------------
    "inst_attr":            False,
    "inst_attr_kind":       "phase",      # phase | cos_phase | frequency | envelope
    "chain_order":          None,
}


FILTER_REGISTRY = {
    "timezero": {
        "label": "Tempo zero",
        "param_key": "timezero",
    },
    "dewow": {
        "label": "Wobble removal (dewow)",
        "param_key": "dewow",
    },
    "bandpass": {
        "label": "Filtro passa-banda",
        "param_key": "bandpass",
    },
    "bg_removal": {
        "label": "Rimozione rumore di fondo",
        "param_key": "bg_removal",
    },
    "migration": {
        "label": "Migrazione 2D (Stolt F-K)",
        "param_key": "migration",
    },
    "pre_agc_gain": {
        "label": "Range gain (TVG)",
        "param_key": "pre_agc_gain",
    },
    "agc": {
        "label": "Guadagno automatico (AGC)",
        "param_key": "agc",
    },
    "envelope": {
        "label": "Inviluppo (Hilbert)",
        "param_key": "envelope",
    },
    "dcshift": {
        "label": "Rimozione DC (dcshift)",
        "param_key": "dcshift",
    },
    "boxcar": {
        "label": "Lisciatura boxcar 2D",
        "param_key": "boxcar",
    },
    "notch": {
        "label": "Filtro notch",
        "param_key": "notch",
    },
    "whiten": {
        "label": "Whitening spettrale",
        "param_key": "whiten",
    },
    "amp_eq": {
        "label": "Equalizzazione d'ampiezza",
        "param_key": "amp_eq",
    },
    "rad_threshold": {
        "label": "Soglia ampiezza (threshold)",
        "param_key": "rad_threshold",
    },
    "power_gain": {
        "label": "Range gain analitico (t^a / exp)",
        "param_key": "power_gain",
    },
    "eigen_filter": {
        "label": "Filtro eigen-image (SVD)",
        "param_key": "eigen_filter",
    },
    "median2d": {
        "label": "Mediana 2D (de-spike)",
        "param_key": "median2d",
    },
    "fk_dip": {
        "label": "Filtro FK a ventaglio (dip)",
        "param_key": "fk_dip",
    },
    "kirchhoff": {
        "label": "Migrazione Kirchhoff 2D",
        "param_key": "kirchhoff",
    },
    "inst_attr": {
        "label": "Attributo istantaneo (Hilbert)",
        "param_key": "inst_attr",
    },
}

DEFAULT_CHAIN_ORDER = [
    "dewow",
    "timezero",
    "dcshift",
    "bg_removal",
    "eigen_filter",
    "notch",
    "bandpass",
    "fk_dip",
    "whiten",
    "boxcar",
    "median2d",
    "migration",
    "kirchhoff",
    "pre_agc_gain",
    "power_gain",
    "agc",
    "amp_eq",
    "rad_threshold",
    "envelope",
    "inst_attr",
]


def _resolve_chain_order(chain_order, params: dict) -> list[str]:
    """Resolve pipeline order from explicit arg or params['chain_order']."""
    raw = chain_order if chain_order is not None else params.get("chain_order")
    if raw is None:
        return list(DEFAULT_CHAIN_ORDER)
    if isinstance(raw, (list, tuple)):
        seen = set()
        out = []
        for step in raw:
            sid = str(step or "").strip()
            if sid in FILTER_REGISTRY and sid not in seen:
                out.append(sid)
                seen.add(sid)
        for sid in DEFAULT_CHAIN_ORDER:
            if sid not in seen:
                out.append(sid)
        return out
    return list(DEFAULT_CHAIN_ORDER)


def _step_fingerprint(
    step: str,
    params: dict,
    input_shape: tuple,
    dt_ns: float,
    bg_ref_present: bool = False,
) -> tuple:
    p = params
    if step == "dewow":
        return ("dewow", bool(p.get("dewow")), int(p.get("dewow_win", 16)), input_shape)
    elif step == "timezero":
        return (
            "timezero", bool(p.get("timezero")), str(p.get("tz_method", "peak")),
            str(p.get("tz_mode", "line_by_line")), float(p.get("tz_threshold", 0.2)),
            int(p.get("tz_backup_nsamp", 4)), input_shape
        )
    elif step == "bg_removal":
        return (
            "bg_removal", bool(p.get("bg_removal")), str(p.get("bg_mode", "line_by_line")),
            str(p.get("bg_method", "mean")), int(p.get("bg_window", 0)),
            int(p.get("bg_sample_start", 0)), int(p.get("bg_sample_end", 0)),
            bg_ref_present, input_shape
        )
    elif step == "bandpass":
        return (
            "bandpass", bool(p.get("bandpass")), float(p.get("bp_low_mhz", 100.0)),
            float(p.get("bp_high_mhz", 1200.0)), float(p.get("bp_taper_frac", 0.12)),
            int(p.get("agc_sample_start", 0)), dt_ns, input_shape
        )
    elif step == "migration":
        return (
            "migration", bool(p.get("migration")), float(p.get("mig_velocity_m_s", 1e8)),
            float(p.get("mig_dx_m", 0.05)), dt_ns, input_shape
        )
    elif step == "pre_agc_gain":
        return (
            "pre_agc_gain", bool(p.get("pre_agc_gain")), float(p.get("pre_agc_surface_gain", 1.0)),
            float(p.get("pre_agc_deep_gain", 1.0)), str(p.get("pre_agc_breakpoints")), input_shape
        )
    elif step == "agc":
        return (
            "agc", bool(p.get("agc")), int(p.get("agc_win", 128)),
            int(p.get("agc_sample_start", 0)), float(p.get("agc_max_gain", 0.0)), input_shape
        )
    elif step == "envelope":
        return ("envelope", bool(p.get("envelope")), input_shape)
    elif step == "dcshift":
        return (
            "dcshift", bool(p.get("dcshift")), int(p.get("dcshift_sample_start", 0)),
            int(p.get("dcshift_sample_end", 0)), input_shape
        )
    elif step == "boxcar":
        return (
            "boxcar", bool(p.get("boxcar")), int(p.get("boxcar_width", 3)),
            int(p.get("boxcar_height", 3)), input_shape
        )
    elif step == "notch":
        return (
            "notch", bool(p.get("notch")), float(p.get("notch_freq_mhz", 0.0)),
            float(p.get("notch_bw_mhz", 50.0)), dt_ns, input_shape
        )
    elif step == "whiten":
        return (
            "whiten", bool(p.get("whiten")), float(p.get("whiten_low_mhz", 0.0)),
            float(p.get("whiten_high_mhz", 0.0)), float(p.get("whiten_gain", 20.0)),
            dt_ns, input_shape
        )
    elif step == "amp_eq":
        return (
            "amp_eq", bool(p.get("amp_eq")), int(p.get("amp_eq_window", 120)),
            float(p.get("amp_eq_gain", 1.0)), input_shape
        )
    elif step == "rad_threshold":
        return (
            "rad_threshold", bool(p.get("rad_threshold")),
            float(p.get("rad_thr_neg", 0.0)), float(p.get("rad_thr_pos", 0.0)), input_shape
        )
    elif step == "power_gain":
        return (
            "power_gain", bool(p.get("power_gain")), str(p.get("power_gain_mode", "power")),
            float(p.get("power_gain_exp", 1.0)), float(p.get("power_gain_t0_ns", 0.0)),
            float(p.get("power_gain_scale", 1.0)), dt_ns, input_shape
        )
    elif step == "eigen_filter":
        return (
            "eigen_filter", bool(p.get("eigen_filter")), int(p.get("eigen_drop", 1)),
            int(p.get("eigen_keep_last", 0)), input_shape
        )
    elif step == "median2d":
        return (
            "median2d", bool(p.get("median2d")), int(p.get("median2d_w_trace", 1)),
            int(p.get("median2d_w_sample", 1)), input_shape
        )
    elif step == "fk_dip":
        return (
            "fk_dip", bool(p.get("fk_dip")), float(p.get("fk_dip_vmin", 0.0)),
            float(p.get("fk_dip_vmax", 0.0)), str(p.get("fk_dip_mode", "reject")),
            float(p.get("fk_dip_taper", 0.15)), float(p.get("fk_dip_dx_m", 0.05)),
            dt_ns, input_shape
        )
    elif step == "kirchhoff":
        _elev = p.get("kirchhoff_elev_m")
        _elev_sig = 0
        if _elev is not None:
            try:
                _ea = np.asarray(_elev, dtype=np.float64).ravel()
                _elev_sig = (int(_ea.size), round(float(_ea.sum()), 3),
                             round(float(_ea.max() - _ea.min()), 4)) if _ea.size else 0
            except Exception:
                _elev_sig = 0
        return (
            "kirchhoff", bool(p.get("kirchhoff")),
            float(p.get("kirchhoff_velocity_m_s", 1.0e8)),
            float(p.get("kirchhoff_dx_m", 0.05)),
            float(p.get("kirchhoff_aperture_m", 0.0)),
            float(p.get("kirchhoff_txrx_m", 0.0)),
            bool(p.get("kirchhoff_topo", False)), _elev_sig, dt_ns, input_shape
        )
    elif step == "inst_attr":
        return (
            "inst_attr", bool(p.get("inst_attr")),
            str(p.get("inst_attr_kind", "phase")), dt_ns, input_shape
        )
    return (step, bool(p.get(step)), input_shape)


def _sanitize_gain_breakpoints(points) -> np.ndarray:
    rows = []
    if isinstance(points, np.ndarray):
        pts_iter = points.tolist()
    elif isinstance(points, (list, tuple)):
        pts_iter = points
    else:
        pts_iter = []
    for it in pts_iter:
        try:
            x = float(it[0])
            y = float(it[1])
        except Exception:
            continue
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        rows.append(
            (
                float(np.clip(x, 0.0, 1.0)),
                float(np.clip(y, 0.0, 100.0)),
            )
        )
    if len(rows) < 2:
        rows = [(0.0, 1.0), (1.0, 1.0)]
    arr = np.asarray(rows, dtype=np.float64)
    arr = arr[np.argsort(arr[:, 0], kind="mergesort")]
    x = arr[:, 0]
    keep = np.concatenate(([True], np.diff(x) > 1e-9))
    arr = arr[keep]
    if arr[0, 0] > 0.0:
        arr = np.vstack([[0.0, arr[0, 1]], arr])
    else:
        arr[0, 0] = 0.0
    if arr[-1, 0] < 1.0:
        arr = np.vstack([arr, [1.0, arr[-1, 1]]])
    else:
        arr[-1, 0] = 1.0
    return arr


def _build_pre_agc_gain(n_samples: int, p: dict) -> np.ndarray:
    n = max(1, int(n_samples))
    g0 = max(float(p.get("pre_agc_surface_gain", 1.0) or 1.0), 0.0)
    g1 = max(float(p.get("pre_agc_deep_gain", 1.0) or 1.0), 0.0)
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)
    pts = _sanitize_gain_breakpoints(p.get("pre_agc_breakpoints"))
    if pts.shape[0] >= 2:
        g = np.interp(t, pts[:, 0], pts[:, 1])
    else:
        g = g0 + (g1 - g0) * t
    g = np.clip(g, 0.0, 1e6).astype(np.float32, copy=False)
    return g.reshape(-1, 1)


def apply_pre_bg_pipeline(
    data: np.ndarray,
    params: dict,
    dt_ns: float = 0.117,
) -> np.ndarray:
    """
    Applica solo le fasi che precedono il BG removal (dewow + time-zero).

    Usata per il calcolo della traccia di riferimento nel modo grid_by_grid
    a due passate:
      1a passata: apply_pre_bg_pipeline su tutti i profili -> media globale
      2a passata: apply_pipeline con bg_reference_trace = media globale
    """
    if data is None:
        return np.zeros((0, 0), dtype=np.float32)
    out = np.asarray(data, dtype=np.float32)
    if out.ndim != 2 or out.size == 0:
        return out.copy()
    out = np.nan_to_num(out, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    p   = {**DEFAULT_PIPELINE, **params}
    if p["dewow"]:
        out = dewow(out, window=int(p["dewow_win"]))
    if p["timezero"]:
        out = time_zero_correction(
            out,
            method       = str(p.get("tz_method",      "peak")),
            mode         = str(p.get("tz_mode",        "line_by_line")),
            threshold    = float(p.get("tz_threshold",  0.2)),
            backup_nsamp = int(p.get("tz_backup_nsamp", 4)),
        )
    return out


def apply_pipeline(
    data: np.ndarray,
    params: dict,
    dt_ns: float = 0.117,
    bg_reference_trace: np.ndarray | None = None,
    normalize_output: bool = True,
    chain_order: list[str] | None = None,
    stage_cache: dict | None = None,
) -> np.ndarray:
    """
    Applica la pipeline di processing completa.

    Parametri
    ----------
    data                : (n_samples, n_traces) float32
    params              : dict con chiavi da DEFAULT_PIPELINE
    dt_ns               : intervallo di campionamento in nanosecondi
    bg_reference_trace  : (n_samples,) traccia media dell'intero grid
    normalize_output    : normalizza a [-1, 1] se True
    chain_order         : ordine dei filtri (opzionale)
    stage_cache         : dizionario per memorizzazione intermedia dei passi (stage caching)
    """
    if data is None:
        return np.zeros((0, 0), dtype=np.float32)
    out = np.asarray(data, dtype=np.float32)
    if out.ndim != 2 or out.size == 0:
        return out.copy()
    out = np.nan_to_num(out, copy=True, nan=0.0, posinf=0.0, neginf=0.0)

    p = {**DEFAULT_PIPELINE, **params}
    order = _resolve_chain_order(chain_order, p)

    def _log(tag: str, arr: np.ndarray) -> None:
        if arr is None or arr.size == 0:
            return
        mn = float(arr.min())
        mx = float(arr.max())
        nz = int(np.count_nonzero(arr))
        _log_gpr(
            f"[GPR] {tag:40s}  min={mn:.4g}  max={mx:.4g}  "
            f"nonzero={nz}/{arr.size}  shape={arr.shape}"
        )

    _log("raw", out)
    _row_std = out.astype(np.float64).std(axis=1)
    _log_gpr(
        f"[GPR] raw_diag: mean_inter-trace_std={_row_std.mean():.6g}  "
        f"rows_zero_std={int(np.count_nonzero(_row_std == 0.0))}/{out.shape[0]}  "
        f"(0 std ovunque = tracce identiche gia' in input, bug a monte nel reader)"
    )

    can_use_cache = isinstance(stage_cache, dict)
    cache_valid = can_use_cache

    for idx, step in enumerate(order):
        fp = _step_fingerprint(
            step, p, out.shape, dt_ns, bg_ref_present=(bg_reference_trace is not None)
        )
        if cache_valid and idx in stage_cache:
            cached_fp, cached_out = stage_cache[idx]
            if cached_fp == fp and isinstance(cached_out, np.ndarray):
                out = cached_out
                continue

        cache_valid = False

        if step == "dewow":
            if p["dewow"]:
                out = dewow(out, window=int(p["dewow_win"]))
                _log(f"dewow(win={p['dewow_win']})", out)

        elif step == "timezero":
            if p["timezero"]:
                out = time_zero_correction(
                    out,
                    method=str(p.get("tz_method", "peak")),
                    mode=str(p.get("tz_mode", "line_by_line")),
                    threshold=float(p.get("tz_threshold", 0.2)),
                    backup_nsamp=int(p.get("tz_backup_nsamp", 4)),
                )
                _log("timezero", out)

        elif step == "bg_removal":
            if p["bg_removal"]:
                bg_mode = str(p.get("bg_mode", "line_by_line"))
                bg_method = str(p.get("bg_method", "mean"))
                bg_window = int(p.get("bg_window", 0))
                bg_sample_start = int(p.get("bg_sample_start", 0))
                bg_sample_end = int(p.get("bg_sample_end", 0))
                out = background_removal(
                    out,
                    mode=bg_mode,
                    window=bg_window,
                    reference_trace=bg_reference_trace,
                    sample_start=bg_sample_start,
                    sample_end=bg_sample_end,
                    method=bg_method,
                )
                win_lbl = "auto" if bg_window <= 0 else str(bg_window)
                s_lbl = f"{bg_sample_start}:{bg_sample_end if bg_sample_end > 0 else 'end'}"
                _log(f"bg_removal(mode={bg_mode} method={bg_method} win={win_lbl} s=[{s_lbl}])", out)

        elif step == "bandpass":
            if p["bandpass"]:
                bp_s0 = int(p.get("agc_sample_start", 0))
                if bp_s0 > 0 and out.shape[0] > bp_s0:
                    out = out.copy()
                    out[:bp_s0, :] = 0.0
                out = bandpass_filter(
                    out,
                    dt_ns,
                    float(p["bp_low_mhz"]),
                    float(p["bp_high_mhz"]),
                    taper_frac=float(p.get("bp_taper_frac", 0.12)),
                )
                _log(
                    f"bandpass({p['bp_low_mhz']}-{p['bp_high_mhz']}MHz "
                    f"taper={float(p.get('bp_taper_frac', 0.12)):.2f})",
                    out,
                )

        elif step == "migration":
            if bool(p.get("migration", False)):
                mig_v = float(p.get("mig_velocity_m_s", 1e8) or 1e8)
                mig_dx = float(p.get("mig_dx_m", 0.05) or 0.05)
                out = stolt_migration_2d(
                    out,
                    dt_ns=dt_ns,
                    dx_m=mig_dx,
                    velocity_m_s=mig_v,
                )
                _log(f"migration(v={mig_v:.3e} dx={mig_dx:.3f})", out)

        elif step == "pre_agc_gain":
            if bool(p.get("pre_agc_gain", False)):
                gain = _build_pre_agc_gain(int(out.shape[0]), p)
                out = (out.astype(np.float32, copy=False) * gain).astype(np.float32, copy=False)
                _log("pre_agc_gain", out)

        elif step == "agc":
            if p["agc"]:
                agc_s0 = int(p.get("agc_sample_start", 0))
                agc_max = float(p.get("agc_max_gain", 0.0) or 0.0)
                out = agc_gain(
                    out,
                    window=int(p["agc_win"]),
                    clip_percentile=99.5,
                    sample_start=agc_s0,
                    max_gain_factor=(agc_max if agc_max > 0 else None),
                )
                s0_lbl = f"s0={agc_s0}" if agc_s0 > 0 else ""
                _log(f"agc(win={p['agc_win']}{(' ' + s0_lbl) if s0_lbl else ''})", out)

        elif step == "envelope":
            if bool(p.get("envelope", False)):
                out = hilbert_envelope(out)
                _log("envelope(hilbert)", out)

        elif step == "dcshift":
            if bool(p.get("dcshift", False)):
                out = dcshift(
                    out,
                    sample_start=int(p.get("dcshift_sample_start", 0)),
                    sample_end=int(p.get("dcshift_sample_end", 0)),
                )
                _log("dcshift", out)

        elif step == "boxcar":
            if bool(p.get("boxcar", False)):
                out = boxcar_smooth(
                    out,
                    width=int(p.get("boxcar_width", 3)),
                    height=int(p.get("boxcar_height", 3)),
                )
                _log(f"boxcar(w={p.get('boxcar_width', 3)} h={p.get('boxcar_height', 3)})", out)

        elif step == "notch":
            if bool(p.get("notch", False)) and float(p.get("notch_freq_mhz", 0.0)) > 0.0:
                out = notch_filter(
                    out,
                    dt_ns,
                    float(p.get("notch_freq_mhz", 0.0)),
                    float(p.get("notch_bw_mhz", 50.0)),
                )
                _log(f"notch(f0={p.get('notch_freq_mhz')}MHz)", out)

        elif step == "whiten":
            if bool(p.get("whiten", False)):
                out = spectral_whitening(
                    out,
                    dt_ns,
                    float(p.get("whiten_low_mhz", 0.0)),
                    float(p.get("whiten_high_mhz", 0.0)),
                    float(p.get("whiten_gain", 20.0)),
                )
                _log("whiten", out)

        elif step == "amp_eq":
            if bool(p.get("amp_eq", False)):
                out = amplitude_equalization(
                    out,
                    window=int(p.get("amp_eq_window", 120)),
                    gain=float(p.get("amp_eq_gain", 1.0)),
                )
                _log(f"amp_eq(win={p.get('amp_eq_window', 120)})", out)

        elif step == "rad_threshold":
            if bool(p.get("rad_threshold", False)):
                out = radar_threshold(
                    out,
                    neg_fraction=float(p.get("rad_thr_neg", 0.0)),
                    pos_fraction=float(p.get("rad_thr_pos", 0.0)),
                )
                _log("rad_threshold", out)

        elif step == "power_gain":
            if bool(p.get("power_gain", False)):
                out = power_gain(
                    out,
                    dt_ns,
                    mode=str(p.get("power_gain_mode", "power")),
                    exponent=float(p.get("power_gain_exp", 1.0)),
                    t0_ns=float(p.get("power_gain_t0_ns", 0.0)),
                    scale=float(p.get("power_gain_scale", 1.0)),
                )
                _log(f"power_gain({p.get('power_gain_mode', 'power')})", out)

        elif step == "eigen_filter":
            if bool(p.get("eigen_filter", False)):
                out = eigen_filter(
                    out,
                    drop=int(p.get("eigen_drop", 1)),
                    keep_last=int(p.get("eigen_keep_last", 0)),
                )
                _log(f"eigen_filter(drop={p.get('eigen_drop', 1)})", out)

        elif step == "median2d":
            if bool(p.get("median2d", False)):
                out = median_filter2d(
                    out,
                    w_trace=int(p.get("median2d_w_trace", 1)),
                    w_sample=int(p.get("median2d_w_sample", 1)),
                )
                _log(
                    f"median2d(wt={p.get('median2d_w_trace', 1)} "
                    f"ws={p.get('median2d_w_sample', 1)})",
                    out,
                )

        elif step == "fk_dip":
            if bool(p.get("fk_dip", False)):
                out = fk_dip_filter(
                    out,
                    dt_ns,
                    float(p.get("fk_dip_dx_m", 0.05)),
                    v_min=float(p.get("fk_dip_vmin", 0.0)),
                    v_max=float(p.get("fk_dip_vmax", 0.0)),
                    mode=str(p.get("fk_dip_mode", "reject")),
                    taper=float(p.get("fk_dip_taper", 0.15)),
                )
                _log(f"fk_dip(mode={p.get('fk_dip_mode', 'reject')})", out)

        elif step == "kirchhoff":
            if bool(p.get("kirchhoff", False)):
                _kw = dict(
                    dt_ns=dt_ns,
                    dx_m=float(p.get("kirchhoff_dx_m", 0.05)),
                    velocity_m_s=float(p.get("kirchhoff_velocity_m_s", 1.0e8)),
                    aperture_m=float(p.get("kirchhoff_aperture_m", 0.0)),
                    tr_offset_m=float(p.get("kirchhoff_txrx_m", 0.0)),
                )
                if bool(p.get("kirchhoff_topo", False)) and p.get("kirchhoff_elev_m") is not None:
                    out = kirchhoff_migration_topo(out, elev_m=p.get("kirchhoff_elev_m"), **_kw)
                    _log("kirchhoff(topo)", out)
                else:
                    out = kirchhoff_migration_2d(out, **_kw)
                    _log(
                        f"kirchhoff(v={float(p.get('kirchhoff_velocity_m_s', 1e8)):.3e} "
                        f"dx={p.get('kirchhoff_dx_m', 0.05)})",
                        out,
                    )

        elif step == "inst_attr":
            if bool(p.get("inst_attr", False)):
                out = instantaneous_attributes(
                    out, dt_ns, str(p.get("inst_attr_kind", "phase"))
                )
                _log(f"inst_attr(kind={p.get('inst_attr_kind', 'phase')})", out)

        if can_use_cache:
            stage_cache[idx] = (fp, out.copy())

    if can_use_cache:
        for k in list(stage_cache.keys()):
            if isinstance(k, int) and k >= len(order):
                stage_cache.pop(k, None)

    if normalize_output:
        out = normalize_display(out)
        _log("normalize (finale)", out)
    else:
        out = out.astype(np.float32, copy=False)
        _log("finale (no normalize)", out)
    return out


# ---------------------------------------------------------------------------
# Auto-tune
# ---------------------------------------------------------------------------

def _peak_freq_mhz(freqs_mhz: np.ndarray, spectrum: np.ndarray, prior_mhz: float = 0.0) -> tuple[float, int]:
    """Trova il picco dello spettro, opzionalmente vincolato attorno a `prior_mhz`.

    Senza un prior (antenna sconosciuta) il picco puo' essere catturato da
    rumore a banda larga con energia superiore al vero segnale (comune con
    antenne a bassa frequenza / basso SNR). Se e' nota la frequenza nominale
    dell'antenna, la ricerca viene ristretta a [0.5x, 2.5x] quella frequenza:
    la banda -6dB tipica di un'antenna GPR impulsiva copre circa un'ottava
    intorno alla frequenza centrale nominale. Una finestra piu' larga
    (usata in precedenza: [0.25x, 4x], due ottave sopra il nominale) lascia
    troppo margine perche' rumore a banda larga/armoniche vengano scambiati
    per il vero picco del segnale, producendo un bandpass che taglia via il
    segnale reale e lascia passare solo rumore.
    """
    if prior_mhz and prior_mhz > 10.0:
        lo = prior_mhz * 0.5
        hi = prior_mhz * 2.5
        mask = (freqs_mhz >= lo) & (freqs_mhz <= hi)
        if mask.any():
            sub_idx = np.nonzero(mask)[0]
            local_idx = int(np.argmax(spectrum[sub_idx]))
            idx = int(sub_idx[local_idx])
            return float(freqs_mhz[idx]), idx
    idx = int(np.argmax(spectrum))
    return float(freqs_mhz[idx]) if idx < len(freqs_mhz) else 0.0, idx


def auto_tune_params(
    raw_data_list: list,
    dt_ns_list: list[float],
    antenna_freq_mhz: float = 0.0,
    max_profiles: int = 5,
) -> dict:
    """Analizza un campione di profili grezzi e suggerisce i parametri ottimali.

    Parametri
    ----------
    raw_data_list     : lista di np.ndarray (n_samples, n_traces) float32, profili grezzi
    dt_ns_list        : lista di float, dt in ns per ogni profilo
    antenna_freq_mhz  : frequenza nominale dell'antenna (0 se sconosciuta). Se
                        presente, ancora la ricerca del picco spettrale
                        attorno a questo valore invece di seguire il massimo
                        assoluto dello spettro (vulnerabile a rumore a banda
                        larga su antenne a bassa frequenza / basso SNR).
    max_profiles      : quanti profili usare al massimo (per velocità)

    Ritorna
    -------
    dict con le stesse chiavi di DEFAULT_PIPELINE + valori suggeriti.
    """
    if not raw_data_list:
        return dict(DEFAULT_PIPELINE)

    # Campiona al massimo max_profiles profili distribuiti uniformemente
    n = len(raw_data_list)
    if n <= max_profiles:
        indices = list(range(n))
    else:
        indices = [int(round(i * (n - 1) / (max_profiles - 1))) for i in range(max_profiles)]

    samples_list = [raw_data_list[i] for i in indices]
    dt_ns_sample = [dt_ns_list[i] for i in indices]

    # Accumula statistiche su tutti i profili campionati
    n_samples_all = []
    n_traces_all = []
    dt_ns_all = []
    energy_profiles = []   # profilo di energia media per sample (normalizzato)
    fft_spectra = []       # spettro di potenza medio per profilo

    for arr, dt in zip(samples_list, dt_ns_sample):
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 2 or arr.size == 0:
            continue
        ns, nt = arr.shape
        n_samples_all.append(ns)
        n_traces_all.append(nt)
        dt_ns_all.append(float(dt))

        # Profilo di energia per sample (media del quadrato su tutte le tracce)
        energy = np.mean(arr.astype(np.float64) ** 2, axis=1)
        peak = float(energy.max()) if energy.max() > 0 else 1.0
        energy_norm = energy / peak
        energy_profiles.append(energy_norm)

        # Stima agc_sample_start per questo profilo (fine del burst superficiale).
        # Serve per limitare il FFT alla zona utile sotto il burst.
        burst_end = 0
        threshold = 0.20
        consecutive = 0
        found = 0
        for i in range(len(energy_norm)):
            if energy_norm[i] < threshold:
                consecutive += 1
                if consecutive >= 8:
                    found = i - 7
                    break
            else:
                consecutive = 0
        burst_end = max(0, min(found, ns - 1))

        # Spettro FFT medio calcolato SOLO sulla zona utile (sotto il burst).
        # Il burst ha energia DC/bassa frequenza che sposta il picco verso il
        # basso e falsifica la stima della frequenza centrale del segnale GPR.
        s0_fft = burst_end if burst_end > 0 else min(ns // 8, ns - 1)
        sub_fft = arr[s0_fft:, :]
        if sub_fft.shape[0] < 8:
            sub_fft = arr  # fallback se il burst occupa quasi tutto
        n_tr_fft = min(64, nt)
        step_tr = max(1, nt // n_tr_fft)
        traces = sub_fft[:, ::step_tr].T.astype(np.float64)  # (n_tr, ns_sub)
        fft_mag = np.abs(np.fft.rfft(traces, axis=1))         # (n_tr, ns_sub//2+1)
        fft_spectra.append(fft_mag.mean(axis=0))

    if not n_samples_all:
        return dict(DEFAULT_PIPELINE)

    ns_median = int(np.median(n_samples_all))
    dt_median = float(np.median(dt_ns_all))

    # ------------------------------------------------------------------ #
    # 1. dewow_win: durata di ~3 cicli alla frequenza centrale stimata.   #
    #    Il dewow è un moving-average lungo l'asse campioni che rimuove   #
    #    la componente DC/lenta; la finestra deve essere comparabile con  #
    #    la durata del pulse (1-3 cicli), NON una frazione di ns_median.  #
    #    Formula: n_cicli / (f_center_MHz * dt_ns * 1e-3)                 #
    #    Fallback se f_center non disponibile: 5% di ns_median, max 31.   #
    # ------------------------------------------------------------------ #
    if dt_median > 0 and fft_spectra:
        # Stima rapida della frequenza centrale dal picco dello spettro
        # (calcolato prima del blocco bandpass per poterlo riusare qui).
        # Riusiamo fft_spectra che sarà disponibile in basso; facciamo un
        # calcolo anticipato della frequenza di picco.
        _fft_arr = np.zeros(max(s.shape[0] for s in fft_spectra), dtype=np.float64)
        _fft_cnt = np.zeros_like(_fft_arr)
        for _sp in fft_spectra:
            _fft_arr[:_sp.shape[0]] += _sp
            _fft_cnt[:_sp.shape[0]] += 1
        _fft_arr /= np.where(_fft_cnt > 0, _fft_cnt, 1)
        _freqs = np.fft.rfftfreq(ns_median, d=dt_median * 1e-3)  # MHz
        if _freqs.shape[0] > _fft_arr.shape[0]:
            _freqs = _freqs[:_fft_arr.shape[0]]
        elif _fft_arr.shape[0] > _freqs.shape[0]:
            _fft_arr = _fft_arr[:_freqs.shape[0]]
        _peak_mhz, _peak_idx = _peak_freq_mhz(_freqs, _fft_arr, antenna_freq_mhz)
        if _peak_mhz > 10.0:
            # 3 cicli alla frequenza centrale, arrotondato a dispari
            _samples_per_cycle = 1.0 / (_peak_mhz * dt_median * 1e-3)  # dt in µs, f in MHz → cicli/campione
            dewow_win = max(5, int(round(3.0 * _samples_per_cycle)))
        else:
            dewow_win = max(5, min(31, ns_median // 20))
    else:
        dewow_win = max(5, min(31, ns_median // 20))
    if dewow_win % 2 == 0:
        dewow_win += 1

    # ------------------------------------------------------------------ #
    # 2. burst_end e pre_agc_gain (TVG per attenuare il burst superficiale)#
    #                                                                      #
    #  Il burst (direct wave / antenna coupling) ha energia 100-1000x     #
    #  superiore al segnale di riflessione utile. Se lasciato inalterato, #
    #  l'AGC (che normalizza localmente) lo amplifica poco ma lascia un   #
    #  salto netto tra la zona burst e la zona utile.                      #
    #                                                                      #
    #  Approccio corretto (fisicamente motivato):                          #
    #  - agc_sample_start = 0: AGC lavora sull'intero profilo             #
    #  - pre_agc_gain attua una curva TVG che attenua il burst prima       #
    #    dell'AGC, portandolo allo stesso ordine di grandezza del segnale  #
    #    utile. Questo permette all'AGC di normalizzare uniformemente.      #
    #                                                                      #
    #  Curva TVG calcolata dal profilo di energia medio:                   #
    #  - gain[0..burst_end]  : da gain_burst → 1.0 (rampa lineare)        #
    #  - gain[burst_end..ns] : 1.0 (nessuna attenuazione aggiuntiva)       #
    #  gain_burst = sqrt(energy_useful / energy_burst)  (RMS ratio)       #
    # ------------------------------------------------------------------ #
    burst_end = 0
    pre_agc_breakpoints = None
    pre_agc_enabled = False

    if energy_profiles:
        # Media dei profili di energia (già normalizzati a [0,1])
        max_len = max(e.shape[0] for e in energy_profiles)
        energy_mean = np.zeros(max_len, dtype=np.float64)
        counts = np.zeros(max_len, dtype=np.float64)
        for e in energy_profiles:
            energy_mean[:e.shape[0]] += e
            counts[:e.shape[0]] += 1
        counts = np.where(counts > 0, counts, 1)
        energy_mean /= counts

        # Picco nella prima metà (burst diretto)
        half = max(1, max_len // 2)
        peak_idx = int(np.argmax(energy_mean[:half]))
        threshold_20 = 0.20 * float(energy_mean[peak_idx])

        # Fine del burst: prima posizione dove l'energia resta < 20% del picco
        # per almeno 8 campioni consecutivi
        found = peak_idx
        consecutive = 0
        for i in range(peak_idx, max_len):
            if energy_mean[i] < threshold_20:
                consecutive += 1
                if consecutive >= 8:
                    found = i - 7
                    break
            else:
                consecutive = 0
        burst_end = max(0, min(found, ns_median - 1))

        if burst_end > 0:
            # Energia media nella zona burst e nella zona utile
            e_burst = float(np.mean(energy_mean[:burst_end])) if burst_end > 0 else 1.0
            e_useful = float(np.mean(energy_mean[burst_end:])) if burst_end < max_len else 1.0
            e_burst = max(e_burst, 1e-12)
            e_useful = max(e_useful, 1e-12)

            # Rapporto di attenuazione (in ampiezza = sqrt del rapporto energetico)
            # Attenua il burst fino al livello medio del segnale utile.
            gain_burst = float(np.sqrt(e_useful / e_burst))
            gain_burst = float(np.clip(gain_burst, 1e-4, 1.0))

            # Breakpoints: [x_normalizzato, gain]
            # x=0 → gain_burst (picco burst)
            # x=burst_end/ns → 1.0 (inizio zona utile, rampa lineare)
            # x=1.0 → 1.0 (fondo profilo)
            x_burst = float(burst_end) / float(max(ns_median, 1))
            pre_agc_breakpoints = [
                [0.0,      gain_burst],
                [x_burst,  1.0],
                [1.0,      1.0],
            ]
            pre_agc_enabled = True

    # ------------------------------------------------------------------ #
    # 3. agc_win: proporzionale all'intera traccia (agc_sample_start=0)   #
    # ------------------------------------------------------------------ #
    agc_sample_start = 0
    agc_win = int(np.clip(ns_median // 8, 16, 256))
    if agc_win % 2 != 0:
        agc_win += 1

    # ------------------------------------------------------------------ #
    # 4. bg_sample_start: 0 — il bg_removal rimuove la componente          #
    #    orizzontale costante (clutter), incluso il burst superficiale.    #
    #    Applicarlo dall'inizio riduce le bande orizzontali visibili.      #
    # ------------------------------------------------------------------ #
    bg_sample_start = 0

    # ------------------------------------------------------------------ #
    # 5. Bandpass: stima le frequenze dal picco dello spettro FFT medio   #
    # ------------------------------------------------------------------ #
    bp_low_mhz = float(DEFAULT_PIPELINE["bp_low_mhz"])
    bp_high_mhz = float(DEFAULT_PIPELINE["bp_high_mhz"])
    if fft_spectra and dt_median > 0:
        # Frequenze in MHz
        max_fft_len = max(s.shape[0] for s in fft_spectra)
        fft_mean = np.zeros(max_fft_len, dtype=np.float64)
        fft_counts = np.zeros(max_fft_len, dtype=np.float64)
        for sp in fft_spectra:
            fft_mean[:sp.shape[0]] += sp
            fft_counts[:sp.shape[0]] += 1
        fft_counts = np.where(fft_counts > 0, fft_counts, 1)
        fft_mean /= fft_counts

        freqs_mhz = np.fft.rfftfreq(ns_median, d=dt_median * 1e-3)  # dt in µs → MHz
        if freqs_mhz.shape[0] > fft_mean.shape[0]:
            freqs_mhz = freqs_mhz[:fft_mean.shape[0]]
        elif fft_mean.shape[0] > freqs_mhz.shape[0]:
            fft_mean = fft_mean[:freqs_mhz.shape[0]]

        peak_freq, peak_freq_idx = _peak_freq_mhz(freqs_mhz, fft_mean, antenna_freq_mhz)

        if peak_freq > 10.0:
            # Banda -6 dB intorno al picco (potenza dimezzata = 0.5 del picco).
            # -20 dB (0.01) è troppo stretta per stimare la banda GPR utile;
            # -6 dB cattura la banda principale del segnale.
            threshold_6db = fft_mean[peak_freq_idx] * 0.5
            lo_idx, hi_idx = peak_freq_idx, peak_freq_idx
            for i in range(peak_freq_idx, -1, -1):
                if fft_mean[i] < threshold_6db:
                    lo_idx = i
                    break
            for i in range(peak_freq_idx, len(fft_mean)):
                if fft_mean[i] < threshold_6db:
                    hi_idx = i
                    break
            lo_freq = float(freqs_mhz[lo_idx]) if lo_idx < len(freqs_mhz) else peak_freq * 0.25
            hi_freq = float(freqs_mhz[hi_idx]) if hi_idx < len(freqs_mhz) else peak_freq * 2.0
            # Margine del 20% verso il basso, 30% verso l'alto.
            # Minimo basso a 50 MHz (non ha senso filtrare DC) ma almeno peak/4.
            bp_low_mhz = max(50.0, round(lo_freq * 0.8, 0))
            bp_high_mhz = min(3000.0, round(hi_freq * 1.3, 0))
            if bp_high_mhz <= bp_low_mhz:
                bp_high_mhz = bp_low_mhz * 4.0

        # Clamp di sicurezza: anche vincolando la ricerca del picco intorno al
        # prior (vedi _peak_freq_mhz), uno spettro molto rumoroso puo' ancora
        # restituire una banda implausibile rispetto alla frequenza nominale
        # dell'antenna. Se e' nota, la banda finale non puo' scostarsi oltre
        # un fattore ragionevole: altrimenti il bandpass rischia di tagliare
        # via il segnale reale e lasciar passare solo rumore a banda larga.
        if antenna_freq_mhz and antenna_freq_mhz > 10.0:
            bp_low_mhz = float(np.clip(bp_low_mhz, antenna_freq_mhz * 0.2, antenna_freq_mhz * 1.2))
            bp_high_mhz = float(np.clip(bp_high_mhz, antenna_freq_mhz * 0.8, antenna_freq_mhz * 4.0))
            if bp_high_mhz <= bp_low_mhz:
                bp_high_mhz = bp_low_mhz * 3.0

    # ------------------------------------------------------------------ #
    # Costruisci il dizionario risultato                                  #
    # ------------------------------------------------------------------ #
    _log_gpr(
        f"[GPR] auto_tune: ns={ns_median} dt={dt_median:.4g}ns "
        f"dewow_win={dewow_win} agc_win={agc_win} agc_s0={agc_sample_start} "
        f"burst_end={burst_end} pre_agc={pre_agc_enabled} "
        f"bg_s0={bg_sample_start} bp=[{bp_low_mhz:.0f},{bp_high_mhz:.0f}]MHz"
    )
    result = dict(DEFAULT_PIPELINE)
    result.update({
        "dewow":                True,
        "dewow_win":            dewow_win,
        "timezero":             True,
        "tz_method":            "peak",
        "tz_mode":              "line_by_line",
        "bg_removal":           True,
        "bg_sample_start":      bg_sample_start,
        "pre_agc_gain":         pre_agc_enabled,
        "pre_agc_breakpoints":  pre_agc_breakpoints,
        "agc":                  True,
        "agc_win":              agc_win,
        "agc_sample_start":     agc_sample_start,
        "bandpass":             True,
        "bp_low_mhz":           bp_low_mhz,
        "bp_high_mhz":          bp_high_mhz,
    })
    return result
