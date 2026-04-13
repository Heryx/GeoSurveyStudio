"""Pipeline di processing per radargram GPR — implementazione puro numpy.

Tutte le funzioni operano su matrice (n_samples, n_slices) float32.
Nessuna dipendenza da scipy: Hilbert e bandpass sono implementati
tramite FFT numpy.
"""

from __future__ import annotations

import numpy as np

try:
    from qgis.core import Qgis, QgsMessageLog
except Exception:  # pragma: no cover - QGIS runtime not available in headless tests
    Qgis = None
    QgsMessageLog = None


def _log_gpr(message: str, level=None) -> None:
    """Log to QGIS panel when available, fallback to stdout otherwise."""
    msg = str(message)
    if QgsMessageLog is not None and Qgis is not None:
        try:
            lvl = Qgis.Info if level is None else level
            QgsMessageLog.logMessage(msg, "RasterLinker", lvl)
            return
        except Exception:
            pass
    print(msg)


# ---------------------------------------------------------------------------
# Primitivi FFT (senza scipy)
# ---------------------------------------------------------------------------

def _hilbert_numpy(x: np.ndarray) -> np.ndarray:
    N  = len(x)
    Xf = np.fft.fft(x)
    h  = np.zeros(N, dtype=np.float64)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1.0
        h[1:N // 2] = 2.0
    else:
        h[0] = 1.0
        h[1:(N + 1) // 2] = 2.0
    return np.fft.ifft(Xf * h)


def _bandpass_fft(
    data: np.ndarray,
    dt_ns: float,
    low_mhz: float,
    high_mhz: float,
) -> np.ndarray:
    N = int(data.shape[0])
    dt_s = float(dt_ns) * 1e-9
    if N <= 1 or (not np.isfinite(dt_s)) or dt_s <= 0.0:
        return np.array(data, dtype=np.float32, copy=True)

    # Reflect padding mitigates circular-convolution edge artifacts that
    # often appear as horizontal striping after FFT filtering.
    pad = int(np.clip(N // 4, 32, 256))
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

    # Tukey-like cosine rolloff (12% per edge) to reduce Gibbs ringing.
    bw = max(1e-9, f_hi - f_lo)
    edge = max(1e-9, 0.12 * bw)
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
    data = np.asarray(data, dtype=np.float32)
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
    data     = data.astype(np.float32)
    n_s, n_t = data.shape
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
) -> np.ndarray:
    """
    Background removal (GPR-SLICE §Background Removal, pag. 166).

    Sottrae una traccia di riferimento (il "clutter" orizzontale costante)
    da ogni traccia del radargram.

    Modalita'
    ---------
    'line_by_line'
        window = 0  (AUTO): media globale sull'intero profilo.
        window > 0        : media scorrevole di `window` tracce.

    'grid_by_grid'
        Sottrae `reference_trace` pre-calcolata su tutti i radargram.
        Il chiamante deve fornirla via apply_pipeline(bg_reference_trace=...).
        Se None: fallback a line_by_line auto.

    Finestra temporale (sample_start / sample_end)
    -----------------------------------------------
    Permette di limitare la sottrazione a un range di campioni [s_lo, s_hi).
    I campioni FUORI dalla finestra vengono lasciati INVARIATI.

    Caso d'uso principale: escludere i primi N campioni (ground coupling /
    onda diretta) dalla rimozione del background.
    Effetto: preserva il segnale nei primi 50 cm, permettendo all'AGC di
    amplificarlo correttamente invece di trovare ~0 dopo la sottrazione.

        sample_start = 0, sample_end = 0  ->  intera traccia (default).
        sample_start = 20, sample_end = 0 ->  salta i primi 20 campioni
                                               (ground coupling), applica
                                               BG removal dal campione 20
                                               in poi.
        sample_start = 0, sample_end = 80 ->  applica solo ai primi 80
                                               campioni (zona superficiale).

    Parametri
    ----------
    data             : (n_samples, n_traces) float32
    mode             : 'line_by_line' | 'grid_by_grid'
    window           : int  lunghezza filtro in tracce (0 = auto)
    reference_trace  : (n_samples,) float64 | None  solo per grid_by_grid
    sample_start     : int  primo campione incluso nella finestra (0 = inizio)
    sample_end       : int  ultimo campione escluso (0 = auto = n_samples)
    """
    d64      = data.astype(np.float64)
    n_s, n_t = d64.shape
    try:
        win_req = int(window)
    except Exception:
        win_req = 0
    win_eff = int(win_req)
    if win_eff > n_t and n_t > 0:
        # Prevent oversized windows from silently behaving inconsistently.
        win_eff = int(n_t)
        _log_gpr(
            f"[GPR] bg_removal: win clampato a n_traces={n_t} "
            f"(richiesto={win_req})."
        )

    # Diagnostico
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
        f"[GPR] bg_diag: mode={mode} win={win_eff if win_eff > 0 else 'auto'}  "
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

    # Sub-matrice su cui operare
    sub = d64[s_lo:s_hi, :]   # (n_win, n_t)
    # Keep perfectly flat rows untouched (e.g., pure zero rows after truncation).
    row_std_sub = np.std(sub, axis=1)
    valid_rows = row_std_sub > 1e-12

    # ---------------------------------------------------------------
    # grid_by_grid
    # ---------------------------------------------------------------
    if mode == "grid_by_grid":
        if reference_trace is not None:
            ref = np.asarray(reference_trace, dtype=np.float64)
            # Adatta ref alla finestra campioni
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
            ref_sub = sub.mean(axis=1)
        out = d64.copy()
        corr = sub - ref_sub.reshape(-1, 1)
        block = out[s_lo:s_hi, :]
        block[valid_rows, :] = corr[valid_rows, :]
        out[s_lo:s_hi, :] = block
        return out.astype(np.float32)

    # ---------------------------------------------------------------
    # line_by_line — auto: media globale del sub-range
    # ---------------------------------------------------------------
    if win_eff <= 0 or win_eff >= n_t:
        mean_trace = sub.mean(axis=1, keepdims=True)
        _log_gpr(f"[GPR] bg_removal: global mean subtracted  n_t={n_t}  s=[{s_lo}:{s_hi}]")
        out = d64.copy()
        corr = sub - mean_trace
        block = out[s_lo:s_hi, :]
        block[valid_rows, :] = corr[valid_rows, :]
        out[s_lo:s_hi, :] = block
        return out.astype(np.float32)

    # ---------------------------------------------------------------
    # line_by_line — finestra scorrevole lungo asse tracce
    # ---------------------------------------------------------------
    half   = win_eff // 2
    n_sub  = sub.shape[1]
    idx    = np.arange(n_sub)
    hi_t   = np.minimum(n_sub - 1, idx + half)
    lo_t   = np.maximum(0,         idx - half)
    cs     = np.cumsum(sub, axis=1)
    sum_hi = cs[:, hi_t]
    lo_prev = np.maximum(0, lo_t - 1)
    sum_lo  = cs[:, lo_prev]
    sum_lo[:, lo_t == 0] = 0.0
    win_len    = (hi_t - lo_t + 1).reshape(1, -1)
    local_mean = (sum_hi - sum_lo) / win_len
    _log_gpr(
        f"[GPR] bg_removal: sliding win={win_eff}  s=[{s_lo}:{s_hi}]  "
        f"eff_win=[{(hi_t-lo_t+1).min()},{(hi_t-lo_t+1).max()}]"
    )
    out = d64.copy()
    corr = sub - local_mean
    block = out[s_lo:s_hi, :]
    block[valid_rows, :] = corr[valid_rows, :]
    out[s_lo:s_hi, :] = block
    return out.astype(np.float32)


def agc_gain(
    data: np.ndarray,
    window: int = 128,
    clip_percentile: float | None = None,
) -> np.ndarray:
    """
    Automatic Gain Control vettorizzato con cumsum.
    O(n_s * n_t) senza loop Python.
    """
    n_s, n_t = data.shape
    if n_s <= 1 or n_t <= 0:
        return np.array(data, dtype=np.float32, copy=True)
    win = max(8, int(window))
    win = min(win, max(8, n_s))
    if win % 2 != 0:
        win += 1
    half = max(4, win // 2)
    d64 = data.astype(np.float64, copy=False)
    sq = d64 ** 2
    # Full window at edges via reflect padding.
    sq_pad = np.pad(sq, ((half, half), (0, 0)), mode="reflect")
    cs = np.vstack([np.zeros((1, n_t), dtype=np.float64), np.cumsum(sq_pad, axis=0)])
    s_idx = np.arange(n_s, dtype=np.int64)
    sum_win = cs[s_idx + win, :] - cs[s_idx, :]
    rms = np.sqrt(np.maximum(sum_win / float(win), 0.0))
    try:
        valid_rms = rms[rms > 1e-10]
        if valid_rms.size > 0:
            rms_global = float(np.percentile(valid_rms, 95.0))
            if np.isfinite(rms_global) and rms_global > 1e-12:
                rms = np.minimum(rms, rms_global * 10.0)
    except Exception:
        pass
    rms = rms + 1e-10
    out = (d64 / rms).astype(np.float32)
    if clip_percentile is not None and out.size > 0:
        clip = float(np.percentile(np.abs(out), clip_percentile))
        if clip > 1e-12:
            return np.clip(out, -clip, clip)
    return out


def bandpass_filter(
    data: np.ndarray,
    dt_ns: float,
    low_mhz: float,
    high_mhz: float,
    order: int = 4,
) -> np.ndarray:
    return _bandpass_fft(data, dt_ns, low_mhz, high_mhz)


def hilbert_envelope(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    n_s, n_t = arr.shape
    if n_s <= 1 or n_t <= 0:
        return np.abs(arr).astype(np.float32, copy=True)
    out = np.empty((n_s, n_t), dtype=np.float32)
    n_freq = int(n_s // 2 + 1)
    w = np.zeros(n_freq, dtype=np.complex128)
    if n_s % 2 == 0:
        if n_freq > 2:
            w[1:-1] = -1j
    else:
        if n_freq > 1:
            w[1:] = -1j
    target_bytes = 64 * 1024 * 1024
    bytes_per_trace = max(1, 16 * (n_s + n_freq))
    cols_per_block = max(1, int(target_bytes // bytes_per_trace))
    cols_per_block = min(cols_per_block, max(1, n_t))
    for start in range(0, n_t, cols_per_block):
        stop = min(n_t, start + cols_per_block)
        block = np.ascontiguousarray(arr[:, start:stop].T, dtype=np.float64)
        R = np.fft.rfft(block, axis=1)
        hilb = np.fft.irfft(R * w.reshape(1, -1), n=n_s, axis=1)
        env = np.sqrt(np.maximum(block * block + hilb * hilb, 0.0))
        out[:, start:stop] = env.T.astype(np.float32, copy=False)
    return out


def normalize_display(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
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
    "dewow":            True,
    "dewow_win":        16,
    "timezero":         False,
    "tz_method":        "peak",
    "tz_mode":          "line_by_line",
    "tz_threshold":     0.2,
    "tz_backup_nsamp":  4,
    "bg_removal":       True,
    "bg_mode":          "line_by_line",
    "bg_window":        0,
    "bg_sample_start":  0,    # 0 = dall'inizio
    "bg_sample_end":    0,    # 0 = auto (intera traccia)
    "agc":              True,
    "agc_win":          128,
    # Optional depth-dependent gain applied BEFORE AGC.
    # Useful to attenuate direct-wave/surface burst before local AGC normalization.
    "pre_agc_gain":     False,
    "pre_agc_surface_gain": 1.0,
    "pre_agc_deep_gain":    1.0,
    "pre_agc_breakpoints":  None,     # [[x(0..1), gain], ...]
    "bandpass":         False,
    "bp_low_mhz":       100.0,
    "bp_high_mhz":      1200.0,
    "envelope":         False,
    "chain_order":      None,
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
}

DEFAULT_CHAIN_ORDER = [
    "dewow",
    "timezero",
    "bg_removal",
    "bandpass",
    "pre_agc_gain",
    "agc",
    "envelope",
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

    Questo garantisce che la traccia sottratta sia la media reale del
    ground coupling comune a TUTTI i radargram del grid, non solo del
    singolo profilo.
    """
    p   = {**DEFAULT_PIPELINE, **params}
    out = data.copy()
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
) -> np.ndarray:
    """
    Applica la pipeline di processing completa.

    Parametri
    ----------
    data                : (n_samples, n_traces) float32
    params              : dict con chiavi da DEFAULT_PIPELINE
    dt_ns               : intervallo di campionamento in nanosecondi
    bg_reference_trace  : (n_samples,) traccia media dell'intero grid.
                          Fornita da _process_profiles() per grid_by_grid
                          a due passate; None per line_by_line.
    """
    p = {**DEFAULT_PIPELINE, **params}
    out = data.copy()
    order = _resolve_chain_order(chain_order, p)

    def _log(tag: str, arr: np.ndarray) -> None:
        mn = float(arr.min())
        mx = float(arr.max())
        nz = int(np.count_nonzero(arr))
        _log_gpr(
            f"[GPR] {tag:40s}  min={mn:.4g}  max={mx:.4g}  "
            f"nonzero={nz}/{arr.size}  shape={arr.shape}"
        )

    _log("raw", out)

    for step in order:
        if step == "dewow":
            if p["dewow"]:
                out = dewow(out, window=int(p["dewow_win"]))
                _log(f"dewow(win={p['dewow_win']})", out)
            continue

        if step == "timezero":
            if p["timezero"]:
                out = time_zero_correction(
                    out,
                    method=str(p.get("tz_method", "peak")),
                    mode=str(p.get("tz_mode", "line_by_line")),
                    threshold=float(p.get("tz_threshold", 0.2)),
                    backup_nsamp=int(p.get("tz_backup_nsamp", 4)),
                )
                _log("timezero", out)
            continue

        if step == "bg_removal":
            if p["bg_removal"]:
                bg_mode = str(p.get("bg_mode", "line_by_line"))
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
                )
                win_lbl = "auto" if bg_window <= 0 else str(bg_window)
                s_lbl = f"{bg_sample_start}:{bg_sample_end if bg_sample_end > 0 else 'end'}"
                _log(f"bg_removal(mode={bg_mode} win={win_lbl} s=[{s_lbl}])", out)
            continue

        if step == "pre_agc_gain":
            if bool(p.get("pre_agc_gain", False)):
                gain = _build_pre_agc_gain(int(out.shape[0]), p)
                out = (out.astype(np.float32, copy=False) * gain).astype(np.float32, copy=False)
                _log("pre_agc_gain", out)
            continue

        if step == "agc":
            if p["agc"]:
                out = agc_gain(out, window=int(p["agc_win"]), clip_percentile=99.5)
                _log(f"agc(win={p['agc_win']})", out)
            continue

        if step == "bandpass":
            if p["bandpass"]:
                out = bandpass_filter(
                    out,
                    dt_ns,
                    float(p["bp_low_mhz"]),
                    float(p["bp_high_mhz"]),
                )
                _log(f"bandpass({p['bp_low_mhz']}-{p['bp_high_mhz']}MHz)", out)
            continue

        if step == "envelope":
            if bool(p.get("envelope", False)):
                out = hilbert_envelope(out)
                _log("envelope(hilbert)", out)
            continue

    if normalize_output:
        out = normalize_display(out)
        _log("normalize (finale)", out)
    else:
        out = out.astype(np.float32, copy=False)
        _log("finale (no normalize)", out)
    return out
