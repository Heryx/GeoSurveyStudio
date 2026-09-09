"""Parser per il formato raw .DT (+ sidecar .gec/.gps.sav/.gps).

Prodotto da acquisizioni georeferenziate GPS (es. antenna TIPASA/Zone1),
organizzate come cartella "raw" di progetto: ogni linea e' un set di file
con lo stesso nome base, es. LA010001.DT / .gec / .gps / .DT.gps.sav.

Layout binario del file .DT (dedotto empiricamente, nessuna doc ufficiale):
  - Header a blocchi ASCII di 1028 byte ciascuno (tag a 2-3 lettere +
    valori scientifici space-padded), tipicamente 10 blocchi (10280 byte).
    Il primo blocco ("V0") contiene i parametri di acquisizione:
      campo 0: n_samples/traccia
      campo 1: NON e' la frequenza reale dell'antenna: l'export "raw" vi
               scrive un segnaposto fisso 1.0E+08 Hz (100 MHz).
               La frequenza vera viene recuperata dal file `install.dat` di
               progetto (risalendo le cartelle) oppure da un file sidecar
               di override "<base>.DT.freq" / "antenna_freq_mhz.txt"
               (un numero, MHz) accanto al .DT. Vedi `_resolve_antenna_frequency_mhz`.
      campo 2/3/5: finestra tempo (s)
      campo 6/7/8: spaziatura traccia (m)
  - Dopo l'header: campioni int16 little-endian, n_samples per traccia,
    tracce contigue senza header per-traccia.
  - La lunghezza esatta dell'header puo' variare leggermente da file a
    file: viene quindi ricavata per via aritmetica (vedi
    `_locate_trace_data`), validando che (filesize - offset) sia un
    multiplo esatto di n_samples * 2 byte.

Posizionamento delle tracce: il file .gec (o, in fallback, il file
.DT.gps.sav) contiene una tabella di "marker" GPS gia' proiettati in UTM
(trace#, UTM_X, UTM_Y, elev). Le posizioni intermedie sono ottenute per
interpolazione lineare sull'indice di traccia.
"""

from __future__ import annotations

import functools
import re
from pathlib import Path

import numpy as np

from .gpr_ogpr_reader import OgprChannel, OgprProfile

HEADER_BLOCK_SIZE = 1028
DEFAULT_HEADER_BLOCKS = 10
_TAG_RE = re.compile(rb"[A-Z][A-Z0-9]\d?\x00")
_V0_FIELD_WIDTH = 16
_V0_N_FIELDS = 10
_SAMPLE_ITEMSIZE = 2  # int16

# Layout "a blocchi": alcuni export .DT scrivono ogni traccia in un blocco di
# HEADER_BLOCK_SIZE byte = intestazione di traccia (8 byte) + campioni int16.
# Il campo 0 del blocco V0 (es. 200) NON e' il numero di campioni reale:
# quello si ricava dal blocco. Intestazione di traccia osservata (8 byte):
#   int32 marker  -> byte0 = 0x52 ('R'), byte1 = flag (0x07 su prima/ultima
#                    traccia, 0x00 altrove), byte2..3 = 0x0000;
#   int32 counter -> contatore per-traccia (odometro/tempo), NON campioni.
_TRACE_BLOCK_SIZE = HEADER_BLOCK_SIZE          # 1028
_TRACE_PREFIX_BYTES = 8
_SAMPLES_PER_TRACE_BLOCK = (_TRACE_BLOCK_SIZE - _TRACE_PREFIX_BYTES) // _SAMPLE_ITEMSIZE  # 510
_TRACE_MARKER_BYTE = 0x52                      # 'R'

# Il campo 1 del blocco V0 nell'export "raw" non contiene la frequenza
# reale dell'antenna: e' un valore fisso 1.0E+08 Hz (100 MHz) scritto come
# segnaposto. La frequenza vera va recuperata dal file `install.dat` di
# progetto o impostata a mano.
_V0_FREQ_PLACEHOLDER_HZ = 1.0e8
_FREQ_PLAUSIBLE_MIN_MHZ = 10.0
_FREQ_PLAUSIBLE_MAX_MHZ = 3000.0
_INSTALL_DAT_NAMES = ("install.dat", "installv5.dat")
_INSTALL_DAT_MAX_PARENTS = 4


class DtSliceReadError(Exception):
    pass


# ---------------------------------------------------------------------------
# Header .DT
# ---------------------------------------------------------------------------

def _parse_v0_header(raw: bytes) -> dict:
    if raw[:2] != b"V0":
        raise DtSliceReadError(f"Tag header non valido, atteso 'V0': {raw[:8]!r}")
    fields_start = 8
    fields_blob = raw[fields_start:fields_start + _V0_FIELD_WIDTH * _V0_N_FIELDS]
    values = []
    for i in range(_V0_N_FIELDS):
        chunk = fields_blob[i * _V0_FIELD_WIDTH:(i + 1) * _V0_FIELD_WIDTH]
        try:
            values.append(float(chunk.decode("ascii").strip()))
        except ValueError:
            values.append(float("nan"))

    # Alcuni file .DT hanno il blocco V0 azzerato/vuoto (nessun parametro
    # scritto per quella linea): n_samples resta NaN e viene gestito dal
    # chiamante (fallback su un file "fratello" valido).
    n_samples_raw = values[0]
    n_samples = int(round(n_samples_raw)) if n_samples_raw == n_samples_raw else 0
    frequency_hz = values[1]
    time_window_s = values[2]
    trace_spacing_m = values[6]
    return {
        "n_samples": n_samples,
        "frequency_hz": frequency_hz,
        "time_window_s": time_window_s,
        "trace_spacing_m": trace_spacing_m,
        "raw_fields": values,
    }


def _header_is_valid(header: dict) -> bool:
    n_samples = header.get("n_samples") or 0
    time_window_s = header.get("time_window_s")
    if n_samples <= 0:
        return False
    if time_window_s is None or time_window_s != time_window_s or time_window_s <= 0:
        return False
    return True


def _freq_is_plausible(mhz: float) -> bool:
    return (
        mhz == mhz  # not NaN
        and _FREQ_PLAUSIBLE_MIN_MHZ <= mhz <= _FREQ_PLAUSIBLE_MAX_MHZ
    )


def _as_float(line: str) -> float | None:
    try:
        return float(line)
    except ValueError:
        return None


def _parse_antenna_frequency_mhz(install_path: Path) -> float | None:
    """Estrae la frequenza nominale antenna (MHz) da un `install.dat` di progetto.

    Nel blocco "radargram information" la frequenza compare come coppia di
    righe identiche (freq iniziale / finale) preceduta dalla velocita' del
    mezzo (m/ns, ~0.05-0.30), da due valori di estensione e dalla finestra
    temporale (ns). Il layout non e' documentato: si ancora la ricerca a una
    riga "velocita'" e si prende la prima coppia di interi consecutivi uguali
    in un intervallo plausibile per un'antenna GPR entro le righe successive.
    """
    try:
        lines = [
            ln.strip()
            for ln in install_path.read_text(encoding="latin1", errors="replace").splitlines()
        ]
    except OSError:
        return None

    for i, ln in enumerate(lines):
        vel = _as_float(ln)
        if vel is None or not (0.02 <= vel <= 0.35) or float(vel).is_integer():
            continue
        window = lines[i + 1:i + 9]
        # richiede una riga "finestra temporale" (float ns) prima della coppia
        if not any(
            (fv := _as_float(w)) is not None and 5.0 <= fv <= 5000.0
            for w in window
        ):
            continue
        prev: float | None = None
        for w in window:
            val = _as_float(w)
            if (
                val is not None
                and prev is not None
                and val == prev
                and float(val).is_integer()
                and _FREQ_PLAUSIBLE_MIN_MHZ <= val <= _FREQ_PLAUSIBLE_MAX_MHZ
            ):
                return float(val)
            prev = val
    return None


def _find_project_install_frequency_mhz(path: Path) -> tuple[float, str] | None:
    """Risale l'albero cartelle cercando un `install.dat` di progetto."""
    seen: set[Path] = set()
    for parent in list(path.parents)[:_INSTALL_DAT_MAX_PARENTS]:
        for name in _INSTALL_DAT_NAMES:
            candidate = parent / name
            if candidate in seen or not candidate.is_file():
                continue
            seen.add(candidate)
            freq = _parse_antenna_frequency_mhz(candidate)
            if freq is not None and _freq_is_plausible(freq):
                return freq, candidate.name
    return None


def _read_frequency_sidecar_mhz(path: Path) -> float | None:
    """Override esplicito: `<base>.freq` o `antenna_freq_mhz.txt` (un numero, MHz)."""
    candidates = [
        Path(str(path) + ".freq"),
        path.with_suffix(".freq"),
        path.parent / "antenna_freq_mhz.txt",
    ]
    for cand in candidates:
        try:
            if not cand.is_file():
                continue
            val = float(cand.read_text(encoding="latin1", errors="replace").split()[0])
        except (OSError, ValueError, IndexError):
            continue
        if _freq_is_plausible(val):
            return val
    return None


def _resolve_antenna_frequency_mhz(path: Path, v0_freq_hz: float) -> tuple[float, str | None]:
    """Determina la frequenza antenna (MHz) e un eventuale warning diagnostico."""
    sidecar = _read_frequency_sidecar_mhz(path)
    if sidecar is not None:
        return sidecar, f"Frequenza antenna {sidecar:.0f} MHz da file sidecar"

    v0_mhz = v0_freq_hz / 1e6 if v0_freq_hz and v0_freq_hz == v0_freq_hz else 0.0
    is_placeholder = abs(v0_freq_hz - _V0_FREQ_PLACEHOLDER_HZ) < 1.0

    if not is_placeholder and _freq_is_plausible(v0_mhz):
        return v0_mhz, None

    found = _find_project_install_frequency_mhz(path)
    if found is not None:
        freq, src = found
        return freq, f"Frequenza antenna {freq:.0f} MHz recuperata da {src} di progetto"

    if is_placeholder:
        return 0.0, (
            "Il file .DT non contiene la frequenza reale dell'antenna (l'header "
            "V0 riporta un segnaposto fisso di 100 MHz) e non e' stato trovato "
            "un install.dat di progetto. Imposta la frequenza manualmente "
            "creando un file di testo con il valore in MHz accanto al .DT "
            f"({path.name}.freq) oppure 'antenna_freq_mhz.txt' nella stessa cartella."
        )
    if v0_mhz > 0.0:
        return v0_mhz, (
            f"Frequenza antenna dall'header V0 ({v0_mhz:.0f} MHz) fuori dall'intervallo "
            "tipico: verificala e all'occorrenza correggila con un file sidecar "
            f"{path.name}.freq (valore in MHz)."
        )
    return 0.0, (
        "Frequenza antenna non disponibile nel .DT ne' in un install.dat del "
        f"progetto: impostala con un file sidecar {path.name}.freq (valore in MHz)."
    )


def _find_fallback_header(path: Path) -> tuple[dict, str] | None:
    """Cerca un file .DT fratello nella stessa cartella con header V0 valido."""
    try:
        siblings = sorted(path.parent.glob("*.DT")) + sorted(path.parent.glob("*.dt"))
    except OSError:
        return None
    seen = set()
    for sib in siblings:
        if sib.resolve() == path.resolve() or str(sib) in seen:
            continue
        seen.add(str(sib))
        try:
            with open(sib, "rb") as f:
                head_bytes = f.read(HEADER_BLOCK_SIZE)
            if len(head_bytes) < HEADER_BLOCK_SIZE:
                continue
            header = _parse_v0_header(head_bytes)
        except (DtSliceReadError, OSError):
            continue
        if _header_is_valid(header):
            return header, sib.name
    return None


def _locate_trace_data(raw: bytes, n_samples: int) -> tuple[int, int]:
    """Ricava (offset_dati, n_tracce) validando la divisibilita' esatta."""
    trace_bytes = n_samples * _SAMPLE_ITEMSIZE
    file_size = len(raw)
    if trace_bytes <= 0:
        raise DtSliceReadError("n_samples non valido nell'header .DT")

    # Stima iniziale: fine dell'ultimo blocco header taggato individuato.
    tag_offsets = [m.start() for m in _TAG_RE.finditer(raw[:32768])]
    if tag_offsets:
        n_blocks_guess = max(DEFAULT_HEADER_BLOCKS, (max(tag_offsets) // HEADER_BLOCK_SIZE) + 2)
    else:
        n_blocks_guess = DEFAULT_HEADER_BLOCKS
    guess = n_blocks_guess * HEADER_BLOCK_SIZE

    # Ricerca del multiplo esatto piu' vicino alla stima entro una finestra
    # di sicurezza (qualche blocco header in piu' o in meno).
    window = HEADER_BLOCK_SIZE * 4
    lo = max(0, guess - window)
    hi = min(file_size, guess + window)
    remainder = file_size % trace_bytes
    # Tutti gli offset in [lo, hi] congruenti a `remainder` modulo trace_bytes:
    # sceglie quello piu' vicino alla stima (`guess`), non necessariamente >=.
    first_candidate = lo + ((remainder - lo) % trace_bytes)
    best = None
    best_dist = None
    candidate = first_candidate
    while candidate <= hi:
        dist = abs(candidate - guess)
        if best_dist is None or dist < best_dist:
            best, best_dist = candidate, dist
        candidate += trace_bytes
    if best is None:
        # Fallback: nessun candidato nella finestra, usa la stima grezza
        # troncando all'ultimo multiplo intero di trace_bytes.
        best = guess
        n_traces = max(0, (file_size - best) // trace_bytes)
        return best, n_traces

    n_traces = (file_size - best) // trace_bytes
    return best, n_traces


def _find_trace_block_start(raw: bytes) -> int | None:
    """Se il .DT usa il layout 'a blocchi' (una traccia per blocco di
    HEADER_BLOCK_SIZE byte), ritorna l'offset della prima traccia; altrimenti
    None. Cerca >=3 blocchi consecutivi con l'intestazione di traccia
    (byte 0 = 0x52, byte 2..3 = 0x0000)."""
    if len(raw) % _TRACE_BLOCK_SIZE != 0:
        return None
    n_blocks = len(raw) // _TRACE_BLOCK_SIZE
    lo = max(1, DEFAULT_HEADER_BLOCKS - 4)
    hi = min(n_blocks - 3, DEFAULT_HEADER_BLOCKS + 8)
    for b in range(lo, hi + 1):
        off = b * _TRACE_BLOCK_SIZE
        ok = True
        for j in range(3):
            o = off + j * _TRACE_BLOCK_SIZE
            if not (raw[o] == _TRACE_MARKER_BYTE and raw[o + 2] == 0 and raw[o + 3] == 0):
                ok = False
                break
        if ok:
            return off
    return None


def _read_dt_blocked(raw: bytes, header: dict) -> tuple[np.ndarray, int, int] | None:
    """Decodifica il layout 'a blocchi'. Ritorna (data, data_start, n_traces)
    oppure None se il layout non e' riconosciuto."""
    data_start = _find_trace_block_start(raw)
    if data_start is None:
        return None
    n_traces = (len(raw) - data_start) // _TRACE_BLOCK_SIZE
    if n_traces <= 0:
        return None
    ns = _SAMPLES_PER_TRACE_BLOCK
    rows = np.empty((n_traces, ns), dtype=np.float32)
    for k in range(n_traces):
        o = data_start + k * _TRACE_BLOCK_SIZE + _TRACE_PREFIX_BYTES
        rows[k] = np.frombuffer(raw, dtype="<i2", count=ns, offset=o)
    header["n_samples"] = ns
    header["trace_layout"] = "blocked"
    header["trace_block_bytes"] = _TRACE_BLOCK_SIZE
    header["trace_prefix_bytes"] = _TRACE_PREFIX_BYTES
    return rows.T.copy(), data_start, int(n_traces)


def _read_dt_binary(path: Path) -> tuple[np.ndarray, dict]:
    raw = path.read_bytes()
    if len(raw) < HEADER_BLOCK_SIZE:
        raise DtSliceReadError(f"File .DT troppo piccolo: {path}")

    header = _parse_v0_header(raw)
    header_warning = None
    if not _header_is_valid(header):
        fallback = _find_fallback_header(path)
        if fallback is None:
            raise DtSliceReadError(
                f"Header V0 vuoto/non valido in {path.name} e nessun file .DT "
                f"fratello con header valido trovato nella stessa cartella"
            )
        fb_header, fb_name = fallback
        header_warning = (
            f"Header V0 vuoto in {path.name}: parametri di acquisizione "
            f"presi da {fb_name}"
        )
        header = {**fb_header}

    # 1) layout "a blocchi" (una traccia per blocco da 1028 byte): il campo
    #    n_samples dell'header e' inaffidabile, si ricava dal blocco.
    blocked = _read_dt_blocked(raw, header)
    if blocked is not None:
        data, data_start, n_traces = blocked
        header["data_start"] = data_start
        header["n_traces"] = n_traces
        if header_warning:
            header["header_warning"] = header_warning
        return data, header

    # 2) layout storico: campioni int16 contigui, n_samples dall'header.
    n_samples = header["n_samples"]
    if n_samples <= 0:
        raise DtSliceReadError(f"n_samples non valido ({n_samples}) in {path}")

    data_start, n_traces = _locate_trace_data(raw, n_samples)
    if n_traces <= 0:
        raise DtSliceReadError(f"Impossibile individuare le tracce dati in {path}")

    samples = np.frombuffer(
        raw, dtype="<i2", count=n_samples * n_traces, offset=data_start
    ).astype(np.float32)
    data = samples.reshape(n_traces, n_samples).T.copy()  # (n_samples, n_traces)
    header["data_start"] = data_start
    header["n_traces"] = n_traces
    header["trace_layout"] = "contiguous"
    if header_warning:
        header["header_warning"] = header_warning
    return data, header


# ---------------------------------------------------------------------------
# Posizionamento: .gec / .DT.gps.sav
# ---------------------------------------------------------------------------

def _parse_gec(path: Path) -> dict:
    """Estrae EPSG, xcell e la tabella VALID_MARKERS_SWEEP da un file .gec."""
    text = path.read_text(encoding="latin1", errors="replace")
    lines = [ln.strip() for ln in text.splitlines()]

    epsg = None
    xcell = None
    markers: list[tuple[int, float, float, float]] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        if line == "<UTM_ZONE>" and i + 1 < len(lines):
            parts = [p.strip() for p in lines[i + 1].split(",")]
            try:
                zone = int(float(parts[0]))
                hemisphere = int(float(parts[1])) if len(parts) > 1 else 1
                epsg = 32600 + zone if hemisphere == 1 else 32700 + zone
            except (ValueError, IndexError):
                pass
        elif line == "<XCELL>" and i + 1 < len(lines):
            try:
                xcell = float(lines[i + 1])
            except ValueError:
                pass
        elif line == "<VALID_MARKERS_SWEEP>" and i + 1 < len(lines):
            try:
                n_markers = int(float(lines[i + 1]))
            except ValueError:
                n_markers = 0
            for j in range(n_markers):
                idx = i + 2 + j
                if idx >= len(lines) or not lines[idx]:
                    break
                parts = [p.strip() for p in lines[idx].split(",")]
                if len(parts) < 6:
                    continue
                try:
                    # gli indici .gec sono 1-based (il primo marker e' la
                    # traccia 1, l'ultimo == n_tracce): riportali a 0-based.
                    trace_idx = max(0, int(float(parts[0])) - 1)
                    x = float(parts[3])
                    y = float(parts[4])
                    z = float(parts[5])
                    markers.append((trace_idx, x, y, z))
                except (ValueError, IndexError):
                    continue
            i += n_markers
        i += 1

    return {"epsg": epsg, "xcell": xcell, "markers": markers}


def _parse_sav(path: Path) -> list[tuple[int, float, float, float]]:
    """Fallback: tabella marker salvata in <base>.DT.gps.sav (CSV semplice).

    Colonne: UTM_X, UTM_Y, elev, marker#, trace#, ... (resto ignorato).
    """
    markers: list[tuple[int, float, float, float]] = []
    for line in path.read_text(encoding="latin1", errors="replace").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            x = float(parts[0])
            y = float(parts[1])
            z = float(parts[2])
            trace_idx = int(float(parts[4]))
            markers.append((trace_idx, x, y, z))
        except (ValueError, IndexError):
            continue
    return markers


def _sibling(path: Path, suffix: str) -> Path | None:
    candidate = path.with_suffix(suffix)
    if candidate.exists():
        return candidate
    # Gestisce doppie estensioni tipo "<base>.DT.gps.sav"
    candidate = Path(str(path) + suffix)
    if candidate.exists():
        return candidate
    return None


def _interp_positions(markers: list[tuple[int, float, float, float]], n_traces: int):
    if not markers:
        raise DtSliceReadError("Nessun marker di posizione disponibile (.gec/.sav mancanti o vuoti)")

    markers = sorted(markers, key=lambda m: m[0])
    trace_idx = np.array([m[0] for m in markers], dtype=np.float64)
    xs = np.array([m[1] for m in markers], dtype=np.float64)
    ys = np.array([m[2] for m in markers], dtype=np.float64)
    zs = np.array([m[3] for m in markers], dtype=np.float64)

    all_idx = np.arange(n_traces, dtype=np.float64)
    easting = np.interp(all_idx, trace_idx, xs)
    northing = np.interp(all_idx, trace_idx, ys)
    altitude = np.interp(all_idx, trace_idx, zs)
    return easting, northing, altitude


# ---------------------------------------------------------------------------
# Reader principale
# ---------------------------------------------------------------------------

def read_dtslice(path: str) -> OgprProfile:
    """Legge una linea raw .DT (+ sidecar) come `OgprProfile`."""
    p = Path(path)
    if not p.exists():
        raise DtSliceReadError(f"File non trovato: {path}")

    parse_warnings: list[str] = []

    data, header = _read_dt_binary(p)
    n_samples, n_traces = data.shape
    if header.get("header_warning"):
        parse_warnings.append(header["header_warning"])

    gec_path = _sibling(p, ".gec")
    sav_path = _sibling(p, ".gps.sav")
    gps_path = _sibling(p, ".gps")

    epsg = 32631
    xcell = header.get("trace_spacing_m")
    markers: list[tuple[int, float, float, float]] = []

    if gec_path is not None:
        gec = _parse_gec(gec_path)
        if gec["epsg"]:
            epsg = gec["epsg"]
        if gec["xcell"]:
            xcell = gec["xcell"]
        markers = gec["markers"]
        if not markers:
            parse_warnings.append(f"Tabella marker vuota in {gec_path.name}")
    else:
        parse_warnings.append(f"File .gec non trovato per {p.name}: EPSG assunto {epsg}")

    if not markers and sav_path is not None:
        markers = _parse_sav(sav_path)
        parse_warnings.append(f"Posizionamento da fallback {sav_path.name}")

    has_gps = bool(markers)
    if has_gps:
        max_marker_trace = max(m[0] for m in markers)
        if max_marker_trace >= n_traces:
            parse_warnings.append(
                f"Indici marker oltre il numero di tracce lette ({max_marker_trace} >= {n_traces})"
            )
        if max_marker_trace < n_traces - 1:
            parse_warnings.append(
                f"Marker GPS coprono solo {max_marker_trace + 1}/{n_traces} tracce: "
                f"posizione estrapolata (costante) oltre l'ultimo marker"
            )
        easting, northing, altitude = _interp_positions(markers, n_traces)
    else:
        # Nessun sidecar di posizionamento GNSS (.gec / .DT.gps.sav assente o
        # tabella marker vuota). Il profilo resta comunque elaborabile: si
        # assegnano coordinate locali sintetiche lungo l'asse X (passo = xcell
        # dell'header, fallback 0.05 m) con origine (0, 0). La geometria reale
        # del rilievo puo' poi essere definita col dialog di
        # "Posizionamento Griglia Manuale" nel viewer profili.
        step_m = float(xcell) if (xcell is not None and xcell == xcell and xcell > 0) else 0.05
        parse_warnings.append(
            f"Nessun dato di posizionamento GNSS per {p.name} "
            f"(.gec / .DT.gps.sav assente o vuoto): assegnate coordinate locali "
            f"sintetiche lungo l'asse X con passo {step_m:.3f} m. Il profilo e' "
            f"comunque elaborabile; usa 'Posizionamento Griglia Manuale' per "
            f"definire la geometria reale del rilievo."
        )
        easting = np.arange(n_traces, dtype=np.float64) * step_m
        northing = np.zeros(n_traces, dtype=np.float64)
        altitude = np.zeros(n_traces, dtype=np.float64)

    d_east = np.diff(easting, prepend=easting[0])
    d_north = np.diff(northing, prepend=northing[0])
    heading = np.degrees(np.arctan2(d_east, d_north)) % 360.0
    distances = np.concatenate(([0.0], np.cumsum(np.hypot(d_east[1:], d_north[1:]))))

    channel = OgprChannel(
        channel_idx=0,
        data=data,
        easting=easting,
        northing=northing,
        altitude=altitude,
        heading=heading,
        distances=distances,
    )

    n_samples_val = header["n_samples"]
    time_window_s = header["time_window_s"]
    dt_ns = (time_window_s / n_samples_val) * 1e9 if n_samples_val else 0.0
    frequency_mhz, freq_warning = _resolve_antenna_frequency_mhz(
        p, float(header.get("frequency_hz") or 0.0)
    )
    if freq_warning:
        parse_warnings.append(freq_warning)
    velocity_m_s = 0.9e8  # default 0.9 m/ns (materiale generico), nessun dato di velocita' nel .gec

    return OgprProfile(
        path=str(p),
        version_major=0,
        version_minor=0,
        swath_name=p.stem,
        swath_id=p.stem,
        array_id=0,
        n_samples=n_samples_val,
        n_channels=1,
        n_slices=n_traces,
        sampling_step_m=xcell or 0.0,
        sampling_time_ns=dt_ns,
        velocity_m_s=velocity_m_s,
        frequency_mhz=frequency_mhz,
        polarization="",
        epsg=epsg,
        value_type="amplitude",
        channels=[channel],
        header_raw=header,
        value_type_raw="int16",
        raw_dtype="int16",
        byte_order="<",
        md5_verified=False,
        md5_scope="",
        parse_warnings=parse_warnings,
    )


@functools.lru_cache(maxsize=32)
def _read_dtslice_cached_impl(path: str, mtime_ns: int, size: int, freq_sidecar_sig: str) -> OgprProfile:
    _ = (mtime_ns, size, freq_sidecar_sig)
    return read_dtslice(path)


def _freq_sidecar_signature(p: Path) -> str:
    """Firma (mtime) degli override di frequenza, per invalidare la cache."""
    sig = []
    for cand in (
        Path(str(p) + ".freq"),
        p.with_suffix(".freq"),
        p.parent / "antenna_freq_mhz.txt",
    ):
        try:
            sig.append(f"{cand.name}:{cand.stat().st_mtime_ns}")
        except OSError:
            sig.append(f"{cand.name}:-")
    return "|".join(sig)


def read_dtslice_cached(path: str) -> OgprProfile:
    p = Path(path).expanduser().resolve()
    st = p.stat()
    mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
    size = int(st.st_size)
    return _read_dtslice_cached_impl(str(p), mtime_ns, size, _freq_sidecar_signature(p))


def clear_dtslice_cache() -> None:
    _read_dtslice_cached_impl.cache_clear()
