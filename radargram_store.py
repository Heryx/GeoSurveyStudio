# -*- coding: utf-8 -*-
"""Accesso in sola lettura ai radargrammi processati di una variante (.npz).

Serve al pannello agganciabile di QGIS (`gpr_radargram_dock`), che deve mostrare
i profili anche a Profile Viewer chiuso: niente Qt, niente QGIS, solo numpy.

- ``VariantStore`` apre una cartella-variante (`Processing/processed/<nome>`),
  legge subito solo i metadati di ogni profilo e carica un canale alla volta
  (i .npz sono uno zip con un array per canale: un canale = pochi MB).
- ``VariantStore.nearest`` trova il profilo/canale/traccia piu' vicino a un
  punto E/N (per il clic sulla mappa).
- ``slice_info_for_path`` ricava da ``sliceset.json`` profondita' e variante di
  una time-slice, cosi' il pannello sa quale finestra di profondita' mostrare.

Geometria: la fonte unica e' il profilo in memoria del Profile Viewer (pagina di
posizionamento). Se il viewer e' aperto si passa ``geometry_provider`` e le
coordinate vengono da li'; a viewer chiuso resta la fotografia scritta nel .npz
al momento del salvataggio (``ChannelData.geometry_source`` dice quale).
"""

from __future__ import annotations

import json
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .gpr_hyperbola_math import m_ns_to_velocity_m_s
from .gpr_npz_io import aligned_geometry_columns, list_variants
from .gpr_velocity_store import read_velocity_overrides, resolve_velocity_m_ns
from .slice_manifest import SliceSetError, has_sliceset, read_sliceset

__all__ = [
    "ProfileInfo",
    "ChannelData",
    "VariantRef",
    "VariantStore",
    "discover_variants",
    "processed_dir_for",
    "slice_info_for_path",
    "variant_dir_from_slice_dir",
]


def _natural_key(text: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(text))]


# ---------------------------------------------------------------------------
# Elenco varianti
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VariantRef:
    name: str
    path: str
    n_files: int


def processed_dir_for(root: str) -> str:
    """`processed/` di un progetto: accetta la radice, la cartella Processing
    o la stessa `processed/`. Stringa vuota se non esiste."""
    root = os.path.normpath(str(root or "").strip())
    if not root or root == ".":
        return ""
    for cand in (
        os.path.join(root, "Processing", "processed"),
        os.path.join(root, "processed"),
        root,
    ):
        if os.path.basename(cand).lower() == "processed" and os.path.isdir(cand):
            return cand
    return ""


def discover_variants(root: str) -> list[VariantRef]:
    """Varianti con almeno un .npz sotto `processed/`, in ordine alfabetico."""
    processed = processed_dir_for(root)
    if not processed:
        return []
    out = []
    for d in list_variants(processed):
        try:
            n = sum(1 for f in os.listdir(d) if f.lower().endswith(".npz"))
        except OSError:
            continue
        if n:
            out.append(VariantRef(os.path.basename(os.path.normpath(d)), os.path.normpath(d), n))
    return out


# ---------------------------------------------------------------------------
# Dati
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProfileInfo:
    path: str                 # .npz
    stem: str
    n_channels: int
    n_traces: tuple
    n_samples: tuple
    dt_ns: float
    velocity_m_s: float
    frequency_mhz: float
    epsg: int
    sampling_step_m: float
    view_geom: object

    @property
    def dz_m(self) -> float:
        """Profondita' (m) di una riga del radargramma."""
        return float(self.dt_ns) * float(self.velocity_m_s) / 1e9 / 2.0


@dataclass
class ChannelData:
    data: np.ndarray          # (n_campioni, n_tracce) float32
    dist: np.ndarray          # (n_tracce,) m dall'inizio, monotona
    east: np.ndarray
    north: np.ndarray
    dz_m: float
    geometry_source: str      # "viewer" (profilo in memoria) | "npz" (fotografia salvata)

    @property
    def n_traces(self) -> int:
        return int(self.data.shape[1])

    @property
    def length_m(self) -> float:
        return float(self.dist[-1]) if self.dist.size else 0.0

    @property
    def depth_max_m(self) -> float:
        return float(self.data.shape[0]) * self.dz_m

    def trace_at_distance(self, x: float) -> int:
        if self.dist.size == 0:
            return 0
        j = int(np.searchsorted(self.dist, float(x)))
        j = int(np.clip(j, 0, self.dist.size - 1))
        if j > 0 and abs(self.dist[j - 1] - x) <= abs(self.dist[j] - x):
            j -= 1
        return j

    def row_at_depth(self, z: float) -> int:
        n = int(self.data.shape[0])
        if n <= 0 or self.dz_m <= 0:
            return 0
        # la riga i occupa [i*dz, (i+1)*dz): stessa convenzione dell'immagine del viewer
        return int(np.clip(np.floor(float(z) / self.dz_m), 0, n - 1))

    def position(self, trace: int):
        """(E, N) della traccia, None se la geometria non e' valida."""
        if not (0 <= trace < self.east.size):
            return None
        e, n = float(self.east[trace]), float(self.north[trace])
        return (e, n) if np.isfinite(e) and np.isfinite(n) else None

    def amplitude_at(self, trace: int, row: int) -> float:
        """Ampiezza campionata al clic (usata per il campo 'amplitude' auto-compilato
        delle geometrie disegnate sul radargramma)."""
        n_rows, n_traces = self.data.shape
        if not (0 <= row < n_rows and 0 <= trace < n_traces):
            return float("nan")
        return float(self.data[row, trace])


def _clean_distance_axis(dist: np.ndarray, n_t: int, step_m: float) -> np.ndarray:
    """Asse distanze monotono che parte da 0; se inservibile, tracce equispaziate."""
    d = np.asarray(dist, dtype=np.float64)
    if d.size != n_t or n_t < 2:
        d = np.zeros(0)
    else:
        finite = np.isfinite(d)
        if not finite.all():
            if not finite.any():
                d = np.zeros(0)
            else:
                idx = np.arange(d.size, dtype=np.float64)
                d = np.interp(idx, idx[finite], d[finite])
    if d.size:
        d = np.maximum.accumulate(d - d[0])
    if d.size < 2 or float(d[-1]) <= 1e-9:
        step = float(step_m) if step_m and step_m > 0 else 0.05
        d = np.arange(int(n_t), dtype=np.float64) * step
    return d


class VariantStore:
    """Una variante processata: profili, canali e ricerca del punto piu' vicino."""

    _MAX_INDEX_POINTS = 1_500_000    # oltre, l'indice spaziale decima le tracce
    _CHANNEL_CACHE = 6

    def __init__(self, variant_dir: str,
                 geometry_provider: Optional[Callable[[str], object]] = None):
        self.path = os.path.normpath(str(variant_dir or ""))
        self.name = os.path.basename(self.path)
        self.geometry_provider = geometry_provider
        self.profiles: list[ProfileInfo] = []
        self._channels: "OrderedDict[tuple, ChannelData]" = OrderedDict()
        self._index = None
        self._signature = self._disk_signature()
        self._scan()

    # -- lettura metadati ------------------------------------------------

    def _scan(self) -> None:
        try:
            names = [f for f in os.listdir(self.path) if f.lower().endswith(".npz")]
        except OSError:
            names = []
        for fname in sorted(names, key=_natural_key):
            fp = os.path.join(self.path, fname)
            try:
                with np.load(fp, allow_pickle=False) as npz:
                    meta = json.loads(str(npz["meta"]))
            except Exception:
                continue        # file illeggibile o non e' un radargramma processato
            n_ch = int(meta.get("n_channels", 0) or 0)
            if n_ch <= 0:
                continue
            self.profiles.append(ProfileInfo(
                path=fp,
                stem=os.path.splitext(fname)[0],
                n_channels=n_ch,
                n_traces=tuple(int(v) for v in (meta.get("n_traces_by_channel") or [])),
                n_samples=tuple(int(v) for v in (meta.get("n_samples_by_channel") or [])),
                dt_ns=float(meta.get("sampling_time_ns", 0.0) or 0.0),
                velocity_m_s=float(meta.get("velocity_m_s", 0.0) or 0.0),
                frequency_mhz=float(meta.get("frequency_mhz", 0.0) or 0.0),
                epsg=int(meta.get("epsg", 0) or 0),
                sampling_step_m=float(meta.get("sampling_step_m", 0.0) or 0.0),
                view_geom=meta.get("view_geom"),
            ))

    def _disk_signature(self) -> tuple:
        """(numero di .npz, data di modifica piu' recente): cambia se la variante viene riscritta."""
        try:
            names = [f for f in os.listdir(self.path) if f.lower().endswith(".npz")]
            return (len(names), max((os.path.getmtime(os.path.join(self.path, f)) for f in names),
                                    default=0.0))
        except OSError:
            return (0, 0.0)

    def is_stale(self) -> bool:
        """True se i .npz sono stati riscritti (o aggiunti/tolti) dopo l'apertura: la cache e' vecchia."""
        return self._disk_signature() != self._signature

    def __len__(self) -> int:
        return len(self.profiles)

    def profile_index_for_stem(self, stem: str) -> int:
        for i, p in enumerate(self.profiles):
            if p.stem == stem:
                return i
        return -1

    # -- geometria --------------------------------------------------------

    def invalidate_geometry(self) -> None:
        """Il posizionamento e' cambiato: si rileggono coordinate e indice spaziale."""
        self._channels.clear()
        self._index = None

    def reference_velocity_m_ns(self, stem: str) -> dict:
        """Velocita' preimpostata (m/ns) per un profilo: override per-profilo, poi
        default di variante, poi la costante condivisa 0.1 m/ns - vedi
        gpr_velocity_store.py. Usata dal popup informativo del puntatore e dai campi
        auto-compilati delle geometrie disegnate in RadargramDock."""
        return resolve_velocity_m_ns(self.path, stem)

    def _explicit_velocity_override_m_ns(self, stem: str):
        """Solo un override esplicito (per-profilo o default di variante) salvato
        dall'utente, None se non c'e' nulla - a differenza di
        `reference_velocity_m_ns` non ripiega mai sulla costante 0.1. Usata per
        l'asse di profondita' del canale: senza un override esplicito si continua a
        usare la velocita' dell'header strumento (`ProfileInfo.dz_m`), cosi' i
        profili mai toccati non cambiano scala sotto l'utente."""
        data = read_velocity_overrides(self.path)
        entry = data["profiles"].get(str(stem or "")) or data["default"]
        if entry and entry.get("velocity_m_ns") is not None:
            return float(entry["velocity_m_ns"])
        return None

    def effective_velocity_m_s(self, pi: int) -> float:
        """Velocita' (m/s) che genera davvero `ChannelData.dz_m` per quel profilo:
        override esplicito se c'e', altrimenti l'header strumento (`velocity_m_s`),
        altrimenti la costante condivisa 0.1 m/ns. Usata dal puntatore "Calcola con
        iperbole" per restare coerente con l'asse di profondita' mostrato (`v_ref`
        deve essere la stessa velocita' che ha prodotto la profondita' sotto
        l'apice, non un valore indipendente)."""
        info = self.profiles[int(pi)]
        v_override_mns = self._explicit_velocity_override_m_ns(info.stem)
        if v_override_mns is not None:
            return m_ns_to_velocity_m_s(v_override_mns)
        v = float(info.velocity_m_s or 0.0)
        if np.isfinite(v) and v > 0:
            return v
        return m_ns_to_velocity_m_s(resolve_velocity_m_ns(self.path, info.stem)["velocity_m_ns"])

    def _geometry(self, npz, info: ProfileInfo, ci: int):
        """(east, north, dist, sorgente) del canale, con `npz` gia' aperto."""
        n_t = int(info.n_traces[ci]) if ci < len(info.n_traces) else 0
        provider = self.geometry_provider
        if provider is not None:
            try:
                src = provider(info.stem)
                n_src = int(getattr(src, "n_channels", 0) or 0)
                if src is not None and n_src > 0:
                    cols = aligned_geometry_columns(
                        src.channel(min(ci, n_src - 1)), n_t, info.view_geom)
                    if cols is not None:
                        return cols["easting"], cols["northing"], cols["distances"], "viewer"
            except Exception:
                pass    # provider guasto: si ripiega sulla fotografia del .npz
        east = np.asarray(npz[f"east_{ci}"], dtype=np.float64)
        north = np.asarray(npz[f"north_{ci}"], dtype=np.float64)
        dist = np.asarray(npz[f"dist_{ci}"], dtype=np.float64)
        return east, north, dist, "npz"

    # -- canali -----------------------------------------------------------

    def channel(self, pi: int, ci: int) -> ChannelData:
        key = (int(pi), int(ci))
        hit = self._channels.get(key)
        if hit is not None:
            self._channels.move_to_end(key)
            return hit
        info = self.profiles[key[0]]
        if not (0 <= key[1] < info.n_channels):
            raise IndexError(f"canale {ci} fuori range (0..{info.n_channels - 1})")
        with np.load(info.path, allow_pickle=False) as npz:
            data = np.asarray(npz[f"data_{key[1]}"], dtype=np.float32)
            east, north, dist, source = self._geometry(npz, info, key[1])
        n_t = int(data.shape[1])
        if east.size != n_t or north.size != n_t:
            east = np.full(n_t, np.nan)
            north = np.full(n_t, np.nan)
        # Profondita' per riga: se l'utente ha salvato esplicitamente una velocita'
        # (manuale o da fit iperbole) per questo profilo/variante la si usa qui,
        # cosi' l'asse del radargramma nel dock riflette il valore verificato;
        # altrimenti si resta sul dz_m di sempre (velocita' dell'header strumento),
        # cosi' i profili mai toccati non cambiano scala sotto l'utente. Isolato a
        # RadargramDock: non tocca il rendering del Profile Viewer ne' lo slice engine.
        v_override_mns = self._explicit_velocity_override_m_ns(info.stem)
        if v_override_mns is not None:
            dz_m = float(info.dt_ns) * m_ns_to_velocity_m_s(v_override_mns) / 1e9 / 2.0
        else:
            dz_m = info.dz_m
        out = ChannelData(
            data=data,
            dist=_clean_distance_axis(dist, n_t, info.sampling_step_m),
            east=east, north=north,
            dz_m=dz_m,
            geometry_source=source,
        )
        self._channels[key] = out
        while len(self._channels) > self._CHANNEL_CACHE:
            self._channels.popitem(last=False)
        return out

    # -- punto piu' vicino -------------------------------------------------

    def _build_index(self):
        xs, ys, ids = [], [], []
        total = 0
        for info in self.profiles:
            total += sum(info.n_traces)
        step = max(1, int(np.ceil(total / float(self._MAX_INDEX_POINTS))))
        for pi, info in enumerate(self.profiles):
            try:
                npz = np.load(info.path, allow_pickle=False)
            except Exception:
                continue
            with npz:
                for ci in range(info.n_channels):
                    try:
                        e, n, _d, _src = self._geometry(npz, info, ci)
                    except Exception:
                        continue
                    if e.size == 0 or e.size != n.size:
                        continue
                    ok = np.isfinite(e) & np.isfinite(n)
                    tr = np.flatnonzero(ok)[::step]
                    if tr.size == 0:
                        continue
                    xs.append(e[tr])
                    ys.append(n[tr])
                    ids.append(np.column_stack([
                        np.full(tr.size, pi), np.full(tr.size, ci), tr,
                    ]))
        if not xs:
            return (np.zeros(0), np.zeros(0), np.zeros((0, 3), dtype=np.int64), step)
        return (np.concatenate(xs), np.concatenate(ys),
                np.concatenate(ids).astype(np.int64), step)

    def nearest(self, east: float, north: float, tol_m: float = 2.0):
        """(distanza m, profilo, canale, traccia) del punto rilevato piu' vicino a
        (east, north), oppure None se nessuno entra in `tol_m`."""
        if self._index is None:
            self._index = self._build_index()
        xs, ys, ids, step = self._index
        if xs.size == 0:
            return None
        d2 = (xs - float(east)) ** 2 + (ys - float(north)) ** 2
        k = int(np.argmin(d2))
        pi, ci, tr = (int(v) for v in ids[k])
        best = float(d2[k]) ** 0.5
        if step > 1:
            # indice decimato: si affina sulla traccia intera del canale trovato
            ch = self.channel(pi, ci)
            lo, hi = max(0, tr - step), min(ch.n_traces, tr + step + 1)
            e, n = ch.east[lo:hi], ch.north[lo:hi]
            dd = (e - float(east)) ** 2 + (n - float(north)) ** 2
            if np.isfinite(dd).any():
                j = int(np.nanargmin(dd))
                tr, best = lo + j, float(dd[j]) ** 0.5
        if best > float(tol_m):
            return None
        return best, pi, ci, tr

    def extent(self):
        """(xmin, ymin, xmax, ymax) di tutte le tracce, None se senza geometria."""
        if self._index is None:
            self._index = self._build_index()
        xs, ys = self._index[0], self._index[1]
        if xs.size == 0:
            return None
        return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


# ---------------------------------------------------------------------------
# Time-slice -> profondita' e variante
# ---------------------------------------------------------------------------

_SLICESET_CACHE: dict = {}


def _norm(p: str) -> str:
    return os.path.normcase(os.path.normpath(str(p or "")))


def slice_info_for_path(raster_path: str) -> Optional[dict]:
    """Profondita' e variante di una time-slice, letti da `sliceset.json` accanto al raster.

    Ritorna ``{"z_from", "z_to", "z_center", "variant", "dir", "epsg"}`` oppure
    None se il raster non fa parte di un set generato dal plugin.
    """
    path = str(raster_path or "").split("|")[0].strip()
    if not path:
        return None
    folder = os.path.dirname(os.path.normpath(path))
    if not folder or not has_sliceset(folder):
        return None
    manifest = os.path.join(folder, "sliceset.json")
    try:
        mtime = os.path.getmtime(manifest)
        cached = _SLICESET_CACHE.get(folder)
        if cached is not None and cached[0] == mtime:
            data = cached[1]
        else:
            data = read_sliceset(folder)
            _SLICESET_CACHE[folder] = (mtime, data)
    except (OSError, SliceSetError):
        return None
    grid = os.path.basename(path)
    for s in data.get("slices") or []:
        if str(s.get("grid") or "") != grid:
            continue
        try:
            z0, z1 = float(s["z_from"]), float(s["z_to"])
        except (KeyError, TypeError, ValueError):
            return None
        try:
            zc = float(s.get("z_center"))
        except (TypeError, ValueError):
            zc = 0.5 * (z0 + z1)
        lo, hi = (z0, z1) if z0 <= z1 else (z1, z0)
        source = data.get("source") if isinstance(data.get("source"), dict) else {}
        return {
            "z_from": lo, "z_to": hi, "z_center": zc,
            "variant": str(source.get("variant_id") or ""),
            "dir": folder,
            "epsg": data.get("crs_epsg"),
        }
    return None


def variant_dir_from_slice_dir(slice_dir: str, variant: str) -> str:
    """Cartella `processed/<variante>` che corrisponde a `Processing/timeslices/<gruppo>/`."""
    variant = str(variant or "").strip()
    if not variant or variant.lower() == "raw":
        return ""
    cur = os.path.normpath(str(slice_dir or ""))
    for _ in range(4):
        cur = os.path.dirname(cur)
        if not cur:
            break
        cand = os.path.join(cur, "processed", variant)
        if os.path.isdir(cand):
            return cand
    return ""
