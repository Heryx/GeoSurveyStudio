# -*- coding: utf-8 -*-
"""Ampiezze per singola time-slice, NON distruttive (pannello principale).

Le stesse rifiniture del viewer (taglio dei picchi, normalizzazione, scala,
despeckle, low-pass, alto-passa) piu' i limiti colore, applicate a *una* immagine
gia' salvata. L'originale non viene mai toccato: le impostazioni stanno in un
file allegato al progetto (``timeslice_amplitude.json``) e il risultato dei
filtri in un raster derivato (cartella ``timeslice_amplitude/``). Alla riapertura
del progetto le immagini con impostazioni vengono caricate dal derivato
(rigenerato se manca o se l'originale e' piu' recente).

Nessuna dipendenza Qt: provabile con qualunque Python che abbia numpy/scipy/GDAL.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, fields
from types import SimpleNamespace

import numpy as np

STORE_NAME = "timeslice_amplitude.json"
DERIVED_DIR = "timeslice_amplitude"
STORE_VERSION = 1

NORMALIZE_MODES = ("none", "median")
AMPLITUDE_SCALES = ("linear", "sqrt", "db")
DESPECKLE_SIZES = (0, 3, 5)
LOWPASS_SIZES = (0, 3, 5, 7, 9, 11, 15)
LIMIT_MODES = ("auto", "percentile", "fixed")


class AmplitudeSettingsError(ValueError):
    pass


@dataclass
class AmplitudeSettings:
    """Rifinitura di una singola slice. I default = nessun cambiamento."""

    # --- valori (cambiano i pixel -> raster derivato) ----------------------
    despike_pct: float = 100.0      # 100 = off; tronca i valori sopra questo percentile
    normalize: str = "none"         # "none" | "median"
    amp_scale: str = "linear"       # "linear" | "sqrt" | "db"  (solo ampiezze >= 0)
    despeckle: int = 0              # 0 | 3 | 5
    smooth_lowpass: int = 0         # 0 | 3 | 5 | 7 | 9 | 11 | 15
    highpass_m: float = 0.0         # 0 = off
    # --- visualizzazione (solo renderer) -----------------------------------
    limits_mode: str = "auto"       # "auto" (min/max) | "percentile" | "fixed"
    p_lo: float = 2.0
    p_hi: float = 98.0
    vmin: float | None = None
    vmax: float | None = None

    def validate(self) -> "AmplitudeSettings":
        if not (50.0 <= float(self.despike_pct) <= 100.0):
            raise AmplitudeSettingsError("despike_pct deve stare in [50, 100]")
        if self.normalize not in NORMALIZE_MODES:
            raise AmplitudeSettingsError(f"normalize {self.normalize!r} non valido")
        if self.amp_scale not in AMPLITUDE_SCALES:
            raise AmplitudeSettingsError(f"amp_scale {self.amp_scale!r} non valido")
        if int(self.despeckle) not in DESPECKLE_SIZES:
            raise AmplitudeSettingsError(f"despeckle {self.despeckle!r} non valido")
        if int(self.smooth_lowpass) not in LOWPASS_SIZES:
            raise AmplitudeSettingsError(f"smooth_lowpass {self.smooth_lowpass!r} non valido")
        if float(self.highpass_m) < 0.0:
            raise AmplitudeSettingsError("highpass_m deve essere >= 0")
        if self.limits_mode not in LIMIT_MODES:
            raise AmplitudeSettingsError(f"limits_mode {self.limits_mode!r} non valido")
        if not (0.0 <= float(self.p_lo) < float(self.p_hi) <= 100.0):
            raise AmplitudeSettingsError("percentili non validi (0 <= lo < hi <= 100)")
        if self.limits_mode == "fixed":
            if self.vmin is None or self.vmax is None or not float(self.vmax) > float(self.vmin):
                raise AmplitudeSettingsError("limiti fissi: serve vmax > vmin")
        return self

    @property
    def pixel_tools_active(self) -> bool:
        """True se almeno un filtro cambia i valori (serve il raster derivato)."""
        return bool(
            float(self.despike_pct) < 100.0 or self.normalize != "none"
            or self.amp_scale != "linear" or int(self.despeckle) > 0
            or int(self.smooth_lowpass) > 0 or float(self.highpass_m) > 0.0
        )

    @property
    def is_default(self) -> bool:
        return self == AmplitudeSettings()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data) -> "AmplitudeSettings":
        known = {f.name for f in fields(cls)}
        clean = {k: v for k, v in dict(data or {}).items() if k in known}
        return cls(**clean).validate()


# ---------------------------------------------------------------------------
# grid processing
# ---------------------------------------------------------------------------


def process_grid(grid, settings: AmplitudeSettings, resolution: float) -> np.ndarray:
    """Filtri di ``settings`` su una griglia (NaN = fuori dati). Il contorno NaN
    resta quello dell'ingresso. Scala/normalizzazione solo se i valori sono >= 0
    (come ``slice_engine.finish_grid`` per le ampiezze con segno)."""
    from .slice_engine import finish_grid
    from .slicing import clip_outliers

    g = np.asarray(grid, dtype=np.float64)
    fin = np.isfinite(g)
    if not fin.any():
        return g.astype(np.float32)
    signed = bool(g[fin].min() < 0.0)
    if float(settings.despike_pct) < 100.0:
        g = clip_outliers(g, float(settings.despike_pct), symmetric=signed)
    lp = int(settings.smooth_lowpass)
    spec = SimpleNamespace(
        amplitude_mode="signed_mean" if signed else "env_peak",
        normalize=settings.normalize,
        amp_scale=settings.amp_scale,
        despeckle=int(settings.despeckle),
        grid_bg_spec="",
        smooth_sigma=(lp / 3.0 if lp else 0.0),
        highpass_m=float(settings.highpass_m),
    )
    return finish_grid(g, spec, float(resolution) or 1.0)


def display_limits(settings: AmplitudeSettings, grid) -> tuple[float, float] | None:
    """(vmin, vmax) della scala colore per ``grid`` (gia' filtrata), o None."""
    a = np.asarray(grid, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    if settings.limits_mode == "fixed":
        lo, hi = float(settings.vmin), float(settings.vmax)
    elif settings.limits_mode == "percentile":
        lo, hi = (float(x) for x in np.percentile(a, [settings.p_lo, settings.p_hi]))
    else:
        lo, hi = float(a.min()), float(a.max())
    if not hi > lo:
        hi = lo + 1e-6
    return lo, hi


# ---------------------------------------------------------------------------
# raster I/O (GDAL), l'originale e' solo letto
# ---------------------------------------------------------------------------


def read_band(path: str):
    """``(array float64 con NaN, resolution)`` della banda 1 (orientamento nativo)."""
    from osgeo import gdal

    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        raise IOError(f"impossibile aprire il raster: {path}")
    try:
        band = ds.GetRasterBand(1)
        arr = np.asarray(band.ReadAsArray(), dtype=np.float64)
        nod = band.GetNoDataValue()
        gt = ds.GetGeoTransform(can_return_null=True) or (0, 1, 0, 0, 0, -1)
    finally:
        ds = None
    if nod is not None and np.isfinite(float(nod)):
        arr[arr == float(nod)] = np.nan
    arr[~np.isfinite(arr)] = np.nan
    return arr, float(abs(gt[1])) or 1.0


def write_adjusted_raster(src_path: str, settings: AmplitudeSettings, out_path: str) -> str:
    """Copia di ``src_path`` (georeferenziazione e metadati inclusi) con la banda 1
    filtrata. Scrive solo ``out_path``; ritorna il percorso."""
    from osgeo import gdal

    arr, res = read_band(src_path)
    new = process_grid(arr, settings, res)
    src = gdal.Open(str(src_path), gdal.GA_ReadOnly)
    if src is None:
        raise IOError(f"impossibile aprire il raster: {src_path}")
    nod = src.GetRasterBand(1).GetNoDataValue()
    nod = -9999.0 if nod is None or not np.isfinite(float(nod)) else float(nod)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp = out_path + ".part.tif"
    drv = gdal.GetDriverByName("GTiff")
    dst = drv.CreateCopy(tmp, src, 0, options=["COMPRESS=LZW", "TILED=YES"])
    src = None
    if dst is None:
        raise IOError(f"impossibile creare il raster derivato: {out_path}")
    band = dst.GetRasterBand(1)
    band.WriteArray(np.where(np.isfinite(new), new, nod).astype(np.float32))
    band.SetNoDataValue(nod)
    band.FlushCache()
    dst.FlushCache()
    dst = None
    os.replace(tmp, out_path)
    return out_path


# ---------------------------------------------------------------------------
# file allegato al progetto
# ---------------------------------------------------------------------------


def _norm(p: str) -> str:
    return os.path.normcase(os.path.normpath(str(p)))


def image_key(project_root: str, src_path: str) -> str:
    """Chiave stabile dell'immagine: percorso relativo alla radice (con '/'),
    assoluto se fuori radice."""
    root = os.path.abspath(project_root)
    p = os.path.abspath(src_path)
    try:
        rel = os.path.relpath(p, root)
    except ValueError:
        return p.replace("\\", "/")
    if rel.startswith(".."):
        return p.replace("\\", "/")
    return rel.replace("\\", "/")


def store_path(project_root: str) -> str:
    return os.path.join(project_root, STORE_NAME)


def load_store(project_root: str) -> dict:
    """``{chiave: {"settings": {...}}}``; vuoto se il file manca o e' illeggibile."""
    try:
        with open(store_path(project_root), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        images = data.get("images", {}) if isinstance(data, dict) else {}
        return images if isinstance(images, dict) else {}
    except (OSError, ValueError):
        return {}


def save_store(project_root: str, images: dict) -> None:
    payload = {"format": "rasterlinker_timeslice_amplitude", "version": STORE_VERSION,
               "images": images}
    path = store_path(project_root)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def get_settings(project_root: str, src_path: str) -> AmplitudeSettings | None:
    entry = load_store(project_root).get(image_key(project_root, src_path))
    if not entry:
        return None
    try:
        return AmplitudeSettings.from_dict(entry.get("settings"))
    except AmplitudeSettingsError:
        return None


def set_settings(project_root: str, src_path: str, settings: AmplitudeSettings) -> None:
    """Registra le impostazioni; quelle di default tolgono la voce."""
    images = load_store(project_root)
    key = image_key(project_root, src_path)
    if settings.is_default:
        images.pop(key, None)
    else:
        images[key] = {"settings": settings.validate().to_dict()}
    save_store(project_root, images)


def derived_path(project_root: str, src_path: str) -> str:
    key = image_key(project_root, src_path)
    tag = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    stem = os.path.splitext(os.path.basename(src_path))[0]
    return os.path.join(project_root, DERIVED_DIR, f"{stem}__{tag}.tif")


def ensure_derived(project_root: str, src_path: str, settings: AmplitudeSettings,
                   *, force: bool = False) -> str:
    """Raster da mostrare per ``src_path`` con ``settings``: l'originale se nessun
    filtro cambia i valori, altrimenti il derivato (rigenerato se manca, se e'
    piu' vecchio dell'originale, o con ``force``; stale se cambiano le impostazioni
    e il chiamante passa ``force=True``)."""
    if not settings.pixel_tools_active:
        return src_path
    out = derived_path(project_root, src_path)
    fresh = (
        not force and os.path.exists(out)
        and os.path.getmtime(out) >= os.path.getmtime(src_path)
    )
    if not fresh:
        write_adjusted_raster(src_path, settings, out)
    return out


def remove_derived(project_root: str, src_path: str) -> None:
    out = derived_path(project_root, src_path)
    try:
        os.remove(out)
    except OSError:
        pass


def is_derived_of(project_root: str, layer_source: str, src_path: str) -> bool:
    return _norm(layer_source.split("|")[0]) == _norm(derived_path(project_root, src_path))


__all__ = [
    "AmplitudeSettings", "AmplitudeSettingsError", "process_grid", "display_limits",
    "read_band", "write_adjusted_raster", "image_key", "load_store", "save_store",
    "get_settings", "set_settings", "derived_path", "ensure_derived", "remove_derived",
    "is_derived_of", "STORE_NAME", "DERIVED_DIR",
]
