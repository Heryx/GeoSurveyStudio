# -*- coding: utf-8 -*-
"""
GPR palettes
============
Caricamento di palette / color-ramp esterne (cpt-city SVG, GIMP .ggr, GMT .cpt)
e registrazione come colormap matplotlib, cosi' da poterle usare ovunque nel
plugin esattamente come le colormap native (per nome).

API principale:

    register_all()            -> list[str]   ri-scansiona le cartelle e registra
    import_palette_file(path)  -> str         importa un file nella cartella utente
    colormap_preview_pixmap(name, w, h) -> QPixmap
    make_cmap_icon(name, w, h) -> QIcon

Le palette "bundled" stanno in  <plugin>/palettes/*.svg
Le palette importate dall'utente vengono copiate in una cartella persistente del
profilo QGIS (indipendente dagli aggiornamenti del plugin).
"""

from __future__ import annotations

import os
import re
import glob
import shutil
from typing import Iterable, List, Optional, Sequence, Tuple

_BUNDLED_DIR = os.path.join(os.path.dirname(__file__), "palettes")

# Nomi registrati con successo nell'ultima chiamata a register_all().
PALETTE_NAMES: List[str] = []

# Prefisso opzionale per evitare collisioni con colormap native matplotlib.
# I nomi cpt-city non collidono normalmente, quindi si usa il nome "nudo";
# in caso di collisione con una colormap builtin si aggiunge il suffisso.
_SUPPORTED_EXT = (".svg", ".ggr", ".cpt")


# ----------------------------------------------------------------------------
# Cartelle
# ----------------------------------------------------------------------------
def user_palette_dir() -> str:
    """Cartella persistente dove finiscono le palette importate dall'utente."""
    base = None
    try:
        from qgis.core import QgsApplication  # type: ignore

        base = QgsApplication.qgisSettingsDirPath()
    except Exception:
        base = None
    if not base:
        base = os.path.expanduser("~")
    d = os.path.join(base, "gpr_palettes")
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return d


def palette_dirs() -> List[str]:
    dirs = []
    if os.path.isdir(_BUNDLED_DIR):
        dirs.append(_BUNDLED_DIR)
    ud = user_palette_dir()
    if os.path.isdir(ud):
        dirs.append(ud)
    return dirs


# ----------------------------------------------------------------------------
# Parser dei formati
# ----------------------------------------------------------------------------
_RE_STOP = re.compile(
    r"<stop\b[^>]*?offset\s*=\s*\"([^\"]+)\"[^>]*?stop-color\s*=\s*\"([^\"]+)\"",
    re.IGNORECASE | re.DOTALL,
)
_RE_RGB = re.compile(r"rgb\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)", re.I)


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _parse_color_token(tok: str) -> Optional[Tuple[float, float, float]]:
    tok = tok.strip()
    m = _RE_RGB.match(tok)
    if m:
        return (
            _clamp01(float(m.group(1)) / 255.0),
            _clamp01(float(m.group(2)) / 255.0),
            _clamp01(float(m.group(3)) / 255.0),
        )
    if tok.startswith("#"):
        h = tok[1:]
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        if len(h) >= 6:
            try:
                return (
                    int(h[0:2], 16) / 255.0,
                    int(h[2:4], 16) / 255.0,
                    int(h[4:6], 16) / 255.0,
                )
            except ValueError:
                return None
    return None


def _parse_svg(path: str) -> List[Tuple[float, Tuple[float, float, float]]]:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        txt = fh.read()
    stops: List[Tuple[float, Tuple[float, float, float]]] = []
    for m in _RE_STOP.finditer(txt):
        off_raw = m.group(1).strip()
        try:
            off = float(off_raw[:-1]) / 100.0 if off_raw.endswith("%") else float(off_raw)
        except ValueError:
            continue
        col = _parse_color_token(m.group(2))
        if col is None:
            continue
        stops.append((_clamp01(off), col))
    if not stops:
        stops = _stops_from_embedded_image(txt)
    return stops


_RE_DATA_IMG = re.compile(r"data:image/(?:png|jpe?g);base64,([A-Za-z0-9+/=\s]+)", re.I)


def _stops_from_embedded_image(txt: str, n: int = 128) -> List[Tuple[float, Tuple[float, float, float]]]:
    """SVG "colorbar" raster (<image> con PNG base64): campiona la barra al centro."""
    m = _RE_DATA_IMG.search(txt)
    if not m:
        return []
    try:
        import base64
        from qgis.PyQt.QtGui import QImage
        img = QImage()
        if not img.loadFromData(base64.b64decode(re.sub(r"\s+", "", m.group(1)))):
            return []
    except Exception:
        return []
    w, h = img.width(), img.height()
    if w < 2 or h < 1:
        return []
    horizontal = w >= h
    length = w if horizontal else h
    n = max(2, min(n, length))
    stops = []
    for i in range(n):
        t = i / (n - 1)
        k = int(round(t * (length - 1)))
        c = img.pixelColor(k, h // 2) if horizontal else img.pixelColor(w // 2, length - 1 - k)
        stops.append((t, (c.redF(), c.greenF(), c.blueF())))
    return stops


def _parse_ggr(path: str) -> List[Tuple[float, Tuple[float, float, float]]]:
    """GIMP gradient (.ggr): usa gli estremi di ogni segmento come stop."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    if not lines or not lines[0].lower().startswith("gimp gradient"):
        return []
    idx = 1
    if idx < len(lines) and lines[idx].lower().startswith("name:"):
        idx += 1
    try:
        nseg = int(lines[idx].split()[0])
    except (ValueError, IndexError):
        return []
    idx += 1
    stops: List[Tuple[float, Tuple[float, float, float]]] = []
    for i in range(nseg):
        if idx + i >= len(lines):
            break
        parts = lines[idx + i].split()
        if len(parts) < 11:
            continue
        try:
            left = float(parts[0])
            right = float(parts[2])
            lcol = (float(parts[3]), float(parts[4]), float(parts[5]))
            rcol = (float(parts[7]), float(parts[8]), float(parts[9]))
        except ValueError:
            continue
        stops.append((_clamp01(left), tuple(_clamp01(c) for c in lcol)))
        stops.append((_clamp01(right), tuple(_clamp01(c) for c in rcol)))
    return stops


def _parse_cpt(path: str) -> List[Tuple[float, Tuple[float, float, float]]]:
    """GMT .cpt (solo COLOR_MODEL RGB, formato numerico)."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        raw = fh.readlines()
    hsv = any("HSV" in ln.upper() for ln in raw if ln.lstrip().startswith("#") and "COLOR_MODEL" in ln.upper())
    if hsv:
        return []
    rows: List[Tuple[float, float, float, float]] = []
    for ln in raw:
        s = ln.strip()
        if not s or s.startswith("#") or s[0] in "BFN":
            continue
        s = s.replace("/", " ")
        parts = s.split()
        if len(parts) < 8:
            continue
        try:
            vals = [float(p) for p in parts[:8]]
        except ValueError:
            continue
        rows.append((vals[0], vals[1], vals[2], vals[3]))
        rows.append((vals[4], vals[5], vals[6], vals[7]))
    if not rows:
        return []
    zmin = min(r[0] for r in rows)
    zmax = max(r[0] for r in rows)
    span = (zmax - zmin) or 1.0
    scale = 1.0 if zmax <= 1.0 and zmin >= 0.0 else 255.0
    stops = [
        (_clamp01((z - zmin) / span), tuple(_clamp01(c / scale) for c in (r, g, b)))
        for (z, r, g, b) in rows
    ]
    return stops


def _load_stops(path: str) -> List[Tuple[float, Tuple[float, float, float]]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".svg":
        stops = _parse_svg(path)
    elif ext == ".ggr":
        stops = _parse_ggr(path)
    elif ext == ".cpt":
        stops = _parse_cpt(path)
    else:
        return []
    # normalizza: ordina, dedup, forza estremi 0..1
    stops = sorted(stops, key=lambda t: t[0])
    cleaned: List[Tuple[float, Tuple[float, float, float]]] = []
    for pos, col in stops:
        if cleaned and abs(cleaned[-1][0] - pos) < 1e-6:
            cleaned[-1] = (pos, col)  # tieni l'ultimo per quella posizione
        else:
            cleaned.append((pos, col))
    if len(cleaned) < 2:
        return []
    if cleaned[0][0] > 0.0:
        cleaned.insert(0, (0.0, cleaned[0][1]))
    if cleaned[-1][0] < 1.0:
        cleaned.append((1.0, cleaned[-1][1]))
    cleaned[0] = (0.0, cleaned[0][1])
    cleaned[-1] = (1.0, cleaned[-1][1])
    return cleaned


# ----------------------------------------------------------------------------
# Registrazione in matplotlib
# ----------------------------------------------------------------------------
def _register_one(path: str) -> Optional[str]:
    try:
        import matplotlib
        from matplotlib.colors import LinearSegmentedColormap
    except Exception:
        return None

    stops = _load_stops(path)
    if not stops:
        return None

    stem = os.path.splitext(os.path.basename(path))[0]
    name = stem

    try:
        existing = set(matplotlib.colormaps)  # mpl >= 3.6
    except Exception:  # pragma: no cover
        from matplotlib import cm as _cm

        existing = set(getattr(_cm, "_cmap_registry", {}))

    # Non sovrascrivere una colormap nativa con lo stesso nome.
    if name in existing and name not in PALETTE_NAMES:
        name = "cpt_" + stem

    try:
        cmap = LinearSegmentedColormap.from_list(name, stops, N=256)
    except Exception:
        return None

    try:
        matplotlib.colormaps.register(cmap, name=name, force=True)
    except Exception:
        try:
            matplotlib.cm.register_cmap(name=name, cmap=cmap)  # mpl vecchio
        except Exception:
            return None
    return name


def register_all(extra_files: Optional[Iterable[str]] = None) -> List[str]:
    """Ri-scansiona le cartelle palette e registra tutto. Idempotente."""
    seen: dict = {}
    for d in palette_dirs():
        for ext in _SUPPORTED_EXT:
            for p in sorted(glob.glob(os.path.join(d, "*" + ext))):
                seen.setdefault(os.path.basename(p).lower(), p)
    for p in extra_files or []:
        if os.path.isfile(p):
            seen.setdefault(os.path.basename(p).lower(), p)

    names: List[str] = []
    for p in seen.values():
        nm = _register_one(p)
        if nm and nm not in names:
            names.append(nm)

    names.sort(key=str.lower)
    PALETTE_NAMES[:] = names
    return list(names)


def import_palette_file(path: str) -> str:
    """Copia il file nella cartella utente, lo registra e ne ritorna il nome.

    Solleva ValueError se il formato non e' riconosciuto / non parsabile.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in _SUPPORTED_EXT:
        raise ValueError("Formato non supportato: %s (usa .svg .ggr .cpt)" % ext)
    if not _load_stops(path):
        raise ValueError("Nessun colore leggibile in: %s" % os.path.basename(path))

    dest_dir = user_palette_dir()
    dest = os.path.join(dest_dir, os.path.basename(path))
    if os.path.abspath(dest) != os.path.abspath(path):
        try:
            shutil.copy2(path, dest)
        except Exception as exc:  # pragma: no cover
            raise ValueError("Copia fallita: %s" % exc)

    name = _register_one(dest)
    if not name:
        raise ValueError("Registrazione fallita per: %s" % os.path.basename(path))
    if name not in PALETTE_NAMES:
        PALETTE_NAMES.append(name)
        PALETTE_NAMES.sort(key=str.lower)
    return name


# ----------------------------------------------------------------------------
# Anteprima
# ----------------------------------------------------------------------------
def colormap_preview_pixmap(name: str, w: int = 180, h: int = 14):
    """QPixmap con la striscia di gradiente della colormap `name`."""
    from qgis.PyQt.QtGui import QImage, QPixmap, QColor
    from qgis.PyQt.QtCore import Qt

    w = max(8, int(w))
    h = max(4, int(h))
    img = QImage(w, h, QImage.Format_RGB888)

    try:
        import numpy as np
        import matplotlib

        cmap = matplotlib.colormaps[name]
        cols = (np.asarray(cmap(np.linspace(0.0, 1.0, w)))[:, :3] * 255.0).astype("uint8")
        for x in range(w):
            r, g, b = int(cols[x, 0]), int(cols[x, 1]), int(cols[x, 2])
            rgb = (r << 16) | (g << 8) | b
            for y in range(h):
                img.setPixel(x, y, rgb)
    except Exception:
        img.fill(QColor(200, 200, 200))

    pm = QPixmap.fromImage(img)
    return pm


def make_cmap_icon(name: str, w: int = 64, h: int = 14):
    """QIcon con la striscia di gradiente, per le voci di una QComboBox."""
    from qgis.PyQt.QtGui import QIcon

    return QIcon(colormap_preview_pixmap(name, w, h))
