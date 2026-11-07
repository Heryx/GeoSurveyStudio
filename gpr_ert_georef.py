# -*- coding: utf-8 -*-
"""Georiferimento di una sezione ERT 2D: dalla distanza lungo la linea a x, y del progetto.

Il nucleo (``ErtGeoref``) e' puro numpy: lavora con coordinate gia' nel CRS del progetto. La
lettura dei KML e la riproiezione (``lonlat_transformer``) stanno in fondo al modulo; la
riproiezione usa QGIS, o pyproj se QGIS non c'e'.

Il modello e' una polilinea con punti di controllo ``(distanza, x, y)``:

- 2 punti (primo e ultimo elettrodo): la sezione e' una retta;
- N punti: la linea segue la polilinea e puo' curvare; con un punto per elettrodo e' il caso
  limite.

Tra due punti di controllo la distanza si interpola linearmente; oltre i punti estremi si
prolunga l'ultimo tratto. Se la lunghezza misurata sulla mappa tra due punti differisce da
quella della sezione, ``check()`` lo segnala: la sezione verrebbe stirata o compressa.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

# Scarto relativo tra lunghezza sulla mappa e lunghezza della sezione oltre il quale si avvisa.
SCALE_TOLERANCE = 0.05


class ErtGeorefError(ValueError):
    """Punti di controllo non validi."""


@dataclass(frozen=True)
class ControlPoint:
    distance: float  # m dall'origine della sezione (primo elettrodo)
    x: float  # coordinate nel CRS del progetto
    y: float


class ErtGeoref:
    """Mappa distanza lungo la sezione -> (x, y) nel CRS del progetto."""

    def __init__(self, points, crs_authid: str = ""):
        triples = [(p.distance, p.x, p.y) if isinstance(p, ControlPoint) else p for p in points]
        pts = sorted((ControlPoint(float(d), float(x), float(y)) for d, x, y in triples),
                     key=lambda p: p.distance)
        if len(pts) < 2:
            raise ErtGeorefError("servono almeno 2 punti di controllo")
        for a, b in zip(pts, pts[1:]):
            if b.distance - a.distance <= 1e-9:
                raise ErtGeorefError("due punti di controllo hanno la stessa distanza")
            if math.hypot(b.x - a.x, b.y - a.y) <= 1e-9:
                raise ErtGeorefError("due punti di controllo consecutivi coincidono sulla mappa")
        self.points = pts
        self.crs_authid = crs_authid
        self._d = np.array([p.distance for p in pts])
        self._x = np.array([p.x for p in pts])
        self._y = np.array([p.y for p in pts])

    # --- costruttori --------------------------------------------------------------------
    @classmethod
    def from_endpoints(cls, start_xy, end_xy, length_m: float, crs_authid: str = ""):
        """Due punti: ``start_xy`` a distanza 0 e ``end_xy`` a ``length_m``."""
        return cls([(0.0, start_xy[0], start_xy[1]), (float(length_m), end_xy[0], end_xy[1])],
                   crs_authid)

    @classmethod
    def from_polyline(cls, xy, length_m: float, crs_authid: str = ""):
        """Vertici senza distanze: la distanza e' proporzionale alla lunghezza sulla mappa e il
        totale e' ``length_m`` (primo vertice = distanza 0, ultimo = ``length_m``)."""
        xy = [(float(a), float(b)) for a, b in xy]
        if len(xy) < 2:
            raise ErtGeorefError("servono almeno 2 vertici")
        seg = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(xy, xy[1:])]
        total = sum(seg)
        if total <= 0:
            raise ErtGeorefError("la polilinea ha lunghezza nulla")
        cum = np.concatenate([[0.0], np.cumsum(seg)]) * (float(length_m) / total)
        return cls([(float(d), p[0], p[1]) for d, p in zip(cum, xy)], crs_authid)

    # --- uso ----------------------------------------------------------------------------
    def locate(self, distance):
        """(x, y) per una distanza o un array di distanze; fuori dagli estremi prolunga il tratto."""
        d = np.asarray(distance, dtype=np.float64)
        out = []
        for coord in (self._x, self._y):
            v = np.interp(d, self._d, coord)
            lo = d < self._d[0]
            hi = d > self._d[-1]
            if np.any(lo):
                s = (coord[1] - coord[0]) / (self._d[1] - self._d[0])
                v = np.where(lo, coord[0] + (d - self._d[0]) * s, v)
            if np.any(hi):
                s = (coord[-1] - coord[-2]) / (self._d[-1] - self._d[-2])
                v = np.where(hi, coord[-1] + (d - self._d[-1]) * s, v)
            out.append(v)
        if d.ndim == 0:
            return float(out[0]), float(out[1])
        return out[0], out[1]

    def project(self, x, y):
        """Proietta punti della mappa sulla linea: (distanza lungo la sezione, scostamento laterale).

        Inverso di ``locate``: serve a sapere dove cade, sulla sezione ERT, ogni traccia GPR.
        Prima dell'inizio e dopo la fine la linea e' prolungata, quindi la distanza esce da
        [primo, ultimo punto di controllo]: tocca a chi chiama mascherare cio' che sta fuori.
        """
        px = np.atleast_1d(np.asarray(x, dtype=np.float64))
        py = np.atleast_1d(np.asarray(y, dtype=np.float64))
        n_seg = len(self.points) - 1
        best_off = np.full(px.shape, np.inf)
        best_dist = np.full(px.shape, np.nan)
        for i in range(n_seg):
            ax_, ay_ = self._x[i], self._y[i]
            dx, dy = self._x[i + 1] - ax_, self._y[i + 1] - ay_
            t = ((px - ax_) * dx + (py - ay_) * dy) / (dx * dx + dy * dy)
            lo = -np.inf if i == 0 else 0.0
            hi = np.inf if i == n_seg - 1 else 1.0
            t = np.clip(t, lo, hi)
            off = np.hypot(px - (ax_ + t * dx), py - (ay_ + t * dy))
            better = off < best_off
            best_off = np.where(better, off, best_off)
            best_dist = np.where(better, self._d[i] + t * (self._d[i + 1] - self._d[i]), best_dist)
        if np.ndim(x) == 0:
            return float(best_dist[0]), float(best_off[0])
        return best_dist, best_off

    def polyline(self, step_m: float = 1.0, start: float | None = None, end: float | None = None):
        """Vertici (x, y) della linea tra due distanze, per disegnarla sulla mappa."""
        a = self._d[0] if start is None else float(start)
        b = self._d[-1] if end is None else float(end)
        n = max(int(math.ceil((b - a) / max(step_m, 1e-6))), 1)
        xs, ys = self.locate(np.linspace(a, b, n + 1))
        return list(zip(xs.tolist(), ys.tolist()))

    def map_length(self) -> float:
        return float(sum(math.hypot(b.x - a.x, b.y - a.y)
                         for a, b in zip(self.points, self.points[1:])))

    def segment_scales(self) -> list:
        """Rapporto lunghezza sulla mappa / lunghezza della sezione, per ogni tratto (1 = esatto)."""
        return [math.hypot(b.x - a.x, b.y - a.y) / (b.distance - a.distance)
                for a, b in zip(self.points, self.points[1:])]

    def check(self, tolerance: float = SCALE_TOLERANCE) -> list:
        """Avvisi (testi) sui tratti la cui scala si scosta troppo da 1."""
        warns = []
        for i, s in enumerate(self.segment_scales(), start=1):
            if abs(s - 1.0) > tolerance:
                warns.append(f"tratto {i}: sulla mappa misura il {s * 100:.0f}% della sezione "
                             f"(oltre {tolerance * 100:.0f}% di scarto): la sezione verrebbe "
                             f"{'stirata' if s > 1 else 'compressa'}")
        return warns

    def flipped(self, length_m: float) -> "ErtGeoref":
        """Stessa linea percorsa al contrario (la distanza 0 passa all'altro estremo)."""
        return ErtGeoref([(float(length_m) - p.distance, p.x, p.y) for p in self.points],
                         self.crs_authid)

    # --- persistenza --------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"crs": self.crs_authid,
                "points": [{"distance": p.distance, "x": p.x, "y": p.y} for p in self.points]}

    @classmethod
    def from_dict(cls, data: dict) -> "ErtGeoref":
        try:
            pts = [(p["distance"], p["x"], p["y"]) for p in data["points"]]
        except (KeyError, TypeError) as exc:
            raise ErtGeorefError(f"georiferimento non leggibile: {exc}") from exc
        return cls(pts, data.get("crs", ""))


# --- KML -----------------------------------------------------------------------------------
@dataclass
class KmlProfile:
    """Un profilo trovato in un KML: la linea e/o i punti con lo stesso nome (lon, lat)."""

    name: str
    line: list = field(default_factory=list)
    points: list = field(default_factory=list)

    def endpoints(self):
        """Primo e ultimo punto (lon, lat): dalla linea se c'e', altrimenti dai punti."""
        src = self.line if len(self.line) >= 2 else self.points
        if len(src) < 2:
            raise ErtGeorefError(f"il profilo '{self.name}' non ha due punti")
        return src[0], src[-1]


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _parse_coords(text: str) -> list:
    out = []
    for tok in (text or "").split():
        parts = tok.split(",")
        if len(parts) >= 2:
            out.append((float(parts[0]), float(parts[1])))
    return out


def read_kml_profiles(path: str) -> list:
    """Profili di un KML, raggruppati per nome (senza distinguere maiuscole/minuscole).

    Una linea (LineString) con un nome e uno o piu' punti (Point) con lo stesso nome finiscono
    nello stesso ``KmlProfile``; l'ordine dei vertici e' quello del file.
    """
    root = ET.parse(path).getroot()
    found: dict = {}
    order: list = []
    for pm in root.iter():
        if _local(pm.tag) != "Placemark":
            continue
        name = ""
        lines, points = [], []
        for el in pm.iter():
            tag = _local(el.tag)
            if tag == "name" and not name:
                name = (el.text or "").strip()
            elif tag == "LineString":
                for c in el.iter():
                    if _local(c.tag) == "coordinates":
                        lines.extend(_parse_coords(c.text))
            elif tag == "Point":
                for c in el.iter():
                    if _local(c.tag) == "coordinates":
                        points.extend(_parse_coords(c.text))
        if not name or not (lines or points):
            continue
        key = name.lower()
        if key not in found:
            found[key] = KmlProfile(name=name)
            order.append(key)
        found[key].line.extend(lines)
        found[key].points.extend(points)
    return [found[k] for k in order]


def georef_from_kml_profile(profile: KmlProfile, length_m: float, transform,
                            crs_authid: str = "", invert: bool = False) -> ErtGeoref:
    """Georiferimento a 2 punti dagli estremi di un profilo KML.

    ``transform`` e' ``(lon, lat) -> (x, y)`` nel CRS del progetto (vedi ``lonlat_transformer``).
    Il primo vertice del KML e' il primo elettrodo (distanza 0), a meno di ``invert``.
    """
    a, b = profile.endpoints()
    if invert:
        a, b = b, a
    return ErtGeoref.from_endpoints(transform(*a), transform(*b), length_m, crs_authid)


def parse_xy(text: str):
    """(x, y) da un testo come ``451270.5 4049823.4``, ``451270,5; 4049823,4`` o ``x; y``; None se non e' una coppia.

    Con il punto e virgola la virgola e' il separatore decimale; senza, si separa per spazi e
    la virgola decimale e' accettata se non c'e' ambiguita'.
    """
    s = str(text or "").strip()
    if not s:
        return None
    parts = [p for p in (s.split(";") if ";" in s else s.split()) if p.strip()]
    if len(parts) != 2:
        return None
    try:
        return float(parts[0].strip().replace(",", ".")), float(parts[1].strip().replace(",", "."))
    except ValueError:
        return None


def georef_from_xy_points(xy, length_m: float, crs_authid: str = "", invert: bool = False) -> ErtGeoref:
    """Georiferimento da punti senza distanze: 2 punti = retta, N punti = polilinea.

    I punti sono in ordine di percorrenza (il primo e' il primo elettrodo, a meno di ``invert``);
    la distanza di ognuno e' proporzionale alla lunghezza sulla mappa e l'ultimo cade a ``length_m``.
    """
    pts = [(float(a), float(b)) for a, b in xy]
    if invert:
        pts.reverse()
    return ErtGeoref.from_polyline(pts, length_m, crs_authid)


def suggested_length(x_max: float) -> float:
    """Lunghezza della linea da un x massimo di centri-cella: arrotonda al decimetro."""
    return round(float(x_max), 1)


# --- Riproiezione ---------------------------------------------------------------------------
def lonlat_transformer(dest_authid: str, source_authid: str = "EPSG:4326"):
    """Funzione ``(lon, lat) -> (x, y)`` nel CRS di destinazione (es. quello del progetto).

    Usa QGIS (``QgsCoordinateTransform``) se disponibile, altrimenti pyproj.
    """
    try:
        from qgis.core import (QgsCoordinateReferenceSystem, QgsCoordinateTransform,
                               QgsPointXY, QgsProject)

        src = QgsCoordinateReferenceSystem(source_authid)
        dst = QgsCoordinateReferenceSystem(dest_authid)
        if not src.isValid() or not dst.isValid():
            raise ErtGeorefError(f"CRS non valido: {source_authid} -> {dest_authid}")
        tr = QgsCoordinateTransform(src, dst, QgsProject.instance().transformContext())

        def _qgis(lon, lat):
            p = tr.transform(QgsPointXY(float(lon), float(lat)))
            return p.x(), p.y()

        return _qgis
    except ImportError:
        pass
    try:
        from pyproj import Transformer
    except ImportError as exc:
        raise ErtGeorefError("serve QGIS o pyproj per riproiettare le coordinate") from exc
    t = Transformer.from_crs(source_authid, dest_authid, always_xy=True)
    return lambda lon, lat: t.transform(float(lon), float(lat))
