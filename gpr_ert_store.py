# -*- coding: utf-8 -*-
"""Sezioni ERT nel progetto e allineamento al profilo GPR (senza Qt ne' QGIS).

Ogni sezione importata sta in ``<radice progetto>/ERT/<nome>/`` con la copia del file
invertito (``section.csv``) e il suo georiferimento (``georef.json``): il progetto resta
autosufficiente anche se l'export di ResIPY viene spostato.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from dataclasses import dataclass

import numpy as np

from .gpr_ert_georef import ErtGeoref
from .gpr_ert_import import ErtSection, load_ert_section

ERT_DIR = "ERT"
_SECTION_FILE = "section.csv"
_GEOREF_FILE = "georef.json"
# Oltre questo angolo tra profilo GPR e linea ERT la proiezione affiancata non e' un confronto valido.
MAX_PARALLEL_ANGLE_DEG = 20.0


def safe_name(name: str) -> str:
    """Nome valido come cartella: lettere, cifre, spazio, punto, trattino, sottolineato."""
    out = re.sub(r"[^\w .\-]", "_", str(name or "").strip(), flags=re.UNICODE).strip(" .")
    return out or "ERT"


def ert_root(project_root: str) -> str:
    return os.path.join(str(project_root), ERT_DIR)


def save_ert(project_root: str, name: str, csv_path: str, georef: ErtGeoref,
             z_mode: str = "auto") -> str:
    """Copia il file invertito nel progetto e scrive il georiferimento. Ritorna la cartella.

    Rifiuta un file illeggibile prima di toccare il disco. Una sezione con lo stesso nome
    viene sostituita.
    """
    load_ert_section(csv_path, z_mode=z_mode)           # solleva ErtImportError se non va
    folder = os.path.join(ert_root(project_root), safe_name(name))
    os.makedirs(folder, exist_ok=True)
    shutil.copyfile(csv_path, os.path.join(folder, _SECTION_FILE))
    payload = {"name": safe_name(name), "source": os.path.basename(csv_path), "z_mode": z_mode,
               "georef": georef.to_dict()}
    with open(os.path.join(folder, _GEOREF_FILE), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return folder


def list_ert(project_root: str) -> list:
    """Nomi delle sezioni ERT del progetto (ordine alfabetico)."""
    root = ert_root(project_root) if project_root else ""
    if not root or not os.path.isdir(root):
        return []
    return sorted(d for d in os.listdir(root)
                  if os.path.isfile(os.path.join(root, d, _SECTION_FILE))
                  and os.path.isfile(os.path.join(root, d, _GEOREF_FILE)))


@dataclass
class ErtEntry:
    name: str
    section: ErtSection
    georef: ErtGeoref
    source: str = ""


def load_ert(project_root: str, name: str) -> ErtEntry:
    folder = os.path.join(ert_root(project_root), name)
    with open(os.path.join(folder, _GEOREF_FILE), "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    section = load_ert_section(os.path.join(folder, _SECTION_FILE),
                               z_mode=payload.get("z_mode", "auto"))
    return ErtEntry(name=name, section=section, georef=ErtGeoref.from_dict(payload["georef"]),
                    source=str(payload.get("source", "")))


def delete_ert(project_root: str, name: str) -> bool:
    folder = os.path.join(ert_root(project_root), name)
    if not os.path.isfile(os.path.join(folder, _GEOREF_FILE)):
        return False
    shutil.rmtree(folder)
    return True


@dataclass
class ErtAlignment:
    """Dove sta una sezione ERT rispetto a un canale GPR: ``x`` GPR <-> distanza ERT."""

    gpr_dist: np.ndarray        # distanza GPR (m) delle tracce valide, crescente
    ert_dist: np.ndarray        # distanza ERT corrispondente
    median_offset_m: float      # scostamento laterale mediano tra tracce GPR e linea ERT
    max_offset_m: float
    overlap_m: float            # tratto di profilo GPR che cade entro la sezione ERT
    angle_deg: float = 0.0      # inclinazione media del profilo GPR sulla linea ERT (0 = paralleli, 90 = ortogonali)

    def is_parallel(self, max_angle_deg: float = MAX_PARALLEL_ANGLE_DEG) -> bool:
        """Vero se la proiezione sulla linea ERT ha senso: profili (quasi) paralleli o sovrapposti."""
        return self.angle_deg <= max_angle_deg

    def ert_to_gpr(self, ert_d):
        """Distanza GPR per distanze ERT; NaN dove il profilo GPR non arriva."""
        order = np.argsort(self.ert_dist)
        return np.interp(np.asarray(ert_d, dtype=np.float64), self.ert_dist[order],
                         self.gpr_dist[order], left=np.nan, right=np.nan)

    def gpr_to_ert(self, gpr_x):
        """Distanza ERT per distanze GPR (prolunga con i valori estremi)."""
        return np.interp(np.asarray(gpr_x, dtype=np.float64), self.gpr_dist, self.ert_dist)


def align_to_channel(georef: ErtGeoref, east, north, dist, ert_length: float | None = None):
    """Allinea il georiferimento ERT alle tracce di un canale GPR (stesso CRS del georef).

    Ogni traccia viene proiettata sulla linea ERT. Ritorna ``None`` se il canale non ha
    coordinate valide. ``overlap_m`` conta solo le tracce che cadono entro
    [0, ``ert_length``] (default: ultimo punto di controllo).
    """
    east = np.asarray(east, dtype=np.float64)
    north = np.asarray(north, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64)
    ok = np.isfinite(east) & np.isfinite(north) & np.isfinite(dist)
    if ok.sum() < 2:
        return None
    ert_d, off = georef.project(east[ok], north[ok])
    gx = dist[ok]
    order = np.argsort(gx)
    gx, ert_d, off = gx[order], ert_d[order], off[order]
    d0 = georef.points[0].distance
    d1 = georef.points[-1].distance if ert_length is None else float(ert_length)
    inside = (ert_d >= d0 - 1e-9) & (ert_d <= d1 + 1e-9)
    overlap = float(gx[inside].max() - gx[inside].min()) if inside.sum() >= 2 else 0.0
    return ErtAlignment(gpr_dist=gx, ert_dist=ert_d, median_offset_m=float(np.median(off)),
                        max_offset_m=float(np.max(off)), overlap_m=overlap,
                        angle_deg=_track_angle_deg(georef, east[ok][order], north[ok][order], ert_d))


def _track_angle_deg(georef: ErtGeoref, east, north, ert_d) -> float:
    """Angolo (0-90) tra la direzione del profilo GPR e quella della linea ERT sotto di esso.

    Direzione del GPR: dal primo all'ultimo punto (la forma fine non conta); direzione ERT: la
    tangente della polilinea al punto di mezzo. Linee percorse in verso opposto contano come parallele.
    """
    gx, gy = float(east[-1] - east[0]), float(north[-1] - north[0])
    if math.hypot(gx, gy) < 1e-9:
        return 0.0
    mid = float(np.median(ert_d))
    pts = georef.points
    k = int(np.clip(np.searchsorted([p.distance for p in pts], mid) - 1, 0, len(pts) - 2))
    ex, ey = pts[k + 1].x - pts[k].x, pts[k + 1].y - pts[k].y
    cos = abs(gx * ex + gy * ey) / (math.hypot(gx, gy) * math.hypot(ex, ey))
    return math.degrees(math.acos(min(1.0, cos)))
