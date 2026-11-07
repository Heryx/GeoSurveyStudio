# -*- coding: utf-8 -*-
"""Import di sezioni ERT 2D gia' invertite (export CSV/XYZ di ResIPY), senza Qt ne' QGIS.

ResIPY esporta i **centri delle celle** della mesh triangolare (x, y, z + resistivita'),
non una griglia: qui si leggono e si ricampionano su una griglia regolare distanza x
profondita', lasciando vuoto (NaN) cio' che sta fuori dalla copertura dell'inversione.

La sezione non e' georiferita: ``x`` e' la distanza progressiva lungo la linea, in m. Il
georiferimento (2 punti o piu') e' un passo a parte.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field

import numpy as np

# Una sezione 2D di ResIPY ha y == 0 ovunque.
_Y_TOL = 1e-6
# Senza topografia ResIPY mette la superficie a z = 0 e le celle a z < 0 (profondita').
_DEPTH_MODE_ZMAX = 0.5

_COLUMN_ALIASES = {
    "x": ("x",),
    "y": ("y",),
    "z": ("z",),
    "resistivity": ("resistivity(ohm.m)", "resistivity", "res", "rho"),
    "log10_res": ("resistivity(log10)", "log10"),
    "conductivity": ("conductivity(ms/m)", "conductivity"),
    "sensitivity": ("sensitivity(log10)", "sensitivity"),
}


class ErtImportError(ValueError):
    """File ERT non leggibile o non 2D."""


@dataclass
class ErtSection:
    """Celle di una sezione ERT: ``x`` distanza (m), ``z`` come nel file, valori per cella."""

    x: np.ndarray
    z: np.ndarray
    values: dict = field(default_factory=dict)
    z_mode: str = "depth"  # "depth": z<=0 sotto la superficie; "elevation": z e' una quota
    source: str = ""

    @property
    def depth(self) -> np.ndarray:
        """Profondita' positiva verso il basso; in modalita' quota, dalla quota piu' alta."""
        if self.z_mode == "depth":
            return -self.z
        return float(np.max(self.z)) - self.z

    @property
    def length_m(self) -> float:
        return float(np.max(self.x) - np.min(self.x))


@dataclass
class ErtGrid:
    """Griglia regolare: ``data[iz, ix]`` con la riga 0 in alto (z decrescente)."""

    x_axis: np.ndarray
    z_axis: np.ndarray  # nella convenzione del file (z_mode)
    data: np.ndarray
    quantity: str
    z_mode: str

    @property
    def depth_axis(self) -> np.ndarray:
        if self.z_mode == "depth":
            return -self.z_axis
        return float(self.z_axis[0]) - self.z_axis


def _find_columns(header: list) -> dict:
    names = [h.strip().lower() for h in header]
    found = {}
    for key, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in names:
                found[key] = names.index(alias)
                break
    return found


def _split(line: str) -> list:
    return [p for p in re.split(r"[,;\t ]+", line.strip()) if p != ""]


def _read_table(path: str):
    """Restituisce (header | None, righe di float). Tollera virgola, punto e virgola, spazi."""
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        lines = [ln for ln in fh if ln.strip()]
    if not lines:
        raise ErtImportError("file vuoto")
    first = _split(lines[0])
    try:
        [float(v) for v in first]
        header = None
        body = lines
    except ValueError:
        header = first
        body = lines[1:]
    rows = []
    for ln in body:
        parts = _split(ln)
        try:
            rows.append([float(v) for v in parts])
        except ValueError:
            continue
    if not rows:
        raise ErtImportError("nessuna riga numerica")
    width = len(rows[0])
    rows = [r for r in rows if len(r) == width]
    return header, np.asarray(rows, dtype=np.float64)


def load_ert_section(path: str, z_mode: str = "auto") -> ErtSection:
    """Legge un CSV/XYZ di ResIPY (con intestazione) o un file numerico ``x z res`` / ``x y z res``.

    ``z_mode`` = "auto" | "depth" | "elevation". In "auto" la z e' una profondita' se tutte le
    celle stanno sotto lo zero (superficie piana a 0, nessuna topografia).
    """
    if z_mode not in ("auto", "depth", "elevation"):
        raise ErtImportError(f"z_mode non valido: {z_mode}")
    header, table = _read_table(path)
    if header is not None:
        cols = _find_columns(header)
        for need in ("x", "z", "resistivity"):
            if need not in cols and not (need == "resistivity" and "log10_res" in cols):
                raise ErtImportError(f"colonna '{need}' non trovata nell'intestazione")
        x = table[:, cols["x"]]
        z = table[:, cols["z"]]
        y = table[:, cols["y"]] if "y" in cols else np.zeros_like(x)
        values = {}
        for key in ("resistivity", "log10_res", "conductivity", "sensitivity"):
            if key in cols:
                values[key] = table[:, cols[key]]
    else:
        ncol = table.shape[1]
        if ncol == 3:
            x, z, res = table[:, 0], table[:, 1], table[:, 2]
            y = np.zeros_like(x)
        elif ncol >= 4:
            x, y, z, res = table[:, 0], table[:, 1], table[:, 2], table[:, 3]
        else:
            raise ErtImportError("servono almeno 3 colonne (x z resistivita')")
        values = {"resistivity": res}

    if np.ptp(y) > _Y_TOL:
        raise ErtImportError("la sezione non e' 2D (y varia): per ora si importano solo sezioni 2D")
    if "resistivity" not in values:
        values["resistivity"] = np.power(10.0, values["log10_res"])
    if "log10_res" not in values:
        with np.errstate(divide="ignore", invalid="ignore"):
            values["log10_res"] = np.log10(values["resistivity"])

    finite = np.isfinite(x) & np.isfinite(z) & np.isfinite(values["resistivity"])
    if not finite.any():
        raise ErtImportError("nessuna cella valida")
    x, z = x[finite], z[finite]
    values = {k: v[finite] for k, v in values.items()}

    if z_mode == "auto":
        z_mode = "depth" if float(np.max(z)) <= _DEPTH_MODE_ZMAX else "elevation"
    return ErtSection(x=x, z=z, values=values, z_mode=z_mode, source=str(path))


def default_spacing(section: ErtSection) -> float:
    """Passo di griglia proposto: meta' della distanza mediana al vicino piu' prossimo."""
    from scipy.spatial import cKDTree

    pts = np.column_stack([section.x, section.z])
    dist, _ = cKDTree(pts).query(pts, k=2)
    return float(max(np.median(dist[:, 1]) / 2.0, 1e-3))


def to_grid(section: ErtSection, dx: float | None = None, dz: float | None = None,
            quantity: str = "log10_res") -> ErtGrid:
    """Ricampiona le celle su una griglia regolare (interpolazione lineare su Delaunay).

    Si interpola ``quantity`` (default il log10 della resistivita': varia di ordini di
    grandezza); fuori dall'involucro convesso dei centri delle celle il valore e' NaN, nessun
    dato viene inventato oltre la copertura.
    """
    from scipy.interpolate import LinearNDInterpolator

    if quantity not in section.values:
        raise ErtImportError(f"grandezza non disponibile: {quantity}")
    step = default_spacing(section)
    dx = float(dx) if dx else step
    dz = float(dz) if dz else step
    if dx <= 0 or dz <= 0:
        raise ErtImportError("passo di griglia non valido")
    x_axis = np.arange(float(np.min(section.x)), float(np.max(section.x)) + dx / 2, dx)
    z_hi, z_lo = float(np.max(section.z)), float(np.min(section.z))
    z_axis = np.arange(z_hi, z_lo - dz / 2, -dz)
    interp = LinearNDInterpolator(np.column_stack([section.x, section.z]),
                                  section.values[quantity], fill_value=np.nan)
    gx, gz = np.meshgrid(x_axis, z_axis)
    data = interp(gx, gz)
    return ErtGrid(x_axis=x_axis, z_axis=z_axis, data=data, quantity=quantity,
                   z_mode=section.z_mode)
