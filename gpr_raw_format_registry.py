"""Registro dei formati raw GPR importabili nella viewer profili.

Ogni formato registra:
  - le estensioni primarie riconosciute (case-insensitive, con il punto),
  - una funzione reader `path -> OgprProfile` (con cache),
  - opzionalmente le estensioni "sidecar" da copiare insieme al file
    principale quando il profilo viene messo in staging nella cartella
    raw del progetto (es. .gec/.gps per il formato .DT).

Per aggiungere un nuovo formato in futuro basta scrivere un reader che
produca un `OgprProfile` (vedi `gpr_ogpr_reader.py`/`gpr_dtslice_reader.py`
come esempi) e registrarlo qui: il resto della pipeline (processing,
viewer, slicing IDW->GeoTIFF, import timeslice in QGIS) e' gia'
format-agnostico.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .gpr_ogpr_reader import OgprProfile, read_ogpr_cached
from .gpr_dtslice_reader import read_dtslice_cached


@dataclass(frozen=True)
class RawFormat:
    key: str
    label: str
    primary_ext: str                 # es. ".ogpr"
    sidecar_exts: tuple               # es. (".gec", ".gps", ".gps.sav")
    reader: Callable[[str], OgprProfile]
    dialog_filter: str                # es. "OGPR files (*.ogpr)"


RAW_FORMATS: tuple[RawFormat, ...] = (
    RawFormat(
        key="ogpr",
        label="OGPR",
        primary_ext=".ogpr",
        sidecar_exts=(),
        reader=lambda p: read_ogpr_cached(p, verify_md5=True),
        dialog_filter="*.ogpr",
    ),
    RawFormat(
        key="dtslice",
        label="DT raw (.DT + sidecar)",
        primary_ext=".dt",
        sidecar_exts=(".gec", ".gps", ".gps.sav"),
        reader=read_dtslice_cached,
        dialog_filter="*.DT",
    ),
)

_BY_EXT = {fmt.primary_ext: fmt for fmt in RAW_FORMATS}


class UnknownRawFormatError(Exception):
    pass


def format_for_path(path: str) -> RawFormat:
    ext = Path(path).suffix.lower()
    fmt = _BY_EXT.get(ext)
    if fmt is None:
        raise UnknownRawFormatError(
            f"Formato non riconosciuto per '{path}' (estensione '{ext}')"
        )
    return fmt


def detect_and_read(path: str, verify_md5: bool = True) -> OgprProfile:
    """Legge un profilo scegliendo automaticamente il reader in base all'estensione."""
    fmt = format_for_path(path)
    if fmt.key == "ogpr":
        return read_ogpr_cached(path, verify_md5=verify_md5)
    return fmt.reader(path)


def sidecar_paths(path: str) -> list[str]:
    """Path sidecar esistenti sul disco associati al file `path` (stesso formato)."""
    fmt = format_for_path(path)
    if not fmt.sidecar_exts:
        return []
    p = Path(path)
    found = []
    for ext in fmt.sidecar_exts:
        for cand in (p.with_suffix(ext), Path(str(p) + ext)):
            if cand.exists() and str(cand) not in found:
                found.append(str(cand))
    return found


def import_dialog_filter() -> str:
    exts = " ".join(fmt.dialog_filter for fmt in RAW_FORMATS)
    return f"Profili GPR ({exts});;All files (*)"
