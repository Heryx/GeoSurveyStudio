# -*- coding: utf-8 -*-
"""Classi di segnale per le annotazioni sul radargramma (iperbole, riflettore, attenuazione...).

Modulo puro, senza Qt ne' QGIS: ogni classe ha un codice stabile (quello che finisce
nel layer), un nome, la geometria di default con cui si disegna e un colore.  L'elenco
sta in ``<project_root>/metadata/annotation_classes.json`` e si puo' personalizzare;
senza file valgono le classi predefinite.  Il codice non cambia mai quando si rinomina
una classe, cosi' le annotazioni gia' salvate restano collegate.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata

GEOMETRIES = ("point", "line", "polygon")
GEOMETRY_LABELS = {"point": "Punto", "line": "Linea", "polygon": "Poligono"}

CONFIDENCE_LEVELS = (
    ("certo", "Certo"),
    ("probabile", "Probabile"),
    ("possibile", "Possibile"),
)
DEFAULT_CONFIDENCE = "probabile"

# Codice della classe usata quando una feature non ne ha (annotazioni precedenti
# all'introduzione delle classi, o classe cancellata dall'elenco).
UNCLASSIFIED_CODE = "non_classificato"
UNCLASSIFIED_COLOR = "#00e5ff"      # lo stesso ciano dell'overlay di prima

MAX_CLASSES = 9                     # i tasti 1-9 scelgono la classe

_FILE_NAME = "annotation_classes.json"

DEFAULT_CLASSES = (
    {"code": "iperbole", "label": "Iperbole di diffrazione", "geometry": "point", "color": "#ffd600"},
    {"code": "riflettore", "label": "Riflettore / interfaccia", "geometry": "line", "color": "#00e5ff"},
    {"code": "attenuazione", "label": "Attenuazione / perdita di segnale", "geometry": "polygon", "color": "#ff5252"},
    {"code": "anomalia_ampiezza", "label": "Anomalia di ampiezza", "geometry": "polygon", "color": "#e040fb"},
    {"code": "struttura", "label": "Struttura (muro, fondazione)", "geometry": "polygon", "color": "#69f0ae"},
    {"code": "disturbo", "label": "Disturbo / rumore", "geometry": "polygon", "color": "#9e9e9e"},
)


def normalize_color(value, default: str = UNCLASSIFIED_COLOR) -> str:
    """'#rrggbb' minuscolo; accetta anche 'rgb' corto, altrimenti ``default``."""
    s = str(value or "").strip()
    if re.fullmatch(r"#[0-9a-fA-F]{6}", s):
        return s.lower()
    if re.fullmatch(r"#[0-9a-fA-F]{3}", s):
        return "#" + "".join(c * 2 for c in s[1:]).lower()
    return default


def slugify(text: str) -> str:
    """Codice stabile da un nome libero: minuscolo, senza accenti, solo [a-z0-9_]."""
    s = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    return s[:40] or "classe"


def clean_class(raw) -> dict | None:
    """Una classe valida o None: codice, nome, geometria e colore sempre presenti."""
    if not isinstance(raw, dict):
        return None
    label = str(raw.get("label") or "").strip()
    code = str(raw.get("code") or "").strip() or slugify(label)
    if not label:
        label = code
    if not code:
        return None
    geometry = str(raw.get("geometry") or "").strip().lower()
    if geometry not in GEOMETRIES:
        geometry = "polygon"
    return {"code": slugify(code), "label": label, "geometry": geometry,
            "color": normalize_color(raw.get("color"))}


def clean_classes(items) -> list:
    """Elenco ripulito: codici unici, al massimo ``MAX_CLASSES`` voci."""
    out, seen = [], set()
    for raw in items or []:
        cls = clean_class(raw)
        if cls is None or cls["code"] in seen or cls["code"] == UNCLASSIFIED_CODE:
            continue
        seen.add(cls["code"])
        out.append(cls)
        if len(out) >= MAX_CLASSES:
            break
    return out


def default_classes() -> list:
    return [dict(c) for c in DEFAULT_CLASSES]


def classes_path(project_root: str) -> str:
    return os.path.join(str(project_root or ""), "metadata", _FILE_NAME)


def load_classes(project_root: str | None) -> list:
    """Classi del progetto; se il file manca o e' illeggibile, quelle predefinite."""
    if project_root:
        try:
            with open(classes_path(project_root), "r", encoding="utf-8") as fh:
                data = json.load(fh)
            items = data.get("classes") if isinstance(data, dict) else data
            cleaned = clean_classes(items)
            if cleaned:
                return cleaned
        except (OSError, ValueError):
            pass
    return default_classes()


def save_classes(project_root: str, classes) -> list:
    """Scrive l'elenco (ripulito) nel progetto e lo restituisce."""
    cleaned = clean_classes(classes) or default_classes()
    path = classes_path(project_root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"version": 1, "classes": cleaned}, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    return cleaned


def class_by_code(classes, code) -> dict | None:
    code = str(code or "")
    for cls in classes or []:
        if cls["code"] == code:
            return cls
    return None


def color_for(classes, code) -> str:
    cls = class_by_code(classes, code)
    return cls["color"] if cls else UNCLASSIFIED_COLOR


def label_for(classes, code) -> str:
    cls = class_by_code(classes, code)
    if cls:
        return cls["label"]
    return str(code or "") or "Non classificato"


def unique_code(classes, label: str) -> str:
    """Codice libero per una nuova classe, derivato dal nome."""
    base = slugify(label)
    taken = {c["code"] for c in classes or []} | {UNCLASSIFIED_CODE}
    code, n = base, 2
    while code in taken:
        code = f"{base}_{n}"
        n += 1
    return code


def confidence_label(code) -> str:
    for c, label in CONFIDENCE_LEVELS:
        if c == str(code or ""):
            return label
    return ""
