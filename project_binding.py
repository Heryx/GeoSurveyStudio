# -*- coding: utf-8 -*-
"""Lega il progetto QGIS aperto al progetto GPR (project_root) a cui appartiene.

Prima la cartella-progetto attiva viveva SOLO nelle QSettings del profilo
QGIS (``GeoSurveyStudio/active_project_root``): un valore globale, "l'ultimo
progetto usato". Aprendo un .qgz esistente il plugin continuava a guardare
quello dell'ultima sessione (o, nel Profile Viewer, la cartella del .qgz),
quindi il catalogo non conteneva i gruppi/time-slice caricati nel canvas e il
viewer non trovava le varianti processate finche' non si riapriva il progetto
dal Project Manager.

Qui la project_root viene scritta DENTRO il progetto QGIS (``writeEntry``) e,
per i progetti salvati prima di questa modifica, dedotta dai file dei layer
raster caricati nei gruppi di GeoSurvey Studio (si risale fino alla cartella
con ``metadata/project_catalog.json`` o ``metadata/project.json``).

``adopt_untracked_timeslices`` registra nel catalogo i raster che stanno gia'
nei gruppi del canvas ma che il catalogo non conosce, cosi' la lista gruppi
del plugin li riconosce."""

from __future__ import annotations

import os
from collections import Counter

try:
    from qgis.core import (
        QgsLayerTreeGroup,
        QgsLayerTreeLayer,
        QgsProject,
        QgsRasterLayer,
    )
except ImportError:  # pragma: no cover - solo per test di logica pura
    QgsLayerTreeGroup = QgsLayerTreeLayer = QgsProject = QgsRasterLayer = None

SCOPE = "GeoSurveyStudio"
KEY_ROOT = "gpr_project_root"

# Gruppi tecnici del plugin (vettoriali / disegno) e contenitore dei
# riferimenti cross-progetto, che per scelta NON entrano nel catalogo.
_SKIP_GROUP_NAMES = {"Line Traces", "Cell grids"}
_CROSS_PROJECT_PREFIX = "Imported — "


def _norm(path: str) -> str:
    raw = str(path or "").strip()
    if not raw:
        return ""
    raw = raw.split("|", 1)[0].strip()
    try:
        return os.path.normcase(os.path.normpath(os.path.abspath(raw)))
    except Exception:
        return os.path.normcase(raw)


def is_gpr_project_root(path: str) -> bool:
    """True se ``path`` e' la cartella di un progetto GPR (ha il catalogo o il
    ``project.json`` del viewer)."""
    p = str(path or "").strip()
    if not p or not os.path.isdir(p):
        return False
    return (
        os.path.isfile(os.path.join(p, "metadata", "project_catalog.json"))
        or os.path.isfile(os.path.join(p, "metadata", "project.json"))
    )


def find_root_above(path: str) -> str:
    """Prima cartella, risalendo da ``path`` (incluso), che e' una project_root
    GPR. '' se non ce n'e'."""
    cur = os.path.normpath(str(path or "").strip())
    if not cur:
        return ""
    if os.path.isfile(cur):
        cur = os.path.dirname(cur)
    seen = set()
    while cur and cur not in seen:
        seen.add(cur)
        if is_gpr_project_root(cur):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return ""


def most_common_root(sources) -> str:
    """Project_root cui appartengono piu' file in ``sources`` ('' se nessuno)."""
    votes = Counter()
    for src in sources or []:
        base = str(src or "").split("|", 1)[0].strip()
        if not base:
            continue
        root = find_root_above(base)
        if root:
            votes[root] += 1
    if not votes:
        return ""
    return votes.most_common(1)[0][0]


# ----------------------------------------------------------------------
# Progetto QGIS <-> project_root
# ----------------------------------------------------------------------

def recall_root() -> str:
    """project_root scritta nel progetto QGIS corrente ('' se assente o se la
    cartella non esiste piu', es. progetto spostato)."""
    if QgsProject is None:
        return ""
    try:
        value, ok = QgsProject.instance().readEntry(SCOPE, KEY_ROOT, "")
    except Exception:
        return ""
    value = str(value or "").strip() if ok else ""
    return os.path.normpath(value) if value and os.path.isdir(value) else ""


def remember_root(root: str) -> bool:
    """Scrive ``root`` nel progetto QGIS corrente. Non lo segna "modificato"
    se non lo era gia' e il valore non cambia."""
    if QgsProject is None:
        return False
    root = os.path.normpath(str(root or "").strip()) if root else ""
    if not root:
        return False
    project = QgsProject.instance()
    try:
        current, ok = project.readEntry(SCOPE, KEY_ROOT, "")
    except Exception:
        current, ok = "", False
    if ok and os.path.normcase(os.path.normpath(current or "x")) == os.path.normcase(root):
        return False
    was_dirty = False
    try:
        was_dirty = bool(project.isDirty())
    except Exception:
        pass
    try:
        project.writeEntry(SCOPE, KEY_ROOT, root)
    except Exception:
        return False
    if not was_dirty and project.fileName():
        # Un progetto appena aperto non va segnato "da salvare" solo perche'
        # gli abbiamo scritto il riferimento: lo portera' con se' al prossimo
        # salvataggio voluto dall'utente.
        try:
            project.setDirty(False)
        except Exception:
            pass
    return True


def _iter_plugin_groups(plugin_root):
    for child in list(plugin_root.children()):
        if QgsLayerTreeGroup is not None and isinstance(child, QgsLayerTreeGroup):
            yield child


def _iter_group_raster_layers(group):
    """Layer raster validi figli diretti di ``group`` (i sottogruppi sono
    gruppi a se': vedi ``_iter_plugin_groups``)."""
    for child in list(group.children()):
        if QgsLayerTreeLayer is None or not isinstance(child, QgsLayerTreeLayer):
            continue
        layer = child.layer()
        if layer is not None and QgsRasterLayer is not None and isinstance(layer, QgsRasterLayer):
            yield layer


def layer_tree_sources(plugin_root) -> list:
    """``source()`` di tutti i raster nei gruppi del plugin, a qualsiasi livello."""
    out = []

    def _walk(node):
        for child in list(node.children()):
            if QgsLayerTreeLayer is not None and isinstance(child, QgsLayerTreeLayer):
                layer = child.layer()
                if layer is not None and QgsRasterLayer is not None and isinstance(layer, QgsRasterLayer):
                    out.append(str(layer.source() or ""))
            elif QgsLayerTreeGroup is not None and isinstance(child, QgsLayerTreeGroup):
                _walk(child)

    if plugin_root is not None:
        _walk(plugin_root)
    return out


def resolve_root(plugin_root=None) -> str:
    """project_root del progetto QGIS aperto: prima quella scritta nel file,
    poi quella dedotta dai raster caricati nei gruppi del plugin."""
    root = recall_root()
    if root:
        return root
    if plugin_root is not None:
        return most_common_root(layer_tree_sources(plugin_root))
    return ""


# ----------------------------------------------------------------------
# Raster del canvas sconosciuti al catalogo
# ----------------------------------------------------------------------

def adopt_untracked_timeslices(project_root: str, plugin_root) -> dict:
    """Allinea il catalogo ai gruppi del canvas.

    Per ogni gruppo figlio di ``plugin_root`` (esclusi i gruppi tecnici e i
    riferimenti cross-progetto) e per ogni raster valido che contiene:
    - se il file non e' nel catalogo viene registrato come time-slice;
    - se non e' elencato nel gruppo omonimo del catalogo (creato se manca)
      ci viene aggiunto.
    Ritorna ``{"registered": n, "assigned": n, "groups": n}``."""
    from .project_catalog import (
        assign_timeslices_to_group,
        create_raster_group,
        load_catalog,
        parse_depth_from_filename,
        register_timeslices_batch,
        utc_now_iso,
    )

    result = {"registered": 0, "assigned": 0, "groups": 0}
    if not project_root or plugin_root is None or not os.path.isdir(project_root):
        return result

    catalog = load_catalog(project_root)
    by_path = {}
    for rec in catalog.get("timeslices", []):
        key = _norm(rec.get("project_path"))
        if key:
            by_path[key] = rec
    groups_by_name = {
        str(g.get("name") or "").strip().lower(): g
        for g in catalog.get("raster_groups", [])
    }

    new_records = []          # record da registrare
    wanted = []               # (nome gruppo, id time-slice)
    stamp = utc_now_iso()
    seq = 0

    for group_node in _iter_plugin_groups(plugin_root):
        gname = str(group_node.name() or "").strip()
        if not gname or gname in _SKIP_GROUP_NAMES or gname.startswith(_CROSS_PROJECT_PREFIX):
            continue
        cat_group = groups_by_name.get(gname.lower())
        in_group = set(cat_group.get("timeslice_ids", [])) if cat_group else set()
        for layer in _iter_group_raster_layers(group_node):
            try:
                if not layer.isValid():
                    continue
            except Exception:
                continue
            src = str(layer.source() or "").split("|", 1)[0].strip()
            if not src or not os.path.isfile(src):
                continue
            key = _norm(src)
            rec = by_path.get(key)
            if rec is None:
                seq += 1
                name = str(layer.name() or os.path.basename(src))
                ext = layer.extent()
                crs = layer.crs()
                rec = {
                    "id": f"timeslice_{stamp}_{seq}_{os.path.basename(src)}",
                    "name": os.path.splitext(name)[0],
                    "normalized_name": name,
                    "source_path": src,
                    "project_path": src,
                    "imported_at": stamp,
                    "is_valid_raster": True,
                    "crs": crs.authid() if crs.isValid() else None,
                    "width": layer.width(),
                    "height": layer.height(),
                    "band_count": layer.bandCount(),
                    "extent": {
                        "xmin": ext.xMinimum(), "xmax": ext.xMaximum(),
                        "ymin": ext.yMinimum(), "ymax": ext.yMaximum(),
                    },
                    "adopted_from_canvas": True,
                }
                try:
                    # sliceset.json / tag GeoTIFF / nome file, in questo ordine
                    from .slice_manifest import depth_window_for_file
                    window = depth_window_for_file(src)
                    if window:
                        rec["depth_from"], rec["depth_to"] = window
                        rec["unit"] = "m"
                    else:
                        depth = parse_depth_from_filename(os.path.basename(src)) or {}
                        for k in ("depth_from", "depth_to", "unit"):
                            if depth.get(k) is not None:
                                rec[k] = depth[k]
                except Exception:
                    pass
                new_records.append(rec)
                by_path[key] = rec
            if rec.get("id") not in in_group:
                wanted.append((gname, rec["id"]))

    if new_records:
        register_timeslices_batch(project_root, new_records)
        result["registered"] = len(new_records)

    by_group = {}
    for gname, tid in wanted:
        by_group.setdefault(gname, []).append(tid)
    for gname, ids in by_group.items():
        group, created = create_raster_group(project_root, gname)
        assign_timeslices_to_group(project_root, group.get("id"), ids)
        result["assigned"] += len(ids)
        result["groups"] += 1
    return result
