# -*- coding: utf-8 -*-
"""Scarica dal canvas QGIS i layer prodotti da una project_root GPR.

Usato per "chiudere" pulitamente il lavoro di una zona/griglia prima di
aprire o creare il progetto GPR di un'altra (vedi gpr_viewer_project.py,
``_maybe_close_previous_project``): i file restano sul disco, vengono solo
scaricati dal progetto QGIS corrente in modo che il canvas non resti
"sporco" della zona precedente, a meno che l'utente non scelga di tenerli.

Nessun registro lega oggi un layer QGIS caricato alla sua project_root di
origine: l'identificazione avviene per prefisso di percorso sul ``source()``
del layer (stesso principio di ``layer_source_utils``), mai con un semplice
``str.startswith`` per evitare falsi positivi tipo ``GPR_ProjectA`` vs
``GPR_ProjectA2``. I layer con modifiche non salvate non vengono mai
rimossi automaticamente."""

from __future__ import annotations

import os

try:
    from qgis.core import QgsProject
except ImportError:  # pragma: no cover - solo per test di logica pura
    QgsProject = None

try:
    from .layer_source_utils import normalize_source_for_compare
except ImportError:
    from layer_source_utils import normalize_source_for_compare


def _is_layer_under_root(layer_source: str, root: str) -> bool:
    src = normalize_source_for_compare(layer_source)
    root_norm = normalize_source_for_compare(root)
    if not src or not root_norm:
        return False
    return src == root_norm or src.startswith(root_norm + os.sep)


def _layer_has_unsaved_changes(layer) -> bool:
    for attr in ("isModified", "isEditable"):
        fn = getattr(layer, attr, None)
        if fn is None:
            continue
        try:
            if bool(fn()):
                return True
        except Exception:
            continue
    return False


def find_layer_ids_under_root(project_root: str) -> dict:
    """``{"removable": [layer_id, ...], "skipped_modified": [nome, ...]}``:
    i layer attualmente caricati il cui ``source()`` e' dentro
    ``project_root``, separando quelli con modifiche non salvate (mai
    rimossi automaticamente)."""
    removable = []
    skipped_modified = []
    if not project_root or QgsProject is None:
        return {"removable": removable, "skipped_modified": skipped_modified}
    project = QgsProject.instance()
    for lid, lyr in project.mapLayers().items():
        try:
            src = lyr.source()
        except Exception:
            continue
        if not _is_layer_under_root(src, project_root):
            continue
        if _layer_has_unsaved_changes(lyr):
            skipped_modified.append(lyr.name())
        else:
            removable.append(lid)
    return {"removable": removable, "skipped_modified": skipped_modified}


def _group_depth(group) -> int:
    depth = 0
    parent = group.parent()
    while parent is not None:
        depth += 1
        parent = parent.parent()
    return depth


def _prune_now_empty_groups(candidate_groups) -> int:
    """Rimuove dall'albero layer i gruppi in ``candidate_groups`` (contenevano
    almeno uno dei layer appena scaricati) che sono rimasti senza figli.
    Non tocca gruppi vuoti per altri motivi (es. creati manualmente
    dall'utente e mai popolati da questo cleanup)."""
    pruned = 0
    for group in sorted(candidate_groups, key=_group_depth, reverse=True):
        try:
            if len(group.children()) == 0:
                parent = group.parent()
                if parent is not None:
                    parent.removeChildNode(group)
                    pruned += 1
        except Exception:
            continue
    return pruned


def remove_layers_under_root(project_root: str, prune_empty_groups: bool = True) -> dict:
    """Scarica dal progetto QGIS corrente i layer prodotti da
    ``project_root``. Ritorna
    ``{"removed": n, "skipped_modified": [nomi], "pruned_groups": n}``."""
    found = find_layer_ids_under_root(project_root)
    ids = found["removable"]
    pruned_groups = 0
    if ids and QgsProject is not None:
        project = QgsProject.instance()
        candidate_groups = set()
        if prune_empty_groups:
            root = project.layerTreeRoot()
            for lid in ids:
                node = root.findLayer(lid)
                if node is None:
                    continue
                parent = node.parent()
                while parent is not None and parent is not root:
                    candidate_groups.add(parent)
                    parent = parent.parent()
        project.removeMapLayers(ids)
        if candidate_groups:
            pruned_groups = _prune_now_empty_groups(candidate_groups)
    return {
        "removed": len(ids),
        "skipped_modified": found["skipped_modified"],
        "pruned_groups": pruned_groups,
    }
