# -*- coding: utf-8 -*-
"""Confronto tra il ``source()`` di un layer QGIS caricato e un percorso file.

Estratto da ``project_manager_import_rollback_mixin.py`` (dove la stessa
logica era duplicata per ogni rollback) cosi' da essere riusabile anche da
altri punti del plugin (es. cancellazione a cascata di una variante nel GPR
Profile Viewer) senza reimplementarla. Funzioni pure, nessuna dipendenza da
``self``."""

from __future__ import annotations

import os

from qgis.core import QgsProject


def normalize_source_for_compare(source: str) -> str:
    raw = str(source or "").strip()
    if not raw:
        return ""
    base = raw.split("|", 1)[0].strip()
    try:
        return os.path.normcase(os.path.abspath(base))
    except Exception:
        return os.path.normcase(base)


def find_layer_ids_for_paths(paths) -> list:
    """Id dei layer attualmente caricati il cui ``source()`` corrisponde
    esattamente (dopo normalizzazione) a uno dei percorsi in ``paths``."""
    wanted = {
        normalize_source_for_compare(p)
        for p in (paths or [])
        if normalize_source_for_compare(p)
    }
    if not wanted:
        return []
    project = QgsProject.instance()
    return [
        lid
        for lid, lyr in project.mapLayers().items()
        if normalize_source_for_compare(lyr.source()) in wanted
    ]


def remove_layers_for_paths(paths) -> int:
    """Scarica dal progetto QGIS corrente i layer il cui source e' in
    ``paths``. Ritorna il numero di layer rimossi."""
    ids = find_layer_ids_for_paths(paths)
    if ids:
        QgsProject.instance().removeMapLayers(ids)
    return len(ids)
