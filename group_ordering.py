# -*- coding: utf-8 -*-
"""
Ordine di gruppi e immagini.

Convenzione unica: il PRIMO elemento sta in CIMA al pannello layer di QGIS (quindi
si vede sopra gli altri) ed e' il primo indice della manopola di navigazione.
Per le time-slice l'ordine naturale e' dalla piu' superficiale alla piu' profonda.

Le funzioni sulle liste sono pure; ``apply_*_to_qgis`` riordinano i nodi
dell'albero layer di QGIS senza ricaricare nulla.
"""

import os

PLUGIN_ROOT_NAMES = ("GeoSurvey Studio", "RasterLinker", "rasterlinker")


def move_items(ids, selected, delta):
    """Sposta di una posizione (``delta`` = -1 su, +1 giu') gli elementi
    ``selected`` dentro ``ids``, mantenendo l'ordine relativo del resto.
    I blocchi selezionati contigui si muovono insieme; ai bordi restano fermi."""
    ids = list(ids)
    sel = set(selected)
    if not sel or delta not in (-1, 1):
        return ids
    rng = range(len(ids)) if delta < 0 else range(len(ids) - 1, -1, -1)
    for i in rng:
        j = i + delta
        if ids[i] in sel and 0 <= j < len(ids) and ids[j] not in sel:
            ids[i], ids[j] = ids[j], ids[i]
    return ids


def depth_sort_key(rec):
    """Chiave: (profondita' iniziale, finale, nome); senza profondita' in fondo."""
    rec = rec or {}

    def _num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    d0, d1 = _num(rec.get("depth_from")), _num(rec.get("depth_to"))
    name = str(rec.get("normalized_name") or rec.get("name") or rec.get("id") or "").lower()
    if d0 is None and d1 is None:
        return (1, 0.0, 0.0, name)
    return (0, d0 if d0 is not None else d1, d1 if d1 is not None else d0, name)


def sort_ids(ids, records_by_id, mode="depth"):
    """``mode``: ``depth`` (superficiale prima), ``depth_desc`` (profonda prima),
    ``name``. Gli id senza record restano in coda nell'ordine originale."""
    known = [t for t in ids if t in records_by_id]
    unknown = [t for t in ids if t not in records_by_id]
    if mode == "name":
        known.sort(key=lambda t: str(records_by_id[t].get("normalized_name")
                                     or records_by_id[t].get("name") or t).lower())
    elif mode == "depth_desc":
        with_d = [t for t in known if depth_sort_key(records_by_id[t])[0] == 0]
        without = [t for t in known if depth_sort_key(records_by_id[t])[0] != 0]
        with_d.sort(key=lambda t: depth_sort_key(records_by_id[t]), reverse=True)
        known = with_d + without
    else:
        known.sort(key=lambda t: depth_sort_key(records_by_id[t]))
    return known + unknown


# ---------------------------------------------------------------- albero QGIS
def _node_key(node):
    from qgis.core import QgsLayerTreeGroup
    if isinstance(node, QgsLayerTreeGroup):
        return ("g", node.name())
    try:
        return ("l", node.layerId())
    except Exception:
        return ("?", id(node))


def reorder_children(parent, desired_keys):
    """Riordina i figli di ``parent`` i cui ``_node_key`` sono in ``desired_keys``
    secondo quell'ordine. I figli non elencati restano dove sono."""
    keys = list(desired_keys)
    children = list(parent.children())
    slots = [i for i, ch in enumerate(children) if _node_key(ch) in keys]
    if len(slots) < 2:
        return False
    by_key = {_node_key(children[i]): children[i] for i in slots}
    ordered_keys = [k for k in keys if k in by_key]
    # sequenza finale: i figli "tracciati" occupano i propri slot nell'ordine voluto
    final = [_node_key(ch) for ch in children]
    for slot, k in zip(slots, ordered_keys):
        final[slot] = k
    changed = False
    for i, want in enumerate(final):
        cur = list(parent.children())
        if _node_key(cur[i]) == want:
            continue
        node = next((ch for ch in cur[i + 1:] if _node_key(ch) == want), None)
        if node is None:
            continue
        clone = node.clone()
        try:
            clone.setExpanded(node.isExpanded())
        except Exception:
            pass
        parent.insertChildNode(i, clone)
        parent.removeChildNode(node)
        changed = True
    return changed


def _plugin_root():
    from qgis.core import QgsLayerTreeGroup, QgsProject
    root = QgsProject.instance().layerTreeRoot()
    for name in PLUGIN_ROOT_NAMES:
        node = next((g for g in root.children()
                     if isinstance(g, QgsLayerTreeGroup) and g.name() == name), None)
        if node is not None:
            return node
    return None


def apply_group_order_to_qgis(ordered_group_names):
    """Mette nell'ordine dato i gruppi con quei nomi, ovunque stiano nell'albero
    (ogni gruppo padre viene riordinato per conto suo)."""
    try:
        from qgis.core import QgsProject
        root = QgsProject.instance().layerTreeRoot()
        pos = {n: i for i, n in enumerate(ordered_group_names)}
        buckets = {}
        for g in root.findGroups(True):
            if g.name() in pos and g.parent() is not None:
                buckets.setdefault(id(g.parent()), (g.parent(), []))[1].append(g.name())
        changed = False
        for parent, names in buckets.values():
            names.sort(key=lambda n: pos[n])
            changed |= reorder_children(parent, [("g", n) for n in names])
        return changed
    except Exception:
        return False


def apply_slice_order_to_qgis(group_name, ordered_paths):
    """Riordina i layer raster secondo ``ordered_paths`` (percorsi dei file).
    Cerca i layer per file in tutto l'albero, quindi funziona anche con
    sottogruppi annidati; ogni gruppo padre viene riordinato per conto suo.
    ``group_name`` non e' piu' necessario e resta solo per compatibilita'."""
    try:
        from qgis.core import QgsProject
        norm = lambda p: os.path.normcase(os.path.abspath(str(p).split("|")[0]))
        pos = {norm(p): i for i, p in enumerate(ordered_paths) if p}
        root = QgsProject.instance().layerTreeRoot()
        buckets = {}
        for node in root.findLayers():
            lyr = node.layer()
            if lyr is None or node.parent() is None:
                continue
            src = None
            try:
                src = lyr.customProperty("rl_amp_src")  # raster derivato dalle ampiezze
            except Exception:
                pass
            for cand in (src, lyr.source()):
                if cand and norm(cand) in pos:
                    buckets.setdefault(id(node.parent()), (node.parent(), []))[1].append(
                        (pos[norm(cand)], _node_key(node)))
                    break
        changed = False
        for parent, items in buckets.values():
            items.sort()
            changed |= reorder_children(parent, [k for _i, k in items])
        return changed
    except Exception:
        return False
