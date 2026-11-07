# -*- coding: utf-8 -*-
"""Cancellazione a cascata di una variante di processing.

Una variante puo' avere varianti "figlie" (create con "Nuova variante da
qui", tracciate in ``variant_graph.json`` via ``variant_provenance``) e
time-slice generate da lei (cartella ``timeslices/<variante>/``, per
convenzione — nessun campo esplicito le lega, va inferito dal path). Questo
modulo calcola cosa cancellare (``plan_cascade_delete``, nessun side-effect,
per mostrare un riepilogo prima di chiedere conferma) ed esegue la
cancellazione (``execute_cascade_delete``): cartelle processed/timeslices,
righe di catalogo, nodi nel grafo di provenienza.

Puro stdlib + ``project_catalog``/``variant_provenance``: nessuna dipendenza
da Qt/QGIS (lo scaricamento dei layer dal canvas resta a carico del
chiamante, vedi ``layer_source_utils.remove_layers_for_paths``, da invocare
sui ``layer_paths`` del piano PRIMA di eseguire la cancellazione)."""

from __future__ import annotations

import os
import shutil

try:
    from . import variant_provenance
    from .project_catalog import ensure_system_group, load_catalog, save_catalog
except ImportError:
    import variant_provenance
    from project_catalog import ensure_system_group, load_catalog, save_catalog


def _abs_norm(path: str) -> str:
    try:
        return os.path.normcase(os.path.abspath(str(path or "")))
    except Exception:
        return os.path.normcase(str(path or ""))


def _is_within_root(path: str, root: str) -> bool:
    p = _abs_norm(path)
    r = _abs_norm(root)
    if not p or not r:
        return False
    return p == r or p.startswith(r + os.sep)


def rename_variant(processed_root: str, old: str, new: str) -> str:
    """Rinomina la variante ``old`` in ``new``: cartella ``processed/<old>/``,
    ``name`` in variant.json e nodo (e riferimenti ``parent`` dei figli) nel
    grafo di provenienza. Le time-slice gia' generate non si toccano (layer e
    catalogo puntano ai loro percorsi). Ritorna '' se ok, altrimenti il
    messaggio d'errore."""
    old, new = str(old or "").strip(), str(new or "").strip()
    if not old or not new:
        return "Nome vuoto."
    if new == old:
        return ""
    if new.lower() == "raw" or any(c in new for c in '\\/:*?"<>|') or new in (".", ".."):
        return "Nome non valido (caratteri vietati o riservato)."
    src = os.path.join(processed_root, old)
    dst = os.path.join(processed_root, new)
    if not os.path.isdir(src):
        return f"Cartella della variante '{old}' non trovata."
    if os.path.exists(dst) and os.path.normcase(dst) != os.path.normcase(src):
        return f"Esiste gia' una variante chiamata '{new}'."
    try:
        os.rename(src, dst)
    except OSError as exc:
        return f"Impossibile rinominare la cartella (file in uso?): {exc}"

    try:
        import json
        mpath = os.path.join(dst, "variant.json")
        if os.path.isfile(mpath):
            with open(mpath, "r", encoding="utf-8") as fh:
                man = json.load(fh)
            if isinstance(man, dict):
                man["name"] = new
                with open(mpath, "w", encoding="utf-8") as fh:
                    json.dump(man, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass
    try:
        entries = variant_provenance.load(processed_root)
        if old in entries:
            entries = {(new if n == old else n): m for n, m in entries.items()}
            for m in entries.values():
                if m.get("parent") == old:
                    m["parent"] = new
            variant_provenance.save(processed_root, entries)
    except Exception:
        pass
    return ""


def list_descendants(entries: dict, name: str) -> list:
    """Discendenti (figli, nipoti, ...) di ``name`` nel grafo ``entries``
    (formato di ``variant_provenance.load()``). A differenza di
    ``variant_provenance.remove_entry``, che ri-aggancia solo i figli
    diretti al nonno, questa funzione elenca l'intero sottoalbero. Robusta
    a cicli malformati nel grafo."""
    name = str(name or "")
    children_by_parent: dict = {}
    for n, meta in (entries or {}).items():
        parent = (meta or {}).get("parent")
        if parent:
            children_by_parent.setdefault(str(parent), []).append(str(n))

    out = []
    seen = {name}
    stack = list(children_by_parent.get(name, []))
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        out.append(n)
        stack.extend(children_by_parent.get(n, []))
    return out


def plan_cascade_delete(
    processed_root: str,
    timeslices_root: str,
    catalog_project_root: str,
    variant_name: str,
) -> dict:
    """Calcola cosa verrebbe cancellato per ``variant_name`` (+ discendenti),
    senza modificare nulla su disco. Ritorna::

        {
            "variant_name": str,
            "names": [variant_name, *discendenti],
            "variant_dirs": {name: path|None},      # processed/<name>/, None se assente
            "timeslice_dirs": {name: path|None},     # timeslices/<name>/, None se assente
            "matched_ts_records": {name: [record, ...]},  # righe catalogo sotto quella cartella
            "layer_paths": [path, ...],              # project_path dei record trovati
        }
    """
    variant_name = str(variant_name or "").strip()
    entries = variant_provenance.load(processed_root) if processed_root else {}
    names = [variant_name] + list_descendants(entries, variant_name)

    variant_dirs = {}
    timeslice_dirs = {}
    for name in names:
        vdir = os.path.join(processed_root, name) if processed_root else ""
        variant_dirs[name] = vdir if vdir and os.path.isdir(vdir) else None
        tdir = os.path.join(timeslices_root, name) if timeslices_root else ""
        timeslice_dirs[name] = tdir if tdir and os.path.isdir(tdir) else None

    matched_ts_records = {name: [] for name in names}
    layer_paths = []
    try:
        catalog = load_catalog(catalog_project_root) if catalog_project_root else {}
    except Exception:
        catalog = {}
    for rec in catalog.get("timeslices", []) or []:
        pth = str(rec.get("project_path") or "")
        if not pth:
            continue
        for name in names:
            tdir = timeslice_dirs.get(name)
            if tdir and _is_within_root(pth, tdir):
                matched_ts_records[name].append(rec)
                layer_paths.append(pth)
                break

    return {
        "variant_name": variant_name,
        "names": names,
        "variant_dirs": variant_dirs,
        "timeslice_dirs": timeslice_dirs,
        "matched_ts_records": matched_ts_records,
        "layer_paths": layer_paths,
    }


def describe_plan(plan: dict) -> str:
    """Testo di riepilogo per l'unico dialogo di conferma prima della cascata."""
    names = list(plan.get("names") or [])
    variant_name = plan.get("variant_name") or (names[0] if names else "")
    children = [n for n in names if n != variant_name]
    n_ts_dirs = sum(1 for d in (plan.get("timeslice_dirs") or {}).values() if d)
    n_records = sum(len(v) for v in (plan.get("matched_ts_records") or {}).values())
    n_layers = len(plan.get("layer_paths") or [])

    lines = [f"Eliminare la variante '{variant_name}'?"]
    if children:
        lines.append(f"Varianti derivate da eliminare in cascata ({len(children)}): {', '.join(children)}")
    lines.append(f"Cartelle time-slice da eliminare: {n_ts_dirs}")
    lines.append(f"Righe di catalogo da rimuovere: {n_records}")
    lines.append(f"Layer da scaricare dal canvas: {n_layers}")
    lines.append("")
    lines.append("I file su disco verranno rimossi. Operazione non annullabile.")
    return "\n".join(lines)


def execute_cascade_delete(
    processed_root: str,
    timeslices_root: str,
    catalog_project_root: str,
    plan: dict,
) -> dict:
    """Esegue la cancellazione pianificata da ``plan_cascade_delete``.

    Ogni cartella (processed/<name>/, timeslices/<name>/) viene rimossa con
    un try/except indipendente: se una fallisce (es. handle di file ancora
    aperto), le altre proseguono comunque. Le righe di catalogo delle
    time-slice la cui cartella non e' stata rimossa vengono spostate nel
    gruppo di sistema "Unassigned" invece di essere cancellate, cosi' non
    restano ne' perse ne' referenziate a un percorso inesistente. Il nodo di
    provenienza di una variante viene rimosso solo se la sua cartella
    ``processed/<name>/`` e' stata eliminata con successo (o era già assente).

    Ritorna ``{deleted_variants, failed_variants, removed_records, warnings}``.
    """
    names = list(plan.get("names") or [])
    variant_dirs = plan.get("variant_dirs") or {}
    timeslice_dirs = plan.get("timeslice_dirs") or {}
    matched_ts_records = plan.get("matched_ts_records") or {}

    deleted_variants = []
    failed_variants = []
    warnings = []
    removed_record_ids = set()
    survivor_records = []

    for name in names:
        processed_ok = True
        timeslice_ok = True

        vdir = variant_dirs.get(name)
        if vdir and os.path.isdir(vdir):
            try:
                shutil.rmtree(vdir)
            except Exception as exc:
                processed_ok = False
                warnings.append(f"Rimozione cartella variante '{name}' fallita: {exc}")

        tdir = timeslice_dirs.get(name)
        if tdir and os.path.isdir(tdir):
            try:
                shutil.rmtree(tdir)
            except Exception as exc:
                timeslice_ok = False
                warnings.append(f"Rimozione cartella time-slice '{name}' fallita: {exc}")

        recs = matched_ts_records.get(name) or []
        if timeslice_ok:
            removed_record_ids.update(r.get("id") for r in recs if r.get("id"))
        else:
            survivor_records.extend(recs)

        if processed_ok:
            deleted_variants.append(name)
        else:
            failed_variants.append(name)

    removed_records = 0
    if catalog_project_root and (removed_record_ids or survivor_records):
        try:
            data = load_catalog(catalog_project_root)
            if removed_record_ids:
                before = len(data.get("timeslices", []) or [])
                data["timeslices"] = [
                    r for r in data.get("timeslices", []) or [] if r.get("id") not in removed_record_ids
                ]
                removed_records = max(0, before - len(data.get("timeslices", []) or []))
                for g in data.get("raster_groups", []) or []:
                    g["timeslice_ids"] = [
                        tid for tid in g.get("timeslice_ids", []) or [] if tid not in removed_record_ids
                    ]
                data["links"] = [
                    lk for lk in data.get("links", []) or [] if lk.get("timeslice_id") not in removed_record_ids
                ]
            if survivor_records:
                survivor_ids = [r.get("id") for r in survivor_records if r.get("id")]
                groups = data.setdefault("raster_groups", [])
                grp = next((g for g in groups if g.get("id") == "grp_imported"), None)
                if grp is None:
                    groups.append(ensure_system_group(catalog_project_root, "grp_imported", "Unassigned"))
                    grp = next(g for g in groups if g.get("id") == "grp_imported")
                grp["timeslice_ids"] = list(dict.fromkeys((grp.get("timeslice_ids") or []) + survivor_ids))
            save_catalog(catalog_project_root, data)
        except Exception as exc:
            warnings.append(f"Aggiornamento catalogo fallito: {exc}")

    if processed_root and deleted_variants:
        try:
            entries = variant_provenance.load(processed_root)
            for name in deleted_variants:
                entries.pop(name, None)
            deleted_set = set(deleted_variants)
            for meta in entries.values():
                if meta.get("parent") in deleted_set:
                    meta["parent"] = None
            variant_provenance.save(processed_root, entries)
        except Exception as exc:
            warnings.append(f"Aggiornamento grafo varianti fallito: {exc}")

    return {
        "deleted_variants": deleted_variants,
        "failed_variants": failed_variants,
        "removed_records": removed_records,
        "warnings": warnings,
    }
