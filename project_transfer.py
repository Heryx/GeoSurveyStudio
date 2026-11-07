# -*- coding: utf-8 -*-
"""Trasferimento di un progetto GeoSurvey Studio in un'altra cartella.

Copia l'intera cartella-progetto (project_root) in una nuova posizione e
riscrive i metadati, cosi' alla riapertura il plugin la riconosce:

- ``metadata/project_catalog.json``: campo ``project_root`` e ogni percorso
  assoluto che puntava dentro la vecchia cartella (i relativi restano tali);
- ``metadata/project.json``, ``sliceset.json``, sidecar ``*.json`` / ``*.qml``
  e simili: ogni occorrenza del vecchio percorso (con barre ``\\``, ``/`` o
  ``\\\\`` come nel JSON) diventa il nuovo;
- progetti QGIS (``.qgs`` e ``.qgz``) dentro la cartella: stessa sostituzione,
  compresa la voce ``GeoSurveyStudio/gpr_project_root``.

Non tocca: i file ``.npz`` (il ``source_path`` interno e' solo informativo) e i
percorsi che puntavano FUORI dalla vecchia cartella (restano com'erano: sono
elencati nel rapporto). Con ``move=True`` la vecchia cartella viene cancellata
solo a copia e riscrittura riuscite.

Modulo senza dipendenze da Qt/QGIS."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import zipfile

TEXT_SUFFIXES = (".json", ".qml", ".qgs", ".qmd", ".txt", ".xml", ".csv", ".prj")
_MAX_TEXT_BYTES = 64 * 1024 * 1024
_CASE_INSENSITIVE = os.name == "nt"


class TransferError(Exception):
    pass


def _norm(path: str) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(str(path or ""))))


def validate_transfer(old_root: str, new_root: str) -> str:
    """Messaggio d'errore ('' se il trasferimento e' possibile)."""
    if not old_root or not os.path.isdir(old_root):
        return "La cartella del progetto attuale non esiste."
    if not new_root or not str(new_root).strip():
        return "Scegli la cartella di destinazione."
    o, n = _norm(old_root), _norm(new_root)
    if o == n:
        return "La destinazione coincide con la cartella attuale."
    if n.startswith(o + os.sep):
        return "La destinazione non puo' stare dentro la cartella attuale."
    if o.startswith(n + os.sep):
        return "La destinazione non puo' contenere la cartella attuale."
    if os.path.exists(new_root) and (not os.path.isdir(new_root) or os.listdir(new_root)):
        return "La cartella di destinazione esiste e non e' vuota."
    return ""


def _path_variants(root: str) -> list:
    """Forme testuali del percorso: native, barre dritte, JSON-escaped."""
    base = os.path.normpath(str(root)).rstrip("\\/")
    forms = []
    for form in (base, base.replace("\\", "/"), base.replace("\\", "\\\\")):
        if form and form not in forms:
            forms.append(form)
    return forms


def _build_replacer(old_root: str, new_root: str):
    """Ritorna ``f(text) -> (text, n)`` che sostituisce il vecchio percorso con
    il nuovo conservando lo stile di barre di ciascuna occorrenza."""
    new_base = os.path.normpath(str(new_root)).rstrip("\\/")
    pairs = [
        (old, new) for old, new in zip(
            _path_variants(old_root),
            (new_base, new_base.replace("\\", "/"), new_base.replace("\\", "\\\\")),
        )
    ]
    # Prima le forme piu' lunghe (``\\\\``) cosi' non si spezzano.
    pairs.sort(key=lambda p: -len(p[0]))
    flags = re.IGNORECASE if _CASE_INSENSITIVE else 0
    # Il percorso deve finire su un separatore o su un delimitatore: "C:\a\p"
    # non deve toccare "C:\a\pippo".
    tail = r"(?=$|[\\/\"'|\s,;:)\]}<>])"
    compiled = [(re.compile(re.escape(o) + tail, flags), n) for o, n in pairs]

    def replace(text: str):
        total = 0
        for rx, new in compiled:
            text, n = rx.subn(lambda _m, _new=new: _new, text)
            total += n
        return text, total

    return replace


def _rewrite_text_file(path: str, replace) -> int:
    try:
        if os.path.getsize(path) > _MAX_TEXT_BYTES:
            return 0
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError:
        return 0
    for enc in ("utf-8", "cp1252"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        return 0
    new_text, n = replace(text)
    if n:
        with open(path, "wb") as fh:
            fh.write(new_text.encode(enc))
    return n


def _rewrite_qgz(path: str, replace) -> int:
    """Riscrive i membri testuali di un .qgz (zip) mantenendo gli altri."""
    total = 0
    fd, tmp = tempfile.mkstemp(suffix=".qgz", dir=os.path.dirname(path))
    os.close(fd)
    try:
        with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename.lower().endswith((".qgs", ".qml", ".qmd")):
                    try:
                        text = data.decode("utf-8")
                    except UnicodeDecodeError:
                        text = None
                    if text is not None:
                        text, n = replace(text)
                        total += n
                        data = text.encode("utf-8")
                zout.writestr(item, data)
        if total:
            os.replace(tmp, path)
        else:
            os.remove(tmp)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return total


def rewrite_metadata(old_root: str, new_root: str) -> dict:
    """Riscrive in ``new_root`` ogni riferimento a ``old_root``.
    Ritorna ``{"files": n_file_modificati, "replacements": n, "changed": [rel]}``."""
    replace = _build_replacer(old_root, new_root)
    changed, total = [], 0
    for folder, _dirs, files in os.walk(new_root):
        for fn in files:
            low = fn.lower()
            path = os.path.join(folder, fn)
            n = 0
            if low.endswith(".qgz"):
                n = _rewrite_qgz(path, replace)
            elif low.endswith(TEXT_SUFFIXES):
                n = _rewrite_text_file(path, replace)
            if n:
                total += n
                changed.append(os.path.relpath(path, new_root))
    return {"files": len(changed), "replacements": total, "changed": changed}


def _fix_catalog(new_root: str) -> bool:
    """Imposta ``project_root`` del catalogo e ne riscrive i percorsi interni
    come relativi (come fa ``save_catalog``)."""
    from .project_catalog import load_catalog, save_catalog, catalog_path
    if not os.path.isfile(catalog_path(new_root)):
        return False
    data = load_catalog(new_root, create_backup_on_migrate=False)
    data["project_root"] = new_root
    save_catalog(new_root, data)
    return True


def external_references(new_root: str) -> list:
    """Percorsi del catalogo che stanno fuori dalla cartella del progetto
    (restano com'erano dopo il trasferimento)."""
    from .project_catalog import load_catalog, _iter_catalog_file_paths, _is_within_root
    out = []
    try:
        catalog = load_catalog(new_root, create_backup_on_migrate=False)
    except Exception:
        return out
    for raw in _iter_catalog_file_paths(catalog):
        p = str(raw or "").strip()
        if p and os.path.isabs(p) and not _is_within_root(p, new_root):
            out.append(p)
    return sorted(set(out))


def _copy_tree(src: str, dst: str, progress=None, should_cancel=None) -> int:
    files = []
    for folder, _dirs, names in os.walk(src):
        for fn in names:
            files.append(os.path.join(folder, fn))
    total = len(files)
    os.makedirs(dst, exist_ok=True)
    for folder, dirs, _names in os.walk(src):
        for d in dirs:
            os.makedirs(os.path.join(dst, os.path.relpath(os.path.join(folder, d), src)), exist_ok=True)
    for i, path in enumerate(files, 1):
        if should_cancel is not None and should_cancel():
            raise TransferError("Trasferimento annullato.")
        target = os.path.join(dst, os.path.relpath(path, src))
        shutil.copy2(path, target)
        if progress is not None:
            progress(i, total, os.path.relpath(path, src))
    return total


def transfer_project(old_root: str, new_root: str, move: bool = False,
                     progress=None, should_cancel=None) -> dict:
    """Copia ``old_root`` in ``new_root`` e aggiorna i metadati.

    ``progress(i, totale, nome)`` e ``should_cancel()`` sono opzionali.
    In caso di errore o annullamento la copia parziale viene rimossa e
    l'originale resta intatto. Ritorna il rapporto con ``old_root``,
    ``new_root``, ``copied``, ``rewrite``, ``external`` e ``moved``."""
    err = validate_transfer(old_root, new_root)
    if err:
        raise TransferError(err)
    created_new = not os.path.exists(new_root)
    try:
        copied = _copy_tree(old_root, new_root, progress, should_cancel)
        rewrite = rewrite_metadata(old_root, new_root)
        _fix_catalog(new_root)
    except Exception as exc:
        shutil.rmtree(new_root, ignore_errors=True)
        if not created_new:
            os.makedirs(new_root, exist_ok=True)
        if isinstance(exc, TransferError):
            raise
        raise TransferError(f"Trasferimento fallito: {exc}") from exc
    moved = False
    if move:
        try:
            shutil.rmtree(old_root)
            moved = True
        except OSError:
            moved = False
    return {
        "old_root": old_root,
        "new_root": new_root,
        "copied": copied,
        "rewrite": rewrite,
        "external": external_references(new_root),
        "moved": moved,
    }
