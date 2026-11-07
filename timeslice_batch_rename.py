# -*- coding: utf-8 -*-
"""
Rinomina in blocco delle time-slice: trova/sostituisci su una PARTE del nome,
prefisso/suffisso, oppure sequenza numerata. Con anteprima prima/dopo.

``compute_new_names`` e' una funzione pura (testabile senza Qt); la finestra
la usa per l'anteprima e per restituire il piano di rinomina.
"""

import os
import re

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QComboBox,
    QLineEdit, QCheckBox, QSpinBox, QStackedWidget, QWidget, QTableWidget,
    QTableWidgetItem, QDialogButtonBox, QHeaderView, QAbstractItemView,
)

from .project_catalog import sanitize_filename

MODE_REPLACE = "replace"
MODE_AFFIX = "affix"
MODE_SEQUENCE = "sequence"


def compute_new_names(paths, mode, params):
    """Nuovi nomi file (estensione inclusa) per ``paths``, nello stesso ordine.

    Ritorna ``(names, error)``: ``error`` e' un testo se i parametri non sono
    validi (es. espressione regolare errata), altrimenti ``None``.
    """
    stems_ext = [os.path.splitext(os.path.basename(p)) for p in paths]
    names = []
    try:
        if mode == MODE_REPLACE:
            find = params.get("find", "")
            repl = params.get("replace", "")
            if not find:
                return [os.path.basename(p) for p in paths], None
            flags = 0 if params.get("case") else re.IGNORECASE
            if params.get("regex"):
                pat = re.compile(find, flags)
                for stem, ext in stems_ext:
                    names.append(pat.sub(repl, stem) + ext)
            else:
                pat = re.compile(re.escape(find), flags)
                for stem, ext in stems_ext:
                    # lambda: il testo sostitutivo e' letterale (niente \1, \g)
                    names.append(pat.sub(lambda _m: repl, stem) + ext)
        elif mode == MODE_AFFIX:
            pre = params.get("prefix", "")
            suf = params.get("suffix", "")
            for stem, ext in stems_ext:
                names.append(f"{pre}{stem}{suf}{ext}")
        elif mode == MODE_SEQUENCE:
            pre = params.get("prefix", "")
            start = int(params.get("start", 1))
            pad = int(params.get("padding", 3))
            for i, (_stem, ext) in enumerate(stems_ext):
                names.append(f"{pre}{str(start + i).zfill(pad)}{ext.lower()}")
        else:
            return [os.path.basename(p) for p in paths], "Unknown mode."
    except (re.error, IndexError) as exc:
        return [os.path.basename(p) for p in paths], f"Invalid pattern: {exc}"
    return [sanitize_filename(n) if n else n for n in names], None


class BatchRenameDialog(QDialog):
    """``entries``: lista di (chiave, percorso) gia' ordinata. Dopo ``exec_``
    il piano e' in ``plan``: lista di (chiave, vecchio_percorso, nuovo_percorso,
    nuovo_nome) per i soli file che cambiano."""

    def __init__(self, entries, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch rename")
        self.resize(760, 560)
        self._entries = list(entries)
        self._paths = [p for _k, p in self._entries]
        self._old_abs = {os.path.normcase(os.path.abspath(p)) for p in self._paths}
        self.plan = []

        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(
            f"{len(self._entries)} time-slice selected. Only the selected files are renamed; "
            "the extension is always kept."
        ))

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Find & replace part of the name", MODE_REPLACE)
        self.mode_combo.addItem("Add prefix / suffix", MODE_AFFIX)
        self.mode_combo.addItem("Numbered sequence (replaces the whole name)", MODE_SEQUENCE)
        lay.addWidget(self.mode_combo)

        self.stack = QStackedWidget()
        lay.addWidget(self.stack)

        # --- replace
        w = QWidget()
        f = QFormLayout(w)
        f.setContentsMargins(0, 0, 0, 0)
        self.find_edit = QLineEdit()
        self.find_edit.setPlaceholderText("text to find in the name")
        self.repl_edit = QLineEdit()
        self.repl_edit.setPlaceholderText("replacement (empty = remove the text)")
        self.case_chk = QCheckBox("Case sensitive")
        self.regex_chk = QCheckBox("Regular expression")
        opts = QHBoxLayout()
        opts.addWidget(self.case_chk)
        opts.addWidget(self.regex_chk)
        opts.addStretch(1)
        f.addRow("Find:", self.find_edit)
        f.addRow("Replace with:", self.repl_edit)
        f.addRow("", opts)
        self.stack.addWidget(w)

        # --- affix
        w = QWidget()
        f = QFormLayout(w)
        f.setContentsMargins(0, 0, 0, 0)
        self.prefix_edit = QLineEdit()
        self.suffix_edit = QLineEdit()
        self.suffix_edit.setPlaceholderText("added before the extension")
        f.addRow("Prefix:", self.prefix_edit)
        f.addRow("Suffix:", self.suffix_edit)
        self.stack.addWidget(w)

        # --- sequence
        w = QWidget()
        f = QFormLayout(w)
        f.setContentsMargins(0, 0, 0, 0)
        self.seq_prefix_edit = QLineEdit("timeslice_")
        self.seq_start = QSpinBox()
        self.seq_start.setRange(0, 1000000)
        self.seq_start.setValue(1)
        self.seq_pad = QSpinBox()
        self.seq_pad.setRange(1, 8)
        self.seq_pad.setValue(3)
        f.addRow("Prefix:", self.seq_prefix_edit)
        f.addRow("Start index:", self.seq_start)
        f.addRow("Number padding:", self.seq_pad)
        self.stack.addWidget(w)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Current name", "New name", "Status"])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        lay.addWidget(self.table, 1)

        self.summary = QLabel("")
        lay.addWidget(self.summary)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        self.buttons.accepted.connect(self._accept)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)

        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        for ed in (self.find_edit, self.repl_edit, self.prefix_edit,
                   self.suffix_edit, self.seq_prefix_edit):
            ed.textChanged.connect(self._update_preview)
        for chk in (self.case_chk, self.regex_chk):
            chk.toggled.connect(self._update_preview)
        for sp in (self.seq_start, self.seq_pad):
            sp.valueChanged.connect(self._update_preview)
        self._on_mode_changed()

    # -- helpers
    def _params(self):
        return {
            "find": self.find_edit.text(),
            "replace": self.repl_edit.text(),
            "case": self.case_chk.isChecked(),
            "regex": self.regex_chk.isChecked(),
            "prefix": (self.prefix_edit.text()
                       if self.mode_combo.currentData() == MODE_AFFIX
                       else self.seq_prefix_edit.text()),
            "suffix": self.suffix_edit.text(),
            "start": self.seq_start.value(),
            "padding": self.seq_pad.value(),
        }

    def _on_mode_changed(self):
        self.stack.setCurrentIndex(self.mode_combo.currentIndex())
        self._update_preview()

    def _update_preview(self):
        mode = self.mode_combo.currentData()
        names, error = compute_new_names(self._paths, mode, self._params())
        targets = [os.path.join(os.path.dirname(p), n) for p, n in zip(self._paths, names)]
        norm = [os.path.normcase(os.path.abspath(t)) for t in targets]
        counts = {}
        for n in norm:
            counts[n] = counts.get(n, 0) + 1

        self.plan = []
        changed = problems = 0
        self.table.setRowCount(len(self._paths))
        for row, (entry, old, new_name, target, key) in enumerate(
                zip(self._entries, self._paths, names, targets, norm)):
            old_abs = os.path.normcase(os.path.abspath(old))
            status, bad = "", False
            if error or not new_name:
                status, bad = "invalid name", True
            elif key == old_abs:
                status = "unchanged"
            elif counts[key] > 1:
                status, bad = "duplicate target", True
            elif key in self._old_abs:
                status, bad = "clashes with another selected file", True
            elif os.path.exists(target):
                status, bad = "file already exists", True
            else:
                status = "will be renamed"
                changed += 1
                self.plan.append((entry[0], old, target, new_name))
            problems += 1 if bad else 0
            items = (os.path.basename(old), new_name, status)
            for col, text in enumerate(items):
                it = QTableWidgetItem(text)
                if bad:
                    it.setForeground(QColor("#c0392b"))
                elif status == "unchanged":
                    it.setForeground(QColor("#888888"))
                self.table.setItem(row, col, it)

        if error:
            self.summary.setText(error)
            self.summary.setStyleSheet("color: #c0392b;")
        elif problems:
            self.summary.setText(f"{problems} conflict(s): fix them before renaming.")
            self.summary.setStyleSheet("color: #c0392b;")
        else:
            self.summary.setText(f"{changed} file(s) will be renamed.")
            self.summary.setStyleSheet("")
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(not error and not problems and changed > 0)

    def _accept(self):
        self._update_preview()
        if self.plan:
            self.accept()
