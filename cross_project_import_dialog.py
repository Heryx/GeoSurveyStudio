# -*- coding: utf-8 -*-
"""Importa le time-slice di un ALTRO progetto GPR nel canvas del progetto
corrente, come riferimenti di sola lettura.

Pensato per chi lavora per zone/griglie separate (un progetto GPR per
zona, vedi ``gpr_viewer_project.py`` ``_new_gpr_project``): quando serve
vedere insieme gli output di due zone, questo dialogo porta nel canvas
corrente le time-slice dell'altra senza cambiare il "progetto attivo" usato
per nuove elaborazioni (``_active_project_root``) e senza copiare file.

Il catalogo del progetto CORRENTE non registra questi layer. Motivo: ogni
percorso di cancellazione esistente (``_delete_selected``/``_delete_group``
in ``timeslice_group_manager_dialog.py``, ``execute_cascade_delete`` in
``variant_cascade.py``) fa ``os.remove()`` diretto sul ``project_path`` di
un record — se un riferimento esterno diventasse una riga catalogo,
un'eliminazione nel progetto corrente cancellerebbe il file sorgente
nell'ALTRO progetto. I layer importati vengono invece taggati (vedi
``project_manager_timeslice_import_mixin._import_cross_project_group``)
cosi' restano riconoscibili e "Chiudi progetto precedente"
(``project_layer_cleanup.py``) li puo' ripulire dal canvas in seguito."""

from __future__ import annotations

import os

from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)
from qgis.PyQt.QtCore import Qt

from .project_catalog import load_catalog


class CrossProjectImportDialog(QDialog):
    """``importer`` e' il ``ProjectManagerDialog`` corrente: espone
    ``_import_cross_project_group(source_root, group_record)``
    (``ProjectManagerTimesliceImportMixin``), riusato qui invece di
    duplicare la logica di caricamento raster/tagging."""

    def __init__(self, current_project_root, importer, parent=None):
        super().__init__(parent)
        self.current_project_root = current_project_root
        self._importer = importer
        self.source_root = ""
        self.setWindowTitle("Import from another GPR project")
        self.resize(480, 420)
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        info = QLabel(
            "Choose another GPR project folder, then pick which raster group(s)\n"
            "(time-slices) to bring into THIS project's canvas as read-only\n"
            "references. Files stay where they are; nothing is copied or\n"
            "registered in this project's own catalog."
        )
        info.setWordWrap(True)
        lay.addWidget(info)

        row = QHBoxLayout()
        self.path_label = QLabel("(no project selected)")
        self.path_label.setWordWrap(True)
        row.addWidget(self.path_label, 1)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_source)
        row.addWidget(browse_btn)
        lay.addLayout(row)

        self.group_list = QListWidget()
        self.group_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        lay.addWidget(self.group_list)

        btn_row = QHBoxLayout()
        self.import_btn = QPushButton("Import selected group(s)")
        self.import_btn.clicked.connect(self._import_selected)
        self.import_btn.setEnabled(False)
        btn_row.addWidget(self.import_btn)
        btn_row.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)
        lay.addLayout(btn_row)

    def _browse_source(self):
        start = (
            os.path.dirname(self.current_project_root)
            if self.current_project_root else os.path.expanduser("~")
        )
        folder = QFileDialog.getExistingDirectory(self, "GPR project to import from", start)
        if not folder:
            return
        folder = os.path.normpath(folder)
        current_norm = os.path.normcase(os.path.normpath(self.current_project_root or ""))
        if os.path.normcase(folder) == current_norm:
            QMessageBox.information(
                self, "Import from another GPR project", "That is the current project.",
            )
            return
        if not os.path.isfile(os.path.join(folder, "metadata", "project_catalog.json")):
            QMessageBox.warning(
                self, "Import from another GPR project",
                "No project catalog found in that folder\n"
                "(expected metadata/project_catalog.json).",
            )
            return
        self.source_root = folder
        self.path_label.setText(folder)
        self._refresh_groups()

    def _refresh_groups(self):
        self.group_list.clear()
        try:
            catalog = load_catalog(self.source_root)
        except Exception as exc:
            QMessageBox.warning(
                self, "Import from another GPR project", f"Could not read catalog:\n{exc}",
            )
            self.import_btn.setEnabled(False)
            return
        by_id = {r.get("id"): r for r in catalog.get("timeslices", []) or [] if r.get("id")}
        for group in catalog.get("raster_groups", []) or []:
            tids = [t for t in (group.get("timeslice_ids") or []) if t in by_id]
            if not tids:
                continue
            item = QListWidgetItem(f"{group.get('name') or group.get('id')}  ({len(tids)} time-slice)")
            item.setData(Qt.UserRole, group)
            self.group_list.addItem(item)
        self.import_btn.setEnabled(self.group_list.count() > 0)
        if self.group_list.count() == 0:
            self.group_list.addItem(QListWidgetItem("(no group with time-slices in that project)"))

    def _import_selected(self):
        items = self.group_list.selectedItems()
        items = [it for it in items if it.data(Qt.UserRole) is not None]
        if not items:
            QMessageBox.information(
                self, "Import from another GPR project", "Select at least one group.",
            )
            return
        total = 0
        for item in items:
            group_record = item.data(Qt.UserRole)
            total += self._importer._import_cross_project_group(self.source_root, group_record)
        QMessageBox.information(
            self, "Import from another GPR project", f"Imported {total} layer(s) into the canvas.",
        )
