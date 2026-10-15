# -*- coding: utf-8 -*-
"""Variant Manager dialog for Project Manager.

Cascade-deletes a processing variant (its `Processing/processed/<name>/`
folder, any variant derived from it, their generated time-slices, matching
catalog rows and loaded canvas layers) without needing the GPR Profile Viewer
open for that project. Mirrors the "Elimina variante..." action already
available there (gpr_viewer_variants.py); both share the same core logic in
variant_cascade.py so the two entry points never diverge.
"""

from __future__ import annotations

import os

from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)
from qgis.PyQt.QtCore import Qt

from . import project_zones, variant_cascade, variant_provenance
from .gpr_npz_io import list_variants, read_variant_manifest
from .layer_source_utils import remove_layers_for_paths


def _describe_plan_en(plan: dict) -> str:
    names = list(plan.get("names") or [])
    variant_name = plan.get("variant_name") or (names[0] if names else "")
    children = [n for n in names if n != variant_name]
    n_ts_dirs = sum(1 for d in (plan.get("timeslice_dirs") or {}).values() if d)
    n_records = sum(len(v) for v in (plan.get("matched_ts_records") or {}).values())
    n_layers = len(plan.get("layer_paths") or [])

    lines = [f"Delete variant '{variant_name}'?"]
    if children:
        lines.append(f"Derived variants deleted in cascade ({len(children)}): {', '.join(children)}")
    lines.append(f"Time-slice folders to delete: {n_ts_dirs}")
    lines.append(f"Catalog rows to remove: {n_records}")
    lines.append(f"Canvas layers to unload: {n_layers}")
    lines.append("")
    lines.append("Files on disk will be removed. This cannot be undone.")
    return "\n".join(lines)


class VariantManagerDialog(QDialog):
    def __init__(self, project_root, parent=None, on_updated=None, zone=None):
        super().__init__(parent)
        self.project_root = project_root
        self.on_updated = on_updated
        # Varianti dell'area indicata (default: quella attiva del progetto).
        self.zone = zone or project_zones.active_zone(project_root)
        base = project_zones.zone_base(project_root, self.zone)
        self.processed_root = os.path.join(base, "processed")
        self.timeslices_root = os.path.join(base, "timeslices")
        self.setWindowTitle(
            "Variant Manager"
            if self.zone == project_zones.DEFAULT_ZONE
            else f"Variant Manager - {self.zone}"
        )
        self.resize(480, 420)
        self._build_ui()
        self._refresh()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        info = QLabel(
            "Processing variants for this project. Deleting a variant also removes\n"
            "any variant derived from it, its generated time-slices, matching catalog\n"
            "rows and any canvas layers loaded from them."
        )
        info.setWordWrap(True)
        lay.addWidget(info)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Variant", "Profiles", "Last step"])
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        lay.addWidget(self.tree)

        row = QHBoxLayout()
        self.delete_btn = QPushButton("Delete variant (cascade)...")
        self.delete_btn.clicked.connect(self._delete_selected)
        self.delete_btn.setEnabled(False)
        row.addWidget(self.delete_btn)
        row.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        row.addWidget(close_btn)
        lay.addLayout(row)

    def _on_selection_changed(self):
        self.delete_btn.setEnabled(bool(self._selected_variant_name()))

    def _refresh(self):
        self.tree.clear()
        self.delete_btn.setEnabled(False)
        if not os.path.isdir(self.processed_root):
            self.tree.addTopLevelItem(QTreeWidgetItem(["(no variant saved for this project)"]))
            return
        dirs = list_variants(self.processed_root)
        if not dirs:
            self.tree.addTopLevelItem(QTreeWidgetItem(["(no variant saved for this project)"]))
            return

        entries = variant_provenance.load(self.processed_root)
        by_name = {}
        for d in dirs:
            man = read_variant_manifest(d) or {}
            nm = str(man.get("name") or os.path.basename(os.path.normpath(d)))
            by_name[nm] = {
                "n_prof": int(man.get("n_profiles") or len(man.get("files") or [])),
            }
        present = list(by_name.keys())
        forest = variant_provenance.build_forest(entries, present)

        def _add(node, parent_item):
            nm = node["name"]
            info = by_name.get(nm, {})
            last_step = (entries.get(nm, {}) or {}).get("step") or ""
            it = QTreeWidgetItem([nm, str(info.get("n_prof", "")), str(last_step)])
            it.setData(0, Qt.UserRole, nm)
            if parent_item is None:
                self.tree.addTopLevelItem(it)
            else:
                parent_item.addChild(it)
            for child in node.get("children") or []:
                _add(child, it)

        for root in forest:
            _add(root, None)
        self.tree.expandAll()

    def _selected_variant_name(self) -> str:
        it = self.tree.currentItem()
        if it is None:
            return ""
        name = it.data(0, Qt.UserRole)
        return str(name) if name else ""

    def _delete_selected(self):
        name = self._selected_variant_name()
        if not name:
            return
        plan = variant_cascade.plan_cascade_delete(
            self.processed_root, self.timeslices_root, self.project_root, name,
        )
        ans = QMessageBox.question(
            self, "Delete variant", _describe_plan_en(plan),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if ans != QMessageBox.Yes:
            return

        remove_layers_for_paths(plan.get("layer_paths"))
        result = variant_cascade.execute_cascade_delete(
            self.processed_root, self.timeslices_root, self.project_root, plan,
        )
        if result.get("warnings"):
            QMessageBox.warning(self, "Delete variant", "\n".join(result["warnings"]))

        self._refresh()
        if callable(self.on_updated):
            self.on_updated()
