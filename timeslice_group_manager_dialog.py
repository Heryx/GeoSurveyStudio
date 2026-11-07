# -*- coding: utf-8 -*-
"""
Time-slice and group manager for GeoSurvey Studio projects.
"""

import os
import shutil

from PyQt5.QtCore import Qt, QItemSelectionModel
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QToolButton,
    QMenu,
    QTabWidget,
    QLabel,
    QComboBox,
    QLineEdit,
    QPushButton,
    QTableView,
    QAbstractItemView,
    QMessageBox,
    QInputDialog,
    QProgressDialog,
    QApplication,
    QDialogButtonBox,
    QListWidget,
    QListWidgetItem,
)
from qgis.core import QgsCoordinateReferenceSystem

from .project_catalog import (
    load_catalog, save_catalog, sanitize_filename, create_raster_group,
    assign_timeslices_to_group, remove_timeslices_from_group,
    detect_timeslice_prefix_groups, move_timeslices_into_group_folder,
    group_folder_name, rescan_group_folders,
)
from .timeslice_group_table_models import TimesliceTableModel, GroupTableModel
from .unassigned_timeslices_dialog import UnassignedTimeslicesDialog
from .timeslice_batch_rename import BatchRenameDialog
from . import group_ordering as _order


class TimesliceGroupManagerDialog(QDialog):
    def __init__(self, project_root, parent=None, on_updated=None, open_catalog_editor_callback=None,
                 on_project_transferred=None):
        super().__init__(parent)
        self.project_root = project_root
        self.on_updated = on_updated
        self.on_project_transferred = on_project_transferred
        self.open_catalog_editor_callback = open_catalog_editor_callback
        self._catalog = {}
        self.setWindowTitle("GeoSurvey Studio Time-slice / Group Manager")
        self.resize(1180, 560)
        self._build_ui()
        self._refresh()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        info = QLabel(
            "Manage time-slice files and raster groups: rename (single/batch), delete, "
            "set assigned CRS, set textual depth range, and assign/remove images from groups."
        )
        info.setWordWrap(True)
        root.addWidget(info)

        tabs = QTabWidget(self)
        root.addWidget(tabs)

        ts_tab = QDialog(self)
        ts_layout = QVBoxLayout(ts_tab)
        ts_layout.setContentsMargins(0, 0, 0, 0)
        ts_layout.setSpacing(6)

        top_filter = QHBoxLayout()
        top_filter.addWidget(QLabel("Show group:"))
        self.filter_group_combo = QComboBox()
        self.filter_group_combo.currentIndexChanged.connect(self._refresh_timeslice_table)
        top_filter.addWidget(self.filter_group_combo, 1)

        top_filter.addWidget(QLabel("Search:"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("name, id, path, group, CRS, depth...")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.textChanged.connect(self._refresh_timeslice_table)
        top_filter.addWidget(self.search_edit, 2)

        select_all_btn = QPushButton("Select all shown")
        select_all_btn.setToolTip("Select every row currently shown (respects group filter and search).")
        select_all_btn.clicked.connect(lambda: self.ts_table.selectAll())
        top_filter.addWidget(select_all_btn)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh)
        top_filter.addWidget(refresh_btn)
        ts_layout.addLayout(top_filter)

        # Azioni raggruppate per categoria; tutte agiscono sulle righe selezionate.
        actions = QHBoxLayout()
        actions.setSpacing(8)

        name_box = QGroupBox("Name")
        name_lay = QHBoxLayout(name_box)
        self.rename_btn = QPushButton("Rename...")
        self.rename_btn.setToolTip("Rename the single selected time-slice.")
        self.rename_btn.clicked.connect(self._rename_selected)
        name_lay.addWidget(self.rename_btn)
        self.batch_rename_btn = QPushButton("Batch rename...")
        self.batch_rename_btn.setToolTip(
            "Rename the selected time-slices: replace part of the name, "
            "add prefix/suffix, or number them. Shows a preview first."
        )
        self.batch_rename_btn.clicked.connect(self._batch_rename_selected)
        name_lay.addWidget(self.batch_rename_btn)
        actions.addWidget(name_box)

        group_box = QGroupBox("Group")
        group_lay = QHBoxLayout(group_box)
        self.group_action_combo = QComboBox()
        self.group_action_combo.setMinimumWidth(160)
        group_lay.addWidget(self.group_action_combo, 1)
        self.assign_btn = QPushButton("Add")
        self.assign_btn.setToolTip("Add the selected time-slices to the chosen group.")
        self.assign_btn.clicked.connect(self._assign_selected_to_group)
        group_lay.addWidget(self.assign_btn)
        self.unassign_btn = QPushButton("Remove")
        self.unassign_btn.setToolTip("Remove the selected time-slices from the chosen group.")
        self.unassign_btn.clicked.connect(self._remove_selected_from_group)
        group_lay.addWidget(self.unassign_btn)
        actions.addWidget(group_box, 1)

        meta_box = QGroupBox("Depth / CRS")
        meta_lay = QHBoxLayout(meta_box)
        depth_btn = QToolButton()
        depth_btn.setText("Depth")
        depth_btn.setPopupMode(QToolButton.InstantPopup)
        depth_menu = QMenu(depth_btn)
        self.set_depth_btn = depth_menu.addAction("Set depth range...")
        self.set_depth_btn.triggered.connect(self._set_depth_range_selected)
        self.sequence_depth_btn = depth_menu.addAction("Auto depth sequence...")
        self.sequence_depth_btn.triggered.connect(self._auto_depth_sequence_selected)
        self.clear_depth_btn = depth_menu.addAction("Clear depth")
        self.clear_depth_btn.triggered.connect(self._clear_depth_range_selected)
        depth_btn.setMenu(depth_menu)
        meta_lay.addWidget(depth_btn)
        crs_btn = QToolButton()
        crs_btn.setText("Assigned CRS")
        crs_btn.setPopupMode(QToolButton.InstantPopup)
        crs_menu = QMenu(crs_btn)
        self.set_crs_btn = crs_menu.addAction("Set assigned CRS...")
        self.set_crs_btn.triggered.connect(self._set_assigned_crs_single)
        self.clear_crs_btn = crs_menu.addAction("Clear assigned CRS")
        self.clear_crs_btn.triggered.connect(self._clear_assigned_crs_selected)
        crs_btn.setMenu(crs_menu)
        meta_lay.addWidget(crs_btn)
        actions.addWidget(meta_box)

        order_box = QGroupBox("Order")
        order_lay = QHBoxLayout(order_box)
        order_tip = (
            "Order inside the group shown in 'Show group'. The first image is on top "
            "in the QGIS layer panel and is the first step of the navigation dial."
        )
        self.move_up_btn = QPushButton("Up")
        self.move_up_btn.setToolTip("Move the selected images up in the group. " + order_tip)
        self.move_up_btn.clicked.connect(lambda: self._move_selected_in_group(-1))
        order_lay.addWidget(self.move_up_btn)
        self.move_down_btn = QPushButton("Down")
        self.move_down_btn.setToolTip("Move the selected images down in the group. " + order_tip)
        self.move_down_btn.clicked.connect(lambda: self._move_selected_in_group(1))
        order_lay.addWidget(self.move_down_btn)
        sort_btn = QToolButton()
        sort_btn.setText("Sort")
        sort_btn.setToolTip(order_tip)
        sort_btn.setPopupMode(QToolButton.InstantPopup)
        sort_menu = QMenu(sort_btn)
        sort_menu.addAction("By depth: shallowest on top").triggered.connect(
            lambda: self._sort_shown_group("depth"))
        sort_menu.addAction("By depth: deepest on top").triggered.connect(
            lambda: self._sort_shown_group("depth_desc"))
        sort_menu.addAction("By name").triggered.connect(lambda: self._sort_shown_group("name"))
        sort_btn.setMenu(sort_menu)
        order_lay.addWidget(sort_btn)
        actions.addWidget(order_box)

        del_box = QGroupBox("Remove")
        del_lay = QHBoxLayout(del_box)
        del_btn = QToolButton()
        del_btn.setText("Remove")
        del_btn.setPopupMode(QToolButton.InstantPopup)
        del_menu = QMenu(del_btn)
        self.remove_btn = del_menu.addAction("From catalog only (keep files)")
        self.remove_btn.triggered.connect(self._remove_selected_catalog_only)
        self.delete_btn = del_menu.addAction("Delete files from disk too...")
        self.delete_btn.triggered.connect(self._delete_selected)
        del_btn.setMenu(del_menu)
        del_lay.addWidget(del_btn)
        actions.addWidget(del_box)
        ts_layout.addLayout(actions)

        self.ts_table = QTableView(self)
        self.ts_model = TimesliceTableModel(self)
        self.ts_table.setModel(self.ts_model)
        self.ts_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.ts_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.ts_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.ts_table.setSortingEnabled(False)
        self.ts_table.verticalHeader().setVisible(True)
        self.ts_table.horizontalHeader().setStretchLastSection(True)
        ts_layout.addWidget(self.ts_table)

        self.ts_count_label = QLabel("")
        ts_layout.addWidget(self.ts_count_label)

        self.ts_manifest_note = QLabel("")
        self.ts_manifest_note.setWordWrap(True)
        self.ts_manifest_note.setStyleSheet("color: #666; font-style: italic;")
        self.ts_manifest_note.setVisible(False)
        ts_layout.addWidget(self.ts_manifest_note)

        ts_selection_model = self.ts_table.selectionModel()
        if ts_selection_model is not None:
            ts_selection_model.selectionChanged.connect(
                lambda _sel, _desel: (self._refresh_manifest_note(), self._update_count_label())
            )

        tabs.addTab(ts_tab, "Time-slices")

        grp_tab = QDialog(self)
        grp_layout = QVBoxLayout(grp_tab)
        grp_layout.setContentsMargins(0, 0, 0, 0)
        grp_layout.setSpacing(6)

        grp_actions = QHBoxLayout()
        create_group_btn = QPushButton("Create Group")
        create_group_btn.clicked.connect(self._create_group)
        grp_actions.addWidget(create_group_btn)

        rename_group_btn = QPushButton("Rename Group")
        rename_group_btn.clicked.connect(self._rename_group)
        grp_actions.addWidget(rename_group_btn)

        delete_group_btn = QPushButton("Delete Group")
        delete_group_btn.clicked.connect(self._delete_group)
        grp_actions.addWidget(delete_group_btn)

        toggle_lock_btn = QPushButton("Toggle Lock")
        toggle_lock_btn.clicked.connect(self._toggle_group_lock)
        grp_actions.addWidget(toggle_lock_btn)

        grp_refresh_btn = QPushButton("Refresh")
        grp_refresh_btn.clicked.connect(self._refresh)
        grp_actions.addWidget(grp_refresh_btn)

        verify_unassigned_btn = QPushButton("Verify Unassigned")
        verify_unassigned_btn.clicked.connect(self._verify_unassigned_clicked)
        grp_actions.addWidget(verify_unassigned_btn)

        detect_groups_btn = QPushButton("Detect Groups")
        detect_groups_btn.setToolTip(
            "Raggruppa le time-slice sciolte in timeslices_2d per inizio del nome "
            "e propone di creare una cartella e un gruppo per ciascun gruppo trovato."
        )
        detect_groups_btn.clicked.connect(self._detect_groups_from_names)
        grp_actions.addWidget(detect_groups_btn)
        rescan_btn = QPushButton("Rescan Folders")
        rescan_btn.setToolTip(
            "Ricollega ai gruppi le time-slice che stanno nelle cartelle "
            "timeslices_2d/<gruppo> ma non sono nel catalogo."
        )
        rescan_btn.clicked.connect(self._rescan_folders_clicked)
        grp_actions.addWidget(rescan_btn)
        grp_order = QHBoxLayout()
        grp_order.addWidget(QLabel("Group order (top of the QGIS panel first):"))
        up_g = QPushButton("Move Up")
        up_g.clicked.connect(lambda: self._move_selected_group(-1))
        grp_order.addWidget(up_g)
        down_g = QPushButton("Move Down")
        down_g.clicked.connect(lambda: self._move_selected_group(1))
        grp_order.addWidget(down_g)
        sort_all_btn = QPushButton("Sort All Groups' Images By Depth")
        sort_all_btn.setToolTip("Shallowest image first (on top) in every group.")
        sort_all_btn.clicked.connect(self._sort_all_groups_by_depth)
        grp_order.addWidget(sort_all_btn)
        grp_order.addStretch(1)
        grp_layout.addLayout(grp_actions)
        grp_layout.addLayout(grp_order)

        self.group_table = QTableView(self)
        self.group_model = GroupTableModel(self)
        self.group_table.setModel(self.group_model)
        self.group_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.group_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.group_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.group_table.setSortingEnabled(False)
        self.group_table.verticalHeader().setVisible(True)
        self.group_table.horizontalHeader().setStretchLastSection(True)
        group_selection_model = self.group_table.selectionModel()
        if group_selection_model is not None:
            group_selection_model.selectionChanged.connect(
                lambda _selected, _deselected: self._sync_filter_from_group_selection()
            )
        grp_layout.addWidget(self.group_table)

        tabs.addTab(grp_tab, "Groups")

        footer = QHBoxLayout()
        if callable(self.open_catalog_editor_callback):
            open_catalog_btn = QPushButton("Open 3D/Radargram Editor")
            open_catalog_btn.clicked.connect(self._open_catalog_editor)
            footer.addWidget(open_catalog_btn)
        transfer_btn = QPushButton("Transfer project...")
        transfer_btn.setToolTip(
            "Copia (o sposta) l'intero progetto in un'altra cartella e aggiorna "
            "i metadati (catalogo, project.json, sliceset, progetti QGIS) "
            "perche' il nuovo percorso venga riconosciuto alla riapertura."
        )
        transfer_btn.clicked.connect(self._transfer_project)
        footer.addWidget(transfer_btn)
        footer.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        footer.addWidget(close_btn)
        root.addLayout(footer)

    def _refresh(self):
        self._catalog = load_catalog(self.project_root)
        self._refresh_group_combos()
        self._refresh_group_table()
        self._refresh_timeslice_table()

    def _save_and_refresh(self):
        save_catalog(self.project_root, self._catalog)
        self._refresh()
        if callable(self.on_updated):
            try:
                self.on_updated()
            except Exception:
                pass

    def _refresh_group_combos(self):
        groups = list(self._catalog.get("raster_groups", []))
        prev_filter = self.filter_group_combo.currentData()
        prev_action = self.group_action_combo.currentData()

        self.filter_group_combo.blockSignals(True)
        self.group_action_combo.blockSignals(True)

        self.filter_group_combo.clear()
        self.filter_group_combo.addItem("All", None)

        self.group_action_combo.clear()
        for g in groups:
            gid = g.get("id")
            name = g.get("name") or gid or "Group"
            # Hide empty system groups like grp_imported and grp_no_crs
            is_system = bool(g.get("system", False))
            times_count = len(g.get("timeslice_ids", []))
            if is_system and times_count == 0:
                # skip showing empty system groups
                continue
            label = f"{name} [{gid}]"
            self.filter_group_combo.addItem(label, gid)
            self.group_action_combo.addItem(label, gid)

        if prev_filter is not None:
            idx = self.filter_group_combo.findData(prev_filter)
            if idx >= 0:
                self.filter_group_combo.setCurrentIndex(idx)
        if prev_action is not None:
            idx = self.group_action_combo.findData(prev_action)
            if idx >= 0:
                self.group_action_combo.setCurrentIndex(idx)

        self.filter_group_combo.blockSignals(False)
        self.group_action_combo.blockSignals(False)

    def _timeslice_membership_map(self):
        membership = {}
        for g in self._catalog.get("raster_groups", []):
            gname = g.get("name") or g.get("id") or "Group"
            for tid in g.get("timeslice_ids", []):
                membership.setdefault(tid, []).append(gname)
        return membership

    def _format_depth_range(self, rec):
        depth_from = rec.get("depth_from")
        depth_to = rec.get("depth_to")
        unit = (rec.get("unit") or "m").strip() or "m"
        if depth_from is None and depth_to is None:
            return ""

        def _fmt(val):
            try:
                txt = f"{float(val):.3f}"
                txt = txt.rstrip("0").rstrip(".")
                return txt
            except Exception:
                return str(val)

        if depth_from is None:
            return f"to {_fmt(depth_to)} {unit}"
        if depth_to is None:
            return f"from {_fmt(depth_from)} {unit}"
        return f"{_fmt(depth_from)} - {_fmt(depth_to)} {unit}"

    def _parse_depth_value(self, text):
        txt = (text or "").strip()
        if not txt:
            return None
        try:
            return float(txt.replace(",", "."))
        except Exception:
            return None

    def _refresh_timeslice_table(self):
        timeslices = list(self._catalog.get("timeslices", []))
        filter_gid = self.filter_group_combo.currentData()
        search_text = (self.search_edit.text() or "").strip().lower()
        selected_tids = set(self._selected_timeslice_ids())
        allowed = None
        group = None
        if filter_gid:
            group = next((g for g in self._catalog.get("raster_groups", []) if g.get("id") == filter_gid), None)
            allowed = set(group.get("timeslice_ids", []) if group else [])

        membership = self._timeslice_membership_map()
        rows = []
        for rec in timeslices:
            tid = rec.get("id")
            if allowed is not None and tid not in allowed:
                continue
            depth_txt = self._format_depth_range(rec)
            if search_text:
                name = rec.get("normalized_name") or rec.get("name") or ""
                pth = rec.get("project_path") or ""
                grp_txt = ", ".join(sorted(membership.get(tid, [])))
                crs_txt = rec.get("crs") or ""
                assigned_txt = rec.get("assigned_crs") or ""
                hay = " | ".join(
                    [
                        str(tid or ""),
                        str(name),
                        str(depth_txt),
                        str(grp_txt),
                        str(crs_txt),
                        str(assigned_txt),
                        str(pth),
                    ]
                ).lower()
                if search_text not in hay:
                    continue
            rows.append(rec)

        model_rows = []
        for rec in rows:
            tid = rec.get("id") or ""
            name = rec.get("normalized_name") or rec.get("name") or ""
            depth_txt = self._format_depth_range(rec)
            pth = rec.get("project_path") or ""
            exists = "Yes" if pth and os.path.exists(pth) else "No"
            groups_txt = ", ".join(sorted(membership.get(tid, [])))
            crs_txt = rec.get("crs") or ""
            assigned_txt = rec.get("assigned_crs") or ""

            model_rows.append(
                {
                    "id": str(tid),
                    "name": str(name),
                    "depth_range": str(depth_txt),
                    "groups": str(groups_txt),
                    "crs": str(crs_txt),
                    "assigned_crs": str(assigned_txt),
                    "project_path": str(pth),
                    "exists": str(exists),
                }
            )

        if filter_gid and group is not None:
            pos = {t: i for i, t in enumerate(group.get("timeslice_ids", []))}
            model_rows.sort(key=lambda r: pos.get(r.get("id"), len(pos)))
        else:
            # 'All': ordine dei gruppi (come nel catalogo), poi ordine nel gruppo
            pos = {}
            for g in self._catalog.get("raster_groups", []):
                for t in g.get("timeslice_ids", []):
                    pos.setdefault(t, len(pos))
            model_rows.sort(key=lambda r: (pos.get(r.get("id"), len(pos)),
                                           (r.get("name") or r.get("id") or "").lower()))
        self.ts_model.set_rows(model_rows)
        self.ts_table.resizeColumnsToContents()

        selection_model = self.ts_table.selectionModel()
        if selection_model is not None:
            selection_model.clearSelection()
            for row, row_data in enumerate(model_rows):
                if row_data.get("id") not in selected_tids:
                    continue
                idx = self.ts_model.index(row, 0)
                if idx.isValid():
                    selection_model.select(
                        idx,
                        QItemSelectionModel.Select | QItemSelectionModel.Rows,
                    )

        self._refresh_manifest_note()
        self._update_count_label()

    def _update_count_label(self):
        label = getattr(self, "ts_count_label", None)
        if label is not None:
            label.setText(
                f"{len(self._selected_timeslice_ids())} selected of {self.ts_model.rowCount()} shown"
            )

    def _refresh_group_table(self):
        groups = list(self._catalog.get("raster_groups", []))
        selected_gid = self._selected_group_id()
        model_rows = []
        for g in groups:
            gid = g.get("id") or ""
            name = g.get("name") or ""
            count = len(g.get("timeslice_ids", []))
            system_flag = "Yes" if bool(g.get("system", False)) else ""
            lock_flag = "Locked" if g.get("locked", False) else "Unlocked"
            model_rows.append(
                {
                    "id": str(gid),
                    "name": str(name),
                    "timeslice_count": count,
                    "system": system_flag,
                    "lock_status": lock_flag,
                }
            )

        self.group_model.set_rows(model_rows)  # ordine del catalogo
        self.group_table.resizeColumnsToContents()

        if selected_gid:
            for row_idx, row_data in enumerate(model_rows):
                if row_data.get("id") == selected_gid:
                    idx = self.group_model.index(row_idx, 0)
                    if idx.isValid():
                        self.group_table.selectRow(row_idx)
                    break

    def _selected_timeslice_ids(self):
        if self.ts_table is None or self.ts_table.selectionModel() is None:
            return []
        rows = sorted({idx.row() for idx in self.ts_table.selectionModel().selectedRows(0)})
        tids = []
        for row in rows:
            payload = self.ts_model.row_payload(row)
            tid = payload.get("id") if isinstance(payload, dict) else None
            if tid:
                tids.append(tid)
        return tids

    def _selected_group_id(self):
        if self.group_table is None or self.group_table.selectionModel() is None:
            return None
        rows = self.group_table.selectionModel().selectedRows(0)
        if not rows:
            return None
        row = rows[0].row()
        payload = self.group_model.row_payload(row)
        if not isinstance(payload, dict):
            return None
        return payload.get("id")

    def _timeslice_record_by_id(self, tid):
        return next((r for r in self._catalog.get("timeslices", []) if r.get("id") == tid), None)

    @staticmethod
    def _record_dir_has_sliceset(rec):
        """True se il record vive in una cartella con ``sliceset.json``
        (la profondita' e' quindi autorevole e non va sovrascritta a mano)."""
        if not isinstance(rec, dict):
            return False
        p = str(rec.get("project_path") or "").strip()
        if not p:
            return False
        d = os.path.dirname(os.path.abspath(p))
        return os.path.isfile(os.path.join(d, "sliceset.json"))

    def _selected_records_with_sliceset(self):
        out = []
        for tid in self._selected_timeslice_ids():
            rec = self._timeslice_record_by_id(tid)
            if rec is not None and self._record_dir_has_sliceset(rec):
                out.append(rec.get("normalized_name") or rec.get("name") or rec.get("id"))
        return out

    def _refresh_manifest_note(self):
        note = getattr(self, "ts_manifest_note", None)
        if note is None:
            return
        names = self._selected_records_with_sliceset()
        if names:
            note.setText(
                "Profondita' da sliceset.json (manifesto autorevole) per "
                f"{len(names)} time-slice selezionate — 'Set Depth Range' e "
                "'Auto Depth Sequence' sono disabilitati per questa selezione."
            )
            note.setVisible(True)
        else:
            note.setVisible(False)
        gated = bool(names)
        for btn in (getattr(self, "set_depth_btn", None),
                    getattr(self, "sequence_depth_btn", None)):
            if btn is not None:
                btn.setEnabled(not gated)

    def _depth_edit_blocked_by_manifest(self, title):
        names = self._selected_records_with_sliceset()
        if not names:
            return False
        QMessageBox.information(
            self, title,
            "La profondita' di queste time-slice proviene da sliceset.json "
            "(manifesto autorevole della generazione) e non va modificata a "
            "mano.\n\nRigenera il set per cambiarla, oppure usa 'Clear Depth' "
            "e reimporta se devi forzare valori diversi.",
        )
        return True

    def _group_record_by_id(self, gid):
        return next((g for g in self._catalog.get("raster_groups", []) if g.get("id") == gid), None)

    def _ensure_imported_group(self):
        imported = self._group_record_by_id("grp_imported")
        if imported is None:
            imported = {
                "id": "grp_imported",
                "name": "Imported",
                "radargram_ids": [],
                "timeslice_ids": [],
            }
            self._catalog.setdefault("raster_groups", []).append(imported)
        imported.setdefault("timeslice_ids", [])
        return imported

    def _worldfile_suffixes(self, path):
        ext = os.path.splitext(path)[1].lower()
        by_ext = {
            ".tif": [".tfw", ".tifw", ".wld"],
            ".tiff": [".tfw", ".tifw", ".wld"],
            ".png": [".pgw", ".wld"],
            ".jpg": [".jgw", ".wld"],
            ".jpeg": [".jgw", ".wld"],
            ".bmp": [".bpw", ".wld"],
            ".img": [".wld"],
            ".asc": [".wld"],
        }
        suffixes = list(by_ext.get(ext, []))
        if ".wld" not in suffixes:
            suffixes.append(".wld")
        return suffixes

    def _existing_sidecars(self, path):
        sidecars = []
        base, _ = os.path.splitext(path)
        for suffix in self._worldfile_suffixes(path):
            wf = base + suffix
            if os.path.exists(wf):
                sidecars.append(wf)
        aux = path + ".aux.xml"
        if os.path.exists(aux):
            sidecars.append(aux)
        return sidecars

    def _is_within_root(self, path, root):
        try:
            p_abs = os.path.normcase(os.path.abspath(path))
            r_abs = os.path.normcase(os.path.abspath(root))
        except Exception:
            return False
        return bool(p_abs and r_abs and (p_abs == r_abs or p_abs.startswith(r_abs + os.sep)))

    def _group_folder_candidate(self, group_name):
        root = os.path.abspath(os.path.join(self.project_root, "timeslices_2d"))
        name = str(group_name or "").strip()
        if not name:
            return None
        candidate = os.path.abspath(os.path.join(root, group_folder_name(name)))
        if not self._is_within_root(candidate, root):
            return None
        if os.path.normcase(candidate) == os.path.normcase(root):
            return None
        return candidate

    def _rename_file_with_sidecars(self, old_path, new_path):
        warnings = []
        old_path_abs = os.path.abspath(old_path)
        new_path_abs = os.path.abspath(new_path)
        sidecars = self._existing_sidecars(old_path_abs)

        if old_path_abs != new_path_abs:
            os.rename(old_path_abs, new_path_abs)

        old_base, _ = os.path.splitext(old_path_abs)
        new_base, _ = os.path.splitext(new_path_abs)
        for old_side in sidecars:
            try:
                if old_side.endswith(".aux.xml"):
                    new_side = new_path_abs + ".aux.xml"
                else:
                    suffix = old_side[len(old_base):]
                    new_side = new_base + suffix
                if os.path.abspath(old_side) == os.path.abspath(new_side):
                    continue
                if os.path.exists(new_side):
                    warnings.append(f"Skipped sidecar rename (target exists): {new_side}")
                    continue
                os.rename(old_side, new_side)
            except Exception as e:
                warnings.append(f"Sidecar rename failed for {old_side}: {e}")
        return warnings

    @staticmethod
    def _detach_layers_for_rename(paths):
        """Toglie dal progetto QGIS i layer dei file in ``paths`` ricordando
        gruppo padre, posizione, stile, nome e visibilita'."""
        saved = []
        try:
            from qgis.core import QgsProject
            from qgis.PyQt.QtXml import QDomDocument
            wanted = {os.path.normcase(os.path.abspath(p)) for p in paths if p}
            project = QgsProject.instance()
            tree = project.layerTreeRoot()
            for layer in list(project.mapLayers().values()):
                src = os.path.normcase(os.path.abspath(str(layer.source()).split("|")[0]))
                if src not in wanted:
                    continue
                node = tree.findLayer(layer.id())
                parent_names, index, visible = [], 0, True
                if node is not None:
                    visible = bool(node.itemVisibilityChecked())
                    parent = node.parent()
                    index = parent.children().index(node) if parent is not None else 0
                    while parent is not None and parent.parent() is not None:
                        parent_names.insert(0, parent.name())
                        parent = parent.parent()
                doc = QDomDocument("qgis")
                layer.exportNamedStyle(doc)
                saved.append({
                    "path": str(layer.source()).split("|")[0],
                    "name": layer.name(),
                    "style": doc.toString(),
                    "visible": visible,
                    "parents": parent_names,
                    "index": index,
                })
                project.removeMapLayer(layer.id())
        except Exception:
            pass
        saved.sort(key=lambda s: (s["parents"], s["index"]))
        return saved

    @staticmethod
    def _reattach_layers_after_rename(saved, moves):
        """Ricrea i layer staccati: dal nuovo percorso se il file e' stato
        rinominato, dal vecchio altrimenti; stessa posizione nell'albero."""
        if not saved:
            return
        try:
            from qgis.core import QgsLayerTreeGroup, QgsProject, QgsRasterLayer
            from qgis.PyQt.QtXml import QDomDocument
            norm = lambda p: os.path.normcase(os.path.abspath(p))
            new_by_old = {norm(o): n for o, n in moves}
            project = QgsProject.instance()
            root = project.layerTreeRoot()
            for item in saved:
                old = item["path"]
                new = new_by_old.get(norm(old), old)
                name = item["name"]
                if new != old and name in (os.path.basename(old), os.path.splitext(os.path.basename(old))[0]):
                    name = os.path.splitext(os.path.basename(new))[0]
                lyr = QgsRasterLayer(new, name)
                if not lyr.isValid():
                    continue
                try:
                    doc = QDomDocument()
                    if doc.setContent(item["style"]):
                        lyr.importNamedStyle(doc)
                except Exception:
                    pass
                target = root
                for gname in item["parents"]:
                    nxt = next((g for g in target.children()
                                if isinstance(g, QgsLayerTreeGroup) and g.name() == gname), None)
                    target = nxt if nxt is not None else target.addGroup(gname)
                project.addMapLayer(lyr, False)
                node = target.insertLayer(min(item["index"], len(target.children())), lyr)
                if node is not None:
                    node.setItemVisibilityChecked(item["visible"])
        except Exception:
            pass

    def _rename_selected(self):
        tids = self._selected_timeslice_ids()
        if len(tids) != 1:
            QMessageBox.information(self, "Time-slice Manager", "Select exactly one time-slice.")
            return
        rec = self._timeslice_record_by_id(tids[0])
        if rec is None:
            QMessageBox.warning(self, "Time-slice Manager", "Selected record not found in catalog.")
            return
        old_path = rec.get("project_path") or ""
        if not old_path or not os.path.exists(old_path):
            QMessageBox.warning(self, "Time-slice Manager", "Selected file does not exist on disk.")
            return

        old_name = os.path.basename(old_path)
        new_name, ok = QInputDialog.getText(self, "Rename time-slice", "New file name:", text=old_name)
        if not ok or not new_name.strip():
            return

        new_name = sanitize_filename(new_name.strip())
        new_path = os.path.join(os.path.dirname(old_path), new_name)
        if os.path.exists(new_path) and os.path.abspath(new_path) != os.path.abspath(old_path):
            QMessageBox.warning(self, "Time-slice Manager", "A file with this name already exists.")
            return

        saved_layers = self._detach_layers_for_rename([old_path])
        moves = []
        try:
            warnings = self._rename_file_with_sidecars(old_path, new_path)
            moves.append((old_path, new_path))
            rec["project_path"] = new_path
            rec["normalized_name"] = new_name
            rec["name"] = os.path.splitext(new_name)[0]
            self._reattach_layers_after_rename(saved_layers, moves)
            self._save_and_refresh()
            if warnings:
                QMessageBox.warning(self, "Rename completed with warnings", "\n".join(warnings[:10]))
        except Exception as e:
            self._reattach_layers_after_rename(saved_layers, moves)
            QMessageBox.critical(self, "Time-slice Manager", f"Rename failed:\n{e}")

    def _batch_rename_selected(self):
        tids = self._selected_timeslice_ids()
        if not tids:
            QMessageBox.information(self, "Time-slice Manager", "Select one or more time-slices.")
            return

        records = [r for r in (self._timeslice_record_by_id(t) for t in tids) if r is not None]
        records = [r for r in records if r.get("project_path") and os.path.exists(r["project_path"])]
        if not records:
            QMessageBox.warning(self, "Time-slice Manager", "No selected files are available on disk.")
            return
        records.sort(key=lambda r: (r.get("normalized_name") or r.get("name") or r.get("id") or "").lower())

        dlg = BatchRenameDialog([(r.get("id"), r["project_path"]) for r in records], parent=self)
        if dlg.exec_() != QDialog.Accepted or not dlg.plan:
            return
        by_id = {r.get("id"): r for r in records}
        plan = [(by_id[k], old, new, name) for k, old, new, name in dlg.plan if k in by_id]

        progress = QProgressDialog("Batch renaming time-slices...", "Cancel", 0, len(plan), self)
        progress.setWindowTitle("Batch Rename")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        warnings = []
        renamed = 0
        moves = []
        # Su Windows un raster aperto in QGIS non si puo' rinominare: si staccano
        # i layer, si rinomina, e si ricreano (stessa posizione, stile, visibilita').
        saved_layers = self._detach_layers_for_rename([p[1] for p in plan])
        for idx, (rec, old_path, new_path, new_name) in enumerate(plan, start=1):
            if progress.wasCanceled():
                break
            progress.setLabelText(f"Renaming {idx}/{len(plan)}: {os.path.basename(old_path)}")
            QApplication.processEvents()
            try:
                warnings.extend(self._rename_file_with_sidecars(old_path, new_path))
                moves.append((old_path, new_path))
                rec["project_path"] = new_path
                rec["normalized_name"] = new_name
                rec["name"] = os.path.splitext(new_name)[0]
                renamed += 1
            except Exception as e:
                warnings.append(f"Rename failed for {os.path.basename(old_path)}: {e}")
            progress.setValue(idx)
            QApplication.processEvents()

        progress.close()
        self._reattach_layers_after_rename(saved_layers, moves)
        self._save_and_refresh()

        msg = f"Batch rename completed: {renamed}/{len(plan)} files."
        if warnings:
            msg += "\n\nWarnings:\n" + "\n".join(warnings[:12])
            extra = len(warnings) - min(len(warnings), 12)
            if extra > 0:
                msg += f"\n... and {extra} more."
            QMessageBox.warning(self, "Batch rename", msg)
        else:
            QMessageBox.information(self, "Batch rename", msg)

    def _remove_selected_catalog_only(self):
        tids = self._selected_timeslice_ids()
        if not tids:
            QMessageBox.information(self, "Time-slice Manager", "Select one or more time-slices.")
            return
        answer = QMessageBox.question(
            self,
            "Remove records",
            (
                f"Remove {len(tids)} record(s) from catalog?\n"
                "Files on disk will NOT be deleted."
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        tids_set = set(tids)
        self._catalog["timeslices"] = [r for r in self._catalog.get("timeslices", []) if r.get("id") not in tids_set]
        for g in self._catalog.get("raster_groups", []):
            g["timeslice_ids"] = [tid for tid in g.get("timeslice_ids", []) if tid not in tids_set]
        self._catalog["links"] = [lk for lk in self._catalog.get("links", []) if lk.get("timeslice_id") not in tids_set]
        self._save_and_refresh()

    def _delete_selected(self):
        tids = self._selected_timeslice_ids()
        if not tids:
            QMessageBox.information(self, "Time-slice Manager", "Select one or more time-slices.")
            return

        answer = QMessageBox.question(
            self,
            "Delete files",
            (
                f"Delete {len(tids)} file(s) from disk and remove them from catalog?\n"
                "Related worldfiles/aux files will be deleted when found."
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        records = [self._timeslice_record_by_id(tid) for tid in tids]
        records = [r for r in records if r is not None]
        if not records:
            return

        progress = QProgressDialog("Deleting time-slice files...", "Cancel", 0, len(records), self)
        progress.setWindowTitle("Delete Time-slices")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        # Un raster aperto in QGIS blocca il file su Windows: tolgo i layer dal
        # progetto prima di cancellare (quelli non cancellati tornano sotto).
        saved_layers = self._detach_layers_for_rename([r.get("project_path") for r in records])
        deleted_ids = set()
        warnings = []
        for idx, rec in enumerate(records, start=1):
            if progress.wasCanceled():
                break
            pth = rec.get("project_path") or ""
            progress.setLabelText(f"Deleting {idx}/{len(records)}: {os.path.basename(pth)}")
            QApplication.processEvents()
            try:
                if pth and os.path.exists(pth):
                    for side in self._existing_sidecars(pth):
                        try:
                            os.remove(side)
                        except Exception as e:
                            warnings.append(f"Sidecar delete failed: {os.path.basename(side)} ({e})")
                    os.remove(pth)
                deleted_ids.add(rec.get("id"))
            except Exception as e:
                warnings.append(f"Delete failed for {os.path.basename(pth)}: {e}")
            progress.setValue(idx)
            QApplication.processEvents()

        progress.close()
        self._reattach_layers_after_rename(
            [s for s in saved_layers if os.path.exists(s["path"])], [])

        if deleted_ids:
            self._catalog["timeslices"] = [
                r for r in self._catalog.get("timeslices", []) if r.get("id") not in deleted_ids
            ]
            for g in self._catalog.get("raster_groups", []):
                g["timeslice_ids"] = [tid for tid in g.get("timeslice_ids", []) if tid not in deleted_ids]
            self._catalog["links"] = [
                lk for lk in self._catalog.get("links", []) if lk.get("timeslice_id") not in deleted_ids
            ]
            self._save_and_refresh()

        if warnings:
            preview = "\n".join(warnings[:12])
            extra = len(warnings) - min(len(warnings), 12)
            if extra > 0:
                preview += f"\n... and {extra} more."
            QMessageBox.warning(self, "Delete warnings", preview)

    def _set_assigned_crs_single(self):
        tids = self._selected_timeslice_ids()
        if len(tids) != 1:
            QMessageBox.information(self, "Set Assigned CRS", "Select exactly one time-slice.")
            return
        rec = self._timeslice_record_by_id(tids[0])
        if rec is None:
            QMessageBox.warning(self, "Set Assigned CRS", "Selected record not found.")
            return
        current = (rec.get("assigned_crs") or rec.get("crs") or "EPSG:32633").strip()
        text, ok = QInputDialog.getText(
            self,
            "Set Assigned CRS",
            "Enter EPSG code (32633) or AUTHID (EPSG:32633):",
            text=current,
        )
        if not ok or not text.strip():
            return

        raw = text.strip().upper()
        authid = raw if raw.startswith("EPSG:") else f"EPSG:{raw}"
        crs = QgsCoordinateReferenceSystem(authid)
        if not crs.isValid():
            QMessageBox.warning(self, "Set Assigned CRS", "Invalid CRS.")
            return

        rec["assigned_crs"] = crs.authid()
        self._save_and_refresh()

    def _clear_assigned_crs_selected(self):
        tids = self._selected_timeslice_ids()
        if not tids:
            QMessageBox.information(self, "Clear Assigned CRS", "Select one or more time-slices.")
            return
        changed = 0
        for tid in tids:
            rec = self._timeslice_record_by_id(tid)
            if rec is None:
                continue
            if rec.get("assigned_crs"):
                rec.pop("assigned_crs", None)
                changed += 1
        if changed <= 0:
            return
        self._save_and_refresh()

    def _set_depth_range_selected(self):
        tids = self._selected_timeslice_ids()
        if not tids:
            QMessageBox.information(self, "Set Depth Range", "Select one or more time-slices.")
            return
        if self._depth_edit_blocked_by_manifest("Set Depth Range"):
            return

        from_txt, ok = QInputDialog.getText(
            self,
            "Set Depth Range",
            "Depth from:",
            text="0.0",
        )
        if not ok:
            return
        to_txt, ok = QInputDialog.getText(
            self,
            "Set Depth Range",
            "Depth to:",
            text="0.1",
        )
        if not ok:
            return
        unit_txt, ok = QInputDialog.getText(
            self,
            "Set Depth Range",
            "Unit:",
            text="m",
        )
        if not ok:
            return

        depth_from = self._parse_depth_value(from_txt)
        depth_to = self._parse_depth_value(to_txt)
        if depth_from is None or depth_to is None:
            QMessageBox.warning(self, "Set Depth Range", "Depth values must be numeric.")
            return
        if depth_to < depth_from:
            depth_from, depth_to = depth_to, depth_from
        unit = (unit_txt or "").strip() or "m"

        changed = 0
        for tid in tids:
            rec = self._timeslice_record_by_id(tid)
            if rec is None:
                continue
            rec["depth_from"] = depth_from
            rec["depth_to"] = depth_to
            rec["unit"] = unit
            changed += 1
        if changed > 0:
            self._save_and_refresh()

    def _clear_depth_range_selected(self):
        tids = self._selected_timeslice_ids()
        if not tids:
            QMessageBox.information(self, "Clear Depth", "Select one or more time-slices.")
            return
        changed = 0
        for tid in tids:
            rec = self._timeslice_record_by_id(tid)
            if rec is None:
                continue
            if rec.get("depth_from") is not None or rec.get("depth_to") is not None:
                rec["depth_from"] = None
                rec["depth_to"] = None
                changed += 1
        if changed > 0:
            self._save_and_refresh()

    def _auto_depth_sequence_selected(self):
        tids = self._selected_timeslice_ids()
        if len(tids) < 1:
            QMessageBox.information(self, "Auto Depth Sequence", "Select one or more time-slices.")
            return
        if self._depth_edit_blocked_by_manifest("Auto Depth Sequence"):
            return

        start_txt, ok = QInputDialog.getText(self, "Auto Depth Sequence", "Start depth:", text="0.0")
        if not ok:
            return
        thickness_txt, ok = QInputDialog.getText(self, "Auto Depth Sequence", "Slice thickness:", text="0.1")
        if not ok:
            return
        unit_txt, ok = QInputDialog.getText(self, "Auto Depth Sequence", "Unit:", text="m")
        if not ok:
            return

        start_depth = self._parse_depth_value(start_txt)
        thickness = self._parse_depth_value(thickness_txt)
        if start_depth is None or thickness is None or thickness <= 0:
            QMessageBox.warning(self, "Auto Depth Sequence", "Start and thickness must be valid numbers.")
            return
        unit = (unit_txt or "").strip() or "m"

        selected = []
        for tid in tids:
            rec = self._timeslice_record_by_id(tid)
            if rec is not None:
                selected.append(rec)
        selected.sort(key=lambda r: (r.get("normalized_name") or r.get("name") or r.get("id") or "").lower())

        for idx, rec in enumerate(selected):
            depth_from = start_depth + idx * thickness
            depth_to = depth_from + thickness
            rec["depth_from"] = depth_from
            rec["depth_to"] = depth_to
            rec["unit"] = unit
        if selected:
            self._save_and_refresh()

    def _assign_selected_to_group(self):
        tids = self._selected_timeslice_ids()
        if not tids:
            QMessageBox.information(self, "Assign To Group", "Select one or more time-slices.")
            return
        gid = self.group_action_combo.currentData()
        if not gid:
            QMessageBox.warning(self, "Assign To Group", "Select a target group.")
            return
        target = self._group_record_by_id(gid)
        if target is None:
            QMessageBox.warning(self, "Assign To Group", "Target group not found.")
            return

        merged = list(dict.fromkeys((target.get("timeslice_ids") or []) + tids))
        target["timeslice_ids"] = merged
        if gid != "grp_imported":
            imported = self._group_record_by_id("grp_imported")
            if imported is not None:
                imported["timeslice_ids"] = [tid for tid in imported.get("timeslice_ids", []) if tid not in set(tids)]
        self._save_and_refresh()

    def _remove_selected_from_group(self):
        tids = self._selected_timeslice_ids()
        if not tids:
            QMessageBox.information(self, "Remove From Group", "Select one or more time-slices.")
            return
        gid = self.group_action_combo.currentData()
        if not gid:
            QMessageBox.warning(self, "Remove From Group", "Select a source group.")
            return
        target = self._group_record_by_id(gid)
        if target is None:
            QMessageBox.warning(self, "Remove From Group", "Group not found.")
            return

        tids_set = set(tids)
        target["timeslice_ids"] = [tid for tid in target.get("timeslice_ids", []) if tid not in tids_set]

        imported = self._ensure_imported_group()
        if gid != "grp_imported":
            membership = {}
            for g in self._catalog.get("raster_groups", []):
                for tid in g.get("timeslice_ids", []):
                    membership[tid] = membership.get(tid, 0) + 1
            for tid in tids:
                if membership.get(tid, 0) <= 0 and tid not in imported.get("timeslice_ids", []):
                    imported["timeslice_ids"].append(tid)
        self._save_and_refresh()

    def _create_group(self):
        name, ok = QInputDialog.getText(self, "Create Group", "Group name:")
        if not ok or not name.strip():
            return
        group, created = create_raster_group(self.project_root, name.strip())
        self._catalog = load_catalog(self.project_root)
        self._refresh()
        if not created:
            QMessageBox.information(self, "Create Group", f"Group '{group.get('name')}' already exists.")

    # ---- ordine gruppi / immagini
    def _shown_group(self):
        """Gruppo su cui agisce l'ordine: quello scelto in 'Show group'; con
        'All', l'unico gruppo che contiene tutte le righe selezionate."""
        gid = self.filter_group_combo.currentData()
        if gid:
            return self._group_record_by_id(gid)
        tids = set(self._selected_timeslice_ids())
        if not tids:
            return None
        owners = [g for g in self._catalog.get("raster_groups", [])
                  if tids <= set(g.get("timeslice_ids", []))]
        non_system = [g for g in owners if g.get("id") != "grp_imported" and not g.get("system")]
        owners = non_system or owners
        return owners[0] if len(owners) == 1 else None

    def _apply_slice_order(self, group):
        recs = {r.get("id"): r for r in self._catalog.get("timeslices", [])}
        paths = [recs[t].get("project_path") for t in group.get("timeslice_ids", []) if t in recs]
        _order.apply_slice_order_to_qgis(group.get("name"), paths)

    def _move_selected_in_group(self, delta):
        group = self._shown_group()
        if group is None:
            QMessageBox.information(
                self, "Order", "Choose a group in 'Show group' first: the order is per group.")
            return
        tids = self._selected_timeslice_ids()
        if not tids:
            QMessageBox.information(self, "Order", "Select one or more time-slices.")
            return
        group["timeslice_ids"] = _order.move_items(group.get("timeslice_ids", []), tids, delta)
        self._apply_slice_order(group)
        self._save_and_refresh()

    def _sort_shown_group(self, mode):
        group = self._shown_group()
        if group is None:
            QMessageBox.information(
                self, "Order",
                "Choose a group in 'Show group' (or select rows of one group) first: "
                "the order is per group.")
            return
        recs ={r.get("id"): r for r in self._catalog.get("timeslices", [])}
        group["timeslice_ids"] = _order.sort_ids(group.get("timeslice_ids", []), recs, mode)
        self._apply_slice_order(group)
        self._save_and_refresh()

    def _sort_all_groups_by_depth(self):
        recs = {r.get("id"): r for r in self._catalog.get("timeslices", [])}
        n = 0
        for group in self._catalog.get("raster_groups", []):
            if not group.get("timeslice_ids"):
                continue
            group["timeslice_ids"] = _order.sort_ids(group["timeslice_ids"], recs, "depth")
            self._apply_slice_order(group)
            n += 1
        self._save_and_refresh()
        QMessageBox.information(self, "Order", f"Images sorted by depth (shallowest on top) in {n} group(s).")

    def _move_selected_group(self, delta):
        gid = self._selected_group_id()
        if not gid:
            QMessageBox.information(self, "Order", "Select one group.")
            return
        groups = self._catalog.get("raster_groups", [])
        ids = _order.move_items([g.get("id") for g in groups], [gid], delta)
        by_id = {g.get("id"): g for g in groups}
        self._catalog["raster_groups"] = [by_id[i] for i in ids]
        _order.apply_group_order_to_qgis([g.get("name") for g in self._catalog["raster_groups"]])
        self._save_and_refresh()

    def _rescan_folders_clicked(self):
        res = rescan_group_folders(self.project_root)
        self._refresh()
        if callable(self.on_updated):
            try:
                self.on_updated()
            except Exception:
                pass
        QMessageBox.information(
            self, "Rescan Folders",
            f"Time-slice ricollegate: {res['registered']} in {res['groups']} gruppo/i."
            if res["registered"] else "Nessuna time-slice da ricollegare: le cartelle sono gia' allineate al catalogo.",
        )

    def _detect_groups_from_names(self):
        """Cerca nel catalogo le time-slice sciolte in timeslices_2d, le raggruppa
        per inizio del nome e, su conferma, crea cartella + gruppo per ognuna."""
        self._dbg_lines = []
        found = detect_timeslice_prefix_groups(self.project_root)
        self._dbg(f"project_root={self.project_root}")
        self._dbg(f"gruppi proposti: {[(e['prefix'], len(e['records'])) for e in found]}")
        if not found:
            self._dbg_flush()
            QMessageBox.information(
                self, "Detect Groups",
                "Nessun gruppo da proporre: le time-slice sciolte in timeslices_2d "
                "non condividono l'inizio del nome (servono almeno 2 file).",
            )
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Detect Groups")
        lay = QVBoxLayout(dlg)
        lbl = QLabel(
            "Trovati questi gruppi dal nome delle time-slice. Quelli spuntati avranno "
            "una cartella timeslices_2d/<nome> e un gruppo, e i file vi verranno spostati."
        )
        lbl.setWordWrap(True)
        lay.addWidget(lbl)
        lst = QListWidget(dlg)
        for entry in found:
            suffix = " (il gruppo esiste gia')" if entry["group"] else ""
            item = QListWidgetItem(f"{entry['prefix']}  -  {len(entry['records'])} file{suffix}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            item.setData(Qt.UserRole, entry["prefix"])
            lst.addItem(item)
        lay.addWidget(lst)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dlg)
        buttons.button(QDialogButtonBox.Ok).setText("Crea cartelle e gruppi")
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)
        dlg.resize(520, 360)
        if dlg.exec_() != QDialog.Accepted:
            return

        chosen = {
            lst.item(i).data(Qt.UserRole)
            for i in range(lst.count()) if lst.item(i).checkState() == Qt.Checked
        }
        if not chosen:
            return

        created, moved_total, failed = 0, 0, []
        for entry in found:
            if entry["prefix"] not in chosen:
                continue
            group, was_created = create_raster_group(self.project_root, entry["prefix"])
            created += 1 if was_created else 0
            ids = [r.get("id") for r in entry["records"] if r.get("id")]
            self._dbg(f"[{entry['prefix']}] gruppo id={group.get('id')} creato={was_created} ids={len(ids)}")
            _recs = {r.get("id"): r for r in entry["records"]}
            saved_layers = self._detach_qgis_layers(
                [_recs[i].get("project_path") for i in ids if i in _recs], self._dbg)
            res = move_timeslices_into_group_folder(self.project_root, group.get("name"), ids)
            moved_total += len(res["moved"])
            failed.extend(res["failed"])
            self._dbg(f"[{entry['prefix']}] file spostati={len(res['moved'])} falliti={res['failed'][:5]}")
            for o, n in res["moved"][:3]:
                self._dbg(f"    esempio: {o} -> {n}")
            self._attach_qgis_layers(saved_layers, res["moved"], group.get("name"), self._dbg)
            moved_ids = {
                r.get("id") for r in load_catalog(self.project_root).get("timeslices", [])
                if r.get("id") in set(ids)
            }
            assign_timeslices_to_group(self.project_root, group.get("id"), sorted(moved_ids))
            _g = next((g for g in load_catalog(self.project_root).get("raster_groups", [])
                       if g.get("id") == group.get("id")), {})
            self._dbg(f"[{entry['prefix']}] catalogo: {len(_g.get('timeslice_ids', []))} slice nel gruppo")
            try:
                remove_timeslices_from_group(self.project_root, "grp_imported", sorted(moved_ids))
            except Exception:
                pass

        self._catalog = load_catalog(self.project_root)
        self._save_and_refresh()
        msg = f"Gruppi creati: {created}. File spostati nelle cartelle: {moved_total}."
        if failed:
            msg += "\n\nNon spostati:\n" + "\n".join(failed[:10])
        log_path = self._dbg_flush()
        if log_path:
            msg += f"\n\nLog di debug: {log_path}"
        QMessageBox.information(self, "Detect Groups", msg)

    def _dbg(self, text):
        lines = getattr(self, "_dbg_lines", None)
        if lines is None:
            lines = self._dbg_lines = []
        lines.append(str(text))

    def _dbg_flush(self):
        """Scrive il log nel progetto e nel pannello Log di QGIS."""
        lines = list(getattr(self, "_dbg_lines", []) or [])
        if not lines:
            return ""
        path = ""
        try:
            from qgis.core import Qgis, QgsMessageLog
            QgsMessageLog.logMessage("\n".join(lines), "GeoSurvey Detect Groups", Qgis.Info)
        except Exception:
            pass
        try:
            path = os.path.join(self.project_root, "detect_groups_debug.log")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
        except Exception:
            path = ""
        return path

    @staticmethod
    def _detach_qgis_layers(paths, dbg=None):
        """Toglie dal progetto QGIS i layer che puntano a ``paths``: su Windows
        un raster aperto non si puo' spostare. Ritorna cio' che serve a
        ricrearli (nome, stile, visibilita', file)."""
        log = dbg if callable(dbg) else (lambda _t: None)
        saved = []
        try:
            from qgis.core import QgsProject
            from qgis.PyQt.QtXml import QDomDocument
            wanted = {os.path.normcase(os.path.abspath(p)) for p in paths if p}
            project = QgsProject.instance()
            tree = project.layerTreeRoot()
            log(f"  layer nel progetto QGIS: {len(project.mapLayers())}; file da spostare: {len(wanted)}")
            for layer in list(project.mapLayers().values()):
                src = os.path.normcase(os.path.abspath(str(layer.source()).split("|")[0]))
                if src not in wanted:
                    if "timeslices_2d" in src.lower():
                        log(f"  layer NON abbinato: {layer.name()} src={src}")
                    continue
                node = tree.findLayer(layer.id())
                doc = QDomDocument("qgis")
                layer.exportNamedStyle(doc)
                saved.append({
                    "path": str(layer.source()).split("|")[0],
                    "name": layer.name(),
                    "style": doc.toString(),
                    "visible": bool(node.itemVisibilityChecked()) if node is not None else True,
                    "crs": layer.crs().authid(),
                })
                project.removeMapLayer(layer.id())
            log(f"  layer staccati dal progetto: {len(saved)}")
        except Exception:
            import traceback
            log("  ERRORE detach: " + traceback.format_exc())
        return saved

    @staticmethod
    def _attach_qgis_layers(saved, moves, group_name, dbg=None):
        """Ricrea i layer staccati dal nuovo percorso nel sottogruppo di
        "GeoSurvey Studio" (dal vecchio se il file non si e' spostato)."""
        log = dbg if callable(dbg) else (lambda _t: None)
        if not saved:
            return
        try:
            from qgis.core import QgsLayerTreeGroup, QgsProject, QgsRasterLayer
            from qgis.PyQt.QtXml import QDomDocument
            new_by_old = {os.path.normcase(os.path.abspath(o)): n for o, n in moves}
            project = QgsProject.instance()
            root = project.layerTreeRoot()
            top = next((g for g in root.children()
                        if isinstance(g, QgsLayerTreeGroup) and g.name() == "GeoSurvey Studio"), None)
            if top is None:
                top = root.addGroup("GeoSurvey Studio")
            target = next((g for g in top.children()
                           if isinstance(g, QgsLayerTreeGroup) and g.name() == group_name), None)
            if target is None:
                target = top.addGroup(group_name)
            for item in saved:
                new = new_by_old.get(os.path.normcase(os.path.abspath(item["path"])), item["path"])
                lyr = QgsRasterLayer(new, item["name"])
                if not lyr.isValid():
                    log(f"  layer NON ricreato (non valido): {new}")
                    continue
                try:
                    doc = QDomDocument()
                    if doc.setContent(item["style"]):
                        lyr.importNamedStyle(doc)
                except Exception:
                    pass
                project.addMapLayer(lyr, False)
                node = target.addLayer(lyr)
                node.setItemVisibilityChecked(item["visible"])
            log(f"  layer ricreati in '{group_name}': {len(saved)}")
        except Exception:
            import traceback
            log("  ERRORE attach: " + traceback.format_exc())

    def _rename_group(self):
        gid = self._selected_group_id()
        if not gid:
            QMessageBox.information(self, "Rename Group", "Select one group in the Groups tab.")
            return
        if gid == "grp_imported":
            QMessageBox.information(self, "Rename Group", "The default 'Imported' group cannot be renamed.")
            return
        rec = self._group_record_by_id(gid)
        if rec is None:
            return
        current = rec.get("name") or ""
        new_name, ok = QInputDialog.getText(self, "Rename Group", "New group name:", text=current)
        if not ok or not new_name.strip():
            return
        new_name = new_name.strip()
        exists = any(
            (g.get("name") or "").strip().lower() == new_name.lower() and g.get("id") != gid
            for g in self._catalog.get("raster_groups", [])
        )
        if exists:
            QMessageBox.warning(self, "Rename Group", "A group with this name already exists.")
            return
        rec["name"] = new_name
        self._save_and_refresh()

    def _delete_group(self):
        gid = self._selected_group_id()
        if not gid:
            QMessageBox.information(self, "Delete Group", "Select one group in the Groups tab.")
            return
        if gid == "grp_imported":
            QMessageBox.information(self, "Delete Group", "The default 'Imported' group cannot be deleted.")
            return
        rec = self._group_record_by_id(gid)
        if rec is None:
            return
        name = rec.get("name") or gid
        tids = list(dict.fromkeys([tid for tid in (rec.get("timeslice_ids") or []) if tid]))
        records_by_id = {
            str(r.get("id") or ""): r
            for r in self._catalog.get("timeslices", [])
            if isinstance(r, dict) and str(r.get("id") or "")
        }
        records = [records_by_id.get(str(tid)) for tid in tids]
        records = [r for r in records if r is not None]
        ts_count = len(records)
        folder_candidate = self._group_folder_candidate(name)
        folder_txt = folder_candidate or f"{os.path.join(self.project_root, 'timeslices_2d', str(name))}"
        answer = QMessageBox.question(
            self,
            "Delete Group",
            (
                f"Delete group '{name}'?\n\n"
                f"Time-slices linked to this group: {ts_count}\n"
                "All linked files will be deleted from disk and removed from catalog.\n"
                f"Associated folder will be removed recursively:\n{folder_txt}"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        progress = QProgressDialog("Deleting group files...", "Cancel", 0, len(records), self)
        progress.setWindowTitle("Delete Group")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        saved_layers = self._detach_layers_for_rename([r.get("project_path") for r in records])
        deleted_ids = set()
        failed_ids = set()
        warnings = []

        for idx, ts_rec in enumerate(records, start=1):
            if progress.wasCanceled():
                break
            tid = str(ts_rec.get("id") or "").strip()
            pth = str(ts_rec.get("project_path") or "").strip()
            progress.setLabelText(f"Deleting {idx}/{len(records)}: {os.path.basename(pth)}")
            QApplication.processEvents()
            try:
                if pth and os.path.exists(pth):
                    pth_abs = os.path.abspath(pth)
                    inside_group_folder = bool(folder_candidate and self._is_within_root(pth_abs, folder_candidate))
                    if not inside_group_folder:
                        for side in self._existing_sidecars(pth_abs):
                            try:
                                os.remove(side)
                            except Exception as e:
                                warnings.append(f"Sidecar delete failed: {os.path.basename(side)} ({e})")
                        os.remove(pth_abs)
                if tid:
                    deleted_ids.add(tid)
            except Exception as e:
                if tid:
                    failed_ids.add(tid)
                warnings.append(f"Delete failed for {os.path.basename(pth)}: {e}")
            progress.setValue(idx)
            QApplication.processEvents()

        progress.close()

        if folder_candidate and os.path.isdir(folder_candidate):
            try:
                shutil.rmtree(folder_candidate)
            except Exception as e:
                warnings.append(f"Folder delete failed for '{folder_candidate}': {e}")
                # Keep records that are still on disk under the folder.
                for ts_rec in records:
                    tid = str(ts_rec.get("id") or "").strip()
                    pth = str(ts_rec.get("project_path") or "").strip()
                    if not tid or tid in failed_ids:
                        continue
                    if pth and os.path.exists(pth) and self._is_within_root(pth, folder_candidate):
                        deleted_ids.discard(tid)
                        failed_ids.add(tid)

        # layer dei file rimasti su disco tornano nel pannello; il gruppo QGIS
        # del gruppo eliminato si toglie se e' vuoto
        self._reattach_layers_after_rename(
            [s for s in saved_layers if os.path.exists(s["path"])], [])
        try:
            top = _order._plugin_root()
            grp = next((g for g in top.children() if g.name() == name), None) if top else None
            if grp is not None and not grp.children():
                top.removeChildNode(grp)
        except Exception:
            pass

        if deleted_ids:
            self._catalog["timeslices"] = [
                r for r in self._catalog.get("timeslices", []) if r.get("id") not in deleted_ids
            ]
            self._catalog["links"] = [
                lk for lk in self._catalog.get("links", []) if lk.get("timeslice_id") not in deleted_ids
            ]
            for g in self._catalog.get("raster_groups", []):
                g["timeslice_ids"] = [tid for tid in g.get("timeslice_ids", []) if tid not in deleted_ids]

        if failed_ids:
            imported = self._ensure_imported_group()
            survivors = [tid for tid in tids if tid in failed_ids]
            imported["timeslice_ids"] = list(dict.fromkeys((imported.get("timeslice_ids") or []) + survivors))

        self._catalog["raster_groups"] = [g for g in self._catalog.get("raster_groups", []) if g.get("id") != gid]
        self._save_and_refresh()

        if warnings:
            preview = "\n".join(warnings[:12])
            extra = len(warnings) - min(len(warnings), 12)
            if extra > 0:
                preview += f"\n... and {extra} more."
            QMessageBox.warning(self, "Delete Group warnings", preview)

    def _toggle_group_lock(self):
        gid = self._selected_group_id()
        if not gid:
            QMessageBox.information(self, "Toggle Group Lock", "Select one group in the Groups tab.")
            return

        rec = self._group_record_by_id(gid)
        if rec is None:
            QMessageBox.warning(self, "Toggle Group Lock", "Selected group not found.")
            return

        # Flip persistent group lock flag used by plugin visibility workflows.
        rec["locked"] = not bool(rec.get("locked", False))
        self._save_and_refresh()

    def _sync_filter_from_group_selection(self):
        gid = self._selected_group_id()
        if gid is None:
            return
        idx = self.filter_group_combo.findData(gid)
        if idx >= 0:
            self.filter_group_combo.setCurrentIndex(idx)

    def _transfer_project(self):
        from PyQt5.QtWidgets import QFileDialog
        from . import project_transfer as pt

        old_root = self.project_root
        parent_dir = QFileDialog.getExistingDirectory(
            self, "Cartella in cui creare il progetto trasferito")
        if not parent_dir:
            return
        default_name = os.path.basename(os.path.normpath(old_root)) or "GeoSurveyStudioProject"
        name, ok = QInputDialog.getText(
            self, "Transfer project", "Nome della nuova cartella-progetto:", text=default_name)
        name = (name or "").strip()
        if not ok or not name:
            return
        new_root = os.path.normpath(os.path.join(parent_dir, name))
        err = pt.validate_transfer(old_root, new_root)
        if err:
            QMessageBox.warning(self, "Transfer project", err)
            return

        box = QMessageBox(self)
        box.setWindowTitle("Transfer project")
        box.setIcon(QMessageBox.Question)
        box.setText(f"Trasferire il progetto\n{old_root}\n\nin\n{new_root}?")
        box.setInformativeText(
            "Copia: l'originale resta dov'e'. Sposta: l'originale viene cancellato "
            "solo dopo che la copia e i metadati sono a posto.")
        copy_btn = box.addButton("Copia", QMessageBox.AcceptRole)
        move_btn = box.addButton("Sposta", QMessageBox.DestructiveRole)
        box.addButton("Annulla", QMessageBox.RejectRole)
        box.exec_()
        clicked = box.clickedButton()
        if clicked not in (copy_btn, move_btn):
            return
        move = clicked is move_btn

        progress = QProgressDialog("Copia del progetto...", "Annulla", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)

        def _progress(i, total, rel):
            progress.setMaximum(max(1, total))
            progress.setValue(i)
            progress.setLabelText(f"Copia ({i}/{total}): {rel}")
            QApplication.processEvents()

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            report = pt.transfer_project(
                old_root, new_root, move=move,
                progress=_progress, should_cancel=progress.wasCanceled)
        except pt.TransferError as exc:
            QApplication.restoreOverrideCursor()
            progress.close()
            QMessageBox.warning(self, "Transfer project", str(exc))
            return
        finally:
            if QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()
        progress.close()

        if callable(self.on_project_transferred):
            try:
                self.on_project_transferred(old_root, new_root)
            except Exception as exc:
                QMessageBox.warning(
                    self, "Transfer project",
                    f"Progetto copiato, ma l'aggiornamento del plugin e' fallito:\n{exc}")
        self.project_root = new_root
        self._refresh()

        rw = report["rewrite"]
        lines = [
            f"Progetto trasferito in:\n{new_root}",
            f"\nFile copiati: {report['copied']}",
            f"File di metadati aggiornati: {rw['files']} ({rw['replacements']} riferimenti)",
        ]
        if move:
            lines.append("Cartella originale cancellata." if report["moved"]
                         else "ATTENZIONE: non sono riuscito a cancellare la cartella originale.")
        ext = report["external"]
        if ext:
            lines.append(
                f"\n{len(ext)} file del catalogo stanno fuori dalla cartella-progetto "
                "e sono rimasti al loro posto (es. " + ext[0] + ").")
        lines.append("\nSalva il progetto QGIS per fissare i nuovi percorsi dei layer.")
        QMessageBox.information(self, "Transfer project", "\n".join(lines))

    def _open_catalog_editor(self):
        if not callable(self.open_catalog_editor_callback):
            return
        self.close()
        try:
            self.open_catalog_editor_callback()
        except Exception as e:
            QMessageBox.warning(
                self,
                "Time-slice Manager",
                f"Unable to open 3D/Radargram Editor:\n{e}",
            )

    def _verify_unassigned_clicked(self):
        # Collect candidate unassigned / no-CRS time-slices and show the management dialog
        catalog = self._catalog or load_catalog(self.project_root)
        membership = self._timeslice_membership_map()

        tids = set()
        # Include explicit built-in groups if present
        imported = next((g for g in catalog.get("raster_groups", []) if g.get("id") == "grp_imported"), None)
        no_crs = next((g for g in catalog.get("raster_groups", []) if g.get("id") == "grp_no_crs"), None)
        if imported:
            tids.update(imported.get("timeslice_ids", []))
        if no_crs:
            tids.update(no_crs.get("timeslice_ids", []))

        # Add timeslices that have no group membership
        for rec in catalog.get("timeslices", []):
            tid = rec.get("id")
            grp_names = membership.get(tid, [])
            if not grp_names:
                tids.add(tid)
            # also include timeslices missing a CRS
            if not rec.get("crs"):
                tids.add(tid)

        rows = [self._timeslice_record_by_id(tid) for tid in tids if self._timeslice_record_by_id(tid) is not None]
        if not rows:
            QMessageBox.information(self, "Verify Unassigned", "No unassigned or no-CRS time-slices found.")
            return

        dlg = UnassignedTimeslicesDialog(self.project_root, rows, parent=self)
        dlg.exec_()
