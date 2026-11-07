import os

from qgis.PyQt.QtCore import Qt, QTimer
from qgis.PyQt.QtWidgets import (
    QAbstractItemView,
    QDockWidget,
    QPushButton,
    QSizePolicy,
    QCheckBox,
)
from qgis.core import (
    QgsMessageLog,
    Qgis,
    QgsLayerTreeGroup,
    QgsLayerTreeLayer,
    QgsProject,
    QgsRasterLayer,
)

from .geosurvey_studio_dialog import GeoSurveyStudioDialog
from .project_manager_dialog import ProjectManagerDialog


class AppRuntimeMixin:
    def _qt_widget_alive(self, widget):
        """Return True only when the underlying Qt/C++ object is still alive."""
        if widget is None:
            return False
        try:
            # Accessing a Qt method raises RuntimeError when wrapped C++ is deleted.
            widget.objectName()
            return True
        except RuntimeError:
            return False
        except Exception:
            return False

    def _on_dock_destroyed(self, *_args):
        """Reset dock/dialog references when the dock gets destroyed by Qt."""
        self.dock_widget = None
        self.dlg = None
        self.first_start = True
        # The separate tools dock is parented to the QGIS main window, so it
        # outlives the main dock; remove it here so run() rebuilds it cleanly.
        tools_dock = getattr(self, "tools_dock_widget", None)
        if tools_dock is not None:
            try:
                self.iface.removeDockWidget(tools_dock)
                tools_dock.deleteLater()
            except Exception:
                pass
            self.tools_dock_widget = None
        # Reset UI references that may still point to deleted Qt objects.
        if hasattr(self, "_init_ui_layout"):
            try:
                self._init_ui_layout()
            except Exception:
                pass
        if hasattr(self, "internal_grid_checkbox"):
            self.internal_grid_checkbox = None

    def _layout_debug_enabled(self):
        """Enable layout debug logs when GEOSURVEY_LAYOUT_DEBUG=1."""
        return str(os.environ.get("GEOSURVEY_LAYOUT_DEBUG", "")).strip().lower() in ("1", "true", "yes", "on")

    def _layout_debug_log(self, stage):
        if not self._layout_debug_enabled() or self.dlg is None:
            return
        try:
            rw = self.dlg.rasterListWidget
            gw = self.dlg.groupListWidget
            dw = self.dlg.widget
            tabs = getattr(self, "tools_tabs", None)
            bw = getattr(self, "bottom_controls_widget", None)
            msg = (
                f"[{stage}] dlg={self.dlg.size().width()}x{self.dlg.size().height()} | "
                f"raster={rw.height()} group={gw.height()} drawopt={dw.height()} "
                f"tabs={(tabs.height() if tabs is not None else -1)} "
                f"bottom={(bw.height() if bw is not None else -1)} "
                f"narrow={self._is_narrow_layout} short={getattr(self, '_is_short_layout', None)}"
            )
            QgsMessageLog.logMessage(msg, "GeoSurvey Studio", Qgis.Warning)
        except Exception:
            pass

    def _sync_depth_mode_on_plugin_start(self):
        """Apply persisted depth mode to derived vertex layers on startup.

        This keeps point label visibility and derived depth values coherent
        even before opening the 2D/3D Draw Panel.
        """
        try:
            if hasattr(self, "_trace_depth_pick_mode"):
                self.trace_depth_pick_mode = self._trace_depth_pick_mode()
        except Exception:
            pass

        # Prefer full resync when available (updates derived depth_val + labels).
        if hasattr(self, "_resync_vertex_layers_for_all_traces"):
            try:
                self._resync_vertex_layers_for_all_traces()
            except Exception:
                pass

        # Always enforce label visibility on all detected vertex layers,
        # including legacy/existing ones that might be outside strict resync scope.
        if hasattr(self, "_apply_vertex_label_mode_to_layers"):
            try:
                self._apply_vertex_label_mode_to_layers()
            except Exception:
                pass
        try:
            self.iface.mapCanvas().refreshAllLayers()
        except Exception:
            pass

    def _split_setting_list(self, value):
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [str(v).strip() for v in value if str(v).strip()]
        text = str(value).strip()
        if not text:
            return []
        return [v.strip() for v in text.split("||") if v.strip()]

    def run(self):
        """Esegue il plugin."""
        if (not bool(self.first_start)) and (
            (not self._qt_widget_alive(self.dock_widget)) or (not self._qt_widget_alive(self.dlg))
        ):
            self._on_dock_destroyed()

        if self.first_start:
            self.first_start = False
            self.dlg = GeoSurveyStudioDialog()
            self.dlg.on_resized = self._on_dialog_resized
            self.dock_widget = QDockWidget(self.tr(u"GeoSurvey Studio"), self.iface.mainWindow())
            self.dock_widget.setObjectName("GeoSurveyStudioDockWidget")
            self.dock_widget.setFeatures(
                QDockWidget.DockWidgetMovable
                | QDockWidget.DockWidgetFloatable
                | QDockWidget.DockWidgetClosable
            )
            self.dock_widget.setWidget(self.dlg)
            self.dock_widget.destroyed.connect(self._on_dock_destroyed)
            self.dock_widget.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
            self.iface.addDockWidget(Qt.RightDockWidgetArea, self.dock_widget)
            self._tabify_with_existing_right_dock()
            self._connect_main_dock_persistence_signals()
            self._ensure_dialog_main_layout()
            self._init_name_raster_panel()
            self.populate_group_list()
            self.dlg.groupListWidget.setSelectionMode(QAbstractItemView.ExtendedSelection)
            self.dlg.groupListWidget.setDragDropMode(QAbstractItemView.InternalMove)
            self.dlg.groupListWidget.setDefaultDropAction(Qt.MoveAction)
            self.dlg.groupListWidget.setDragEnabled(True)
            self.dlg.groupListWidget.setAcceptDrops(True)
            self.dlg.groupListWidget.setDropIndicatorShown(True)
            self.dlg.createGridButton.setEnabled(True)
            self.dlg.selectGridPointsButton.setEnabled(True)
            self.dlg.selectGridPointsButton.setText("Set Orientation")
            self.dlg.selectGridPointsButton.setMinimumWidth(120)
            self.dlg.selectGridPointsButton.setToolTip(
                "Set grid orientation with 3 clicks: P0 (origin), X1 (X direction), Y1 (Y direction)."
            )
            self.dlg.lineEditDistanceX.setEnabled(True)
            self.dlg.lineEditDistanceY.setEnabled(True)
            self.dlg.lineEditAreaNames.setEnabled(True)
            self.dlg.lineEditX0Y0.setEnabled(True)
            self.dlg.lineEditX1Y0.setEnabled(True)
            self.dlg.lineEditY0.setEnabled(True)
            self.dlg.lineEditX0Y1.setEnabled(True)
            if self.internal_grid_checkbox is None:
                self.internal_grid_checkbox = QCheckBox("Internal grid", self.dlg)
                self.internal_grid_checkbox.setToolTip(
                    "If enabled, create internal cells using Cell X/Cell Y. "
                    "If disabled, create only the base area from x0/x1/y0/y1."
                )
                self.internal_grid_checkbox.toggled.connect(self._on_internal_grid_toggled)

            # Collega i segnali ai metodi
            self.dlg.createGroupButton.clicked.connect(self.create_group)
            #self.dlg.groupListWidget.itemClicked.connect(self.on_group_selected)
            self.dlg.groupListWidget.itemSelectionChanged.connect(self.on_group_selection_changed)
            self.dlg.groupListWidget.currentItemChanged.connect(self._sync_raster_lock_flag_for_current_group)
            group_model = self.dlg.groupListWidget.model()
            if group_model is not None:
                group_model.rowsMoved.connect(self._on_group_list_rows_moved)
            self.dlg.selectGridPointsButton.clicked.connect(self.activate_grid_selection_tool)

            self.dlg.createGridButton.clicked.connect(self.create_grid_from_polygon_layer)

            self.dlg.zoomSelectedGroupsButton.clicked.connect(self.zoom_to_selected_groups)
            self.dlg.zoomSelectedGroupsButton.setText("Zoom Groups")
            self.dlg.zoomSelectedGroupsButton.setToolTip(
                "Zoom map canvas to the extent of selected plugin groups."
            )
            self.dlg.zoomSelectedGroupsButton.setStatusTip("Zoom selected plugin groups")
            self.load_groups_button = QPushButton("Load Groups")
            self.load_groups_button.setToolTip("Load selected plugin groups into the QGIS layer tree.")
            self.load_groups_button.setStatusTip("Load selected plugin groups")
            self.load_groups_button.clicked.connect(lambda: self.load_raster(show_message=True))
            self.dlg.gridLayout.addWidget(self.load_groups_button, 8, 0, 1, 1)
            self.import_groups_button = QPushButton("Manage")
            self.import_groups_button.setToolTip("Open group manager to load/unload plugin groups.")
            self.import_groups_button.setStatusTip("Open group manager")
            self.import_groups_button.clicked.connect(self.open_group_import_dialog)
            self.dlg.gridLayout.addWidget(self.import_groups_button, 8, 1, 1, 1)
            self.enhance_minmax_button = QPushButton("Min/Max")
            self.enhance_minmax_button.setToolTip("Apply Min/Max enhancement to loaded rasters.")
            self.enhance_minmax_button.setStatusTip("Apply min/max enhancement")
            self.enhance_minmax_button.clicked.connect(self.enhance_loaded_images_minmax)
            self.dlg.gridLayout.addWidget(self.enhance_minmax_button, 5, 0, 1, 1)
            self.enhance_batch_button = QPushButton("Batch Enhance")
            self.enhance_batch_button.setToolTip("Apply batch enhancement with selected method.")
            self.enhance_batch_button.setStatusTip("Apply batch enhancement")
            self.enhance_batch_button.clicked.connect(self.enhance_batch_options)
            self.dlg.gridLayout.addWidget(self.enhance_batch_button, 5, 1, 1, 1)
            self.save_style_button = QPushButton("Save Style")
            self.save_style_button.setToolTip("Save style for active group.")
            self.save_style_button.setStatusTip("Save style for active group")
            self.save_style_button.clicked.connect(self.save_selected_group_style)
            self.dlg.gridLayout.addWidget(self.save_style_button, 6, 0, 1, 1)
            self.load_style_button = QPushButton("Load Style")
            self.load_style_button.setToolTip("Load saved style for active group.")
            self.load_style_button.setStatusTip("Load style for active group")
            self.load_style_button.clicked.connect(self.load_selected_group_style)
            self.dlg.gridLayout.addWidget(self.load_style_button, 6, 1, 1, 1)
            self.export_layout_button = QPushButton("Export PDF")
            self.export_layout_button.setToolTip("Quick PDF export for selected group.")
            self.export_layout_button.setStatusTip("Export quick layout PDF")
            self.export_layout_button.clicked.connect(self.export_group_layout_quick)

            if hasattr(self.dlg, "groupNameEdit"):
                self.dlg.groupNameEdit.setPlaceholderText("New group name")
                self.dlg.groupNameEdit.setToolTip("Enter a name and click Create Group.")
            self.dlg.dial2.setToolTip("Navigation dial: switch visible rasters in selected groups.")
            self.dlg.Dial.setToolTip("Navigation slider: switch visible rasters in selected groups.")

            # Preimposta valori predefiniti
            self.dlg.lineEditDistanceX.setText("1.0")  # Valore predefinito per distanza X
            self.dlg.lineEditDistanceY.setText("1.0")  # Valore predefinito per distanza Y
            self.dlg.lineEditAreaNames.setPlaceholderText("Area name | cell prefix")
            self.dlg.lineEditAreaNames.setToolTip(
                "Area naming field. Format: AreaName|CellPrefix (prefix optional)."
            )
            self.dlg.lineEditAreaNames.setAccessibleName("lineEditAreaNames")
            self._build_tools_tabs()
            self._build_group_actions_panel()
            self._build_bottom_controls_layout()
            self._load_ui_settings()
            if self.internal_grid_checkbox is not None:
                self.internal_grid_checkbox.setChecked(bool(self.grid_internal_enabled))
            self._build_grid_options_controls()
            self._swap_drawing_and_navigation_sections()
            if getattr(self, "raster_lock_flag_checkbox", None) is not None:
                self.raster_lock_flag_checkbox.setToolTip(
                    "Lock current raster for the active group while the dial keeps browsing other selected groups."
                )
                self.raster_lock_flag_checkbox.toggled.connect(self._on_raster_lock_flag_toggled)
            self._sync_grid_options_from_controls()
            self._connect_persistent_fields()
            if hasattr(self, "_restore_group_selection_from_settings"):
                try:
                    self._restore_group_selection_from_settings(trigger_update=True)
                except Exception:
                    pass
            self._tune_visual_layout()
            self._apply_responsive_main_layout(self.dlg.width())
            self._apply_button_icons()
            if hasattr(self, "_bootstrap_trace_layer_from_project"):
                try:
                    self._bootstrap_trace_layer_from_project()
                except Exception:
                    pass
            self._sync_depth_mode_on_plugin_start()
            self.refresh_trace_info_table()
            self._sync_raster_lock_flag_for_current_group()

            # Collega il dial alla funzione di aggiornamento
            self.dlg.Dial.valueChanged.connect(self.update_visibility_with_dial)
            self.dlg.dial2.valueChanged.connect(self.update_visibility_with_dial)

        self.dock_widget.show()
        restored = self._restore_main_dock_state()
        if not restored:
            self._ensure_dock_in_right_area()
            self._tabify_with_existing_right_dock()
        self.dock_widget.raise_()
        self._apply_dock_constraints()
        if self.dlg is not None:
            self._apply_responsive_main_layout(self.dlg.width())
        if self.dlg is not None:
            self._apply_button_icons()
            self._sync_raster_lock_flag_for_current_group()
            if hasattr(self, "_bootstrap_trace_layer_from_project"):
                try:
                    self._bootstrap_trace_layer_from_project()
                except Exception:
                    pass
            self._sync_depth_mode_on_plugin_start()
            self.refresh_trace_info_table()
        if not self._active_project_root():
            self._notify_info(
                "No active project linked. Open 'GeoSurvey Studio Project Manager' and create/open a project.",
                duration=8,
            )
        if hasattr(self, "maybe_check_for_updates_on_start"):
            try:
                self.maybe_check_for_updates_on_start()
            except Exception:
                pass

    def open_project_manager(self):
        if self.project_manager_dialog is None:
            self.project_manager_dialog = ProjectManagerDialog(
                self.iface,
                self.iface.mainWindow(),
                on_project_updated=self._on_project_manager_updated,
            )
        self.project_manager_dialog.show()
        self.project_manager_dialog.raise_()
        if self.dlg is not None:
            self.populate_group_list()

    def _on_project_manager_updated(self):
        self.remember_active_project_in_qgis()
        if self.dlg is not None:
            self.populate_group_list()
            self.populate_raster_list_from_selected_groups()
            self.refresh_trace_info_table()

    def _on_qgis_project_read(self):
        """Invoked after QGIS loads a project file.

        Groups saved in the .qgs layer tree are restored by QGIS, but raster
        layers inside them may have broken data-source paths (e.g. because QGIS
        stored relative paths that no longer resolve).  This handler:
          1. Binds the plugin to the GPR project this QGIS project belongs to
             (stored in the .qgz, or deduced from the loaded rasters).
          2. Registers in the catalog the rasters already sitting in the
             plugin groups but unknown to it.
          3. Repairs broken raster sources using the plugin catalog as the
             authoritative path database.
          4. Refreshes the plugin group/raster list widgets.
        """
        QTimer.singleShot(500, self._bind_project_and_repair)

    def remember_active_project_in_qgis(self):
        """Writes the active GPR project root into the open QGIS project, so a
        later reopen finds the right project instead of "the last one used"."""
        try:
            from . import project_binding
            root = self._active_project_root()
            if root and os.path.isdir(root):
                project_binding.remember_root(root)
        except Exception as exc:
            QgsMessageLog.logMessage(
                f"remember_active_project_in_qgis: {exc}", "GeoSurvey Studio", Qgis.Warning)

    def _apply_active_project_root(self, root):
        """Makes ``root`` the active GPR project everywhere (settings, Project
        Manager, open Profile Viewer, radargram dock)."""
        root = os.path.normpath(str(root or "").strip())
        if not root or not os.path.isdir(root):
            return False
        current = self._active_project_root()
        if current and os.path.normcase(os.path.normpath(current)) == os.path.normcase(root):
            return False
        self.settings.setValue(self.settings_key_active_project, root)
        pm = self.project_manager_dialog
        if pm is not None:
            try:
                pm.path_edit.setText(root)
                pm.project_root = root
            except Exception:
                pass
        viewer = getattr(self, "_gpr_profile_viewer", None)
        if viewer is not None:
            try:
                viewer.on_active_project_changed()
            except Exception:
                pass
        dock = getattr(self, "_radargram_dock", None)
        if dock is not None:
            try:
                dock.refresh_variants()
            except Exception:
                pass
        return True

    def _bind_project_and_repair(self):
        try:
            from . import project_binding
            plugin_root = self._find_plugin_root_group()
            root = project_binding.resolve_root(plugin_root)
            if root:
                if self._apply_active_project_root(root):
                    QgsMessageLog.logMessage(
                        f"Progetto GPR attivo dal progetto QGIS: {root}",
                        "GeoSurvey Studio", Qgis.Info)
                project_binding.remember_root(root)
                if plugin_root is not None:
                    adopted = project_binding.adopt_untracked_timeslices(root, plugin_root)
                    if adopted.get("registered") or adopted.get("assigned"):
                        QgsMessageLog.logMessage(
                            "Time-slice del canvas registrate nel catalogo: "
                            f"{adopted.get('registered', 0)} nuove, "
                            f"{adopted.get('assigned', 0)} assegnate a "
                            f"{adopted.get('groups', 0)} gruppo/i.",
                            "GeoSurvey Studio", Qgis.Info)
        except Exception as exc:
            QgsMessageLog.logMessage(
                f"_bind_project_and_repair: {exc}", "GeoSurvey Studio", Qgis.Warning)
        self._repair_broken_group_layers_from_catalog()
        try:
            self._amp_reconcile_project_layers()
        except Exception as exc:
            QgsMessageLog.logMessage(f"Ampiezze: ripristino fallito: {exc}", "GeoSurvey Studio", Qgis.Warning)

    def _repair_broken_group_layers_from_catalog(self):
        """Repair raster layers with broken sources inside plugin layer-tree groups.

        For each layer node that lives under a plugin group:
        - If the layer is already valid, leave it untouched.
        - If the layer is invalid/missing, look up a timeslice record in the
          catalog whose ``project_path`` matches the stored source path (or
          whose ``normalized_name`` matches the layer name), then replace the
          broken layer with a freshly constructed QgsRasterLayer pointing to
          the correct absolute path.
        """
        try:
            plugin_root = self._find_plugin_root_group()
            if plugin_root is None:
                return

            project_root = self._active_project_root()
            if not project_root:
                return

            from .project_catalog import load_catalog
            catalog = load_catalog(project_root)
            timeslices = catalog.get("timeslices", [])

            # Index timeslices by project_path (normalised) and normalised_name.
            by_path = {}
            by_name = {}
            for ts in timeslices:
                pth = os.path.normpath(str(ts.get("project_path") or "").strip())
                if pth and pth != ".":
                    by_path[pth] = ts
                nn = str(ts.get("normalized_name") or "").strip().lower()
                if nn:
                    by_name.setdefault(nn, ts)

            repaired = 0
            for group_node in list(plugin_root.children()):
                if not isinstance(group_node, QgsLayerTreeGroup):
                    continue
                for child in list(group_node.children()):
                    if not isinstance(child, QgsLayerTreeLayer):
                        continue
                    layer = child.layer()
                    if layer is not None and layer.isValid():
                        continue

                    # Determine stored source path and layer name.
                    stored_source = ""
                    layer_name = ""
                    if layer is not None:
                        stored_source = str(layer.source() or "").strip()
                        layer_name = str(layer.name() or "").strip()
                    else:
                        try:
                            stored_source = str(child.layerId() or "").strip()
                        except Exception:
                            pass

                    ts_rec = None
                    # Try to match by stored source path.
                    if stored_source:
                        norm_src = os.path.normpath(stored_source)
                        ts_rec = by_path.get(norm_src)
                        if ts_rec is None:
                            basename = os.path.basename(norm_src).lower()
                            ts_rec = next(
                                (t for t in timeslices
                                 if os.path.basename(str(t.get("project_path") or "")).lower() == basename),
                                None,
                            )
                    # Fall back to layer name match.
                    if ts_rec is None and layer_name:
                        ts_rec = by_name.get(layer_name.lower())

                    if ts_rec is None:
                        continue

                    correct_path = str(ts_rec.get("project_path") or "").strip()
                    if not correct_path or not os.path.exists(correct_path):
                        continue

                    # Skip if source is already correct.
                    if stored_source and os.path.normpath(stored_source) == os.path.normpath(correct_path):
                        continue

                    display_name = str(ts_rec.get("normalized_name") or layer_name or os.path.basename(correct_path))
                    new_layer = QgsRasterLayer(correct_path, display_name)
                    if not new_layer.isValid():
                        continue

                    assigned_authid = str(ts_rec.get("assigned_crs") or "").strip()
                    if assigned_authid:
                        try:
                            from qgis.core import QgsCoordinateReferenceSystem
                            crs = QgsCoordinateReferenceSystem(assigned_authid)
                            if crs.isValid():
                                new_layer.setCrs(crs)
                        except Exception:
                            pass

                    # Replace the broken node with the repaired layer.
                    try:
                        parent_group = child.parent()
                        idx = parent_group.children().index(child)
                        QgsProject.instance().addMapLayer(new_layer, False)
                        parent_group.insertLayer(idx, new_layer)
                        parent_group.removeChildNode(child)
                        repaired += 1
                    except Exception as exc:
                        QgsMessageLog.logMessage(
                            f"_repair_broken_group_layers: could not replace layer '{display_name}': {exc}",
                            "GeoSurvey Studio",
                            Qgis.Warning,
                        )

            if repaired:
                QgsMessageLog.logMessage(
                    f"Repaired {repaired} broken raster layer(s) after project load.",
                    "GeoSurvey Studio",
                    Qgis.Info,
                )
        except Exception as exc:
            QgsMessageLog.logMessage(
                f"_repair_broken_group_layers_from_catalog error: {exc}",
                "GeoSurvey Studio",
                Qgis.Warning,
            )
        finally:
            # Always refresh the plugin UI regardless of repair outcome.
            if self.dlg is not None:
                try:
                    self.populate_group_list()
                except Exception:
                    pass

    def _apply_dock_constraints(self):
        if self.dock_widget is None or self.dlg is None:
            return
        # Keep constraints lightweight to avoid disturbing QGIS global dock layout.
        self.dlg.setMinimumWidth(340)
        self.dlg.setMinimumHeight(0)
        self.dlg.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.dock_widget.setMinimumWidth(340)
        self.dock_widget.setMinimumHeight(0)

    def _on_dialog_resized(self, size):
        width = size.width() if hasattr(size, "width") else (self.dlg.width() if self.dlg is not None else 0)
        self._apply_responsive_main_layout(width)
        self._layout_debug_log("resize")

    def _apply_responsive_main_layout(self, width):
        if self.dlg is None or not hasattr(self.dlg, "gridLayout_3"):
            return
        # Guard: resize events can fire before runtime UI composition is complete.
        # If we apply responsive rules too early, we freeze a broken layout state.
        if (
            getattr(self, "tools_panel_widget", None) is None
            or getattr(self, "bottom_controls_widget", None) is None
            or getattr(self, "left_nav_widget", None) is None
        ):
            self._layout_debug_log("responsive-skipped-not-ready")
            return
        height = int(self.dlg.height())
        # Keep wide layout active on medium dock widths; narrow only when really constrained.
        is_narrow = width < 640
        is_short = height < 860
        prev_narrow = getattr(self, "_is_narrow_layout", None)
        prev_short = getattr(self, "_is_short_layout", None)
        if prev_narrow is is_narrow and prev_short is is_short:
            return

        gl3 = self.dlg.gridLayout_3
        # Single stacked layout at every width: the lists/dial sit on top
        # (row 0), a small elastic gap (row 1), and the "open tools panel"
        # bar button (row 2). The tools panel itself lives in its own dock.
        gl3.addLayout(self.dlg.verticalLayout_3, 0, 0, 1, 4)
        if getattr(self, "tools_collapse_button", None) is not None:
            gl3.addWidget(self.tools_collapse_button, 2, 0, 1, 4)
        if hasattr(self.dlg, "line"):
            self.dlg.line.hide()
        gl3.setColumnStretch(0, 1)
        gl3.setColumnStretch(1, 0)
        gl3.setColumnStretch(2, 0)
        gl3.setColumnStretch(3, 0)
        # Row 0 claims most vertical growth so the raster/group lists expand
        # toward the bottom; row 1 stays a small elastic gap.
        gl3.setRowStretch(0, 3)
        gl3.setRowStretch(1, 1)
        gl3.setRowStretch(2, 0)
        gl3.setRowStretch(3, 0)
        if hasattr(self, "_distribute_left_column_stretch"):
            self._distribute_left_column_stretch()

        # Re-assert the tools-panel collapsed state (row stretches above
        # would otherwise re-expand a collapsed panel).
        if hasattr(self, "_apply_tools_panel_collapsed"):
            self._apply_tools_panel_collapsed(
                getattr(self, "_tools_panel_collapsed", True), persist=False
            )

        self._is_narrow_layout = is_narrow
        self._is_short_layout = is_short
        self._layout_debug_log("responsive-applied")

    def _ensure_dock_in_right_area(self):
        if self.dock_widget is None:
            return
        if self.dock_widget.isFloating():
            return
        main_window = self.iface.mainWindow()
        try:
            area = main_window.dockWidgetArea(self.dock_widget)
            if area != Qt.RightDockWidgetArea:
                main_window.removeDockWidget(self.dock_widget)
                main_window.addDockWidget(Qt.RightDockWidgetArea, self.dock_widget)
        except Exception:
            pass

    def _tabify_with_existing_right_dock(self):
        if self.dock_widget is None:
            return
        main_window = self.iface.mainWindow()
        try:
            main_window.setDockNestingEnabled(True)
        except Exception:
            pass

        candidates = []
        for dock in main_window.findChildren(QDockWidget):
            if dock is self.dock_widget:
                continue
            if dock.isFloating():
                continue
            if main_window.dockWidgetArea(dock) != Qt.RightDockWidgetArea:
                continue
            candidates.append(dock)

        if not candidates:
            return

        tokens = (
            "browser",
            "processing",
            "style",
            "layer",
            "strumenti di processing",
            "stile layer",
        )

        def dock_score(dock):
            text = f"{dock.windowTitle()} {dock.objectName()}".lower()
            score = 0
            for token in tokens:
                if token in text:
                    score += 10
            if dock.isVisible():
                score += 3
            return score

        target = sorted(candidates, key=dock_score, reverse=True)[0]
        try:
            main_window.tabifyDockWidget(target, self.dock_widget)
        except Exception:
            # Fallback: keep standard dock placement.
            return

    def _settings_key(self, key):
        return f"{self.settings_group}/{key}"

    def _dock_area_to_text(self, area):
        if area == Qt.LeftDockWidgetArea:
            return "left"
        return "right"

    def _dock_text_to_area(self, text):
        if str(text or "").strip().lower() == "left":
            return Qt.LeftDockWidgetArea
        return Qt.RightDockWidgetArea

    def _connect_main_dock_persistence_signals(self):
        if self.dock_widget is None:
            return
        try:
            self.dock_widget.topLevelChanged.connect(self._on_main_dock_top_level_changed)
        except Exception:
            pass
        try:
            self.dock_widget.dockLocationChanged.connect(self._on_main_dock_location_changed)
        except Exception:
            pass

    def _on_main_dock_top_level_changed(self, _is_floating):
        self._save_main_dock_state()

    def _on_main_dock_location_changed(self, _area):
        self._save_main_dock_state()

    def _save_main_dock_state(self):
        if self.dock_widget is None:
            return
        try:
            main_window = self.iface.mainWindow()
            area = main_window.dockWidgetArea(self.dock_widget)
            self.settings.setValue(
                self._settings_key("ui/main_dock/area"),
                self._dock_area_to_text(area),
            )
            self.settings.setValue(
                self._settings_key("ui/main_dock/was_floating"),
                bool(self.dock_widget.isFloating()),
            )
        except Exception:
            pass

    def _restore_main_dock_state(self):
        if self.dock_widget is None:
            return False
        restore_on_open = self.settings.value(
            self._settings_key("ui/main_dock/restore_on_open"),
            True,
            type=bool,
        )
        if not restore_on_open:
            return False

        area_text = str(self.settings.value(self._settings_key("ui/main_dock/area"), "") or "").strip()
        if not area_text:
            return False

        target_area = self._dock_text_to_area(area_text)
        try:
            main_window = self.iface.mainWindow()
            try:
                main_window.removeDockWidget(self.dock_widget)
            except Exception:
                pass
            main_window.addDockWidget(target_area, self.dock_widget)
            self.dock_widget.setFloating(False)
            self.dock_widget.show()
            if target_area == Qt.RightDockWidgetArea:
                self._tabify_with_existing_right_dock()
            return True
        except Exception:
            return False

    def _load_ui_settings(self):
        self.grid_use_snap = self.settings.value(self._settings_key("grid/use_snap"), True, type=bool)
        self.grid_snap_mode = self.settings.value(self._settings_key("grid/snap_mode"), "all")
        self.grid_snap_tolerance = self.settings.value(self._settings_key("grid/snap_tolerance"), 12.0, type=float)
        self.grid_snap_units = self.settings.value(self._settings_key("grid/snap_units"), "pixels")
        self.grid_force_orthogonal = self.settings.value(self._settings_key("grid/force_orthogonal"), False, type=bool)
        self.grid_relative_orthogonal = self.settings.value(self._settings_key("grid/relative_orthogonal"), False, type=bool)
        self.keep_source_polygon = self.settings.value(self._settings_key("grid/keep_source_polygon"), True, type=bool)
        mode = self.settings.value(self._settings_key("grid/dimension_mode"), "ask")
        self.grid_dimension_mode = mode if mode in ("ask", "manual", "canvas") else "ask"
        self.grid_internal_enabled = self.settings.value(
            self._settings_key("grid/internal_enabled"),
            True,
            type=bool,
        )

        self.dlg.lineEditDistanceX.setText(
            self.settings.value(self._settings_key("grid/distance_x"), self.dlg.lineEditDistanceX.text())
        )
        self.dlg.lineEditDistanceY.setText(
            self.settings.value(self._settings_key("grid/distance_y"), self.dlg.lineEditDistanceY.text())
        )
        self.dlg.lineEditAreaNames.setText(
            self.settings.value(self._settings_key("grid/area_names"), "")
        )

        # Persisted UI state (groups/navigation/tools tab).
        self._saved_group_selection_ids = self._split_setting_list(
            self.settings.value(self._settings_key("ui/selected_group_ids"), "")
        )
        self._saved_current_group_id = str(
            self.settings.value(self._settings_key("ui/current_group_id"), "") or ""
        ).strip()
        self._saved_selected_timeslice_ids = self._split_setting_list(
            self.settings.value(self._settings_key("ui/selected_timeslice_ids"), "")
        )
        self._saved_current_timeslice_id = str(
            self.settings.value(self._settings_key("ui/current_timeslice_id"), "") or ""
        ).strip()
        saved_raster_row = self.settings.value(self._settings_key("ui/current_raster_row"), -1)
        try:
            self._saved_current_raster_row = int(saved_raster_row)
        except Exception:
            self._saved_current_raster_row = -1
        nav_index = self.settings.value(self._settings_key("ui/navigation_index"), 0)
        try:
            nav_index = int(nav_index)
        except Exception:
            nav_index = 0
        nav_index = max(0, nav_index)
        self.dlg.Dial.setValue(nav_index)
        self.dlg.dial2.setValue(nav_index)

        tools_tab_index = self.settings.value(self._settings_key("ui/tools_tab_index"), 0)
        try:
            tools_tab_index = int(tools_tab_index)
        except Exception:
            tools_tab_index = 0
        if self.tools_tabs is not None and self.tools_tabs.count() > 0:
            self.tools_tabs.setCurrentIndex(max(0, min(tools_tab_index, self.tools_tabs.count() - 1)))

    def _save_ui_settings(self):
        self._sync_grid_options_from_controls()
        self.settings.setValue(self._settings_key("grid/distance_x"), self.dlg.lineEditDistanceX.text().strip())
        self.settings.setValue(self._settings_key("grid/distance_y"), self.dlg.lineEditDistanceY.text().strip())
        self.settings.setValue(self._settings_key("grid/area_names"), self.dlg.lineEditAreaNames.text().strip())
        self.settings.setValue(self._settings_key("grid/use_snap"), self.grid_use_snap)
        self.settings.setValue(self._settings_key("grid/snap_mode"), self.grid_snap_mode)
        self.settings.setValue(self._settings_key("grid/snap_tolerance"), float(self.grid_snap_tolerance))
        self.settings.setValue(self._settings_key("grid/snap_units"), self.grid_snap_units)
        self.settings.setValue(self._settings_key("grid/force_orthogonal"), self.grid_force_orthogonal)
        self.settings.setValue(self._settings_key("grid/relative_orthogonal"), self.grid_relative_orthogonal)
        self.settings.setValue(self._settings_key("grid/keep_source_polygon"), self.keep_source_polygon)
        self.settings.setValue(self._settings_key("grid/dimension_mode"), self.grid_dimension_mode)
        self.settings.setValue(self._settings_key("grid/internal_enabled"), self.grid_internal_enabled)

        selected_group_ids = []
        current_group_id = ""
        if self.dlg is not None and hasattr(self.dlg, "groupListWidget"):
            selected_group_ids = [
                str(item.data(Qt.UserRole)).strip()
                for item in self.dlg.groupListWidget.selectedItems()
                if item is not None and str(item.data(Qt.UserRole) or "").strip()
            ]
            current_item = self.dlg.groupListWidget.currentItem()
            if current_item is not None:
                current_group_id = str(current_item.data(Qt.UserRole) or "").strip()
        self.settings.setValue(
            self._settings_key("ui/selected_group_ids"),
            "||".join(dict.fromkeys(selected_group_ids)),
        )
        self.settings.setValue(self._settings_key("ui/current_group_id"), current_group_id)
        selected_timeslice_ids = []
        current_timeslice_id = ""
        current_raster_row = -1
        if self.dlg is not None and hasattr(self.dlg, "rasterListWidget"):
            for item in self.dlg.rasterListWidget.selectedItems():
                payload = item.data(Qt.UserRole) if item is not None else None
                if isinstance(payload, dict):
                    tid = str(payload.get("timeslice_id") or "").strip()
                    if tid:
                        selected_timeslice_ids.append(tid)
            current_item = self.dlg.rasterListWidget.currentItem()
            if current_item is not None:
                payload = current_item.data(Qt.UserRole)
                if isinstance(payload, dict):
                    current_timeslice_id = str(payload.get("timeslice_id") or "").strip()
                current_raster_row = int(self.dlg.rasterListWidget.currentRow())
        self.settings.setValue(
            self._settings_key("ui/selected_timeslice_ids"),
            "||".join(dict.fromkeys(selected_timeslice_ids)),
        )
        self.settings.setValue(self._settings_key("ui/current_timeslice_id"), current_timeslice_id)
        self.settings.setValue(self._settings_key("ui/current_raster_row"), int(current_raster_row))
        # Keep in-memory snapshot aligned with current UI.
        self._saved_group_selection_ids = list(dict.fromkeys(selected_group_ids))
        self._saved_current_group_id = current_group_id
        self._saved_selected_timeslice_ids = list(dict.fromkeys(selected_timeslice_ids))
        self._saved_current_timeslice_id = current_timeslice_id
        self._saved_current_raster_row = int(current_raster_row)
        if self.dlg is not None:
            self.settings.setValue(self._settings_key("ui/navigation_index"), int(self.dlg.Dial.value()))
        if self.tools_tabs is not None:
            self.settings.setValue(self._settings_key("ui/tools_tab_index"), int(self.tools_tabs.currentIndex()))
        self._save_main_dock_state()
        if hasattr(self, "_save_trace_info_ui_state"):
            try:
                self._save_trace_info_ui_state()
            except Exception:
                pass

    def _connect_persistent_fields(self):
        self.dlg.lineEditDistanceX.editingFinished.connect(self._save_ui_settings)
        self.dlg.lineEditDistanceY.editingFinished.connect(self._save_ui_settings)
        self.dlg.lineEditAreaNames.editingFinished.connect(self._save_ui_settings)
        if self.internal_grid_checkbox is not None:
            self.internal_grid_checkbox.toggled.connect(self._save_ui_settings)
        if self.snap_checkbox is not None:
            self.snap_checkbox.toggled.connect(self._save_ui_settings)
        if self.snap_mode_combo is not None:
            self.snap_mode_combo.currentIndexChanged.connect(self._save_ui_settings)
        if self.snap_tolerance_spin is not None:
            self.snap_tolerance_spin.valueChanged.connect(self._save_ui_settings)
        if self.snap_units_combo is not None:
            self.snap_units_combo.currentIndexChanged.connect(self._save_ui_settings)
        if self.ortho_checkbox is not None:
            self.ortho_checkbox.toggled.connect(self._save_ui_settings)
        if self.ortho_base_checkbox is not None:
            self.ortho_base_checkbox.toggled.connect(self._save_ui_settings)
        if self.keep_area_checkbox is not None:
            self.keep_area_checkbox.toggled.connect(self._save_ui_settings)
        if self.dimension_mode_combo is not None:
            self.dimension_mode_combo.currentIndexChanged.connect(self._save_ui_settings)
        if self.help_button is not None:
            self.help_button.clicked.connect(self.show_grid_help)
        if self.export_button is not None:
            self.export_button.clicked.connect(self.export_last_grid_to_gpkg)
        if self.tools_tabs is not None:
            self.tools_tabs.currentChanged.connect(self._save_ui_settings)
        if hasattr(self.dlg, "groupListWidget"):
            self.dlg.groupListWidget.itemSelectionChanged.connect(self._save_ui_settings)
        if hasattr(self.dlg, "rasterListWidget"):
            self.dlg.rasterListWidget.itemSelectionChanged.connect(self._save_ui_settings)
            self.dlg.rasterListWidget.currentRowChanged.connect(self._save_ui_settings)
        if hasattr(self.dlg, "Dial"):
            self.dlg.Dial.valueChanged.connect(self._save_ui_settings)
        if hasattr(self.dlg, "dial2"):
            self.dlg.dial2.valueChanged.connect(self._save_ui_settings)

