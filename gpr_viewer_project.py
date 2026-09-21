# -*- coding: utf-8 -*-
"""Progetto GPR del GPR Profile Viewer: workspace, project.json, ripristino della sessione.

Estratto da gpr_profile_viewer.py senza modifiche ai metodi."""

from __future__ import annotations

from .gpr_viewer_common import (
    Optional,
    QFileDialog,
    QInputDialog,
    QMessageBox,
    QgsProject,
    datetime,
    detect_and_read,
    json,
    os,
    shutil,
    sidecar_paths,
)


class ProjectPersistenceMixin:
    """Progetto GPR del GPR Profile Viewer: workspace, project.json, ripristino della sessione.

    Mixin di GprProfileViewer: i metodi usano `self` della finestra.
    """


    def _open_project_manager(self):
        if self.plugin is not None and hasattr(self.plugin, "open_project_manager"):
            try:
                self.plugin.open_project_manager()
                return
            except Exception:
                pass
        QMessageBox.information(
            self,
            "Project Manager",
            "Project Manager non disponibile in questo contesto.",
        )

    def _save_project_file(self):
        try:
            ok = bool(QgsProject.instance().write())
        except Exception:
            ok = False
        meta_ok = False
        try:
            meta_ok = bool(self._save_gpr_processing_project_meta())
        except Exception:
            meta_ok = False
        if ok and meta_ok:
            self._safe_set_text(self._lbl_status, "Progetto QGIS + stato GPR salvati.")
        elif ok and (not meta_ok):
            self._safe_set_text(self._lbl_status, "Progetto QGIS salvato, stato GPR non salvato.")
        else:
            QMessageBox.warning(
                self,
                "Salva progetto",
                "Impossibile salvare il progetto QGIS corrente.",
            )

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------

    def _prompt_workspace_dir(self) -> None:
        """Chiede all'utente dove salvare il workspace GPR (raw/processed/timeslices).

        Imposta self._gpr_workspace_override sulla cartella scelta. Se l'utente
        annulla, mantiene l'eventuale workspace precedente (o il default
        automatico sotto il progetto attivo).
        """
        start = str(getattr(self, "_gpr_workspace_override", "") or "").strip()
        if not start:
            root = self._active_project_root()
            start = (
                os.path.join(root, "GPR_Processing") if root else os.path.expanduser("~")
            )
        folder = QFileDialog.getExistingDirectory(
            self, "Cartella di lavoro GPR (dove salvare raw/processed/timeslices)", start
        )
        if folder:
            self._set_active_workspace(folder)

    def _project_manager_root(self) -> str:
        """Progetto attualmente caricato nel Project Manager ('' se nessuno)."""
        root = ""
        try:
            settings = getattr(self.plugin, "settings", None)
            key = getattr(self.plugin, "settings_key_active_project", None)
            if settings is not None and key:
                root = (settings.value(key, "", type=str) or "").strip()
        except Exception:
            root = ""
        return os.path.normpath(root) if root and os.path.isdir(root) else ""

    def _new_gpr_project(self) -> bool:
        """Crea un nuovo progetto GPR (struttura completa Project Manager + GPR_Processing).

        Verifica prima se nel Project Manager c'e' gia' un progetto caricato e
        avvisa l'utente; altrimenti propone di registrare il nuovo percorso
        anche nel Project Manager. Ritorna True se il progetto e' stato creato.
        """
        current = self._project_manager_root()
        register_in_pm = True
        if current:
            ret = QMessageBox.warning(
                self, "Nuovo progetto GPR",
                "Nel Project Manager e' gia' caricato un progetto:\n"
                f"{current}\n\n"
                "Per evitare progetti annidati, di norma conviene usare "
                "'Apri progetto GPR' su quella cartella.\n\n"
                "Vuoi comunque creare un NUOVO progetto separato? "
                "Il Project Manager passera' al nuovo progetto.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if ret != QMessageBox.Yes:
                return False
        else:
            ret = QMessageBox.question(
                self, "Nuovo progetto GPR",
                "Nessun progetto e' caricato nel Project Manager.\n"
                "Stai per creare un progetto nuovo.\n\n"
                "Vuoi caricare il percorso del nuovo progetto anche nel "
                "Project Manager?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes,
            )
            if ret == QMessageBox.Cancel:
                return False
            register_in_pm = ret == QMessageBox.Yes

        start = str(getattr(self, "_gpr_workspace_override", "") or "").strip() \
            or current or os.path.expanduser("~")
        parent_dir = QFileDialog.getExistingDirectory(
            self, "Nuovo progetto GPR — scegli la cartella che contiene il progetto", start
        )
        if not parent_dir:
            return False
        name, ok = QInputDialog.getText(
            self, "Nuovo progetto GPR", "Nome della cartella del progetto:",
            text="GPR_Project",
        )
        name = str(name or "").strip()
        if not ok or not name:
            return False
        folder = os.path.normpath(os.path.join(parent_dir, name))
        if os.path.isfile(os.path.join(folder, "GPR_Processing", "project.json")):
            QMessageBox.information(
                self, "Nuovo progetto GPR",
                "La cartella contiene gia' un progetto GPR. Usa 'Apri progetto GPR'.",
            )
            return False

        try:
            from .project_catalog import ensure_project_structure
            ensure_project_structure(folder)
        except Exception as e:
            QMessageBox.warning(self, "Nuovo progetto GPR", f"Impossibile creare le cartelle:\n{e}")
            return False
        self._set_active_workspace(folder, register_in_project_manager=register_in_pm)
        self._reset_session_state()
        self._ensure_gpr_processing_layout()
        try:
            self._save_gpr_processing_project_meta()
        except Exception:
            pass
        self._safe_set_text(
            self._lbl_status,
            f"Nuovo progetto GPR: {folder}. Importa i profili per iniziare.",
        )
        return True

    def _open_gpr_project(self):
        """Apre un progetto GPR esistente da una cartella scelta."""
        start = str(getattr(self, "_gpr_workspace_override", "") or "").strip() \
            or os.path.expanduser("~")
        folder = QFileDialog.getExistingDirectory(
            self, "Apri progetto GPR — scegli la cartella del progetto", start
        )
        if not folder:
            return
        folder = os.path.normpath(folder)
        # Accetta sia la cartella-progetto (contiene GPR_Processing) sia la
        # cartella GPR_Processing stessa (in tal caso usa la cartella superiore).
        if (not os.path.isfile(os.path.join(folder, "GPR_Processing", "project.json"))
                and os.path.isfile(os.path.join(folder, "project.json"))):
            folder = os.path.dirname(folder)
        self._set_active_workspace(folder)
        if not self._try_import_saved_gpr_processing_project(force=True):
            QMessageBox.information(
                self, "Apri progetto GPR",
                "Nessun profilo trovato nel progetto selezionato.\n"
                "Verifica che la cartella contenga GPR_Processing/project.json.",
            )

    def _active_project_root(self) -> str:
        root = ""
        if self.plugin is not None:
            settings = getattr(self.plugin, "settings", None)
            settings_key = getattr(self.plugin, "settings_key_active_project", None)
            if settings is not None and settings_key:
                try:
                    root = (settings.value(settings_key, "", type=str) or "").strip()
                except Exception:
                    root = ""
        if root:
            return root
        # Fallback: directory of current QGIS project file.
        try:
            qgs_file = str(QgsProject.instance().fileName() or "").strip()
        except Exception:
            qgs_file = ""
        if qgs_file:
            try:
                return os.path.dirname(os.path.normpath(qgs_file))
            except Exception:
                return ""
        return ""

    def _set_active_workspace(self, root: str, register_in_project_manager: bool = True) -> None:
        """Imposta la cartella-progetto attiva (dove vive GPR_Processing).

        Aggiorna sia l'override locale del viewer sia il setting del plugin,
        cosi' viewer e plugin (slicer/catalogo) puntano allo stesso progetto.
        Il valore persiste tra sessioni tramite le impostazioni del plugin.
        """
        root = os.path.normpath(str(root or "").strip())
        if not root:
            return
        self._gpr_workspace_override = root
        if not register_in_project_manager:
            return
        try:
            settings = getattr(self.plugin, "settings", None)
            key = getattr(self.plugin, "settings_key_active_project", None)
            if settings is not None and key:
                settings.setValue(key, root)
        except Exception:
            pass

    def _gpr_processing_paths(self) -> dict:
        # Workspace scelto dall'utente (Nuovo/Apri/import): priorita' sul setting.
        root = str(getattr(self, "_gpr_workspace_override", "") or "").strip()
        if not root:
            root = self._active_project_root()
        if not root:
            return {}
        base = os.path.normpath(os.path.join(root, "GPR_Processing"))
        return {
            "base": base,
            "raw": os.path.join(base, "raw"),
            "processed": os.path.join(base, "processed"),
            "timeslices": os.path.join(base, "timeslices"),
            "project_json": os.path.join(base, "project.json"),
        }

    def _ensure_gpr_processing_layout(self) -> dict:
        paths = self._gpr_processing_paths()
        if not paths:
            return {}
        try:
            os.makedirs(paths["raw"], exist_ok=True)
            os.makedirs(paths["processed"], exist_ok=True)
            os.makedirs(paths["timeslices"], exist_ok=True)
            os.makedirs(paths["base"], exist_ok=True)
        except Exception:
            return {}
        return paths

    def _path_to_gpr_processing_store(self, path: str) -> str:
        p = os.path.normpath(str(path or "").strip())
        if not p:
            return ""
        paths = self._gpr_processing_paths()
        base = paths.get("base", "")
        if not base:
            return p
        try:
            cp = os.path.commonpath([os.path.abspath(p), os.path.abspath(base)])
            if cp == os.path.abspath(base):
                rel = os.path.relpath(os.path.abspath(p), os.path.abspath(base))
                return rel.replace("\\", "/")
        except Exception:
            pass
        return p

    def _path_from_gpr_processing_store(self, stored_path: str) -> str:
        raw = str(stored_path or "").strip()
        if not raw:
            return ""
        paths = self._gpr_processing_paths()
        base = paths.get("base", "")
        if os.path.isabs(raw) or not base:
            return os.path.normpath(raw)
        return os.path.normpath(os.path.join(base, raw))

    def _load_gpr_processing_project_meta(self) -> dict:
        paths = self._gpr_processing_paths()
        pj = paths.get("project_json", "")
        if not pj or (not os.path.exists(pj)):
            return {}
        try:
            with open(pj, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_gpr_processing_project_meta(
        self,
        last_processing: Optional[str] = None,
        last_timeslice: Optional[str] = None,
    ) -> bool:
        paths = self._ensure_gpr_processing_layout()
        if not paths:
            return False
        now_iso = datetime.now().isoformat(timespec="seconds")
        meta = self._load_gpr_processing_project_meta()
        created = str(meta.get("created") or now_iso)
        profiles_payload = []
        for prof in list(getattr(self, "_profiles", []) or []):
            try:
                p_abs = os.path.normpath(str(getattr(prof, "path", "") or ""))
                if not p_abs:
                    continue
                profiles_payload.append({
                    "filename": os.path.basename(p_abs),
                    "path": self._path_to_gpr_processing_store(p_abs),
                    "channels": list(range(int(getattr(prof, "n_channels", 0) or 0))),
                    "loaded": True,
                })
            except Exception:
                continue
        ui_state = {
            "colormap": (
                str(self._cb_cmap.currentText())
                if self._is_qt_alive(getattr(self, "_cb_cmap", None))
                else ""
            ),
            "ve": (
                float(self._spin_vertical_exag.value())
                if self._is_qt_alive(getattr(self, "_spin_vertical_exag", None))
                else 1.0
            ),
            "render": (
                str(self._cb_interp.currentText())
                if self._is_qt_alive(getattr(self, "_cb_interp", None))
                else "Smooth"
            ),
            "show_wiggle": bool(
                self._chk_show_wiggle.isChecked()
                if self._is_qt_alive(getattr(self, "_chk_show_wiggle", None))
                else self._show_wiggle
            ),
        }
        # Fase D: persist the SliceSpec form verbatim.
        _panel = getattr(self, "_slice_spec_panel", None)
        if _panel is not None and self._is_qt_alive(_panel):
            try:
                ui_state["slice_spec"] = _panel.get_spec().to_dict()
            except Exception:
                pass
        if last_timeslice is None:
            try:
                outdir = str(self._le_slice_outdir.text() or "").strip()
            except Exception:
                outdir = ""
            if outdir:
                last_timeslice = self._path_to_gpr_processing_store(os.path.normpath(outdir))
        payload = {
            "version": "1.0",
            "created": created,
            "updated": now_iso,
            "profiles": profiles_payload,
            "last_processing": (
                last_processing
                if last_processing is not None
                else meta.get("last_processing")
            ),
            "last_timeslice": (
                last_timeslice
                if last_timeslice is not None
                else meta.get("last_timeslice")
            ),
            "ui_state": ui_state,
            # Preserve manual grid positioning params across sessions
            "grid_positioning": getattr(self, "_grid_positioning_params", None) or meta.get("grid_positioning"),
            # Ricetta completa della pipeline: evita di riprocessare da zero a
            # ogni riapertura del viewer.
            "processing_state": (
                self._collect_full_processing_state()
                if self._is_qt_alive(getattr(self, "_filter_chain_widget", None))
                else meta.get("processing_state")
            ),
        }
        try:
            with open(paths["project_json"], "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False

    def _stage_profile_in_project_raw(self, src_path: str) -> str:
        p_src = os.path.normpath(str(src_path or "").strip())
        if (not p_src) or (not os.path.exists(p_src)):
            return p_src
        paths = self._ensure_gpr_processing_layout()
        raw_dir = paths.get("raw", "")
        if not raw_dir:
            return p_src
        try:
            cp = os.path.commonpath([os.path.abspath(p_src), os.path.abspath(raw_dir)])
            if cp == os.path.abspath(raw_dir):
                return p_src
        except Exception:
            pass

        name = os.path.basename(p_src)
        base, ext = os.path.splitext(name)
        candidate = os.path.join(raw_dir, name)
        i = 1
        while os.path.exists(candidate):
            try:
                if os.path.getsize(candidate) == os.path.getsize(p_src):
                    return candidate
            except Exception:
                pass
            candidate = os.path.join(raw_dir, f"{base}_{i}{ext}")
            i += 1
            if i > 999:
                break

        # Prefer lightweight link, fallback to real copy.
        staged = None
        try:
            rel_src = os.path.relpath(p_src, raw_dir)
            os.symlink(rel_src, candidate)
            staged = candidate
        except Exception:
            pass
        if staged is None:
            try:
                shutil.copy2(p_src, candidate)
                staged = candidate
            except Exception:
                return p_src

        # Alcuni formati (es. .DT) hanno file sidecar (.gec/.gps/.gps.sav)
        # indispensabili per il posizionamento GNSS: li porta con se' nella
        # stessa cartella raw, accanto al file principale.
        try:
            for sc_src in sidecar_paths(p_src):
                sc_name = os.path.basename(sc_src)
                sc_dst = os.path.join(raw_dir, sc_name)
                if os.path.exists(sc_dst):
                    continue
                try:
                    rel_sc = os.path.relpath(sc_src, raw_dir)
                    os.symlink(rel_sc, sc_dst)
                except Exception:
                    try:
                        shutil.copy2(sc_src, sc_dst)
                    except Exception:
                        pass
        except Exception:
            pass
        return staged

    def _apply_gpr_processing_ui_state(self, ui_state: dict) -> None:
        if not isinstance(ui_state, dict):
            return
        try:
            cmap = str(ui_state.get("colormap") or "").strip()
            if cmap and self._is_qt_alive(getattr(self, "_cb_cmap", None)):
                self._cb_cmap.setCurrentText(cmap)
        except Exception:
            pass
        try:
            if self._is_qt_alive(getattr(self, "_spin_vertical_exag", None)):
                self._spin_vertical_exag.setValue(float(ui_state.get("ve", self._spin_vertical_exag.value())))
        except Exception:
            pass
        try:
            render = str(ui_state.get("render") or "").strip()
            if render and self._is_qt_alive(getattr(self, "_cb_interp", None)):
                self._cb_interp.setCurrentText(render)
        except Exception:
            pass
        try:
            if self._is_qt_alive(getattr(self, "_chk_show_wiggle", None)):
                self._chk_show_wiggle.setChecked(bool(ui_state.get("show_wiggle", self._chk_show_wiggle.isChecked())))
        except Exception:
            pass
        try:
            spec_d = ui_state.get("slice_spec")
            panel = getattr(self, "_slice_spec_panel", None)
            if isinstance(spec_d, dict) and panel is not None and self._is_qt_alive(panel):
                from .slice_spec import SliceSpec
                panel.set_spec(SliceSpec.from_dict(spec_d))
        except Exception as exc:
            print(f"[GPR viewer][WARN] ripristino SliceSpec fallito: {exc}")

    def _try_import_saved_gpr_processing_project(self, force: bool = False) -> bool:
        if self._profiles and (not bool(force)):
            return False
        if self._profiles and bool(force):
            ans = QMessageBox.question(
                self,
                "Import progetto GPR",
                "Sostituire i profili correnti con quelli del progetto GPR salvato?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if ans != QMessageBox.Yes:
                return False
        meta = self._load_gpr_processing_project_meta()
        entries = list(meta.get("profiles") or [])
        if not entries:
            return False
        paths = []
        for e in entries:
            if not isinstance(e, dict):
                continue
            p = self._path_from_gpr_processing_store(str(e.get("path") or ""))
            if p and os.path.exists(p):
                paths.append(p)
        if not paths:
            return False

        self._restoring_project = True
        try:
            return self._restore_saved_project(meta, paths)
        finally:
            self._restoring_project = False
            self._flush_recipe_autosave()

    def _show_restore_bar(self) -> None:
        bar = getattr(self, "_restore_bar", None)
        if not self._is_qt_alive(bar):
            return
        try:
            from qgis.core import QgsProject
            name = os.path.splitext(os.path.basename(QgsProject.instance().fileName()))[0]
        except Exception:
            name = ""
        self._lbl_restore_bar.setText(
            f"Ripristinato l'ultimo stato di processing"
            f"{' di ' + name if name else ''}.")
        bar.show()
        self._set_recipe_state_label(saved=True)

    def _restore_saved_project(self, meta: dict, paths: list) -> bool:
        self._reset_session_state()
        errors = []
        for p in paths:
            try:
                self._profiles.append(detect_and_read(p, verify_md5=True))
            except Exception:
                try:
                    self._profiles.append(detect_and_read(p, verify_md5=False))
                except Exception as exc:
                    errors.append(f"{os.path.basename(p)}: {exc}")
        if not self._profiles:
            if errors:
                self._safe_set_text(self._lbl_status, f"Import progetto: {errors[0]}")
            return False
        self._prof_idx = 0
        # Variante salvata: se la ricetta ripristinata coincide, il dato
        # processato viene letto dal disco invece di ricalcolare la pipeline.
        self._active_variant_dir = self._path_from_gpr_processing_store(
            str(meta.get("last_processing") or ""))
        self._load_current_profile()
        self._apply_gpr_processing_ui_state(meta.get("ui_state") or {})
        try:
            ps = meta.get("processing_state")
            if isinstance(ps, dict) and ps:
                try:
                    self._pre_restore_state = self._collect_full_processing_state()
                except Exception:
                    self._pre_restore_state = None
                self._apply_full_processing_state(ps)
                self._show_restore_bar()
        except Exception as exc:
            print(f"[GPR viewer][WARN] ripristino processing_state fallito: {exc}")
        self._update_slice_profile_info()
        try:
            ts_saved = self._path_from_gpr_processing_store(str(meta.get("last_timeslice") or ""))
            if ts_saved and self._is_qt_alive(getattr(self, "_le_slice_outdir", None)):
                self._le_slice_outdir.setText(ts_saved)
        except Exception:
            pass
        self._safe_set_text(self._lbl_status, f"Progetto GPR importato: {len(self._profiles)} profili.")
        return True
