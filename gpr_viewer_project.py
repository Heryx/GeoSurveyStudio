# -*- coding: utf-8 -*-
"""Progetto GPR del GPR Profile Viewer: workspace, project.json, ripristino della sessione.

Estratto da gpr_profile_viewer.py senza modifiche ai metodi."""

from __future__ import annotations

from . import project_zones
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
                os.path.join(root, "Processing") if root else os.path.expanduser("~")
            )
        folder = QFileDialog.getExistingDirectory(
            self, "Cartella di lavoro GPR (dove salvare raw/processed/timeslices)", start
        )
        if not folder:
            return
        folder = os.path.normpath(folder)
        enclosing = self._find_enclosing_gpr_project(folder)
        if enclosing and enclosing != folder:
            ret = QMessageBox.warning(
                self, "Cartella di lavoro GPR",
                "La cartella scelta e' dentro un progetto GPR gia' esistente:\n"
                f"{enclosing}\n\n"
                "Usarla come nuovo workspace crea un progetto annidato dentro l'altro "
                "(di solito e' un errore: probabilmente intendevi 'Apri progetto GPR' "
                "su quella cartella).\n\nVuoi usarla comunque?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if ret != QMessageBox.Yes:
                return
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
        """Crea un nuovo progetto GPR (struttura completa Project Manager + Processing).

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
        enclosing = self._find_enclosing_gpr_project(folder)
        if enclosing:
            if enclosing == folder:
                QMessageBox.information(
                    self, "Nuovo progetto GPR",
                    "La cartella contiene gia' un progetto GPR. Usa 'Apri progetto GPR'.",
                )
            else:
                QMessageBox.warning(
                    self, "Nuovo progetto GPR",
                    "La cartella scelta e' dentro un progetto GPR gia' esistente:\n"
                    f"{enclosing}\n\n"
                    "Creare un progetto qui lo annida dentro quello (di solito e' un "
                    "errore capitato scegliendo per sbaglio la sua cartella "
                    "'Processing'). Scegli una cartella fuori da li', oppure usa "
                    "'Apri progetto GPR' su quella cartella per aggiungere una nuova zona.",
                )
            return False

        if not self._maybe_close_previous_project(self._active_project_root(), folder):
            return False

        try:
            from .project_catalog import ensure_project_structure
            ensure_project_structure(folder)
        except Exception as e:
            QMessageBox.warning(self, "Nuovo progetto GPR", f"Impossibile creare le cartelle:\n{e}")
            return False
        self._set_active_workspace(folder, register_in_project_manager=register_in_pm)
        self._reset_session_state()
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
        # Accetta sia la cartella-progetto (contiene metadata/project.json) sia
        # la cartella Processing stessa (in tal caso usa la cartella superiore).
        if (not os.path.isfile(os.path.join(folder, "metadata", "project.json"))
                and os.path.basename(folder) == "Processing"
                and os.path.isfile(os.path.join(os.path.dirname(folder), "metadata", "project.json"))):
            folder = os.path.dirname(folder)
        if not self._maybe_close_previous_project(self._active_project_root(), folder):
            return
        self._set_active_workspace(folder)
        if not self._try_import_saved_gpr_processing_project(force=True):
            QMessageBox.information(
                self, "Apri progetto GPR",
                "Nessun profilo trovato nel progetto selezionato.\n"
                "Verifica che la cartella contenga metadata/project.json.",
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
        # Progetto GPR a cui appartiene il progetto QGIS aperto (scritto nel .qgz).
        try:
            from . import project_binding
            bound = project_binding.recall_root()
        except Exception:
            bound = ""
        if bound:
            return bound
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

    @staticmethod
    def _find_enclosing_gpr_project(folder: str) -> str:
        """Risale gli antenati di ``folder`` (``folder`` incluso) cercando un
        progetto GPR gia' esistente (una cartella con ``metadata/project.json``).

        Ritorna quella cartella, o '' se non trovata. Usato prima di creare o
        impostare un workspace per evitare di annidare un progetto dentro
        un altro (es. scegliendo per sbaglio la ``Processing`` di un
        progetto esistente come "cartella che contiene" un progetto nuovo).
        """
        cur = os.path.normpath(str(folder or "").strip())
        if not cur:
            return ""
        seen = set()
        while cur and cur not in seen:
            seen.add(cur)
            if os.path.isfile(os.path.join(cur, "metadata", "project.json")):
                return cur
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
        return ""

    def _maybe_close_previous_project(self, old_root: str, new_root: str) -> bool:
        """Se un progetto GPR diverso da ``new_root`` e' ancora attivo e ha
        layer nel canvas, chiede se chiuderlo (scaricarli) prima di passare
        al nuovo, o se tenerli (utile per vedere due zone insieme). Ritorna
        ``False`` solo se l'utente annulla il cambio di progetto."""
        old_root = os.path.normpath(str(old_root or "").strip()) if old_root else ""
        new_root_norm = os.path.normpath(str(new_root or "").strip())
        if not old_root or os.path.normcase(old_root) == os.path.normcase(new_root_norm):
            return True

        try:
            from . import project_layer_cleanup
            found = project_layer_cleanup.find_layer_ids_under_root(old_root)
        except Exception:
            return True
        if not found["removable"] and not found["skipped_modified"]:
            return True

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Question)
        msg.setWindowTitle("Progetto GPR precedente")
        msg.setText(
            f"Il progetto GPR precedente ha ancora layer nel canvas:\n{old_root}\n\n"
            f"Layer da quel progetto: {len(found['removable'])}.\n\n"
            "Chiuderlo li' rimuove dal canvas (i file su disco non vengono "
            "toccati). Per vedere piu' zone insieme, scegli 'Mantieni layer aperti'."
        )
        close_btn = msg.addButton("Chiudi progetto precedente", QMessageBox.AcceptRole)
        keep_btn = msg.addButton("Mantieni layer aperti", QMessageBox.NoRole)
        msg.addButton(QMessageBox.Cancel)
        msg.setDefaultButton(keep_btn)
        msg.exec_()
        clicked = msg.clickedButton()

        if clicked is close_btn:
            result = project_layer_cleanup.remove_layers_under_root(old_root)
            if result.get("skipped_modified"):
                QMessageBox.warning(
                    self, "Progetto GPR precedente",
                    "Alcuni layer hanno modifiche non salvate: non sono stati "
                    "rimossi dal canvas:\n" + "\n".join(result["skipped_modified"]),
                )
            return True
        if clicked is keep_btn:
            return True
        return False

    def on_active_project_changed(self) -> None:
        """Il progetto GPR attivo e' cambiato fuori dal viewer (es. e' stato
        aperto un altro progetto QGIS): abbandona il workspace locale e
        rilegge varianti, cartella time-slice e, se il viewer e' vuoto, i
        profili salvati del nuovo progetto."""
        self._gpr_workspace_override = ""
        self._invalidate_zone_context()
        self._synced_plugin_root = self._active_project_root()
        self._refresh_zone_combo()
        if not self._profiles:
            try:
                self._try_import_saved_gpr_processing_project()
            except Exception:
                pass
        try:
            out = self._default_slice_output_dir()
            if out and self._is_qt_alive(getattr(self, "_le_slice_outdir", None)):
                self._le_slice_outdir.setText(out)
        except Exception:
            pass
        try:
            self._refresh_variants_list()
        except Exception:
            pass
        if self._profiles:
            self._safe_set_text(
                self._lbl_status,
                "Progetto GPR attivo cambiato: i profili caricati sono quelli di prima "
                "(File > Ricarica ultimo progetto GPR per caricare quelli del nuovo).",
            )

    def sync_with_active_project(self) -> None:
        """Chiamato quando il viewer viene riaperto: se nel frattempo il
        progetto attivo del plugin e' cambiato, si riallinea."""
        now = self._active_project_root()
        before = getattr(self, "_synced_plugin_root", None)
        if before is None:
            self._synced_plugin_root = now
            return
        if os.path.normcase(os.path.normpath(now or "")) != os.path.normcase(os.path.normpath(before or "")):
            self.on_active_project_changed()

    def _set_active_workspace(self, root: str, register_in_project_manager: bool = True) -> None:
        """Imposta la cartella-progetto attiva (dove vive Processing).

        Aggiorna sia l'override locale del viewer sia il setting del plugin,
        cosi' viewer e plugin (slicer/catalogo) puntano allo stesso progetto.
        Il valore persiste tra sessioni tramite le impostazioni del plugin.
        """
        root = os.path.normpath(str(root or "").strip())
        if not root:
            return
        self._gpr_workspace_override = root
        self._invalidate_zone_context()
        self._refresh_zone_combo()
        if not register_in_project_manager:
            return
        try:
            settings = getattr(self.plugin, "settings", None)
            key = getattr(self.plugin, "settings_key_active_project", None)
            if settings is not None and key:
                settings.setValue(key, root)
            self._synced_plugin_root = root
        except Exception:
            pass
        try:
            from . import project_binding
            project_binding.remember_root(root)
        except Exception:
            pass

    def _gpr_processing_paths(self) -> dict:
        # Workspace scelto dall'utente (Nuovo/Apri/import): priorita' sul setting.
        root = str(getattr(self, "_gpr_workspace_override", "") or "").strip()
        if not root:
            root = self._active_project_root()
        if not root:
            return {}
        zone, folder = self._zone_context(root)
        base = os.path.join(root, "Processing")
        if folder:
            base = os.path.join(base, *folder.split("/"))
        base = os.path.normpath(base)
        return {
            "root": root,
            "zone": zone,
            "base": base,
            "raw": os.path.join(base, "raw"),
            "processed": os.path.join(base, "processed"),
            "timeslices": os.path.join(base, "timeslices"),
            "project_json": os.path.join(root, "metadata", "project.json"),
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
        """Stato salvato dell'AREA attiva (le altre aree restano nel file)."""
        paths = self._gpr_processing_paths()
        pj = paths.get("project_json", "")
        if not pj or (not os.path.exists(pj)):
            return {}
        try:
            state = project_zones.read_state(paths["root"])
            payload = dict(state["zones"].get(paths["zone"]) or {})
            payload["created"] = state.get("created")
            return payload
        except Exception:
            return {}

    def _save_gpr_processing_project_meta(
        self,
        last_processing: Optional[str] = None,
        last_timeslice: Optional[str] = None,
    ) -> bool:
        paths = self._gpr_processing_paths()
        if not paths:
            return False
        try:
            os.makedirs(os.path.dirname(paths["project_json"]), exist_ok=True)
        except Exception:
            return False
        meta = self._load_gpr_processing_project_meta()
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
        try:
            _style = self._slice_display_style()
            if _style:
                ui_state["slice_style"] = _style
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
            return bool(project_zones.set_payload(paths["root"], payload, zone=paths["zone"]))
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
        try:
            style = ui_state.get("slice_style")
            if isinstance(style, dict):
                cb = getattr(self, "_cb_slice_cmap", None)
                name = str(style.get("cmap") or "").strip()
                if name and self._is_qt_alive(cb):
                    idx = cb.findText(name)
                    if idx >= 0:
                        cb.setCurrentIndex(idx)
                for key, attr in (("clip_lo", "_spin_slice_vmin_pct"),
                                  ("clip_hi", "_spin_slice_vmax_pct")):
                    sp = getattr(self, attr, None)
                    if key in style and self._is_qt_alive(sp):
                        sp.setValue(float(style[key]))
                if hasattr(self, "_cb_slice_scale_mode"):
                    if style.get("vmin") is not None and style.get("vmax") is not None:
                        self._set_slice_fixed_spins(float(style["vmin"]), float(style["vmax"]))
                        self._set_slice_scale_mode("fixed")
                    else:
                        self._set_slice_scale_mode("percentile")
                    self._redraw_slice_view(force=True)
        except Exception as exc:
            print(f"[GPR viewer][WARN] ripristino stile timeslice fallito: {exc}")

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
        ts_note = ""
        try:
            ts_saved = self._path_from_gpr_processing_store(str(meta.get("last_timeslice") or ""))
            # Se la cartella salvata non esiste piu' (progetto spostato/copiato,
            # o puntava a un percorso di un altro progetto) non fidarsi del
            # percorso assoluto stantio: ripartire dalla cartella timeslices del
            # progetto corrente, cosi' l'output segue sempre la cartella attiva.
            if ts_saved and os.path.isdir(ts_saved):
                target = ts_saved
            else:
                target = self._default_slice_output_dir()
                if ts_saved:
                    ts_note = " Cartella time-slice salvata non trovata: ripristinata quella del progetto."
            if target and self._is_qt_alive(getattr(self, "_le_slice_outdir", None)):
                self._le_slice_outdir.setText(target)
        except Exception:
            pass
        try:
            self._refresh_variants_list()
            self._refresh_zone_combo()
        except Exception:
            pass
        self._safe_set_text(
            self._lbl_status,
            f"Progetto GPR importato: {len(self._profiles)} profili.{ts_note}",
        )
        return True
