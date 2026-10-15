# -*- coding: utf-8 -*-
"""Aree indagate del Profile Viewer: import separato, cambio area.

Ogni area ha cartelle e stato propri (vedi ``project_zones``). Qui vivono il
selettore in toolbar, il passaggio da un'area all'altra e la scelta di dove
mettere i profili che si importano."""

from __future__ import annotations

from . import project_zones
from .gpr_viewer_common import (
    QComboBox,
    QInputDialog,
    QLabel,
    QMessageBox,
    os,
)

_NEW_ZONE_ITEM = "＋ Nuova area..."


class ZonesMixin:
    # ------------------------------------------------------------------
    # Contesto area
    # ------------------------------------------------------------------

    def _gpr_project_root(self) -> str:
        root = str(getattr(self, "_gpr_workspace_override", "") or "").strip()
        return root or self._active_project_root()

    def _zone_context(self, root: str):
        """``(nome area, cartella relativa a Processing/)`` dell'area attiva di
        ``root``, letta da project.json una volta e poi tenuta in cache."""
        cached = getattr(self, "_zone_ctx", None)
        if cached and cached[0] == root:
            return cached[1], cached[2]
        state = project_zones.read_state(root)
        zone = state["active_zone"]
        folder = str((state["zones"].get(zone) or {}).get("folder") or "")
        self._zone_ctx = (root, zone, folder)
        return zone, folder

    def _invalidate_zone_context(self) -> None:
        self._zone_ctx = None

    def _current_zone_name(self) -> str:
        root = self._gpr_project_root()
        if not root:
            return project_zones.DEFAULT_ZONE
        return self._zone_context(root)[0]

    # ------------------------------------------------------------------
    # Selettore in toolbar
    # ------------------------------------------------------------------

    def _zone_add_toolbar_widgets(self, tb) -> None:
        tb.addWidget(QLabel(" Area:"))
        self._cb_zone = QComboBox()
        self._cb_zone.setMinimumWidth(120)
        self._cb_zone.setToolTip(
            "Area indagata attiva. Ogni area ha i suoi profili, le sue varianti\n"
            "e le sue time-slice: cambiare area non tocca le altre."
        )
        self._cb_zone.activated.connect(self._on_zone_activated)
        tb.addWidget(self._cb_zone)
        self._refresh_zone_combo()

    def _refresh_zone_combo(self) -> None:
        combo = getattr(self, "_cb_zone", None)
        if not self._is_qt_alive(combo):
            return
        root = self._gpr_project_root()
        combo.blockSignals(True)
        try:
            combo.clear()
            if not root:
                combo.setEnabled(False)
                return
            combo.setEnabled(True)
            current = self._zone_context(root)[0]
            names = project_zones.list_zones(root)
            if current not in names:
                names.insert(0, current)
            for name in names:
                combo.addItem(name, name)
            combo.addItem(_NEW_ZONE_ITEM, None)
            combo.setCurrentIndex(max(0, combo.findData(current)))
        finally:
            combo.blockSignals(False)

    def _on_zone_activated(self, row: int) -> None:
        combo = self._cb_zone
        name = combo.itemData(int(row))
        if name is None:
            self._refresh_zone_combo()
            self._new_zone()
        else:
            self._switch_zone(str(name))

    # ------------------------------------------------------------------
    # Cambio / creazione area
    # ------------------------------------------------------------------

    def _busy_for_zone_change(self) -> bool:
        if getattr(self, "_variant_job", None) is not None:
            QMessageBox.information(
                self, "Aree",
                "Un salvataggio di variante e' in corso: attendi la fine prima di cambiare area.",
            )
            return True
        return False

    def _save_current_zone_state(self) -> None:
        """Scrive lo stato dell'area attiva (solo se ha profili caricati: non
        si sovrascrive mai con uno stato vuoto)."""
        self._flush_recipe_autosave()

    def _switch_zone(self, name: str) -> bool:
        root = self._gpr_project_root()
        if not root or not name:
            return False
        if name == self._zone_context(root)[0]:
            return True
        if self._busy_for_zone_change():
            self._refresh_zone_combo()
            return False
        self._save_current_zone_state()
        if not project_zones.set_active_zone(root, name):
            self._refresh_zone_combo()
            return False
        self._invalidate_zone_context()
        self._after_zone_change()
        return True

    def _suggest_zone_name(self) -> str:
        root = self._gpr_project_root()
        n = len(project_zones.list_zones(root)) + 1 if root else 1
        return f"Area {n}"

    def _ask_new_zone_name(self) -> str:
        name, ok = QInputDialog.getText(
            self, "Nuova area", "Nome della nuova area indagata:",
            text=self._suggest_zone_name(),
        )
        return str(name or "").strip() if ok else ""

    def _create_zone(self, name: str) -> str:
        """Salva l'area attuale, crea ``name`` e ci passa. Ritorna il nome
        effettivo ('' se non e' stato possibile)."""
        root = self._gpr_project_root()
        if not root or not name:
            return ""
        if self._busy_for_zone_change():
            return ""
        self._save_current_zone_state()
        final = project_zones.add_zone(root, name, make_active=True)
        self._invalidate_zone_context()
        self._after_zone_change()
        return final

    def _new_zone(self) -> str:
        if not self._gpr_project_root():
            QMessageBox.information(
                self, "Nuova area", "Crea o apri prima un progetto GPR."
            )
            return ""
        name = self._ask_new_zone_name()
        if not name:
            return ""
        final = self._create_zone(name)
        if final:
            self._safe_set_text(
                self._lbl_status,
                f"Area '{final}' creata e attiva. Importa i profili per iniziare.",
            )
        return final

    def _after_zone_change(self) -> None:
        """Ricarica tutto cio' che dipende dall'area attiva."""
        self._grid_positioning_params = None
        self._variant_parent_ctx = None
        self._pre_restore_state = None
        self._reset_session_state()
        restored = False
        try:
            restored = bool(self._try_import_saved_gpr_processing_project())
        except Exception as exc:
            self._note_swallowed("ripristino dell'area", exc)
        if not restored:
            try:
                out = self._default_slice_output_dir()
                if out and self._is_qt_alive(getattr(self, "_le_slice_outdir", None)):
                    self._le_slice_outdir.setText(out)
            except Exception:
                pass
        try:
            self._sync_navigator()
            self._update_slice_profile_info()
            self._refresh_variants_list()
        except Exception:
            pass
        self._refresh_zone_combo()
        plugin = getattr(self, "plugin", None)
        dock = getattr(plugin, "_radargram_dock", None) if plugin is not None else None
        if dock is not None:
            try:
                dock.refresh_variants()
            except Exception:
                pass
        zone = self._current_zone_name()
        if restored:
            self._safe_set_text(
                self._lbl_status,
                f"Area '{zone}': {len(self._profiles)} profili.",
            )
        else:
            self._safe_set_text(
                self._lbl_status, f"Area '{zone}': nessun profilo importato."
            )

    # ------------------------------------------------------------------
    # Import: dove mettere i nuovi profili
    # ------------------------------------------------------------------

    def _ask_import_target(self):
        """Se nell'area attiva ci sono gia' profili chiede dove metterne di
        nuovi. Ritorna ``"append"``, ``"replace"``, ``("new", nome)`` o
        ``None`` (annullato). Con l'area vuota ritorna ``"replace"``, cioe' il
        comportamento di sempre."""
        if not self._profiles:
            return "replace"
        zone = self._current_zone_name()
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle("Importa profili")
        box.setText(
            f"L'area '{zone}' ha gia' {len(self._profiles)} profili.\n\n"
            "Dove vuoi importare i nuovi?"
        )
        b_new = box.addButton("Nuova area...", QMessageBox.AcceptRole)
        b_add = box.addButton(f"Aggiungi a '{zone}'", QMessageBox.AcceptRole)
        b_rep = box.addButton(f"Sostituisci i profili di '{zone}'", QMessageBox.DestructiveRole)
        box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(b_new)
        box.exec_()
        clicked = box.clickedButton()
        if clicked is b_add:
            return "append"
        if clicked is b_rep:
            return "replace"
        if clicked is b_new:
            name = self._ask_new_zone_name()
            return ("new", name) if name else None
        return None
