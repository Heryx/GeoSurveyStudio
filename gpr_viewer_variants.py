# -*- coding: utf-8 -*-
"""Varianti e 'Lavoro' del GPR Profile Viewer: salvataggio automatico del processo, varianti su disco, pannello Varianti.

Estratto da gpr_profile_viewer.py senza modifiche ai metodi."""

from __future__ import annotations

import threading
import time

from qgis.PyQt.QtCore import QEventLoop

from .gpr_profile_gain import profile_gain_key, resolve_profile_params
from .gpr_variant_runner import run_channels
from .gpr_velocity_store import write_velocity_override
from .gpr_viewer_common import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QObject,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QThread,
    QTimer,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    Qt,
    glob,
    json,
    list_variants,
    log_exception,
    multifreq_blend,
    np,
    os,
    pyqtSignal,
    read_variant_manifest,
    save_processed_profile_npz,
    shutil,
    write_variant_manifest,
)


def _fmt_duration(seconds: float) -> str:
    s = max(0, int(round(seconds)))
    if s >= 3600:
        return f"{s // 3600} h {(s % 3600) // 60:02d} min"
    if s >= 60:
        return f"{s // 60} min {s % 60:02d} s"
    return f"{s} s"


class _VariantSaveWorker(QObject):
    """Esegue il salvataggio della variante fuori dal thread della GUI.

    Orchestra `run_channels` (pool di processi, ripiego seriale): i canali si
    calcolano in parallelo, i profili completi si scrivono su disco qui."""

    progress = pyqtSignal(int, int, int)  # canali fatti, canali totali, profili salvati
    finished = pyqtSignal(object)         # dict di run_channels

    def __init__(self, counts, make_payload, on_profile_done):
        super().__init__()
        self._counts = counts
        self._make_payload = make_payload
        self._on_profile_done = on_profile_done
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        try:
            result = run_channels(
                self._counts, self._make_payload, self._on_profile_done,
                on_progress=lambda done, total, saved, _pi: self.progress.emit(done, total, saved),
                is_cancelled=self._cancel.is_set,
            )
        except Exception as exc:
            try:
                log_exception("salvataggio variante", exc)
            except Exception:
                pass
            result = {"fatal": str(exc), "errors": {}, "cancelled": False, "mode": "seriale",
                      "workers": 1, "note": "", "elapsed": 0.0, "done": 0, "total": 0}
        self.finished.emit(result)


class _VariantProgressDialog(QDialog):
    """Avanzamento del salvataggio: resta aperto e responsivo, annullabile.

    Non si chiude da solo ne' con Esc/X: l'annullamento e' una richiesta che il
    worker onora al canale successivo, e la finestra resta finche' non ha finito."""

    cancel_requested = pyqtSignal()

    def __init__(self, parent, total: int):
        super().__init__(parent)
        self.setWindowTitle("Salva variante")
        self.setWindowModality(Qt.ApplicationModal)
        self.setMinimumWidth(480)
        self._finished = False
        self._cancel_note = "\nI profili gia' scritti restano su disco."
        lay = QVBoxLayout(self)
        self._lbl = QLabel("Avvio del calcolo in parallelo...")
        self._lbl.setWordWrap(True)
        lay.addWidget(self._lbl)
        self._bar = QProgressBar()
        self._bar.setRange(0, max(1, int(total)))
        self._bar.setValue(0)
        lay.addWidget(self._bar)
        self._btn = QPushButton("Annulla")
        self._btn.clicked.connect(self._request_cancel)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self._btn)
        lay.addLayout(row)

    def _request_cancel(self) -> None:
        if self._finished or not self._btn.isEnabled():
            return
        self._btn.setEnabled(False)
        self._lbl.setText("Annullamento in corso..." + self._cancel_note)
        self.cancel_requested.emit()

    def set_progress(self, done: int, text: str) -> None:
        if not self._btn.isEnabled():
            return  # annullamento richiesto: non sovrascrivere il messaggio
        self._bar.setValue(int(done))
        self._lbl.setText(text)

    def finish(self) -> None:
        self._finished = True
        self.close()

    def reject(self) -> None:
        self._request_cancel()

    def closeEvent(self, event) -> None:
        if self._finished:
            event.accept()
        else:
            event.ignore()
            self._request_cancel()


class VariantsMixin:
    """Varianti e 'Lavoro' del GPR Profile Viewer: salvataggio automatico del processo, varianti su disco, pannello Varianti.

    Mixin di GprProfileViewer: i metodi usano `self` della finestra.
    """


    def _schedule_recipe_autosave(self) -> None:
        """Salva la ricetta di processing in project.json poco dopo l'ultima
        modifica (debounce), cosi' non dipende dalla chiusura del viewer."""
        if getattr(self, "_restoring_project", False) or not self._profiles:
            return
        timer = getattr(self, "_recipe_autosave_timer", None)
        if timer is None:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.setInterval(1500)
            timer.timeout.connect(self._flush_recipe_autosave)
            self._recipe_autosave_timer = timer
        self._set_recipe_state_label(saved=False)
        timer.start()

    def _set_recipe_state_label(self, saved: bool) -> None:
        lbl = getattr(self, "_lbl_recipe_state", None)
        if not self._is_qt_alive(lbl):
            return
        if saved:
            lbl.setText("Processo: salvato ✓")
            lbl.setStyleSheet("color: #2e7d32;")
        else:
            lbl.setText("Processo: modificato…")
            lbl.setStyleSheet("color: #b26a00;")

    def _flush_recipe_autosave(self) -> None:
        if getattr(self, "_restoring_project", False) or not self._profiles:
            return
        try:
            self._save_gpr_processing_project_meta()
            self._set_recipe_state_label(saved=True)
        except Exception:
            pass

    def _undo_restore(self) -> None:
        prev = getattr(self, "_pre_restore_state", None)
        self._pre_restore_state = None
        if self._is_qt_alive(getattr(self, "_restore_bar", None)):
            self._restore_bar.hide()
        if not isinstance(prev, dict) or not prev:
            return
        try:
            self._apply_full_processing_state(prev)
            self._safe_set_text(self._lbl_status, "Ripristino annullato.")
        except Exception as exc:
            print(f"[GPR viewer][WARN] annulla ripristino fallito: {exc}")

    @staticmethod
    def _recipe_signature(params: dict, geom=None) -> str:
        """Firma stabile della ricetta (+ trim/flip); JSON-normalizzata cosi'
        da poter confrontare i parametri correnti con quelli letti da un .npz."""
        try:
            clean = {k: v for k, v in dict(params or {}).items()
                     if not isinstance(v, np.ndarray)}
            norm = json.loads(json.dumps(clean, default=str))
            return json.dumps([norm, geom], sort_keys=True, default=str)
        except Exception:
            return ""

    def _view_geom_sig(self):
        return [
            int(self._trim_start_traces), int(self._trim_end_traces),
            bool(getattr(self, "_chk_flip_profile", None)
                 and self._chk_flip_profile.isChecked()),
        ]

    def _variant_npz_for(self, prof) -> str:
        """.npz della variante attiva corrispondente al profilo (o '')."""
        vdir = str(getattr(self, "_active_variant_dir", "") or "")
        if not vdir or not os.path.isdir(vdir):
            return ""
        base = os.path.splitext(os.path.basename(str(getattr(prof, "path", "") or "")))[0]
        fp = os.path.join(vdir, base + ".npz")
        return fp if base and os.path.isfile(fp) else ""

    def _load_variant_channel(self, prof, ch_idx: int, sig: str, shape):
        """Dato processato dal disco se la variante attiva e' stata calcolata con
        esattamente la ricetta corrente; altrimenti None."""
        fp = self._variant_npz_for(prof)
        if not fp:
            return None
        try:
            with np.load(fp, allow_pickle=False) as npz:
                meta = json.loads(str(npz["meta"]))
                saved = self._recipe_signature(
                    meta.get("pipeline_params") or {}, meta.get("view_geom"))
                if saved != sig:
                    return None
                key = f"data_{int(ch_idx)}"
                if key not in npz.files:
                    return None
                data = np.asarray(npz[key], dtype=np.float32)
        except Exception as exc:
            # File di variante illeggibile/corrotto: si ricalcola, ma va segnalato.
            self._note_swallowed("lettura della variante salvata", exc)
            return None
        if shape is not None and tuple(data.shape) != tuple(shape):
            return None
        return data

    def _variant_matches_current(self, params: dict) -> bool:
        if not self._profiles:
            return False
        sig = self._recipe_signature(params, self._view_geom_sig())
        for prof in self._profiles:
            fp = self._variant_npz_for(prof)
            if not fp:
                return False
            try:
                with np.load(fp, allow_pickle=False) as npz:
                    meta = json.loads(str(npz["meta"]))
                if self._recipe_signature(
                        meta.get("pipeline_params") or {}, meta.get("view_geom")) != sig:
                    return False
            except Exception:
                return False
        return True

    def _save_working_variant(self) -> bool:
        """Scrive/sovrascrive l'unica variante di lavoro (nome fisso) con la
        pipeline corrente. Non crea mai varianti nuove e non altera la catena
        di provenienza delle varianti salvate dall'utente."""
        parent_ctx = getattr(self, "_variant_parent_ctx", None)
        try:
            return self._save_processed_variant(
                self.WORKING_VARIANT_NAME, quiet=True, overwrite=True
            ) is not None
        except Exception as exc:
            print(f"[GPR viewer][WARN] variante di lavoro non salvata: {exc}")
            return False
        finally:
            self._variant_parent_ctx = parent_ctx

    def _confirm_save_before_close(self) -> bool:
        """Chiusura senza domande. Il processo (parametri) e' salvato in
        automatico in project.json; i radargrammi filtrati vengono scritti
        nell'unica variante di lavoro (sempre sovrascritta). Le altre varianti
        si creano solo su richiesta esplicita. Non annulla mai la chiusura."""
        try:
            self._flush_recipe_autosave()
        except Exception:
            pass
        if not self._profiles or self._raw_data is None:
            return True
        try:
            params = self._current_processing_params()
            enabled = [k for k, v in params.items()
                       if isinstance(v, bool) and v and k != "manual_gain"]
            if enabled and not self._variant_matches_current(params):
                self._save_working_variant()
        except Exception as exc:
            print(f"[GPR viewer][WARN] aggiornamento variante di lavoro: {exc}")
        return True

    def _save_processed_variant(self, name=None, quiet: bool = False, overwrite: bool = False):
        # triggered(bool) passa un bool come primo argomento: accetta solo str
        name = name if isinstance(name, str) else None
        """Applica la pipeline corrente a TUTTI i profili/canali dal grezzo e
        salva i radargrammi processati come .npz in processed/<nome variante>.

        A differenza di _apply_processing (solo RAM/anteprima), questo scrive
        davvero i dati filtrati su disco, cosi' da poterli riusare per le
        time-slice e confrontare varianti diverse.
        """
        if getattr(self, "_variant_job", None) is not None:
            QMessageBox.information(
                self, "Salva variante", "Un salvataggio di variante e' gia' in corso."
            )
            return
        if not self._profiles:
            QMessageBox.information(
                self, "Salva variante", "Importa almeno un profilo .ogpr prima."
            )
            return
        paths = self._ensure_gpr_processing_layout()
        if not paths:
            QMessageBox.warning(
                self, "Salva variante",
                "Nessun workspace disponibile. Reimporta scegliendo una cartella di lavoro.",
            )
            return

        if name is None:
            name, ok = QInputDialog.getText(
                self, "Salva variante processata",
                "Nome variante (es. bp_bg, solo_bg):",
            )
            if not ok:
                return
        variant = "".join(
            c if c.isalnum() or c in "_-" else "_" for c in str(name).strip()
        ).strip("_")
        if not variant:
            QMessageBox.warning(self, "Salva variante", "Nome variante non valido.")
            return

        params = self._current_processing_params()
        chain = params.get("chain_order")
        if not quiet:
            params = self._survey_time_zero_check(params)
            if params is None:
                return
        variant_dir = os.path.join(paths["processed"], variant)
        if (not overwrite) and os.path.isdir(variant_dir) and any(
            f.lower().endswith(".npz") for f in os.listdir(variant_dir)
        ):
            ans = QMessageBox.question(
                self, "Variante esistente",
                f"La variante '{variant}' esiste gia'. Sovrascrivere i file processati?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if ans != QMessageBox.Yes:
                return
        os.makedirs(variant_dir, exist_ok=True)

        n = len(self._profiles)
        max_ch = max((int(getattr(p_, "n_channels", 0) or 0) for p_ in self._profiles), default=0)
        bg_refs = {ci: self._grid_bg_reference(params, ci) for ci in range(max_ch)}

        # Trim/flip di ogni canale: leggono i controlli Qt, quindi si calcolano qui
        # (thread della GUI) e il worker riceve solo indici gia' pronti.
        view_geom = self._view_geom_sig()
        plan = []
        for prof in self._profiles:
            per_channel = []
            for ci in range(int(getattr(prof, "n_channels", 0) or 0)):
                n_cols = int(np.asarray(prof.channel(ci).data).shape[1])
                tidx = self._trace_indices_for(n_cols)
                trimmed = not (tidx.size == n_cols and np.array_equal(tidx, np.arange(n_cols)))
                per_channel.append((tidx, trimmed))
            plan.append(per_channel)

        job = self._run_variant_channels(variant_dir, params, chain, view_geom, bg_refs, plan)
        files = job["files"]
        errors = job["errors"]
        cancelled = bool(job["cancelled"])

        if files:
            try:
                write_variant_manifest(variant_dir, variant, params, chain, files)
            except Exception:
                pass
            try:
                from . import variant_provenance as vp
                parent = getattr(self, "_variant_parent_ctx", None)
                if parent == variant:
                    parent = None
                vp.set_entry(
                    paths["processed"], variant,
                    parent=parent,
                    chain=chain or params.get("chain_order"),
                    step=self._last_enabled_step(params, chain),
                    n_profiles=len(files),
                )
                # la prossima variante salvata discende da questa
                self._variant_parent_ctx = variant
            except Exception:
                pass
            # Se l'utente ha impostato un apice Hyperbola in questa sessione, la
            # velocita'/RDP corrente non si perde piu' alla chiusura del viewer:
            # diventa il default preimpostato della variante appena salvata (letto
            # da RadargramDock e da questo stesso pannello - vedi gpr_velocity_store.py).
            try:
                if self._is_qt_alive(getattr(self, "_spin_hyperbola_rdp", None)) and self._hyper_apex_x is not None:
                    rdp = float(self._spin_hyperbola_rdp.value())
                    v_mns = self._rdp_to_velocity_m_s(rdp) * 1e-9
                    write_velocity_override(variant_dir, velocity_m_ns=v_mns, rdp=rdp, source="hyperbola_fit")
            except Exception:
                pass
            self._refresh_slice_variant_choices()
        how = (f"{job['workers']} calcoli in parallelo" if job["mode"] == "processi" else "in serie")
        took = f"{_fmt_duration(job['elapsed'])}, {how}"
        note = f"\n\n{job['note']}." if job.get("note") else ""
        self._safe_set_text(
            self._lbl_status,
            f"Variante '{variant}': {len(files)}/{n} profili"
            + (" (annullata)" if cancelled else "") + f" - {took}."
            + (f" {job['note']}." if job.get("note") else ""),
        )
        if errors:
            QMessageBox.warning(
                self, "Salva variante",
                ("Annullato. " if cancelled else "")
                + f"Salvati {len(files)}/{n} profili.\nErrori:\n" + "\n".join(errors[:8]) + note,
            )
        elif cancelled:
            if not quiet:
                QMessageBox.information(
                    self, "Salva variante",
                    f"Salvataggio annullato: {len(files)}/{n} profili scritti in\n{variant_dir}",
                )
        elif files and quiet:
            pass
        elif files:
            QMessageBox.information(
                self, "Salva variante",
                f"Variante '{variant}' salvata: {len(files)} profili in\n{variant_dir}\n\n"
                f"Tempo: {took}.{note}",
            )
        else:
            QMessageBox.warning(self, "Salva variante", "Nessun profilo salvato.")
        if files:
            self._active_variant_dir = variant_dir
        try:
            self._save_gpr_processing_project_meta(
                last_processing=self._path_to_gpr_processing_store(variant_dir)
            )
        except Exception:
            pass
        return variant if files else None

    def _run_variant_channels(self, variant_dir, params, chain, view_geom, bg_refs, plan) -> dict:
        """Calcola e scrive i profili della variante fuori dal thread della GUI.

        I canali girano in parallelo in processi separati (vedi
        gpr_variant_runner); la GUI resta libera e la finestra di avanzamento
        permette di annullare tra un canale e l'altro. La funzione resta
        SINCRONA per chi la chiama (chiusura del viewer, time-slice dal grezzo):
        aspetta con un event loop locale, senza bloccare la GUI.

        Ritorna: files (nell'ordine dei profili), errors (righe "profilo: motivo"),
        cancelled, mode, workers, note, elapsed.
        """
        profiles = list(self._profiles)
        counts = [len(per_channel) for per_channel in plan]
        total = int(sum(counts))
        topo = bool(params.get("kirchhoff") and params.get("kirchhoff_topo"))
        files_by_idx: dict = {}

        def make_payload(pi, ci):
            prof = profiles[pi]
            tidx, trimmed = plan[pi][ci]
            raw = np.asarray(prof.channel(ci).data, dtype=np.float32)
            if trimmed:
                raw = raw[:, tidx]
            # TVG: ogni profilo con la propria curva.
            p_ch = resolve_profile_params(params, profile_gain_key(prof))
            if topo:
                elev = self._profile_channel_elevation(prof, ci)
                try:
                    if trimmed and elev is not None and len(elev) > int(np.max(tidx)):
                        elev = np.asarray(elev)[tidx]
                except Exception:
                    pass
                p_ch["kirchhoff_elev_m"] = elev
            return (raw, p_ch, prof.dt_ns, bg_refs.get(ci), chain)

        def on_profile_done(pi, proc_by_ch):
            idx_by_ch = [tidx if trimmed else None for tidx, trimmed in plan[pi]]
            files_by_idx[pi] = save_processed_profile_npz(
                variant_dir, profiles[pi], proc_by_ch,
                resolve_profile_params(params, profile_gain_key(profiles[pi])), chain,
                view_geom=view_geom,
                trace_indices_by_channel=(idx_by_ch if any(t for _i, t in plan[pi]) else None),
            )

        dlg = _VariantProgressDialog(self, total)
        worker = _VariantSaveWorker(counts, make_payload, on_profile_done)
        thread = QThread(self)
        worker.moveToThread(thread)
        loop = QEventLoop(self)
        self._variant_job = {
            "dlg": dlg, "loop": loop, "t0": time.perf_counter(),
            "n_profiles": sum(1 for c in counts if c > 0), "result": None,
        }
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_variant_job_progress)
        worker.finished.connect(self._on_variant_job_finished)
        dlg.cancel_requested.connect(lambda: worker.cancel())
        try:
            thread.start()
            dlg.show()
            loop.exec_()
        finally:
            result = (self._variant_job or {}).get("result")
            self._variant_job = None
            thread.quit()
            thread.wait(10000)
            dlg.finish()
            dlg.deleteLater()
            worker.deleteLater()
            thread.deleteLater()

        result = result or {"errors": {}, "cancelled": True, "mode": "seriale", "workers": 1,
                            "note": "", "elapsed": 0.0}
        errors = []
        for pi in sorted(result.get("errors", {})):
            name = os.path.basename(str(getattr(profiles[pi], "path", "?")))
            errors.append(f"{name}: {result['errors'][pi]}")
        if result.get("fatal"):
            errors.append(f"errore imprevisto: {result['fatal']}")
        return {
            "files": [files_by_idx[i] for i in sorted(files_by_idx)],
            "errors": errors,
            "cancelled": bool(result.get("cancelled")),
            "mode": result.get("mode", "seriale"),
            "workers": int(result.get("workers", 1) or 1),
            "note": str(result.get("note", "") or ""),
            "elapsed": float(result.get("elapsed", 0.0) or 0.0),
        }

    def _on_variant_job_progress(self, done: int, total: int, saved: int) -> None:
        st = getattr(self, "_variant_job", None)
        if not st:
            return
        elapsed = time.perf_counter() - st["t0"]
        text = (f"Canali elaborati: {done}/{total}  -  profili scritti: {saved}/{st['n_profiles']}\n"
                f"Trascorso: {_fmt_duration(elapsed)}")
        if 0 < done < total:
            text += f"  -  rimanente stimato: circa {_fmt_duration(elapsed / done * (total - done))}"
        st["dlg"].set_progress(done, text)

    def _on_variant_job_finished(self, result) -> None:
        st = getattr(self, "_variant_job", None)
        if not st:
            return
        st["result"] = result
        st["loop"].quit()

    def _survey_time_zero_check(self, params: dict):
        """Se il time-zero e' attivo confronta la superficie rilevata tra i profili.

        Ritorna i params (eventualmente con t0 uniforme) o None se l'utente annulla."""
        if not params.get("timezero") or str(params.get("tz_mode")) == "fixed":
            return params
        try:
            from .gpr_processing import track_ground_surface, estimate_edge_skip, dewow as _dewow
            picks = []
            for prof in self._profiles:
                ci = min(int(getattr(prof, "n_channels", 1) or 1) - 1, max(0, int(getattr(self, "_ch_idx", 0) or 0)))
                d = np.asarray(prof.channel(ci).data, dtype=np.float32)
                _skip = estimate_edge_skip(d)
                if params.get("dewow"):
                    d = _dewow(d, window=int(params.get("dewow_win", 16)))
                picks.append(int(np.median(track_ground_surface(d, edge_skip=_skip))))
        except Exception:
            return params
        if len(picks) < 2 or (max(picks) - min(picks)) <= 12:
            return params
        groups: dict[int, int] = {}
        for v in picks:
            key = int(round(v / 6.0) * 6)
            groups[key] = groups.get(key, 0) + 1
        desc = ", ".join(f"~{k} campioni: {c} profili" for k, c in sorted(groups.items()))
        med = int(np.median(picks))
        box = QMessageBox(self)
        box.setWindowTitle("Time-zero: superficie diversa tra profili")
        box.setText(
            "La superficie (onda diretta) rilevata cade a campioni diversi nei profili:\n"
            f"{desc}.\n\n"
            "Se l'altezza dell'antenna e' costante e il suolo e' alla stessa quota, "
            "il t0 per-profilo puo' disallineare le time-slice tra linee.\n\n"
            f"Per-profilo: ogni linea usa il proprio t0 (default).\nUniforme: t0 = {med} campioni per tutte."
        )
        b_per = box.addButton("Per-profilo", QMessageBox.AcceptRole)
        b_uni = box.addButton(f"Uniforme ({med})", QMessageBox.ActionRole)
        box.addButton("Annulla", QMessageBox.RejectRole)
        box.exec_()
        clicked = box.clickedButton()
        if clicked is b_uni:
            out = dict(params)
            out["tz_mode"] = "fixed"
            out["tz_fixed_sample"] = int(max(0, med - int(params.get("tz_backup_nsamp", 3))))
            return out
        if clicked is b_per:
            return params
        return None

    def _blend_multifreq_to_variant(self):
        """Fonde antenne dual/triple frequenza in un unico radargramma
        (somma pesata + allineamento tempo-zero) e salva il risultato come
        variante a 1 canale.

        Due sorgenti:
          * i canali del profilo corrente (dati multicanale in un solo file);
          * piu' profili caricati (antenna dual-freq esportata come file
            separati, es. L1_LF + L1_HF): li si seleziona e vengono fusi in
            un unico profilo di uscita.
        La frequenza nominale dell'antenna NON serve per la fusione: conta solo
        che le sorgenti siano lo stesso tracciato.
        """
        if not self._profiles:
            QMessageBox.information(self, "Combina multi-frequenza",
                                   "Importa almeno un profilo prima.")
            return
        n_ch = int(getattr(self._profiles[self._prof_idx], "n_channels", 0) or 0)
        n_prof = len(self._profiles)
        if n_ch < 2 and n_prof < 2:
            QMessageBox.information(
                self, "Combina multi-frequenza",
                "Serve un profilo con >=2 canali, oppure >=2 profili caricati\n"
                "(un file per frequenza dello stesso tracciato).",
            )
            return
        paths = self._ensure_gpr_processing_layout()
        if not paths:
            QMessageBox.warning(self, "Combina multi-frequenza",
                                "Nessun workspace disponibile.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Combina multi-frequenza")
        outer = QVBoxLayout(dlg)
        cb_source = QComboBox()
        if n_ch >= 2:
            cb_source.addItem(f"Canali del profilo corrente ({n_ch})", "channels")
        if n_prof >= 2:
            cb_source.addItem(f"Profili caricati ({n_prof})", "profiles")
        form_top = QFormLayout()
        form_top.addRow("Sorgente:", cb_source)
        outer.addLayout(form_top)
        outer.addWidget(QLabel(
            "Seleziona SOLO sorgenti dello stesso tracciato.\n"
            "Peso e shift tempo-zero (campioni) per sorgente:"
        ))

        rows_box = QWidget()
        rows_lay = QVBoxLayout(rows_box)
        rows_lay.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(rows_box)
        cb_target = QComboBox()
        le_name = QLineEdit("blend")
        form_bot = QFormLayout()
        form_bot.addRow("Sorgente target (asse tempi/geometria):", cb_target)
        form_bot.addRow("Nome variante:", le_name)
        outer.addLayout(form_bot)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        outer.addWidget(bb)

        state = {"rows": []}

        def _rebuild_rows():
            for i in reversed(range(rows_lay.count())):
                it = rows_lay.takeAt(i)
                if it.widget():
                    it.widget().setParent(None)
            state["rows"] = []
            cb_target.clear()
            src = str(cb_source.currentData() or "channels")
            if src == "channels":
                labels = [f"Canale {ci}" for ci in range(n_ch)]
            else:
                labels = [
                    os.path.basename(getattr(p, "path", f"prof {i}") or f"prof {i}")
                    for i, p in enumerate(self._profiles)
                ]
            for k, lab in enumerate(labels):
                chk = QCheckBox(lab)
                chk.setChecked(src == "channels" or k < 2)
                ws = QDoubleSpinBox(); ws.setRange(0.0, 10.0); ws.setSingleStep(0.05)
                ws.setDecimals(3); ws.setValue(round(1.0 / max(2, len(labels)), 3))
                ss = QSpinBox(); ss.setRange(-8192, 8192); ss.setValue(0)
                rw = QWidget(); rl = QHBoxLayout(rw); rl.setContentsMargins(0, 0, 0, 0)
                rl.addWidget(chk, 1)
                rl.addWidget(QLabel("peso")); rl.addWidget(ws)
                rl.addWidget(QLabel("shift t0")); rl.addWidget(ss)
                rows_lay.addWidget(rw)
                state["rows"].append((k, lab, chk, ws, ss))
                cb_target.addItem(lab, k)

        cb_source.currentIndexChanged.connect(lambda _i: _rebuild_rows())
        _rebuild_rows()
        if dlg.exec_() != QDialog.Accepted:
            return

        src = str(cb_source.currentData() or "channels")
        chosen = [(k, w.value(), s.value()) for (k, _l, c, w, s) in state["rows"] if c.isChecked()]
        if len(chosen) < 2:
            QMessageBox.warning(self, "Combina multi-frequenza",
                                "Seleziona almeno 2 sorgenti.")
            return
        idxs = [k for (k, _w, _s) in chosen]
        weights = [float(w) for (_k, w, _s) in chosen]
        shifts = [int(s) for (_k, _w, s) in chosen]
        target_k = int(cb_target.currentData() or idxs[0])
        try:
            target_pos = idxs.index(target_k)
        except ValueError:
            target_pos = 0
        variant = "".join(
            c if c.isalnum() or c in "_-" else "_" for c in str(le_name.text()).strip()
        ).strip("_") or "blend"
        variant_dir = os.path.join(paths["processed"], variant)
        os.makedirs(variant_dir, exist_ok=True)
        meta = {"blend": {"source": src, "indices": idxs, "weights": weights,
                          "t0_shift_samples": shifts, "target": target_k}}

        files, errors = [], []
        try:
            if src == "channels":
                prof = self._profiles[self._prof_idx]
                arrs = [np.asarray(prof.channel(ci).data, dtype=np.float32) for ci in idxs]
                dts = [float(getattr(prof, "dt_ns", 0.1) or 0.1)] * len(arrs)
                blend = multifreq_blend(arrs, weights, dts, shifts, target=target_pos)
                # scrivo un profilo mono-canale usando la geometria del canale target
                fp = save_processed_profile_npz(variant_dir, prof, [blend], meta, [])
                files.append(fp)
            else:
                sel_profs = [self._profiles[k] for k in idxs]
                arrs = [np.asarray(p.channel(0).data, dtype=np.float32) for p in sel_profs]
                dts = [float(getattr(p, "dt_ns", 0.1) or 0.1) for p in sel_profs]
                blend = multifreq_blend(arrs, weights, dts, shifts, target=target_pos)
                fp = save_processed_profile_npz(
                    variant_dir, sel_profs[target_pos], [blend], meta, []
                )
                files.append(fp)
        except Exception as e:
            errors.append(str(e))

        if files:
            try:
                write_variant_manifest(variant_dir, variant, meta, [], files)
            except Exception:
                pass
            QMessageBox.information(
                self, "Combina multi-frequenza",
                f"Variante '{variant}' salvata ({len(files)} radargramma/i).\n"
                "Aprila dalla scheda 'Varianti' per generare le time-slice.",
            )
            try:
                self._refresh_variants_list()
            except Exception:
                pass
        else:
            QMessageBox.warning(self, "Combina multi-frequenza",
                                "Fusione non riuscita:\n" + "\n".join(errors[:5]))

    # ------------------------------------------------------------------
    # Pannello "Varianti" — elenco delle varianti processate su disco
    # ------------------------------------------------------------------
    def _build_variants_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)

        lay.addWidget(QLabel(
            "Varianti di radargrammi processati in Processing/processed/.\n"
            "L'albero segue la provenienza (variant_graph.json): ogni variante\n"
            "e' figlia di quella da cui e' stata derivata. Tasto destro per le azioni."
        ))
        self._tree_variants = QTreeWidget()
        self._tree_variants.setHeaderLabels(["Variante", "Profili", "Ultimo step"])
        self._tree_variants.setAlternatingRowColors(True)
        self._tree_variants.setContextMenuPolicy(Qt.CustomContextMenu)
        self._tree_variants.customContextMenuRequested.connect(self._on_variants_context_menu)
        self._tree_variants.itemSelectionChanged.connect(self._on_variant_selection_changed)
        # Doppio click = attiva la variante (carica la pipeline); la
        # generazione time-slice resta esplicita (pulsante / menu).
        self._tree_variants.itemDoubleClicked.connect(
            lambda _it, _c: self._view_selected_variant()
        )
        lay.addWidget(self._tree_variants, 1)

        row = QHBoxLayout()
        self._btn_variants_view = QPushButton("Carica (visualizza)")
        self._btn_variants_view.setEnabled(False)
        self._btn_variants_view.setToolTip(
            "Mostra nel viewer i radargrammi di questa variante (letti dal disco)\n"
            "e ne carica la pipeline nei controlli. Doppio click = stessa azione."
        )
        self._btn_variants_view.clicked.connect(self._view_selected_variant)
        self._btn_variants_refresh = QPushButton("Aggiorna")
        self._btn_variants_refresh.clicked.connect(self._refresh_variants_list)
        self._btn_variants_newfrom = QPushButton("Nuova variante da qui")
        self._btn_variants_newfrom.setEnabled(False)
        self._btn_variants_newfrom.setToolTip(
            "Carica la pipeline di questa variante nei controlli di processing\n"
            "e la registra come genitore della prossima variante salvata."
        )
        self._btn_variants_newfrom.clicked.connect(self._new_variant_from_selected)
        self._btn_variants_slice = QPushButton("Genera time-slice")
        self._btn_variants_slice.setEnabled(False)
        self._btn_variants_slice.setToolTip(
            "Genera le time-slice dai .npz della variante selezionata,\n"
            "senza riapplicare la pipeline (i dati sono gia' filtrati)."
        )
        self._btn_variants_slice.clicked.connect(self._slice_selected_variant)
        self._btn_variants_open = QPushButton("Apri cartella")
        self._btn_variants_open.setEnabled(False)
        self._btn_variants_open.clicked.connect(self._open_selected_variant_folder)
        row.addWidget(self._btn_variants_view)
        row.addWidget(self._btn_variants_refresh)
        row.addWidget(self._btn_variants_newfrom)
        row.addWidget(self._btn_variants_slice)
        row.addWidget(self._btn_variants_open)
        lay.addLayout(row)

        self._lbl_variant_detail = QLabel("—")
        self._lbl_variant_detail.setWordWrap(True)
        self._lbl_variant_detail.setStyleSheet("color: palette(mid);")
        lay.addWidget(self._lbl_variant_detail)
        return w

    def _refresh_variants_list(self):
        tree = getattr(self, "_tree_variants", None)
        if not self._is_qt_alive(tree):
            return
        tree.clear()
        if self._is_qt_alive(getattr(self, "_lbl_variant_detail", None)):
            self._lbl_variant_detail.setText("—")
        paths = self._gpr_processing_paths()
        processed = paths.get("processed", "") if paths else ""
        if not processed or not os.path.isdir(processed):
            tree.addTopLevelItem(QTreeWidgetItem(["(nessun progetto GPR attivo)"]))
            return
        try:
            dirs = list_variants(processed)
        except Exception:
            dirs = []
        if not dirs:
            tree.addTopLevelItem(QTreeWidgetItem(["(nessuna variante salvata)"]))
            return

        try:
            from . import variant_provenance as vp
        except Exception:
            vp = None
        entries = vp.load(processed) if vp is not None else {}

        by_name = {}
        for d in dirs:
            man = read_variant_manifest(d) or {}
            nm = str(man.get("name") or os.path.basename(os.path.normpath(d)))
            chain = [s for s in (man.get("chain_order") or []) if s]
            n_prof = int(man.get("n_profiles") or len(man.get("files") or []))
            by_name[nm] = {
                "dir": os.path.normpath(d),
                "chain": chain,
                "n_prof": n_prof,
                "created": man.get("created", ""),
            }

        present = list(by_name.keys())
        if vp is not None:
            forest = vp.build_forest(entries, present)
        else:
            forest = [{"name": n, "meta": {}, "children": []} for n in present]

        def _add(node, parent_item):
            nm = node["name"]
            info = by_name.get(nm, {})
            chain = info.get("chain") or []
            last_step = (entries.get(nm, {}) or {}).get("step") or (chain[-1] if chain else "")
            it = QTreeWidgetItem([
                nm, str(info.get("n_prof", "")), str(last_step or ""),
            ])
            it.setData(0, Qt.UserRole, info.get("dir", ""))
            it.setData(0, Qt.UserRole + 1, {"chain": chain, "created": info.get("created", "")})
            it.setToolTip(0, f"{info.get('dir','')}\ncatena: "
                             f"{', '.join(chain) if chain else '(non registrata)'}")
            if parent_item is None:
                self._tree_variants.addTopLevelItem(it)
            else:
                parent_item.addChild(it)
            for ch in node.get("children") or []:
                _add(ch, it)

        for root in forest:
            _add(root, None)
        tree.expandAll()
        self._refresh_slice_variant_choices()

    def _selected_variant_dir(self) -> str:
        tree = getattr(self, "_tree_variants", None)
        if not self._is_qt_alive(tree):
            return ""
        it = tree.currentItem()
        if it is None:
            return ""
        d = it.data(0, Qt.UserRole)
        return str(d) if d else ""

    def _selected_variant_name(self) -> str:
        tree = getattr(self, "_tree_variants", None)
        if not self._is_qt_alive(tree):
            return ""
        it = tree.currentItem()
        return str(it.text(0)) if it is not None else ""

    def _on_variant_selection_changed(self):
        d = self._selected_variant_dir()
        has = bool(d)
        for b in ("_btn_variants_slice", "_btn_variants_open", "_btn_variants_newfrom",
                  "_btn_variants_view"):
            w = getattr(self, b, None)
            if self._is_qt_alive(w):
                w.setEnabled(has)
        tree = getattr(self, "_tree_variants", None)
        it = tree.currentItem() if (has and self._is_qt_alive(tree)) else None
        meta = it.data(0, Qt.UserRole + 1) if it is not None else None
        if isinstance(meta, dict):
            chain = ", ".join(meta.get("chain") or []) or "(catena non registrata)"
            self._lbl_variant_detail.setText(f"Catena: {chain}\n{d}")
        else:
            self._lbl_variant_detail.setText("—")

    def _on_variants_context_menu(self, pos):
        tree = getattr(self, "_tree_variants", None)
        if not self._is_qt_alive(tree):
            return
        it = tree.itemAt(pos)
        if it is None or not it.data(0, Qt.UserRole):
            return
        tree.setCurrentItem(it)
        menu = QMenu(tree)
        act_view = menu.addAction("Carica (visualizza radargrammi)")
        act_new = menu.addAction("Nuova variante da qui")
        act_slice = menu.addAction("Genera time-slice")
        act_open = menu.addAction("Apri cartella")
        menu.addSeparator()
        act_ren = menu.addAction("Rinomina variante…")
        act_del = menu.addAction("Elimina variante…")
        chosen = menu.exec_(tree.viewport().mapToGlobal(pos))
        if chosen == act_ren:
            self._rename_selected_variant()
        elif chosen == act_view:
            self._view_selected_variant()
        elif chosen == act_new:
            self._new_variant_from_selected()
        elif chosen == act_slice:
            self._slice_selected_variant()
        elif chosen == act_open:
            self._open_selected_variant_folder()
        elif chosen == act_del:
            self._delete_selected_variant()

    def _view_selected_variant(self):
        """Attiva la variante selezionata per la visualizzazione: i radargrammi
        vengono letti dai suoi .npz (senza rieseguire la pipeline) e i suoi
        parametri compaiono nei controlli. Non cambia la provenienza."""
        d = self._selected_variant_dir()
        name = self._selected_variant_name()
        if not d or not os.path.isdir(d):
            return
        man = read_variant_manifest(d) or {}
        params = man.get("pipeline_params") if isinstance(man.get("pipeline_params"), dict) else {}
        if not params:
            QMessageBox.warning(self, "Carica variante",
                                "La variante non registra i parametri della pipeline "
                                "(variant.json privo di 'pipeline_params').")
            return
        chain = [s for s in (man.get("chain_order") or params.get("chain_order") or []) if s]
        # prima la variante attiva, cosi' il riprocessamento trova i .npz
        self._active_variant_dir = d
        try:
            self._apply_pipeline_params_dict(dict(params), chain_order=chain)
        except Exception as exc:
            QMessageBox.critical(self, "Carica variante",
                                 f"Impossibile caricare la variante:\n{exc}")
            return
        self._safe_set_text(
            self._lbl_status,
            f"Variante '{name}' caricata per la visualizzazione.",
        )

    def _new_variant_from_selected(self):
        d = self._selected_variant_dir()
        name = self._selected_variant_name()
        if not d or not os.path.isdir(d):
            return
        man = read_variant_manifest(d) or {}
        params = man.get("pipeline_params") if isinstance(man.get("pipeline_params"), dict) else {}
        if not params:
            QMessageBox.warning(self, "Nuova variante da qui",
                                "La variante non registra i parametri della pipeline "
                                "(variant.json privo di 'pipeline_params').")
            return
        chain = [s for s in (man.get("chain_order") or params.get("chain_order") or []) if s]
        try:
            self._apply_pipeline_params_dict(dict(params), chain_order=chain)
        except Exception as exc:
            QMessageBox.critical(self, "Nuova variante da qui",
                                 f"Impossibile caricare la pipeline:\n{exc}")
            return
        self._variant_parent_ctx = str(name)
        self._safe_set_text(
            self._lbl_status,
            f"Pipeline della variante '{name}' caricata. La prossima variante "
            f"salvata sara' registrata come sua figlia.",
        )

    def _rename_selected_variant(self):
        d = self._selected_variant_dir()
        old = os.path.basename(os.path.normpath(d)) if d else ""
        if not d or not os.path.isdir(d):
            return
        if getattr(self, "_variant_job", None) is not None:
            QMessageBox.information(
                self, "Rinomina variante",
                "Un salvataggio di variante e' in corso: attendi la fine.",
            )
            return
        new, ok = QInputDialog.getText(
            self, "Rinomina variante", f"Nuovo nome per la variante '{old}':", text=old
        )
        new = str(new or "").strip()
        if not ok or not new or new == old:
            return
        processed = os.path.dirname(os.path.normpath(d))
        from . import variant_cascade
        err = variant_cascade.rename_variant(processed, old, new)
        if err:
            QMessageBox.warning(self, "Rinomina variante", err)
            return
        new_dir = os.path.join(processed, new)
        active = str(getattr(self, "_active_variant_dir", "") or "")
        if active and os.path.normcase(os.path.normpath(active)) == os.path.normcase(os.path.normpath(d)):
            self._active_variant_dir = new_dir
        if getattr(self, "_variant_parent_ctx", None) == old:
            self._variant_parent_ctx = new
        try:
            self._flush_recipe_autosave()
        except Exception:
            pass
        self._refresh_variants_list()
        plugin = getattr(self, "plugin", None)
        dock = getattr(plugin, "_radargram_dock", None) if plugin is not None else None
        if dock is not None:
            try:
                dock.refresh_variants()
            except Exception:
                pass
        self._safe_set_text(
            self._lbl_status,
            f"Variante '{old}' rinominata in '{new}' (le time-slice gia' generate "
            f"restano nella loro cartella).",
        )

    def _delete_selected_variant(self):
        d = self._selected_variant_dir()
        name = self._selected_variant_name()
        if not d or not os.path.isdir(d):
            return
        paths = self._gpr_processing_paths()
        processed = paths.get("processed", "") if paths else ""
        timeslices_root = paths.get("timeslices", "") if paths else ""
        project_root = self._active_project_root()

        from . import variant_cascade
        plan = variant_cascade.plan_cascade_delete(processed, timeslices_root, project_root, name)
        ans = QMessageBox.question(
            self, "Elimina variante", variant_cascade.describe_plan(plan),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if ans != QMessageBox.Yes:
            return

        try:
            from . import layer_source_utils
            layer_source_utils.remove_layers_for_paths(plan.get("layer_paths"))
        except Exception:
            pass

        result = variant_cascade.execute_cascade_delete(processed, timeslices_root, project_root, plan)
        if result.get("warnings"):
            QMessageBox.warning(self, "Elimina variante", "\n".join(result["warnings"]))
        self._refresh_variants_list()

    def _slice_selected_variant(self):
        d = self._selected_variant_dir()
        if d:
            self._slice_from_variant_folder(d)

    def _open_selected_variant_folder(self):
        d = self._selected_variant_dir()
        if not d:
            return
        try:
            from qgis.PyQt.QtGui import QDesktopServices
            from qgis.PyQt.QtCore import QUrl
            QDesktopServices.openUrl(QUrl.fromLocalFile(d))
        except Exception:
            QMessageBox.information(self, "Varianti", d)

    def _build_slices_from_variant(self):
        """Genera time-slice a partire da una variante di radargrammi gia'
        processati (.npz), senza riapplicare la pipeline. Sceglie la cartella
        con un dialog; per la selezione dal pannello Varianti vedi
        `_slice_from_variant_folder`."""
        paths = self._gpr_processing_paths()
        start = paths.get("processed", "") if paths else ""
        if not start or not os.path.isdir(start):
            start = (paths.get("base", "") if paths else "") or os.path.expanduser("~")
        folder = QFileDialog.getExistingDirectory(
            self, "Seleziona cartella variante processata (.npz)", start
        )
        if not folder:
            return
        self._slice_from_variant_folder(folder)

    def _variant_names_for_picker(self) -> list[str]:
        """Nomi delle varianti salvate (cartelle in processed/ con .npz)."""
        paths = self._gpr_processing_paths()
        processed = paths.get("processed", "") if paths else ""
        if not processed or not os.path.isdir(processed):
            return []
        out = []
        try:
            for d in list_variants(processed):
                man = read_variant_manifest(d) or {}
                nm = str(man.get("name") or os.path.basename(os.path.normpath(d))).strip()
                # solo varianti con radargrammi affettabili
                if nm and glob.glob(os.path.join(d, "*.npz")):
                    out.append(nm)
        except Exception:
            return []
        return sorted(set(out))

    def _refresh_slice_variant_choices(self) -> None:
        panel = getattr(self, "_slice_spec_panel", None)
        if panel is not None and self._is_qt_alive(panel) and hasattr(panel, "set_variant_choices"):
            try:
                panel.set_variant_choices(self._variant_names_for_picker())
            except Exception:
                pass

    def _variant_dir_for_name(self, name: str) -> str:
        name = str(name or "").strip()
        if not name or name.lower() == "raw":
            return ""
        paths = self._gpr_processing_paths()
        processed = paths.get("processed", "") if paths else ""
        if not processed:
            return ""
        d = os.path.join(processed, name)
        return d if os.path.isdir(d) and glob.glob(os.path.join(d, "*.npz")) else ""
