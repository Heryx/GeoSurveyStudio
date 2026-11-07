# -*- coding: utf-8 -*-
"""Radargramma del GPR Profile Viewer: disegno, cursore, A-scan, zoom, pick e sync col canvas QGIS.

Estratto da gpr_profile_viewer.py senza modifiche ai metodi."""

from __future__ import annotations

from .gpr_viewer_common import (
    Figure,
    HAS_MPL,
    QColor,
    QFileDialog,
    QInputDialog,
    QMessageBox,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsPointXY,
    QgsProject,
    QgsRubberBand,
    QgsWkbTypes,
    Qt,
    _dbg,
    manual_gain,
    normalize_display,
    np,
    os,
)


class _ImageLod:
    """Livello di dettaglio per radargrammi lunghi.

    matplotlib ricampiona l'INTERO array a ogni draw, anche con lo zoom: con
    decine di migliaia di tracce ogni zoom/scroll costa centinaia di ms.
    Qui l'immagine mostra solo la porzione visibile (piu' un margine), decimata
    a ~OVERSAMPLE x i pixel disponibili; i livelli decimati si calcolano una
    volta sola e si riusano finche' non cambia il dato.  Logica pura (senza Qt).
    """

    MIN_COLS = 8000      # sotto questa soglia l'immagine resta com'e' (nessuna differenza)
    OVERSAMPLE = 2.0     # colonne per pixel dopo la decimazione
    MARGIN = 0.5         # ritaglio piu' largo della vista: i piccoli pan non lo rifanno

    def __init__(self, src, x_length):
        self.src = src                      # (n_campioni, n_tracce) su griglia uniforme
        self.x_length = float(x_length)     # lunghezza in metri = estensione X dell'immagine
        self._levels = {}
        self.state = None                   # (fattore, colonna_iniziale, colonna_finale) mostrato

    @property
    def n_cols(self) -> int:
        return int(self.src.shape[1])

    @property
    def active(self) -> bool:
        return self.n_cols >= self.MIN_COLS and self.x_length > 0.0

    def level(self, f: int) -> np.ndarray:
        """Array decimato di un fattore f (media di f colonne; f=1 e' il dato originale)."""
        if f <= 1:
            return self.src
        lv = self._levels.get(f)
        if lv is None:
            n_s, n_t = self.src.shape
            n = (n_t // f) * f
            lv = self.src[:, :n].reshape(n_s, n // f, f).mean(axis=2, dtype=np.float32)
            if n < n_t:   # coda che non riempie un blocco: la sua media, non la si perde
                tail = self.src[:, n:].mean(axis=1, keepdims=True, dtype=np.float32)
                lv = np.concatenate([lv, tail], axis=1)
            self._levels[f] = lv
        return lv

    def plan(self, x0: float, x1: float, px: float):
        """(fattore, colonne_necessarie, colonne_da_caricare) per la vista [x0, x1].

        Le colonne sono nel livello scelto: (inizio, fine) con fine esclusa.
        """
        n_t, length = self.n_cols, self.x_length
        lo, hi = (x0, x1) if x0 <= x1 else (x1, x0)
        c_lo = int(np.clip(np.floor(lo * n_t / length), 0, n_t - 1))
        c_hi = int(np.clip(np.ceil(hi * n_t / length), c_lo + 1, n_t))
        width = c_hi - c_lo
        target = max(float(px), 200.0) * self.OVERSAMPLE
        f = 1
        if width > target:
            f = 2 ** int(np.floor(np.log2(width / target)))
        margin = int(width * self.MARGIN)
        m_lo, m_hi = max(0, c_lo - margin), min(n_t, c_hi + margin)
        n_lv = -(-n_t // f)
        need = (c_lo // f, min(n_lv, -(-c_hi // f)))
        want = (m_lo // f, min(n_lv, -(-m_hi // f)))
        return f, need, want

    def covers(self, f: int, need) -> bool:
        st = self.state
        return st is not None and st[0] == f and st[1] <= need[0] and st[2] >= need[1]

    def crop(self, f: int, a: int, b: int):
        """(sottoarray, (x_sinistra, x_destra)) del livello f, colonne [a, b)."""
        lv = self.level(f)
        b = min(int(b), lv.shape[1])
        a = min(int(a), b - 1)
        scale = self.x_length / self.n_cols
        return lv[:, a:b], (a * f * scale, min(b * f, self.n_cols) * scale)


class RadargramViewMixin:
    """Radargramma del GPR Profile Viewer: disegno, cursore, A-scan, zoom, pick e sync col canvas QGIS.

    Mixin di GprProfileViewer: i metodi usano `self` della finestra.
    """


    def _refresh_blit_artists(self) -> None:
        """Registra gli artisti che cambiano a ogni movimento del cursore."""
        blit = getattr(self, "_blit", None)
        if blit is None:
            return
        arts = [self._vline, self._hline]
        if self._wiggle_fil_line is not None and getattr(self, "_ax_wiggle", None) is not None:
            arts += [
                self._wiggle_raw_line,
                self._wiggle_fil_line,
                self._wiggle_depth_line,
                self._ax_wiggle.title,
            ]
        blit.set_artists(arts)

    def _blit_update(self) -> None:
        """Aggiorna cursore e A-scan senza ridisegnare il radargramma."""
        canvas = getattr(self, "_canvas_mpl", None)
        if not self._is_qt_alive(canvas):
            return
        blit = getattr(self, "_blit", None)
        try:
            if blit is None:
                canvas.draw_idle()
            else:
                blit.update()
        except Exception:
            self._safe_draw_idle(canvas)

    def _apply_wiggle_visibility_layout(self):
        if not HAS_MPL or not hasattr(self, "_ax") or not hasattr(self, "_ax_wiggle"):
            return
        if self._ax is None or self._ax_wiggle is None:
            return
        if self._ax_main_bounds_default is None or self._ax_wiggle_bounds_default is None:
            try:
                self._ax_main_bounds_default = list(self._ax.get_position().bounds)
                self._ax_wiggle_bounds_default = list(self._ax_wiggle.get_position().bounds)
            except Exception:
                return

        main = list(self._ax_main_bounds_default)
        wig = list(self._ax_wiggle_bounds_default)
        if bool(getattr(self, "_show_wiggle", True)):
            self._ax_wiggle.set_visible(True)
            self._ax.set_position(main)
            self._ax_wiggle.set_position(wig)
        else:
            self._ax_wiggle.set_visible(False)
            right = float(wig[0] + wig[2])
            self._ax.set_position([float(main[0]), float(main[1]), right - float(main[0]), float(main[3])])

    def _channel_trace_vector(self, values, n_t: int) -> np.ndarray:
        target = max(0, int(n_t))
        if target <= 0:
            return np.zeros(0, dtype=np.float64)
        arr = np.asarray(values if values is not None else [], dtype=np.float64)
        idx = self._trace_source_indices
        if (
            isinstance(idx, np.ndarray)
            and idx.ndim == 1
            and idx.size == target
            and arr.size > 0
        ):
            try:
                imin = int(np.nanmin(idx))
                imax = int(np.nanmax(idx))
            except Exception:
                imin, imax = 0, -1
            if imin >= 0 and imax < arr.size:
                try:
                    return np.asarray(arr[idx.astype(np.int64)], dtype=np.float64)
                except Exception:
                    pass
        if arr.size == target:
            return arr.astype(np.float64, copy=False)
        if arr.size <= 0:
            return np.full(target, np.nan, dtype=np.float64)
        if arr.size == 1:
            return np.full(target, float(arr[0]), dtype=np.float64)
        src = np.linspace(0.0, 1.0, arr.size, dtype=np.float64)
        dst = np.linspace(0.0, 1.0, target, dtype=np.float64)
        return np.interp(dst, src, arr).astype(np.float64)

    def _display_distance_axis(self, prof, n_traces: int) -> np.ndarray:
        n_t = max(1, int(n_traces))
        try:
            ch = prof.channel(self._ch_idx)
            distances = self._channel_trace_vector(getattr(ch, "distances", []), n_t)
        except Exception:
            distances = np.zeros(0, dtype=np.float64)
        if distances.size != n_t:
            if distances.size > 1:
                src = np.linspace(0.0, 1.0, distances.size, dtype=np.float64)
                dst = np.linspace(0.0, 1.0, n_t, dtype=np.float64)
                distances = np.interp(dst, src, distances).astype(np.float64)
            else:
                step = float(getattr(prof, "sampling_step_m", 0.1) or 0.1)
                step = max(step, 1e-6)
                distances = (np.arange(n_t, dtype=np.float64) * step).astype(np.float64)
        finite = np.isfinite(distances)
        if not finite.any():
            return np.linspace(0.0, float(max(1, n_t - 1)), n_t, dtype=np.float64)
        if not finite.all():
            idx = np.arange(distances.size, dtype=np.float64)
            ok = np.where(finite)[0].astype(np.float64)
            distances = np.interp(idx, ok, distances[finite]).astype(np.float64)
        d0 = float(distances[0]) if np.isfinite(distances[0]) else 0.0
        distances = distances - d0
        if distances.size > 1:
            # Enforce monotonic axis for noisy/raw cumulative distance vectors.
            distances = np.maximum.accumulate(distances)
        span = float(distances[-1] - distances[0]) if distances.size > 1 else 0.0
        if (not np.isfinite(span)) or span <= 1e-9:
            step = float(getattr(prof, "sampling_step_m", 0.1) or 0.1)
            step = max(step, 1e-6)
            distances = (np.arange(n_t, dtype=np.float64) * step).astype(np.float64)
        return distances

    def _draw_hyperbola_overlay_on_axis(self, ax, prof, dist_max: float, depth_max: float):
        if ax is None:
            return
        if not bool(getattr(self, "_chk_hyperbola", None) and self._chk_hyperbola.isChecked()):
            return
        if self._hyper_apex_x is None or self._hyper_apex_depth is None:
            return
        x0 = float(self._hyper_apex_x)
        z0_disp = float(self._hyper_apex_depth)
        if not (np.isfinite(x0) and np.isfinite(z0_disp) and np.isfinite(dist_max) and np.isfinite(depth_max)):
            return
        if z0_disp < 0 or z0_disp > depth_max:
            return

        v_ref = self._profile_reference_velocity(prof)
        rdp = float(self._spin_hyperbola_rdp.value())
        v_fit = self._rdp_to_velocity_m_s(rdp)
        s = self._hyperbola_txrx_offset()
        # Apex sample time from current displayed depth scale.
        t0 = (2.0 * z0_disp) / max(v_ref, 1e-9)
        x = np.linspace(0.0, float(dist_max), 600, dtype=np.float64)
        dx = x - x0
        # Bistatic model (s=0 -> monostatico), poi rimappato in profondita' display.
        z_disp = self._hyperbola_curve_depth(dx, t0, v_fit, v_ref, s)
        finite = np.isfinite(z_disp) & (z_disp >= 0.0) & (z_disp <= (depth_max * 1.02))
        if not finite.any():
            return
        ax.plot(x[finite], z_disp[finite], color="#ffd000", lw=1.5, alpha=0.95, zorder=7)
        ax.scatter([x0], [z0_disp], s=28, color="#ff8c00", edgecolor="#1f1f1f", linewidths=0.5, zorder=8)
        ax.text(
            x0,
            min(depth_max, z0_disp + 0.03 * max(depth_max, 1e-6)),
            f"RDP {rdp:.2f}",
            color="#ffd000",
            fontsize=8,
            ha="left",
            va="bottom",
            zorder=8,
        )

    def _draw_hyperbola_overlay(self, prof, dist_max: float, depth_max: float):
        self._draw_hyperbola_overlay_on_axis(self._ax, prof, dist_max, depth_max)

    def _build_depth_range_gain(self, n_samples: int) -> np.ndarray:
        n = max(1, int(n_samples))
        if not bool(getattr(self, "_chk_range_gain", None) and self._chk_range_gain.isChecked()):
            return np.ones((n, 1), dtype=np.float32)
        t = np.linspace(0.0, 1.0, n, dtype=np.float64)
        pts = self._sanitize_range_gain_breakpoints(self._range_gain_breakpoints)
        x = np.asarray(pts[:, 0], dtype=np.float64)
        y = np.asarray(pts[:, 1], dtype=np.float64)
        if str(getattr(self, "_range_gain_interp", "linear")) == "log":
            g = np.exp(np.interp(t, x, np.log(np.maximum(y, 1e-6))))
        else:
            g = np.interp(t, x, y)
        g = np.clip(g, 1e-6, 1e6).astype(np.float32)
        return g.reshape(-1, 1)

    def _compose_display_data(self) -> np.ndarray | None:
        if self._proc_data is None:
            return None
        proc = np.asarray(self._proc_data, dtype=np.float32)
        if proc.ndim != 2 or proc.size <= 0:
            return None

        # Il range-gain e' gia' dentro _proc_data solo se la pipeline e' stata
        # eseguita con pre_agc_gain E il risultato e' stato davvero usato
        # (self._proc_pre_agc, impostato dove si assegna _proc_data). Non si
        # deduce dai checkbox: dopo un fallback sul dato grezzo, o un toggle
        # successivo, widget e dato non coincidono piu'.
        pre_agc_in_proc = bool(self._proc_pre_agc)
        depth_gain = None
        if not pre_agc_in_proc:
            # Apply depth gain as display-only transform BEFORE normalize_display.
            depth_gain = self._build_depth_range_gain(int(proc.shape[0]))
        gain_key = None if depth_gain is None else np.asarray(depth_gain).tobytes()

        # La normalizzazione (percentile su tutto il radargramma) non dipende dal
        # gain scalare/manuale, applicati dopo: si riusa finche' non cambiano il
        # dato processato (identita' dell'array) o il range-gain.
        cache = self._display_base_cache
        if cache is not None and cache[0] is self._proc_data and cache[1] == gain_key:
            base = cache[2]
        else:
            proc_disp = proc
            if depth_gain is not None:
                proc_disp = (proc * depth_gain).astype(np.float32, copy=False)
            base = normalize_display(proc_disp).astype(np.float32, copy=False)
            self._display_base_cache = (self._proc_data, gain_key, base)

        # Normalize to ±1 range first, then apply display gains on top.
        # The scalar gain and manual_gain are applied AFTER normalization so that
        # normalize_display does not cancel them.  Values that exceed ±1 after
        # the gains are intentionally saturated by the final clip — this is the
        # correct visual behaviour (deep samples clamp to full black/white).
        #
        # The burst zone is now attenuated by pre_agc_gain (TVG) before AGC,
        # so the amplitude is already uniform across the full profile here.
        # No special zeroing of the burst zone is needed.

        # Scalar display gain (brightness knob).
        gain = float(self._spin_gain.value()) if hasattr(self, "_spin_gain") else 1.0
        disp = (base * gain).astype(np.float32, copy=False)

        # Manual depth-varying gain — applied last so it is not undone by
        # normalize_display.  The pipeline no longer runs manual_gain; the
        # function is called directly here as a display-only transform.
        chk_mg = getattr(self, "_chk_manual_gain", None)
        if chk_mg is not None and bool(chk_mg.isChecked()):
            try:
                mg_type = str(
                    self._cb_manual_gain_type.currentData() or "linear"
                ) if getattr(self, "_cb_manual_gain_type", None) else "linear"
                mg_start = float(
                    self._spin_manual_gain_start.value()
                ) if getattr(self, "_spin_manual_gain_start", None) else 1.0
                mg_end = float(
                    self._spin_manual_gain_end.value()
                ) if getattr(self, "_spin_manual_gain_end", None) else 10.0
                mg_s0 = int(
                    self._spin_manual_gain_sample_start.value()
                ) if getattr(self, "_spin_manual_gain_sample_start", None) else 0
                mg_s1 = int(
                    self._spin_manual_gain_sample_end.value()
                ) if getattr(self, "_spin_manual_gain_sample_end", None) else 0
                disp = manual_gain(
                    disp,
                    gain_type=mg_type,
                    gain_start=mg_start,
                    gain_end=mg_end,
                    sample_start=mg_s0,
                    sample_end=mg_s1,
                ).astype(np.float32, copy=False)
            except Exception as exc:
                # Il manual gain non viene applicato: l'utente lo vede spento
                # senza spiegazione, quindi almeno lo si scrive nel log.
                self._note_swallowed("manual gain (visualizzazione)", exc)

        # `disp` e' sempre un array nuovo (mai `base`): clip in place.
        return np.clip(disp, -1.0, 1.0, out=disp).astype(np.float32, copy=False)

    def _apply_gain_only(self):
        if self._proc_data is None:
            if self._proc_thread_running():
                return  # il risultato in arrivo lo mostra _finish_processing: non bloccare la GUI
            self._apply_processing()
            return
        self._disp_data = self._compose_display_data()
        if self._disp_data is None:
            return
        gain = float(self._spin_gain.value())
        disp_arr = np.asarray(self._disp_data, dtype=np.float64)
        finite = np.isfinite(disp_arr)
        if finite.any():
            disp_min = float(np.nanmin(disp_arr))
            disp_max = float(np.nanmax(disp_arr))
        else:
            disp_min = float("nan")
            disp_max = float("nan")
        self._safe_set_text(
            self._lbl_status,
            f"proc: min={self._proc_data.min():.3f}  "
            f"max={self._proc_data.max():.3f}  "
            f"disp: min={disp_min:.3f} max={disp_max:.3f}  "
            f"gain={gain:.1f}x  "
            f"range_gain={'on' if self._chk_range_gain.isChecked() else 'off'}"
        )
        if not self._refresh_image_only():
            self._redraw()
        self._apply_pending_cursor()

    @staticmethod
    def _clamp_axis_limits(
        lim0: float, lim1: float, axis_min: float, axis_max: float, min_span_ratio: float = 1e-3
    ) -> tuple[float, float]:
        if not (np.isfinite(lim0) and np.isfinite(lim1)):
            return axis_min, axis_max
        span_total = max(axis_max - axis_min, 1e-12)
        forward = lim1 >= lim0
        lo = min(lim0, lim1)
        hi = max(lim0, lim1)
        min_span = span_total * min_span_ratio
        if (hi - lo) < min_span:
            c = 0.5 * (lo + hi)
            lo = c - 0.5 * min_span
            hi = c + 0.5 * min_span
        if lo < axis_min:
            shift = axis_min - lo
            lo += shift
            hi += shift
        if hi > axis_max:
            shift = hi - axis_max
            lo -= shift
            hi -= shift
        lo = max(axis_min, lo)
        hi = min(axis_max, hi)
        return (lo, hi) if forward else (hi, lo)

    def _reset_zoom(self):
        self._view_xlim = None
        self._view_ylim = None
        # Explicit reset should restore full extent, not the initial short-window view.
        self._profile_view_initialized = True
        self._slice_view_xlim = None
        self._slice_view_ylim = None
        self._redraw()

    def _axis_pixel_size(self) -> tuple[float, float]:
        fig_w_px = fig_h_px = None
        try:
            if self._canvas_mpl is not None:
                fig_w_px, fig_h_px = self._canvas_mpl.get_width_height()
        except Exception:
            fig_w_px = fig_h_px = None
        if not (fig_w_px and fig_h_px):
            fig_w_px = float(self._fig.get_figwidth() * 100.0)
            fig_h_px = float(self._fig.get_figheight() * 100.0)
        bbox = self._ax.get_position()
        ax_w_px = max(1.0, float(fig_w_px) * float(bbox.width))
        ax_h_px = max(1.0, float(fig_h_px) * float(bbox.height))
        return float(ax_w_px), float(ax_h_px)

    def _compute_initial_xview(
        self,
        dist_max: float,
        depth_max: float,
        aspect_factor: float,
    ) -> tuple[float, float]:
        if not (np.isfinite(dist_max) and np.isfinite(depth_max)):
            return (0.0, 1.0)
        if dist_max <= 0.0 or depth_max <= 0.0:
            return (0.0, float(max(1.0, dist_max)))
        ax_w_px, ax_h_px = self._axis_pixel_size()
        if ax_h_px <= 1e-6:
            return (0.0, float(dist_max))
        af = float(np.clip(aspect_factor, 1e-6, 1e6))
        ideal_dist_window = float((ax_w_px * depth_max) / (ax_h_px * af))
        ideal_dist_window = float(np.clip(ideal_dist_window, 1.0, dist_max))
        # Keep full extent for short profiles; constrain only when profile is much longer.
        if dist_max <= (ideal_dist_window * 1.05):
            return (0.0, float(dist_max))
        return (0.0, float(ideal_dist_window))

    def _full_x_limits(self) -> tuple[float, float]:
        if self._disp_data is None or not self._profiles:
            return 0.0, 1.0
        prof = self._profiles[self._prof_idx]
        dist_arr = self._display_distance_axis(prof, int(self._disp_data.shape[1]))
        if dist_arr.size and np.isfinite(dist_arr).any():
            x_full_min = float(np.nanmin(dist_arr))
            x_full_max = float(np.nanmax(dist_arr))
        else:
            x_full_min = 0.0
            x_full_max = float(max(1.0, prof.sampling_step_m * max(self._disp_data.shape[1] - 1, 1)))
        if not (np.isfinite(x_full_min) and np.isfinite(x_full_max)) or x_full_max <= x_full_min:
            x_full_min, x_full_max = 0.0, 1.0
        return x_full_min, x_full_max

    def _sync_xpan_slider(self):
        if not hasattr(self, "_xpan_slider") or self._xpan_slider is None:
            return
        x_full_min, x_full_max = self._full_x_limits()
        full_w = float(max(x_full_max - x_full_min, 1e-9))
        x_view = self._view_xlim if self._view_xlim is not None else (x_full_min, x_full_max)
        x_view = self._clamp_axis_limits(x_view[0], x_view[1], x_full_min, x_full_max)
        view_w = float(max(abs(x_view[1] - x_view[0]), 1e-9))

        can_pan = view_w < (full_w - 1e-6)
        self._updating_xpan = True
        try:
            self._xpan_slider.setEnabled(can_pan)
            if not can_pan:
                self._xpan_slider.setValue(0)
                if self._lbl_xpan is not None:
                    self._safe_set_text(self._lbl_xpan, "Full")
                return
            movable = full_w - view_w
            ratio = (float(min(x_view)) - x_full_min) / max(movable, 1e-9)
            ratio = float(np.clip(ratio, 0.0, 1.0))
            self._xpan_slider.setValue(int(round(ratio * 1000.0)))
            if self._lbl_xpan is not None:
                self._safe_set_text(self._lbl_xpan, f"{min(x_view):.1f}-{max(x_view):.1f} m")
        finally:
            self._updating_xpan = False

    def _on_xpan_slider_changed(self, value: int):
        if bool(getattr(self, "_updating_xpan", False)):
            return
        if self._disp_data is None or self._ax is None:
            return
        x_full_min, x_full_max = self._full_x_limits()
        full_w = float(max(x_full_max - x_full_min, 1e-9))
        x_view = self._view_xlim if self._view_xlim is not None else (x_full_min, x_full_max)
        x_view = self._clamp_axis_limits(x_view[0], x_view[1], x_full_min, x_full_max)
        view_w = float(max(abs(x_view[1] - x_view[0]), 1e-9))
        movable = full_w - view_w
        if movable <= 1e-9:
            self._sync_xpan_slider()
            return
        ratio = float(np.clip(float(value) / 1000.0, 0.0, 1.0))
        lo = x_full_min + ratio * movable
        hi = lo + view_w
        self._view_xlim = (lo, hi)
        self._ax.set_xlim(*self._view_xlim)
        if self._canvas_mpl is not None:
            self._safe_draw_idle(self._canvas_mpl)
        self._sync_xpan_slider()

    def _export_radargram_image(self):
        if not HAS_MPL or self._disp_data is None or not self._profiles:
            QMessageBox.information(self, "Export Radargram", "Nessun radargramma da esportare.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Radargram",
            "radargram_export.png",
            "PNG (*.png);;TIFF (*.tif *.tiff)",
        )
        if not path:
            return
        dpi, ok = QInputDialog.getInt(
            self,
            "Export Radargram",
            "DPI:",
            300,
            72,
            2400,
            10,
        )
        if not ok:
            return
        prof = self._profiles[self._prof_idx]
        arr = np.asarray(self._disp_data, dtype=np.float64)
        if arr.ndim != 2 or arr.size <= 0:
            QMessageBox.information(self, "Export Radargram", "Dati radargramma non validi.")
            return
        n_s, n_t = arr.shape
        dist_arr = self._display_distance_axis(prof, n_t)
        if dist_arr.size > 0 and np.isfinite(dist_arr).any():
            dist_max = float(np.nanmax(dist_arr))
        else:
            dist_max = float(max(1.0, getattr(prof, "sampling_step_m", 0.1) * max(n_t - 1, 1)))
        depth_max = self._depth_max_display(prof)
        if not np.isfinite(depth_max) or depth_max <= 0:
            depth_max = float(max(1, n_s))

        finite = arr[np.isfinite(arr)]
        if finite.size:
            vmin = float(np.percentile(finite, 1.0))
            vmax = float(np.percentile(finite, 99.0))
            if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax - vmin < 1e-6):
                vmin = float(np.nanmin(finite))
                vmax = float(np.nanmax(finite))
            if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax - vmin < 1e-6):
                vmin, vmax = -1.0, 1.0
        else:
            vmin, vmax = -1.0, 1.0
        cmap = self._cb_cmap.currentText()
        interpolation_mode = str(self._cb_interp.currentData() or "bilinear")
        real_aspect = bool(self._chk_real_aspect.isChecked())
        vertical_exag = 1.0
        if hasattr(self, "_spin_vertical_exag") and self._spin_vertical_exag is not None:
            try:
                vertical_exag = float(self._spin_vertical_exag.value())
            except Exception:
                vertical_exag = 1.0
        vertical_exag = float(np.clip(vertical_exag, 0.5, 20.0))

        fig = Figure(figsize=(12.0, 5.5), tight_layout=True)
        ax = fig.add_subplot(111)
        aspect_export = "auto"
        if real_aspect and dist_max > 1e-9 and depth_max > 1e-9:
            # VE=1.0 preserves metric scale; VE>1 increases vertical readability.
            aspect_export = float(vertical_exag)
        disp_export = self._resample_disp_to_uniform_grid(self._disp_data, dist_arr)
        ax.imshow(
            disp_export,
            aspect=aspect_export,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=[0, dist_max, depth_max, 0],
            interpolation=interpolation_mode,
            resample=True,
        )
        ax.set_xlim(0.0, dist_max)
        ax.set_ylim(depth_max, 0.0)
        ax.set_xlabel("Distanza (m)")
        ax.set_ylabel("Profondita' (m)")
        ax.set_title(
            f"{os.path.basename(prof.path)}  |  Ch {self._ch_idx}  |  {float(getattr(prof, 'frequency_mhz', 0.0)):.0f} MHz"
        )
        self._draw_hyperbola_overlay_on_axis(ax, prof, dist_max, depth_max)

        ext = os.path.splitext(path)[1].strip().lower()
        if ext not in {".png", ".tif", ".tiff"}:
            path = f"{path}.png"
        try:
            fig.savefig(path, dpi=int(dpi), bbox_inches="tight", facecolor="white")
            self._safe_set_text(self._lbl_status, f"Radargram export: {path}  dpi={int(dpi)}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Radargram", f"Errore export:\n{exc}")

    # ------------------------------------------------------------------
    # Disegno
    # ------------------------------------------------------------------

    @staticmethod
    def _resample_disp_to_uniform_grid(
        disp: np.ndarray,
        dist_arr: np.ndarray,
    ) -> np.ndarray:
        """Resample disp (n_samples, n_traces) onto a uniform distance grid.

        Each column of *disp* corresponds to a trace at the cumulative distance
        dist_arr[i].  Because surveys are driven at variable speed the trace
        spacing is usually non-uniform.  imshow maps every column to an equal-
        width pixel, so non-uniform spacing produces a geometrically distorted
        radargram.  This function corrects that by interpolating each sample row
        onto a regular grid before rendering.
        """
        n_s, n_t = disp.shape
        if n_t < 2 or dist_arr.size < 2:
            return disp
        d = np.asarray(dist_arr, dtype=np.float64)
        if d.size != n_t:
            return disp
        d_min = float(d[0])
        d_max = float(d[-1])
        if not (np.isfinite(d_min) and np.isfinite(d_max)) or d_max <= d_min:
            return disp
        # Check whether the grid is already sufficiently uniform (saves CPU).
        steps = np.diff(d)
        step_mean = float(np.mean(steps))
        if step_mean > 0.0 and float(np.std(steps)) / step_mean < 0.01:
            return disp  # < 1 % variation — no resampling needed
        uniform = np.linspace(d_min, d_max, n_t, dtype=np.float64)
        # Indici e pesi dell'interpolazione lineare sono gli stessi per tutte le
        # righe: si calcolano una volta sola (un np.interp per riga costava
        # ~30 ms su 512x6000). Stessa semantica di np.interp, anche con distanze
        # ripetute (tratti fermi) e ai bordi.
        idx = np.clip(np.searchsorted(d, uniform, side="right") - 1, 0, n_t - 2)
        dd = d[idx + 1] - d[idx]
        w = np.where(dd > 0.0, (uniform - d[idx]) / np.where(dd > 0.0, dd, 1.0), 1.0)
        w0 = np.clip(1.0 - w, 0.0, 1.0).astype(np.float32)
        w1 = np.clip(w, 0.0, 1.0).astype(np.float32)
        src = np.asarray(disp, dtype=np.float32)
        out = np.empty((n_s, n_t), dtype=np.float32)
        idx1 = idx + 1
        # A blocchi di righe: limita i temporanei a pochi MB anche su profili enormi.
        rows = max(1, 4_000_000 // n_t)
        for r0 in range(0, n_s, rows):
            blk = src[r0:r0 + rows]
            left = np.take(blk, idx, axis=1)
            right = np.take(blk, idx1, axis=1)
            left *= w0
            right *= w1
            np.add(left, right, out=out[r0:r0 + rows])
        return out

    def _image_geometry(self, prof) -> tuple[np.ndarray, float, float]:
        """(asse distanze, dist_max, depth_max) dell'immagine del radargramma."""
        dist_arr = self._display_distance_axis(prof, int(self._disp_data.shape[1]))
        if dist_arr.size and np.isfinite(dist_arr).any():
            dist_max = float(np.nanmax(dist_arr))
        else:
            dist_max = float(max(1.0, prof.sampling_step_m * max(self._disp_data.shape[1] - 1, 1)))
        if not np.isfinite(dist_max) or dist_max <= 0:
            dist_max = float(max(1.0, prof.sampling_step_m * max(self._disp_data.shape[1] - 1, 1)))

        n_out     = self._disp_data.shape[0]
        n_orig    = prof.n_samples
        depth_base = float(prof.depth_max_m) if np.isfinite(prof.depth_max_m) else 0.0
        depth_max = depth_base * (n_out / n_orig) if n_orig > 0 else depth_base
        if not np.isfinite(depth_max) or depth_max <= 0:
            depth_max = max(1.0, float(n_out))
        return dist_arr, dist_max, depth_max

    def _image_key(self, dist_max: float, depth_max: float) -> tuple:
        """Cio' che, se invariato, permette di aggiornare solo i pixel dell'immagine."""
        interp = "bilinear"
        if getattr(self, "_cb_interp", None) is not None:
            interp = str(self._cb_interp.currentData() or "bilinear")
        return (
            tuple(self._disp_data.shape),
            round(float(dist_max), 9),
            round(float(depth_max), 9),
            self._cb_cmap.currentText(),
            interp,
        )

    # -- livello di dettaglio (profili lunghi) ---------------------------------

    def _lod_set_source(self, disp_uniform, dist_max) -> bool:
        """Imposta il dato di riferimento; True se il livello di dettaglio e' attivo."""
        lod = _ImageLod(disp_uniform, dist_max)
        self._lod = lod if lod.active else None
        return self._lod is not None

    def _lod_connect(self) -> None:
        """Aggiorna il ritaglio a ogni cambio di limiti X e a ogni resize."""
        cid = getattr(self, "_lod_cid", None)
        if cid is not None:
            try:
                self._ax.callbacks.disconnect(cid)
            except Exception:
                pass
        self._lod_cid = self._ax.callbacks.connect("xlim_changed", self._on_xlim_changed_lod)
        if getattr(self, "_lod_resize_cid", None) is None and self._canvas_mpl is not None:
            self._lod_resize_cid = self._canvas_mpl.mpl_connect(
                "resize_event", lambda _e: self._lod_update()
            )

    def _on_xlim_changed_lod(self, _ax) -> None:
        if not self._lod_busy:
            self._lod_update()

    def _lod_update(self, force: bool = False) -> None:
        """Mostra nell'immagine il ritaglio (e la decimazione) adatti alla vista attuale."""
        lod = getattr(self, "_lod", None)
        im = getattr(self, "_im", None)
        if lod is None or im is None or self._lod_busy or getattr(im, "axes", None) is not self._ax:
            return
        x0, x1 = self._ax.get_xlim()
        f, need, want = lod.plan(x0, x1, self._axis_pixel_size()[0])
        if not force and lod.covers(f, need):
            return
        arr, (left, right) = lod.crop(f, *want)
        _l, _r, y_bot, y_top = im.get_extent()
        xl, yl = self._ax.get_xlim(), self._ax.get_ylim()
        self._lod_busy = True
        try:
            im.set_data(arr)
            im.set_extent((left, right, y_bot, y_top))
            # set_extent puo' riadattare i limiti della vista: la vista non si tocca.
            if self._ax.get_xlim() != xl:
                self._ax.set_xlim(*xl)
            if self._ax.get_ylim() != yl:
                self._ax.set_ylim(*yl)
        finally:
            self._lod_busy = False
        lod.state = (f, want[0], want[1])

    def _refresh_image_only(self) -> bool:
        """Aggiorna i soli pixel (es. dopo un cambio di gain) senza ricostruire gli assi.

        Restituisce False se la geometria/colormap sono cambiate e serve un
        `_redraw()` completo.
        """
        im = getattr(self, "_im", None)
        key = getattr(self, "_im_key", None)
        if (
            im is None or key is None or not HAS_MPL
            or self._disp_data is None or not self._profiles
            or getattr(im, "axes", None) is not self._ax
        ):
            return False
        prof = self._profiles[self._prof_idx]
        dist_arr, dist_max, depth_max = self._image_geometry(prof)
        if self._image_key(dist_max, depth_max) != key:
            return False
        disp_uniform = self._resample_disp_to_uniform_grid(self._disp_data, dist_arr)
        if self._lod_set_source(disp_uniform, dist_max):
            self._lod_update(force=True)  # l'immagine mostra solo il ritaglio visibile
        else:
            im.set_data(disp_uniform)
        self._safe_draw_idle(self._canvas_mpl)
        return True

    def _redraw(self):
        if not HAS_MPL or self._disp_data is None:
            return
        prof = self._profiles[self._prof_idx]
        ch   = prof.channel(self._ch_idx)
        cmap = self._cb_cmap.currentText()
        dist_arr, dist_max, depth_max = self._image_geometry(prof)

        # Resample to uniform distance grid so imshow is not geometrically distorted
        # by variable survey speed (non-uniform trace spacing).
        disp_uniform = self._resample_disp_to_uniform_grid(self._disp_data, dist_arr)

        self._ax.clear()
        # Keep fixed display limits so gain controls affect what the user sees.
        vmin, vmax = -1.0, 1.0
        real_aspect = bool(
            getattr(self, "_chk_real_aspect", None)
            and self._chk_real_aspect.isChecked()
        )
        interpolation_mode = "bilinear"
        if hasattr(self, "_cb_interp") and self._cb_interp is not None:
            interpolation_mode = str(self._cb_interp.currentData() or "bilinear")
        ve = 1.0
        if hasattr(self, "_spin_vertical_exag") and self._spin_vertical_exag is not None:
            try:
                ve = float(self._spin_vertical_exag.value())
            except Exception:
                ve = 1.0
        ve = float(np.clip(ve, 0.5, 20.0))
        lod_on = self._lod_set_source(disp_uniform, dist_max)
        # Con il livello di dettaglio l'immagine parte da un segnaposto: il ritaglio
        # visibile lo imposta _lod_update() dopo aver fissato i limiti della vista.
        init_data = disp_uniform[:, :8] if lod_on else disp_uniform
        self._im = self._ax.imshow(
            init_data,
            # Mantieni il riempimento del pannello; la scala reale e' gestita sotto con set_aspect+datalim.
            aspect="auto",
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            extent=[0, dist_max, depth_max, 0],
            interpolation=interpolation_mode,
            # Interpolare i dati prima della colormap: stesso risultato visivo
            # ma ~2.7x piu' veloce a ogni draw (zoom, pan, gain).
            interpolation_stage="data",
            resample=True,
        )
        self._im_key = self._image_key(dist_max, depth_max)
        self._im_full_extent = (0.0, float(dist_max), float(depth_max), 0.0)
        if lod_on:
            self._lod_connect()
        x_full = (0.0, dist_max)
        y_full = (depth_max, 0.0)
        if self._view_xlim is None:
            if not bool(getattr(self, "_profile_view_initialized", True)):
                init_af = ve if real_aspect else 1.0
                self._view_xlim = self._compute_initial_xview(dist_max, depth_max, init_af)
            else:
                self._view_xlim = x_full
        x_view = self._view_xlim
        y_view = self._view_ylim if self._view_ylim is not None else y_full
        x_view = self._clamp_axis_limits(x_view[0], x_view[1], x_full[0], x_full[1])
        y_view = self._clamp_axis_limits(y_view[0], y_view[1], 0.0, depth_max)
        self._view_xlim = x_view
        self._view_ylim = y_view
        self._ax.set_xlim(*x_view)
        self._ax.set_ylim(*y_view)
        if lod_on:
            self._lod_update()
        if real_aspect and dist_max > 1e-9 and depth_max > 1e-9:
            # VE=1.0 keeps metric proportions, VE>1.0 increases vertical emphasis.
            self._ax.set_aspect(float(np.clip(ve, 0.5, 20.0)), adjustable="box")
        else:
            # In auto mode avoid excessive horizontal stretch on short profiles.
            ax_w_px, ax_h_px = self._axis_pixel_size()
            x_range = max(1e-9, float(abs(x_view[1] - x_view[0])))
            y_range = max(1e-9, float(abs(y_view[0] - y_view[1])))
            px_per_m_x = ax_w_px / x_range
            px_per_m_y = ax_h_px / y_range
            max_ratio = 4.0
            if px_per_m_x > (px_per_m_y * max_ratio):
                ratio = float((y_range / x_range) / max_ratio)
                self._ax.set_aspect(float(np.clip(ratio, 1e-4, 1e4)), adjustable="box")
            else:
                self._ax.set_aspect("auto", adjustable="box")
        self._ax.set_xlabel("Distanza (m)")
        self._ax.set_ylabel("Profondit\u00e0 (m)")
        self._ax.set_title(
            f"{os.path.basename(prof.path)}  |  "
            f"Ch {self._ch_idx}  |  "
            f"{prof.frequency_mhz:.0f} MHz"
        )
        self._draw_hyperbola_overlay(prof, dist_max, depth_max)
        self._vline = self._ax.axvline(x=0, color="yellow", lw=1, visible=False)
        self._hline = self._ax.axhline(y=0, color="cyan",   lw=1, linestyle="--", visible=False)
        if self._pinned_cursor and self._cursor_x is not None and self._cursor_z is not None:
            # cursore impostato dal clic sulla time-slice: sopravvive ai ridisegni
            # finche' il mouse non entra nel radargramma.
            self._update_cursor_lines(draw=False)
        self._apply_wiggle_visibility_layout()
        self._sync_xpan_slider()
        self._update_wiggle_plot(draw=False, force=True)
        self._refresh_blit_artists()
        self._profile_view_initialized = True
        self._safe_draw_idle(self._canvas_mpl)
        self._redraw_slice_view()

    def _depth_max_display(self, prof) -> float:
        if self._disp_data is not None:
            n_out = int(self._disp_data.shape[0])
        elif self._proc_data is not None:
            n_out = int(self._proc_data.shape[0])
        elif self._raw_data is not None:
            n_out = int(self._raw_data.shape[0])
        else:
            n_out = int(getattr(prof, "n_samples", 0) or 0)
        n_orig = int(getattr(prof, "n_samples", 0) or 0)
        depth_base = float(getattr(prof, "depth_max_m", 0.0) or 0.0)
        depth_max = depth_base * (n_out / n_orig) if n_orig > 0 else depth_base
        if not np.isfinite(depth_max) or depth_max <= 0:
            depth_max = max(1.0, float(n_out))
        return float(depth_max)

    @staticmethod
    def _nearest_index(sorted_values: np.ndarray, x: float) -> int:
        """Indice del valore piu' vicino a x in un vettore ordinato crescente."""
        n = int(sorted_values.size)
        if n <= 1:
            return 0
        i = int(np.searchsorted(sorted_values, x))
        if i <= 0:
            return 0
        if i >= n:
            return n - 1
        # searchsorted da' il vicino a destra: scegli il piu' vicino dei due.
        return i if (sorted_values[i] - x) < (x - sorted_values[i - 1]) else i - 1

    def _cursor_metrics(self):
        if (
            not self._profiles
            or self._disp_data is None
            or self._cursor_x is None
            or self._cursor_z is None
        ):
            return None
        try:
            prof = self._profiles[self._prof_idx]
            ch = prof.channel(self._ch_idx)
        except Exception:
            return None

        n_tr = int(self._disp_data.shape[1])
        distances = self._channel_trace_vector(getattr(ch, "distances", []), n_tr)
        if distances.size != n_tr:
            step = float(getattr(prof, "sampling_step_m", 0.1) or 0.1)
            distances = (np.arange(n_tr, dtype=np.float64) * max(step, 1e-6)).astype(np.float64)
        if distances.size <= 0:
            return None

        # Normalize distances to start at 0, matching the display axis extent
        # built by _display_distance_axis (which subtracts d0).
        if distances.size > 0 and np.isfinite(distances[0]):
            distances = distances - distances[0]
        finite_d = np.isfinite(distances)
        if not finite_d.any():
            return None
        if not finite_d.all():
            idx_arr = np.arange(distances.size, dtype=np.float64)
            ok = np.where(finite_d)[0].astype(np.float64)
            distances = np.interp(idx_arr, ok, distances[finite_d]).astype(np.float64)
        if distances.size > 1:
            distances = np.maximum.accumulate(distances)

        depth_max = self._depth_max_display(prof)
        idx_t = self._nearest_index(distances, float(self._cursor_x))
        idx_s = self._depth_to_sample_index(float(self._cursor_z), depth_max)
        if idx_s is None:
            return None

        amp = None
        try:
            a = float(self._disp_data[idx_s, idx_t])
            amp = a if np.isfinite(a) else None
        except Exception:
            amp = None

        east = north = None
        try:
            e_vec = self._channel_trace_vector(getattr(ch, "easting", []), n_tr)
            n_vec = self._channel_trace_vector(getattr(ch, "northing", []), n_tr)
            if idx_t < e_vec.size:
                east = float(e_vec[idx_t])
            if idx_t < n_vec.size:
                north = float(n_vec[idx_t])
            if east is not None and not np.isfinite(east):
                east = None
            if north is not None and not np.isfinite(north):
                north = None
        except Exception as exc:
            self._note_swallowed("coordinate del cursore", exc)
            east = north = None

        return {
            "idx_trace": int(idx_t),
            "idx_sample": int(idx_s),
            "distance_m": float(distances[idx_t]),
            "depth_m": float(self._cursor_z),
            "time_ns": float(idx_s * float(getattr(prof, "dt_ns", 0.0) or 0.0)),
            "amplitude": amp,
            "east": east,
            "north": north,
            "depth_max_m": depth_max,
        }

    def _wiggle_reset_artists(self) -> None:
        """Dimentica gli artisti dell'A-scan (asse svuotato o dati assenti)."""
        self._wiggle_trace_idx = None
        self._wiggle_depth_line = None
        self._wiggle_raw_line = None
        self._wiggle_fil_line = None
        self._refresh_blit_artists()

    def _wiggle_show_empty(self, draw: bool) -> None:
        axw = self._ax_wiggle
        axw.clear()
        axw.set_xticks([])
        # Asse y condiviso col radargramma: set_yticks([]) cancellerebbe anche i suoi tick.
        axw.tick_params(axis="y", left=False, labelleft=False)
        self._wiggle_reset_artists()
        if draw:
            self._safe_draw_idle(self._canvas_mpl)

    @staticmethod
    def _wiggle_scale(trace: np.ndarray) -> float:
        """Ampiezza di normalizzazione (99-esimo percentile) di una traccia."""
        finite = np.isfinite(trace)
        if not finite.any():
            return 1.0
        vals = np.abs(trace[finite])
        try:
            scale = float(np.percentile(vals, 99.0))
        except Exception:
            scale = float(vals.max())
        return scale if (np.isfinite(scale) and scale > 1e-9) else 1.0

    def _update_wiggle_plot(self, draw: bool = True, force: bool = False):
        if not HAS_MPL or not hasattr(self, "_ax_wiggle"):
            return
        axw = self._ax_wiggle
        if not bool(getattr(self, "_show_wiggle", True)):
            try:
                axw.set_visible(False)
            except Exception:
                pass
            return
        else:
            try:
                axw.set_visible(True)
            except Exception:
                pass
        if getattr(self, "_gain_panel", None) is not None:
            # Editor del guadagno aperto: qui si confrontano le ampiezze prima/dopo.
            self._draw_gain_amplitudes(draw)
            return
        if self._disp_data is None or not self._profiles:
            self._wiggle_show_empty(draw)
            return

        metrics = self._cursor_metrics()
        if metrics is None:
            idx_t = int(np.clip(self._disp_data.shape[1] // 2, 0, max(0, self._disp_data.shape[1] - 1)))
            idx_s = 0
            depth_max = self._depth_max_display(self._profiles[self._prof_idx])
        else:
            idx_t = int(metrics["idx_trace"])
            idx_s = int(metrics["idx_sample"])
            depth_max = float(metrics["depth_max_m"])

        filt_src = self._proc_data if self._proc_data is not None else self._disp_data
        raw_src = self._raw_data if self._raw_data is not None else filt_src
        if filt_src is None or raw_src is None:
            self._wiggle_show_empty(draw)
            return

        idx_t_f = int(np.clip(idx_t, 0, max(0, int(filt_src.shape[1]) - 1)))
        idx_t_r = int(np.clip(idx_t, 0, max(0, int(raw_src.shape[1]) - 1)))
        trace_f = np.asarray(filt_src[:, idx_t_f], dtype=np.float64)
        trace_r = np.asarray(raw_src[:, idx_t_r], dtype=np.float64)
        if trace_f.size <= 0 or trace_r.size <= 0:
            self._wiggle_show_empty(draw)
            return

        # Align raw to filtered length without distorting waveform:
        # if processing shortens the trace (e.g. time-zero), drop the leading
        # raw samples instead of interpolating the whole raw trace.
        n = int(trace_f.size)
        n_raw = int(trace_r.size)
        if n_raw >= n:
            off = max(0, n_raw - n)
            trace_r_plot = trace_r[off:off + n]
        else:
            trace_r_plot = np.full((n,), np.nan, dtype=np.float64)
            trace_r_plot[:n_raw] = trace_r
        trace_f_plot = trace_f[:n]
        depth_axis = np.linspace(0.0, float(depth_max), n, dtype=np.float64)
        y_line = float(depth_axis[idx_s]) if 0 <= idx_s < depth_axis.size else None

        have_lines = (
            self._wiggle_raw_line is not None
            and self._wiggle_fil_line is not None
            and self._wiggle_depth_line is not None
        )
        # Stessa traccia: si sposta solo la linea di profondita'.
        if (
            (not force)
            and have_lines
            and self._wiggle_trace_idx is not None
            and int(self._wiggle_trace_idx) == int(idx_t)
        ):
            if y_line is not None:
                self._wiggle_depth_line.set_ydata([y_line, y_line])
            if draw:
                self._blit_update()
            return

        # Separate normalization keeps RAW visible/stable across filter toggles.
        fil_norm = np.nan_to_num(
            trace_f_plot / self._wiggle_scale(trace_f_plot), nan=0.0, posinf=0.0, neginf=0.0
        )
        raw_norm = np.nan_to_num(
            trace_r_plot / self._wiggle_scale(trace_r_plot), nan=0.0, posinf=0.0, neginf=0.0
        )
        xlim = float(np.max(np.abs(np.r_[raw_norm, fil_norm]))) if raw_norm.size > 0 else 1.0
        if (not np.isfinite(xlim)) or xlim <= 0.1:
            xlim = 1.0
        xlim = float(np.clip(xlim * 1.05, 0.5, 5.0))

        if force or not have_lines:
            # Parte statica (sfondo, zero, spine) + artisti che poi si aggiornano
            # con set_data: il "clear()" completo a ogni traccia costava molto.
            axw.clear()
            axw.set_facecolor("#1a1a1a")
            axw.axvline(0.0, color="#555555", lw=0.5)
            axw.set_xticks([])
            axw.tick_params(axis="y", left=False, labelleft=False)
            axw.set_xlabel("")
            axw.set_ylabel("")
            axw.grid(False)
            for spine in axw.spines.values():
                spine.set_edgecolor("#333333")
                spine.set_linewidth(0.5)
            (self._wiggle_raw_line,) = axw.plot(
                raw_norm, depth_axis, color="#dc5050", lw=0.9, alpha=0.45, antialiased=True
            )
            (self._wiggle_fil_line,) = axw.plot(
                fil_norm, depth_axis, color="#50dc78", lw=0.9, alpha=0.95, antialiased=True
            )
            self._wiggle_depth_line = axw.axhline(
                y_line if y_line is not None else 0.0,
                color="goldenrod", lw=0.8, ls="--", visible=y_line is not None,
            )
            # L'asse Y e' condiviso col radargramma: lo si legge, non lo si
            # impone (altrimenti azzererebbe lo zoom in profondita').
            y0, y1 = self._ax.get_ylim()
            axw.set_ylim(y0, y1)
            self._refresh_blit_artists()
        else:
            self._wiggle_raw_line.set_data(raw_norm, depth_axis)
            self._wiggle_fil_line.set_data(fil_norm, depth_axis)
            if y_line is not None:
                self._wiggle_depth_line.set_ydata([y_line, y_line])
                self._wiggle_depth_line.set_visible(True)
            else:
                self._wiggle_depth_line.set_visible(False)
        axw.set_xlim(-xlim, xlim)
        axw.set_title(f"A-scan T{idx_t}", fontsize=7, pad=2)
        self._wiggle_trace_idx = int(idx_t)
        if draw:
            self._blit_update()

    # ------------------------------------------------------------------
    # Cursore
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Pick-mode: click sul profilo → imposta un campo sample
    # ------------------------------------------------------------------

    def _set_pick_mode(self, target: str | None, source_btn=None):
        """Attiva/disattiva la modalità pick-from-profile.

        target: nome logico del campo da popolare ('agc_sample_start', ...).
                None = disattiva.
        source_btn: il QPushButton che ha scatenato l'azione (per sincronizzare
                    lo stato checked se la disattivazione arriva dall'esterno).
        """
        self._pick_mode_target = target
        # Deseleziona tutti i bottoni pick tranne quello sorgente
        for btn_name in ("_btn_pick_agc_s0",):
            btn = getattr(self, btn_name, None)
            if btn is not None and self._is_qt_alive(btn) and btn is not source_btn:
                btn.blockSignals(True)
                btn.setChecked(False)
                btn.blockSignals(False)
        # Cambia il cursore del canvas per dare feedback visivo
        if HAS_MPL and getattr(self, "_canvas_mpl", None) and self._is_qt_alive(self._canvas_mpl):
            if target is not None:
                self._canvas_mpl.setCursor(Qt.CrossCursor)
            else:
                self._canvas_mpl.unsetCursor()

    def _depth_to_sample_index(self, depth_m: float, depth_max: float | None = None) -> int | None:
        """Profondita' del canvas (m) -> indice sample nel dato mostrato.

        Unica conversione usata da cursore e pick: usa l'asse di profondita'
        *visualizzato* (che si accorcia dopo il time-zero), non quello del
        profilo originale.
        """
        if self._disp_data is None or not self._profiles:
            return None
        n_s = int(self._disp_data.shape[0])
        if n_s <= 0:
            return None
        if depth_max is None:
            depth_max = self._depth_max_display(self._profiles[self._prof_idx])
        try:
            frac = float(depth_m) / max(float(depth_max), 1e-9)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(frac):
            return None
        return int(np.clip(int(np.rint(frac * (n_s - 1))), 0, n_s - 1))

    def _sample_from_ydata(self, ydata: float) -> int | None:
        """Converte la coordinata Y del canvas (profondita') in indice sample."""
        return self._depth_to_sample_index(ydata)

    def _on_mouse_move(self, event):
        if event.inaxes != self._ax or self._disp_data is None:
            return
        self._pinned_cursor = False
        self._cursor_x = event.xdata
        self._cursor_z = event.ydata
        self._update_cursor_lines(draw=False)
        self._update_wiggle_plot(draw=False)
        self._update_status()
        self._blit_update()
        self._canvas_timer.start()

    def _on_mouse_press(self, event):
        if event.inaxes == self._ax and event.button == 1:
            self._cursor_x = event.xdata
            self._cursor_z = event.ydata

            # Pick-mode: popola il campo target e disattiva la modalità
            if self._pick_mode_target is not None and event.ydata is not None:
                sample_idx = self._sample_from_ydata(float(event.ydata))
                if sample_idx is not None:
                    if self._pick_mode_target == "agc_sample_start":
                        if getattr(self, "_spin_agc_sample_start", None) and self._is_qt_alive(self._spin_agc_sample_start):
                            self._spin_agc_sample_start.setValue(sample_idx)
                    # Disattiva pick-mode dopo la selezione
                    self._set_pick_mode(None, None)
                return

            self._update_wiggle_plot(draw=False)
            self._blit_update()
            if bool(getattr(event, "dblclick", False)) and bool(self._chk_hyperbola.isChecked()):
                self._set_hyperbola_apex_from_cursor()
            self._update_dial()
            self._emit_cursor_moved()

    def _on_scroll_zoom(self, event):
        if event.inaxes != self._ax or self._im is None:
            return
        if event.xdata is None or event.ydata is None:
            return

        step = float(getattr(event, "step", 0.0) or 0.0)
        btn = str(getattr(event, "button", "") or "").lower()
        if step > 0 or btn == "up":
            scale = 1.0 / 1.2
        elif step < 0 or btn == "down":
            scale = 1.2
        else:
            return

        x0, x1 = self._ax.get_xlim()
        y0, y1 = self._ax.get_ylim()
        x = float(event.xdata)
        y = float(event.ydata)

        nx0 = x - (x - x0) * scale
        nx1 = x + (x1 - x) * scale
        ny0 = y - (y - y0) * scale
        ny1 = y + (y1 - y) * scale

        # Estensione dell'immagine INTERA: con il livello di dettaglio quella di _im e' il ritaglio.
        ex0, ex1, ey0, ey1 = getattr(self, "_im_full_extent", None) or self._im.get_extent()
        x_min, x_max = float(min(ex0, ex1)), float(max(ex0, ex1))
        y_min, y_max = float(min(ey0, ey1)), float(max(ey0, ey1))

        self._view_xlim = self._clamp_axis_limits(nx0, nx1, x_min, x_max)
        self._view_ylim = self._clamp_axis_limits(ny0, ny1, y_min, y_max)

        self._ax.set_xlim(*self._view_xlim)
        self._ax.set_ylim(*self._view_ylim)
        self._sync_xpan_slider()
        self._safe_draw_idle(self._canvas_mpl)

    def _on_axes_leave(self, event):
        if self._vline: self._vline.set_visible(False)
        if self._hline: self._hline.set_visible(False)
        self._blit_update()

    def _update_cursor_lines(self, draw: bool = True):
        if self._vline and self._cursor_x is not None:
            self._vline.set_xdata([self._cursor_x])
            self._vline.set_visible(True)
        if self._hline and self._cursor_z is not None:
            self._hline.set_ydata([self._cursor_z])
            self._hline.set_visible(True)
        if draw:
            self._blit_update()

    def _update_status(self):
        if not self._profiles:
            return
        m = self._cursor_metrics()
        if not m:
            z_str = f"{self._cursor_z:.3f} m" if self._cursor_z is not None else "\u2014"
            x_str = f"{self._cursor_x:.2f} m" if self._cursor_x is not None else "\u2014"
            self._safe_set_text(
                self._lbl_status,
                f"Dist: {x_str}  |  Profondita': {z_str}"
            )
            return
        amp = m.get("amplitude")
        amp_str = f"{float(amp):.6g}" if amp is not None else "n/a"
        east = m.get("east")
        north = m.get("north")
        east_str = f"{float(east):.2f}" if east is not None else "n/a"
        north_str = f"{float(north):.2f}" if north is not None else "n/a"
        self._safe_set_text(
            self._lbl_status,
            f"Trace={int(m['idx_trace'])}  Sample={int(m['idx_sample'])}  "
            f"Amp={amp_str}  Time={float(m['time_ns']):.3f} ns  "
            f"Depth={float(m['depth_m']):.3f} m  Dist={float(m['distance_m']):.3f} m  "
            f"E {east_str}  N {north_str}"
        )

    def _cursor_world_position(self):
        m = self._cursor_metrics()
        if not m:
            return None, None, None
        east = m.get("east")
        north = m.get("north")
        depth = float(m.get("depth_m", np.nan))
        if east is None or north is None:
            return None, None, None
        if not (np.isfinite(float(east)) and np.isfinite(float(north)) and np.isfinite(depth)):
            return None, None, None
        return float(east), float(north), float(depth)

    def _emit_cursor_moved(self):
        east, north, depth = self._cursor_world_position()
        if east is None or north is None or depth is None:
            return
        try:
            self.cursor_moved.emit(float(east), float(north), float(depth))
        except Exception:
            pass
        if not self._suppress_3d_depth_sync and self._gpr_3d_viewer is not None:
            try:
                self._gpr_3d_viewer.set_depth_from_external(float(depth))
            except Exception:
                pass

    def _on_3d_depth_changed(self, depth: float):
        if not np.isfinite(depth):
            return
        if self._profiles:
            try:
                prof = self._profiles[self._prof_idx]
                n_out = int(self._disp_data.shape[0]) if self._disp_data is not None else int(prof.n_samples)
                n_orig = int(prof.n_samples) if int(prof.n_samples) > 0 else max(1, n_out)
                depth_base = float(prof.depth_max_m) if np.isfinite(prof.depth_max_m) else 0.0
                depth_max = depth_base * (n_out / n_orig) if n_orig > 0 else depth_base
                if not np.isfinite(depth_max) or depth_max <= 0:
                    depth_max = max(1.0, float(n_out))
                depth = float(np.clip(float(depth), 0.0, depth_max))
            except Exception:
                depth = float(depth)
        else:
            depth = float(depth)

        self._suppress_3d_depth_sync = True
        try:
            self._cursor_z = depth
            self._update_cursor_lines(draw=False)
            self._update_wiggle_plot(draw=False)
            self._blit_update()
            self._update_status()
            self._update_dial()
            self._emit_cursor_moved()
            self._canvas_timer.start()
        finally:
            self._suppress_3d_depth_sync = False

    # ------------------------------------------------------------------
    # Bridge QGIS
    # ------------------------------------------------------------------

    def _flush_canvas_update(self):
        if not self._is_qt_alive(self):
            return
        try:
            self._update_rubber_band()
            self._update_dial()
            self._update_timeslice_visibility_in_canvas()
            self._update_timeslice_crosshair()
            self._emit_cursor_moved()
            if bool(getattr(self, "_preview_mode_active", False)):
                self._request_preview_update(immediate=False)
            if self._is_qt_alive(getattr(self, "_canvas_slice", None)):
                self._redraw_slice_view()
        except RuntimeError:
            pass

    def _to_canvas_point(self, prof, east: float, north: float) -> QgsPointXY:
        pt = QgsPointXY(float(east), float(north))
        if self.iface is None:
            return pt
        canvas = self.iface.mapCanvas()
        if canvas is None:
            return pt
        try:
            epsg = int(getattr(prof, "epsg", 0) or 0)
        except Exception:
            epsg = 0
        if epsg <= 0:
            # Fallback: prova CRS progetto QGIS quando l'EPSG profilo manca.
            try:
                src = QgsProject.instance().crs()
                dst = canvas.mapSettings().destinationCrs()
                if src.isValid() and dst.isValid() and src != dst:
                    tr = QgsCoordinateTransform(src, dst, QgsProject.instance())
                    return tr.transform(pt)
            except Exception:
                pass
            return pt
        try:
            src = QgsCoordinateReferenceSystem.fromEpsgId(epsg)
            dst = canvas.mapSettings().destinationCrs()
            if not src.isValid() or not dst.isValid() or src == dst:
                return pt
            tr = QgsCoordinateTransform(src, dst, QgsProject.instance())
            return tr.transform(pt)
        except Exception:
            return pt

    def _update_rubber_band(self):
        if self._cursor_x is None or not self._profiles or self.iface is None:
            return
        m = self._cursor_metrics()
        if not m:
            return
        east = m.get("east")
        north = m.get("north")
        if east is None or north is None:
            return
        prof = self._profiles[self._prof_idx]
        canvas = self.iface.mapCanvas()
        if self._rb_point is None:
            self._rb_point = QgsRubberBand(canvas, QgsWkbTypes.PointGeometry)
            self._rb_point.setColor(QColor(255, 220, 0))
        self._rb_point.setIconSize(12)
        self._rb_point.setWidth(3)
        self._rb_point.setVisible(True)
        self._rb_point.reset(QgsWkbTypes.PointGeometry)
        pt_canvas = self._to_canvas_point(prof, float(east), float(north))
        self._rb_point.addPoint(
            pt_canvas, True
        )

    def _draw_profile_line_on_canvas(self, pan_if_hidden: bool = False):
        if not self._profiles or self.iface is None:
            return
        prof   = self._profiles[self._prof_idx]
        ch     = prof.channel(self._ch_idx)
        canvas = self.iface.mapCanvas()
        if self._rb_line is None:
            self._rb_line = QgsRubberBand(canvas, QgsWkbTypes.LineGeometry)
            self._rb_line.setColor(QColor(255, 120, 0))
            self._rb_line.setWidth(4)
            self._rb_line.setVisible(True)
        else:
            self._rb_line.reset(QgsWkbTypes.LineGeometry)
            self._rb_line.setVisible(True)
        if self._disp_data is not None and np.asarray(self._disp_data).ndim == 2:
            n_t = int(self._disp_data.shape[1])
        else:
            try:
                n_t = int(np.asarray(ch.data).shape[1])
            except Exception:
                n_t = 0
        e_vec = self._channel_trace_vector(getattr(ch, "easting", []), n_t)
        n_vec = self._channel_trace_vector(getattr(ch, "northing", []), n_t)
        _dbg.log_once(("viewer-track", id(ch), n_t), "VIEWER", "traccia mostrata: " + _dbg.describe_track(e_vec, n_vec))
        finite = np.isfinite(e_vec) & np.isfinite(n_vec)
        if int(np.count_nonzero(finite)) < 2:
            return
        e_use = e_vec[finite]
        n_use = n_vec[finite]
        for i in range(len(e_use)):
            pt_canvas = self._to_canvas_point(prof, e_use[i], n_use[i])
            self._rb_line.addPoint(
                pt_canvas,
                i == len(e_use) - 1
            )
        if pan_if_hidden:
            # Profilo fuori dalla vista della mappa: la linea "non si vede" anche se disegnata.
            # Solo pan (mai zoom): la scala scelta dall'utente resta.
            try:
                box = self._rb_line.asGeometry().boundingBox()
                if not canvas.extent().intersects(box):
                    canvas.setCenter(box.center())
                    canvas.refresh()
            except Exception as exc:
                self._note_swallowed("centratura mappa sul profilo", exc)
