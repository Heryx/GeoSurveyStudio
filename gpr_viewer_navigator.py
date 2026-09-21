# -*- coding: utf-8 -*-
"""Navigazione del GPR Profile Viewer: scelta rapida di profilo e canale.

- combo "Profilo" nella toolbar e sidebar sinistra nascondibile (profili + canali);
- tracce dei profili disegnate sulla time-slice (il profilo corrente evidenziato);
- puntatore inverso: un click sulla time-slice porta il radargramma sulla traccia
  piu' vicina (profilo, canale, posizione) alla profondita' della slice.

Tutte le vie di cambio profilo passano da `_select_profile`; la geometria si legge
dai profili in memoria (unica fonte), mai dai .npz.
"""

from __future__ import annotations

from .gpr_viewer_common import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QLabel,
    QListWidget,
    QVBoxLayout,
    QWidget,
    Qt,
    np,
    os,
)


class NavigatorMixin:
    # Tolleranza (m) del click sulla time-slice: oltre non c'e' nessun profilo "sotto" il punto.
    _NAV_CLICK_TOL_M = 2.0

    _nav_updating = False
    _nav_sig = None
    _nav_chan_sig = None
    _nav_dock = None
    _nav_track_artists = ()
    _nav_track_key = None
    _slice_press_px = None
    _pending_cursor = False
    _pinned_cursor = False

    # ------------------------------------------------------------------
    # Costruzione UI
    # ------------------------------------------------------------------

    def _nav_build_dock(self):
        """Sidebar sinistra: elenco profili, elenco canali, opzioni del puntatore inverso."""
        host = QWidget()
        lay = QVBoxLayout(host)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)

        lay.addWidget(QLabel("Profili"))
        self._lst_profiles = QListWidget()
        self._lst_profiles.setToolTip("Clic (o frecce su/giu') per aprire il profilo.")
        self._lst_profiles.currentRowChanged.connect(self._on_nav_profile_picked)
        lay.addWidget(self._lst_profiles, stretch=3)

        lay.addWidget(QLabel("Canali"))
        self._lst_channels = QListWidget()
        self._lst_channels.currentRowChanged.connect(self._on_nav_channel_picked)
        lay.addWidget(self._lst_channels, stretch=2)

        self._chk_nav_slice_click = QCheckBox("Clic sulla time-slice → vai al profilo")
        self._chk_nav_slice_click.setChecked(True)
        self._chk_nav_slice_click.setToolTip(
            "Un clic (senza trascinare) sulla time-slice apre il profilo e il canale\n"
            "piu' vicini a quel punto, con il cursore alla profondita' della slice."
        )
        lay.addWidget(self._chk_nav_slice_click)

        self._chk_nav_slice_tracks = QCheckBox("Tracce dei profili sulla time-slice")
        self._chk_nav_slice_tracks.setChecked(True)
        self._chk_nav_slice_tracks.toggled.connect(lambda _c: self._update_slice_track_overlay())
        lay.addWidget(self._chk_nav_slice_tracks)

        dock = QDockWidget("Profili", self)
        dock.setObjectName("GprNavigatorDock")
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        dock.setFeatures(
            QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
            | QDockWidget.DockWidgetClosable
        )
        dock.setWidget(host)
        # Barra stretta: profili e canali sono elenchi brevi, il resto va al radargramma.
        host.setMinimumWidth(170)
        dock.setMaximumWidth(340)
        self._restore_dock_state(dock, "navigator", Qt.LeftDockWidgetArea)
        self.resizeDocks([dock], [230], Qt.Horizontal)
        self._nav_dock = dock
        return dock

    def _nav_add_toolbar_widgets(self, tb):
        """Combo di selezione rapida del profilo (dopo i tasti precedente/successivo)."""
        self._cb_profile = QComboBox()
        self._cb_profile.setMinimumWidth(240)
        self._cb_profile.setMaxVisibleItems(30)
        self._cb_profile.setToolTip("Seleziona rapidamente il profilo.")
        self._cb_profile.currentIndexChanged.connect(self._on_nav_profile_picked)
        tb.addWidget(self._cb_profile)

    def _nav_toggle_action(self):
        act = self._nav_dock.toggleViewAction()
        act.setText("☰ Profili")
        act.setToolTip("Mostra/nascondi la barra laterale con profili e canali.")
        return act

    # ------------------------------------------------------------------
    # Selezione
    # ------------------------------------------------------------------

    def _select_profile(self, idx: int, channel: int | None = None, deferred: bool = True):
        """Unico punto di cambio profilo (tasti, combo, sidebar, click sulla slice)."""
        if not self._profiles:
            return
        idx = int(np.clip(int(idx), 0, len(self._profiles) - 1))
        if idx != self._prof_idx or self._raw_data is None:
            self._prof_idx = idx
            self._load_current_profile(deferred=deferred, channel=channel)
        elif channel is not None and int(channel) != int(self._ch_idx):
            if 0 <= int(channel) < self._cb_channel.count():
                self._cb_channel.setCurrentIndex(int(channel))  # -> _on_channel_changed

    def _on_nav_profile_picked(self, row: int):
        if self._nav_updating or row is None or int(row) < 0:
            return
        self._select_profile(int(row))

    def _on_nav_channel_picked(self, row: int):
        if self._nav_updating or row is None or int(row) < 0:
            return
        if int(row) != int(self._ch_idx) and int(row) < self._cb_channel.count():
            self._cb_channel.setCurrentIndex(int(row))

    def _sync_navigator(self):
        """Allinea combo e liste a profilo/canale correnti (senza riemettere segnali)."""
        combo = getattr(self, "_cb_profile", None)
        lst_p = getattr(self, "_lst_profiles", None)
        lst_c = getattr(self, "_lst_channels", None)
        if not all(self._is_qt_alive(w) for w in (combo, lst_p, lst_c)):
            return
        profiles = list(self._profiles or [])
        n = len(profiles)
        self._nav_updating = True
        try:
            sig = tuple((id(p), str(getattr(p, "path", ""))) for p in profiles)
            if sig != self._nav_sig:
                self._nav_sig = sig
                labels = [
                    f"{i + 1}/{n}  {os.path.basename(str(getattr(p, 'path', '')))}"
                    for i, p in enumerate(profiles)
                ]
                for w in (combo, lst_p):
                    w.clear()
                combo.addItems(labels)
                lst_p.addItems(labels)
            if n:
                cur = int(np.clip(self._prof_idx, 0, n - 1))
                combo.setCurrentIndex(cur)
                lst_p.setCurrentRow(cur)
            n_ch = int(getattr(profiles[self._prof_idx], "n_channels", 0) or 0) if n else 0
            chan_sig = (self._prof_idx, n_ch)
            if chan_sig != self._nav_chan_sig:
                self._nav_chan_sig = chan_sig
                lst_c.clear()
                lst_c.addItems([f"Ch {i}" for i in range(n_ch)])
            if n_ch:
                lst_c.setCurrentRow(int(np.clip(self._ch_idx, 0, n_ch - 1)))
        except Exception as exc:
            self._note_swallowed("sincronizzazione navigatore profili", exc)
        finally:
            self._nav_updating = False

    # ------------------------------------------------------------------
    # Tracce dei profili sulla time-slice
    # ------------------------------------------------------------------

    @staticmethod
    def _nav_track_xy(prof, ci: int):
        ch = prof.channel(int(ci))
        e = np.asarray(getattr(ch, "easting", []), dtype=np.float64)
        n = np.asarray(getattr(ch, "northing", []), dtype=np.float64)
        if e.size < 2 or e.size != n.size:
            return None
        ok = np.isfinite(e) & np.isfinite(n)
        if int(ok.sum()) < 2:
            return None
        e, n = e[ok], n[ok]
        step = max(1, e.size // 200)
        return e[::step], n[::step]

    def _update_slice_track_overlay(self) -> bool:
        """Disegna sulla time-slice le tracce dei profili; il corrente e' in evidenza."""
        ax = getattr(self, "_ax_slice", None)
        if ax is None or not self._profiles:
            return False
        chk = getattr(self, "_chk_nav_slice_tracks", None)
        enabled = bool(chk is None or chk.isChecked())
        arts = list(self._nav_track_artists)
        key = (enabled, tuple(id(p) for p in self._profiles), int(self._prof_idx), int(self._ch_idx))
        alive = bool(arts) and all(getattr(a, "axes", None) is ax for a in arts)
        if key == self._nav_track_key and (alive or not enabled):
            return False
        for a in arts:
            try:
                a.remove()
            except Exception:
                pass
        arts = []
        if enabled:
            try:
                for i, prof in enumerate(self._profiles):
                    if i == self._prof_idx:
                        continue
                    xy = self._nav_track_xy(prof, int(prof.n_channels) // 2)
                    if xy is not None:
                        arts += ax.plot(*xy, color="white", lw=0.8, alpha=0.55, zorder=3,
                                        scalex=False, scaley=False)
                cur = self._profiles[self._prof_idx]
                xy = self._nav_track_xy(cur, min(int(self._ch_idx), int(cur.n_channels) - 1))
                if xy is not None:
                    arts += ax.plot(*xy, color="#ff8c00", lw=2.4, zorder=5, scalex=False, scaley=False)
                    arts += ax.plot(xy[0][:1], xy[1][:1], "o", color="#ff8c00", ms=5, zorder=6,
                                    scalex=False, scaley=False)
            except Exception as exc:
                self._note_swallowed("tracce dei profili sulla time-slice", exc)
        self._nav_track_artists = tuple(arts)
        self._nav_track_key = key
        self._safe_draw_idle(self._canvas_slice)
        return True

    # ------------------------------------------------------------------
    # Puntatore inverso: time-slice -> radargramma
    # ------------------------------------------------------------------

    def _on_slice_click(self, east: float, north: float):
        chk = getattr(self, "_chk_nav_slice_click", None)
        if chk is not None and not chk.isChecked():
            return
        self._goto_world_point(float(east), float(north))

    def _nav_nearest_trace(self, east: float, north: float):
        """(distanza m, indice profilo, canale, traccia) del punto rilevato piu' vicino."""
        best = None
        for pi, prof in enumerate(self._profiles):
            for ci in range(int(getattr(prof, "n_channels", 0) or 0)):
                try:
                    ch = prof.channel(ci)
                    e = np.asarray(getattr(ch, "easting", []), dtype=np.float64)
                    n = np.asarray(getattr(ch, "northing", []), dtype=np.float64)
                except Exception:
                    continue
                if e.size == 0 or e.size != n.size:
                    continue
                d2 = (e - east) ** 2 + (n - north) ** 2
                if not np.isfinite(d2).any():
                    continue
                ti = int(np.nanargmin(d2))
                if best is None or d2[ti] < best[0]:
                    best = (float(d2[ti]), pi, ci, ti)
        if best is None:
            return None
        return (best[0] ** 0.5, best[1], best[2], best[3])

    def _nav_slice_depth(self):
        """Profondita' (m) della slice mostrata, None se non nota."""
        try:
            if bool(getattr(self, "_preview_mode_active", False)):
                z0 = float(getattr(self, "_preview_slice_z_from", 0.0) or 0.0)
                z1 = float(getattr(self, "_preview_slice_z_to", z0) or z0)
                return 0.5 * (z0 + z1)
            cat = list(getattr(self, "_slice_catalog", []) or [])
            i = getattr(self, "_slice_current_idx", None)
            if cat and i is not None:
                s = cat[int(np.clip(int(i), 0, len(cat) - 1))]
                zc = s.get("z_center", None)
                if zc is None:
                    z0 = float(s.get("z_top", 0.0) or 0.0)
                    z1 = float(s.get("z_bot", z0) or z0)
                    zc = 0.5 * (z0 + z1)
                return float(zc)
        except Exception:
            pass
        return None

    def _nav_trace_cursor_x(self, prof, ci: int, trace: int) -> float:
        """Ascissa (m) sul radargramma della traccia `trace` del dato completo."""
        idx = self._trace_source_indices
        n_tr = int(idx.size) if isinstance(idx, np.ndarray) and idx.ndim == 1 else 0
        if n_tr <= 0:
            return 0.0
        j = int(np.argmin(np.abs(idx - int(trace))))
        dist = self._channel_trace_vector(getattr(prof.channel(int(ci)), "distances", []), n_tr)
        if dist.size != n_tr or not np.isfinite(dist).all():
            step = float(getattr(prof, "sampling_step_m", 0.1) or 0.1)
            return float(j * step)
        return float(dist[j] - dist[0])

    def _goto_world_point(self, east: float, north: float) -> bool:
        if not self._profiles:
            return False
        hit = self._nav_nearest_trace(east, north)
        if hit is None or hit[0] > self._NAV_CLICK_TOL_M:
            self._safe_set_text(
                self._lbl_status,
                "Nessun profilo entro %.1f m dal punto cliccato." % self._NAV_CLICK_TOL_M,
            )
            return False
        dist, pi, ci, ti = hit
        self._select_profile(pi, channel=ci)
        prof = self._profiles[pi]
        z = self._nav_slice_depth()
        if z is None:
            z = self._cursor_z if self._cursor_z is not None else 0.0
        z = float(np.clip(z, 0.0, self._depth_max_display(prof)))
        self._cursor_x = self._nav_trace_cursor_x(prof, ci, ti)
        self._cursor_z = z
        self._pinned_cursor = True
        self._pending_cursor = True
        self._safe_set_text(
            self._lbl_status,
            f"Profilo {pi + 1}/{len(self._profiles)}  Ch {ci}  traccia {ti}  "
            f"(a {dist:.2f} m dal punto)  profondita' {z:.2f} m",
        )
        if self._disp_data is not None:
            self._apply_pending_cursor()
        return True

    def _apply_pending_cursor(self):
        """Mostra il cursore impostato dal puntatore inverso appena il dato e' pronto."""
        if not self._pending_cursor or self._disp_data is None:
            return
        self._pending_cursor = False
        try:
            x = float(self._cursor_x)
            ax = self._ax
            lo, hi = sorted(float(v) for v in ax.get_xlim())
            if not (lo <= x <= hi):
                full = getattr(self, "_im_full_extent", None)
                if full is not None:
                    span = hi - lo
                    self._view_xlim = self._clamp_axis_limits(
                        x - span / 2, x + span / 2, 0.0, float(full[1])
                    )
                    ax.set_xlim(*self._view_xlim)
                    self._sync_xpan_slider()
            self._update_cursor_lines(draw=False)
            self._update_wiggle_plot(draw=False)
            self._blit_update()
            self._safe_draw_idle(self._canvas_mpl)
            self._emit_cursor_moved()
        except Exception as exc:
            self._note_swallowed("cursore da time-slice", exc)
