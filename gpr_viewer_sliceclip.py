# -*- coding: utf-8 -*-
"""Strumento di disegno del limite della griglia sulla vista time-slice.

Il poligono vive nello ``SliceSpec`` (``clip_polygons``, coordinate mondo) e il
pannello dei parametri e' la sua unica copia; qui c'e' solo l'interazione con la
canvas matplotlib della slice e il disegno del contorno.

Mentre si disegna, la preview viene calcolata SENZA ritaglio (si deve vedere
tutta la slice per decidere dove tagliare); a disegno concluso torna ritagliata.
"""

from __future__ import annotations

_CLIP_COLOR = "#e6007e"


class SliceClipMixin:
    """Mixin di ``GprProfileViewer`` (stato ``_clip_*`` creato in modo pigro)."""

    # ------------------------------------------------------------------
    # stato
    # ------------------------------------------------------------------
    def _clip_state(self) -> dict:
        st = getattr(self, "_clip_ui", None)
        if st is None:
            st = {"drawing": False, "pending": [], "cursor": None, "artists": []}
            self._clip_ui = st
        return st

    def _clip_is_drawing(self) -> bool:
        return bool(self._clip_state()["drawing"])

    def _clip_panel(self):
        panel = getattr(self, "_slice_spec_panel", None)
        if panel is not None and self._is_qt_alive(panel):
            return panel
        return None

    def _wire_slice_clip_panel(self) -> None:
        """Collega il pannello dei parametri allo strumento (dopo averlo creato)."""
        panel = self._clip_panel()
        if panel is None:
            return
        panel.clip_draw_toggled.connect(self._on_clip_draw_toggled)
        panel.clip_undo_requested.connect(self._on_clip_undo)
        panel.spec_changed.connect(self._refresh_clip_overlay)
        canvas = getattr(self, "_canvas_slice", None)
        if canvas is not None:
            canvas.mpl_connect("key_press_event", self._on_clip_key)

    # ------------------------------------------------------------------
    # accensione / spegnimento
    # ------------------------------------------------------------------
    def _on_clip_draw_toggled(self, on: bool) -> None:
        st = self._clip_state()
        on = bool(on)
        if on and getattr(self, "_slice_im", None) is None:
            # senza una slice a video non ci sono coordinate su cui disegnare
            panel = self._clip_panel()
            if panel is not None:
                panel.set_clip_drawing(False)
            self._safe_set_text(
                self._lbl_status,
                "Limite griglia: attiva la preview (o apri una time-slice) prima di disegnare.",
            )
            return
        st["drawing"] = on
        st["pending"] = []
        st["cursor"] = None
        if on:
            self._safe_set_text(
                self._lbl_status,
                "Limite griglia: clic = vertice, doppio clic / clic destro = chiudi, Esc = annulla.",
            )
        # la preview cambia (senza ritaglio mentre si disegna, con ritaglio dopo)
        try:
            self._request_preview_update(immediate=True)
        except Exception:
            pass
        self._refresh_clip_overlay()

    def _stop_clip_drawing(self) -> None:
        panel = self._clip_panel()
        if panel is not None:
            panel.set_clip_drawing(False)
        self._on_clip_draw_toggled(False)

    # ------------------------------------------------------------------
    # eventi della canvas (chiamati da _on_slice_press / _on_slice_mouse_move)
    # ------------------------------------------------------------------
    def _clip_handle_press(self, event) -> bool:
        """True se l'evento e' stato consumato dallo strumento."""
        if not self._clip_is_drawing():
            return False
        if event is None or event.inaxes != self._ax_slice:
            return True
        if event.xdata is None or event.ydata is None:
            return True
        btn = int(getattr(event, "button", 0) or 0)
        st = self._clip_state()
        if btn == 3 or bool(getattr(event, "dblclick", False)):
            # il primo clic del doppio clic ha gia' messo il vertice
            self._close_clip_polygon()
            return True
        if btn == 1:
            st["pending"].append([float(event.xdata), float(event.ydata)])
            self._refresh_clip_overlay()
        return True

    def _clip_handle_move(self, event) -> bool:
        st = self._clip_state()
        if not st["drawing"] or not st["pending"]:
            return False
        if event is None or event.inaxes != self._ax_slice or event.xdata is None:
            return True
        st["cursor"] = (float(event.xdata), float(event.ydata))
        self._refresh_clip_overlay()
        return True

    def _on_clip_key(self, event) -> None:
        if event is not None and str(getattr(event, "key", "")) == "escape" and self._clip_is_drawing():
            st = self._clip_state()
            if st["pending"]:
                st["pending"] = []
                st["cursor"] = None
                self._refresh_clip_overlay()
            else:
                self._stop_clip_drawing()
        elif event is not None and str(getattr(event, "key", "")) == "enter" and self._clip_is_drawing():
            self._close_clip_polygon()

    def _on_clip_undo(self) -> None:
        st = self._clip_state()
        if st["drawing"] and st["pending"]:
            st["pending"].pop()
            self._refresh_clip_overlay()
            return
        panel = self._clip_panel()
        if panel is not None:
            panel.pop_clip_polygon()

    def _close_clip_polygon(self) -> None:
        st = self._clip_state()
        pts = list(st["pending"])
        st["pending"] = []
        st["cursor"] = None
        panel = self._clip_panel()
        if len(pts) < 3:
            self._safe_set_text(self._lbl_status, "Limite griglia: servono almeno 3 vertici, poligono scartato.")
            self._refresh_clip_overlay()
            return
        # il pannello torna a "non sto disegnando" PRIMA di aggiungere il poligono:
        # cosi' la preview che parte dall'aggiunta gia' lo applica
        st["drawing"] = False
        if panel is not None:
            panel.set_clip_drawing(False)
            panel.add_clip_polygon(pts)
        if bool(getattr(self, "_preview_mode_active", False)):
            self._safe_set_text(self._lbl_status, f"Limite griglia: poligono chiuso ({len(pts)} vertici).")
        else:
            self._safe_set_text(
                self._lbl_status,
                "Limite griglia salvato nei parametri: vale dal prossimo calcolo / salvataggio del gruppo.",
            )
        self._refresh_clip_overlay()

    # ------------------------------------------------------------------
    # disegno
    # ------------------------------------------------------------------
    def _refresh_clip_overlay(self, *_args) -> None:
        """(Ri)disegna contorni e poligono in corso sopra la slice."""
        ax = getattr(self, "_ax_slice", None)
        canvas = getattr(self, "_canvas_slice", None)
        if ax is None or canvas is None:
            return
        self._draw_clip_artists()
        try:
            self._safe_draw_idle(canvas)
        except Exception:
            pass

    def _draw_clip_artists(self) -> None:
        """Aggiorna gli artisti senza forzare il ridisegno (lo fa il chiamante)."""
        ax = getattr(self, "_ax_slice", None)
        if ax is None:
            return
        st = self._clip_state()
        for art in st["artists"]:
            try:
                art.remove()
            except Exception:
                pass  # l'asse e' stato svuotato da clear(): l'artista non c'e' piu'
        st["artists"] = []
        panel = self._clip_panel()
        rings = panel.clip_polygons() if panel is not None else []
        kw = dict(color=_CLIP_COLOR, linewidth=1.6, scalex=False, scaley=False, zorder=20)
        for ring in rings:
            xs = [p[0] for p in ring] + [ring[0][0]]
            ys = [p[1] for p in ring] + [ring[0][1]]
            st["artists"].extend(ax.plot(xs, ys, linestyle="--" if st["drawing"] else "-", **kw))
        pend = st["pending"]
        if st["drawing"] and pend:
            xs = [p[0] for p in pend]
            ys = [p[1] for p in pend]
            if st["cursor"] is not None:
                xs.append(st["cursor"][0])
                ys.append(st["cursor"][1])
            st["artists"].extend(ax.plot(xs, ys, linestyle="-", marker="o", markersize=4, **kw))
            if len(pend) >= 3:
                # lato di chiusura, tratteggiato
                st["artists"].extend(ax.plot(
                    [pend[-1][0] if st["cursor"] is None else st["cursor"][0], pend[0][0]],
                    [pend[-1][1] if st["cursor"] is None else st["cursor"][1], pend[0][1]],
                    linestyle=":", **kw))
