# -*- coding: utf-8 -*-
"""Preset di processing per strumento nel GPR Profile Viewer.

Il menu a tendina elenca tutti i preset e funziona con qualunque dato; il
pulsante rapido di uno strumento (es. "Preset Stream C", "Preset Opera Duo")
compare solo quando il profilo corrente e' stato acquisito da quello
strumento (vedi gpr_instrument_presets.detect_instrument). Un preset scrive i
valori nei controlli normali del pannello Processing, quindi resta modificabile."""

from __future__ import annotations

from .gpr_instrument_presets import (
    PRESETS,
    describe_profile_instrument,
    detect_instrument,
    preset_by_key,
)
from .gpr_viewer_common import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
    Qt,
)


class PresetsMixin:
    def _build_preset_controls(self, fl) -> None:
        """Aggiunge al pannello Processing la riga: [menu preset] [Applica] [Preset <strumento>]."""
        self._preset_quick_buttons: dict[str, QPushButton] = {}

        self._cb_preset = QComboBox()
        self._cb_preset.addItem("Scegli un preset...", None)
        for preset in PRESETS:
            self._cb_preset.addItem(preset.label, preset.key)
            self._cb_preset.setItemData(
                self._cb_preset.count() - 1, preset.description, Qt.ToolTipRole
            )
        self._cb_preset.setToolTip(
            "Preset di processing per strumento. Carica catena e parametri nei controlli\n"
            "sottostanti (modificabili); sostituisce la catena attuale."
        )

        btn_apply = QPushButton("Applica")
        btn_apply.setToolTip("Applica il preset scelto nel menu.")
        btn_apply.clicked.connect(self._on_apply_selected_preset)

        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lay.addWidget(self._cb_preset, 1)
        lay.addWidget(btn_apply)
        for preset in PRESETS:
            if not (preset.instrument and preset.button_label):
                continue
            btn = QPushButton(preset.button_label)
            btn.setToolTip(preset.description)
            btn.setVisible(False)
            btn.clicked.connect(lambda _checked=False, k=preset.key: self._apply_named_preset(k))
            lay.addWidget(btn)
            self._preset_quick_buttons[preset.key] = btn
        fl.addRow(QLabel("Preset:"), row)

        # Resta visibile dopo l'applicazione: la barra di stato viene sovrascritta
        # subito dal messaggio del processing.
        self._lbl_preset_info = QLabel("")
        self._lbl_preset_info.setWordWrap(True)
        self._lbl_preset_info.setStyleSheet("color: #666; font-style: italic;")
        self._lbl_preset_info.setVisible(False)
        fl.addRow("", self._lbl_preset_info)

    def _refresh_preset_controls(self, profile) -> None:
        """Mostra il pulsante dello strumento rilevato e lo propone nel menu (senza applicarlo)."""
        buttons = getattr(self, "_preset_quick_buttons", None)
        if buttons is None:
            return
        instrument = detect_instrument(profile) if profile is not None else None
        for key, btn in buttons.items():
            preset = preset_by_key(key)
            if preset is None:
                continue
            visible = instrument is not None and preset.instrument == instrument
            btn.setVisible(bool(visible))
            if visible:
                btn.setToolTip(
                    f"Rilevato: {describe_profile_instrument(profile)}.\n\n{preset.description}"
                )
        cb = getattr(self, "_cb_preset", None)
        if cb is not None and instrument is not None:
            for preset in PRESETS:
                if preset.instrument == instrument:
                    idx = cb.findData(preset.key)
                    if idx >= 0:
                        cb.setCurrentIndex(idx)
                    break

    def _on_apply_selected_preset(self) -> None:
        key = self._cb_preset.currentData()
        if key is None:
            self._safe_set_text(self._lbl_status, "Scegli un preset dal menu.")
            return
        self._apply_named_preset(str(key))

    def _apply_named_preset(self, key: str) -> None:
        preset = preset_by_key(key)
        if preset is None:
            return
        if not self._profiles or not (0 <= self._prof_idx < len(self._profiles)):
            self._safe_set_text(self._lbl_status, "Nessun profilo caricato: preset non applicato.")
            return
        prof = self._profiles[self._prof_idx]
        try:
            setup = preset.build(prof)
        except Exception as exc:
            self._safe_set_text(self._lbl_status, f"Preset {preset.label}: errore ({exc}).")
            return

        steps = [str(s) for s in setup["steps"]]
        order = steps + [s for s in self._current_chain_order() if s not in steps]
        params = dict(setup["params"])
        for sid in order:
            params[self._CHAIN_ENABLE_KEY.get(sid, sid)] = sid in steps
        params["chain_order"] = order

        # "Auto" del fondo non e' nei parametri salvati: lo imposto a mano, senza
        # segnali (evita un riprocessamento in piu' dal timer dei parametri).
        auto_chk = getattr(self, "_chk_bg_auto", None)
        if self._is_qt_alive(auto_chk):
            blocked = auto_chk.blockSignals(True)
            try:
                auto_chk.setChecked(bool(setup.get("bg_auto", False)))
            finally:
                auto_chk.blockSignals(blocked)

        self._apply_pipeline_params_dict(params, chain_order=order)

        spin = getattr(self, "_spin_bg_window", None)
        if self._is_qt_alive(spin) and self._is_qt_alive(auto_chk) and self._is_qt_alive(self._chk_bg):
            spin.setEnabled(bool(self._chk_bg.isChecked() and not auto_chk.isChecked()))

        msg = (
            f"Preset {preset.label} applicato: {setup['summary']}. "
            "Gli altri passi sono disattivati; tutti i valori restano modificabili."
        )
        self._safe_set_text(self._lbl_status, msg)
        info = getattr(self, "_lbl_preset_info", None)
        if self._is_qt_alive(info):
            info.setText(msg)
            info.setVisible(True)
