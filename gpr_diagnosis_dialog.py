"""Diagnosi automatica del radargramma, visualizzazione log e assistente AI."""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.request

import numpy as np
from qgis.PyQt.QtCore import QSettings, QThread, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPlainTextEdit, QPushButton, QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)

from .gpr_ai_panel import AiPanel, read_log_tail
from .gpr_processing import LOG_FILE_PATH, diagnose_radargram

RECOMMENDED_CHAIN = [
    "dewow", "timezero", "bg_removal", "bandpass",
    "migration", "pre_agc_gain", "agc",
]

CASCADE_HELP = (
    "Sequenza consigliata dei filtri:\n"
    "  1. Dewow            - toglie la deriva a bassa frequenza (prima di tutto)\n"
    "  2. Tempo zero       - allinea il suolo a profondita' 0, scarta l'aria sopra\n"
    "  3. Rimozione fondo  - toglie le bande orizzontali (aria/ringing), mediana line_by_line\n"
    "  4. Passa-banda      - toglie rumore fuori dalla banda dell'antenna\n"
    "  5. Migrazione       - solo se serve collassare le iperboli (velocita' nota)\n"
    "  6. Guadagno         - Range gain (TVG) oppure AGC, non entrambi; con AGC usare\n"
    "                        'Da campione' subito dopo il burst dell'onda diretta\n"
    "Altri filtri (DC, notch, SVD, FK, ...) vanno aggiunti con il tasto +, solo se servono."
)

class DiagnosisDialog(QWidget):
    """Pannello Diagnosi / Log / AI: usabile come dock laterale o finestra."""

    def __init__(self, viewer):
        super().__init__()
        self._v = viewer
        self._report: dict = {}
        self._worker = None
        self.setWindowTitle("Diagnosi automatica / Log / Assistente AI")
        self.resize(560, 760)
        tabs = QTabWidget(self)
        lay = QVBoxLayout(self)
        lay.addWidget(tabs)

        # --- Diagnosi ---
        w = QWidget()
        wl = QVBoxLayout(w)
        self._txt = QTextEdit()
        self._txt.setReadOnly(True)
        wl.addWidget(self._txt)
        row = QHBoxLayout()
        b1 = QPushButton("Ricalcola")
        b1.clicked.connect(self.run_diagnosis)
        row.addWidget(b1)
        wl.addLayout(row)
        tabs.addTab(w, "Diagnosi")

        # --- Log ---
        w = QWidget()
        wl = QVBoxLayout(w)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setLineWrapMode(QPlainTextEdit.NoWrap)
        wl.addWidget(self._log)
        row = QHBoxLayout()
        b = QPushButton("Aggiorna")
        b.clicked.connect(self.refresh_log)
        row.addWidget(b)
        row.addWidget(QLabel(f"File: {LOG_FILE_PATH}"))
        row.addStretch(1)
        wl.addLayout(row)
        tabs.addTab(w, "Log")

        tabs.addTab(AiPanel(viewer, lambda: self._report), "Assistente AI")

        self.run_diagnosis()
        self.refresh_log()

    # ------------------------------------------------------------------
    def _dt_ns(self) -> float:
        try:
            return float(self._v._profiles[self._v._prof_idx].dt_ns)
        except Exception:
            return 0.117

    def run_diagnosis(self):
        raw = self._v._raw_data
        if raw is None:
            self._txt.setPlainText("Nessun profilo caricato.")
            return
        try:
            self._report = diagnose_radargram(raw, self._dt_ns())
        except Exception as exc:
            self._txt.setPlainText(f"Errore diagnosi: {exc}")
            return
        r = self._report
        lines = ["DIAGNOSI", ""] + [f"- {n}" for n in r.get("notes", [])]
        p = r.get("params", {})
        if p:
            lines += ["", "PARAMETRI SUGGERITI"]
            lines += [f"- {k} = {v}" for k, v in p.items()]
        lines += ["", CASCADE_HELP]
        self._txt.setPlainText("\n".join(lines))

    def refresh_log(self):
        self._log.setPlainText(read_log_tail())
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())
