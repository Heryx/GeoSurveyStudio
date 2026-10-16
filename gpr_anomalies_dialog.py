# -*- coding: utf-8 -*-
"""Tabella delle anomalie: le annotazioni dello stesso codice, su tutti i profili, come un oggetto."""

from __future__ import annotations

from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from .gpr_annotation_analysis import anomalies_to_csv

_COLUMNS = ("Codice", "Classi", "Annotazioni", "Profili", "Profondità (m)", "Contrasto medio")


class AnomaliesDialog(QDialog):
    """Una riga per codice anomalia. Doppio clic (o "Vai") porta alla prima annotazione.

    ``goto(profile_stem, channel, feature_id)`` e' il segnale di navigazione: il pannello
    apre quel profilo/canale e seleziona l'annotazione.
    """

    goto = pyqtSignal(str, int, str)

    def __init__(self, rows, without_code: int = 0, class_labels: dict | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Anomalie")
        self.setMinimumSize(620, 320)
        self._rows = list(rows)
        self._labels = dict(class_labels or {})
        lay = QVBoxLayout(self)
        if self._rows:
            note = "Lo stesso codice su più profili è un solo oggetto. Doppio clic per andare alla prima annotazione."
        else:
            note = "Nessuna annotazione ha ancora un codice anomalia: scrivilo nella scheda dell'annotazione."
        if without_code:
            note += f"\n{without_code} annotazioni sono ancora senza codice."
        lbl = QLabel(note)
        lbl.setWordWrap(True)
        lay.addWidget(lbl)
        self._table = QTableWidget(len(self._rows), len(_COLUMNS))
        self._table.setHorizontalHeaderLabels(list(_COLUMNS))
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setSelectionMode(QTableWidget.SingleSelection)
        self._table.horizontalHeader().setStretchLastSection(True)
        for r, row in enumerate(self._rows):
            for c, text in enumerate(self._cells(row)):
                item = QTableWidgetItem(text)
                if c >= 2:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self._table.setItem(r, c, item)
        self._table.resizeColumnsToContents()
        self._table.cellDoubleClicked.connect(lambda r, _c: self._go(r))
        lay.addWidget(self._table, 1)
        buttons = QHBoxLayout()
        self._btn_go = QPushButton("Vai")
        self._btn_go.clicked.connect(lambda: self._go(self._table.currentRow()))
        self._btn_csv = QPushButton("Esporta CSV…")
        self._btn_csv.setToolTip("Tabella delle anomalie in CSV (separatore «;», si apre in Excel).")
        self._btn_csv.clicked.connect(self._export_csv)
        self._btn_csv.setEnabled(bool(self._rows))
        self._btn_close = QPushButton("Chiudi")
        self._btn_close.clicked.connect(self.accept)
        buttons.addWidget(self._btn_go)
        buttons.addWidget(self._btn_csv)
        buttons.addStretch(1)
        buttons.addWidget(self._btn_close)
        lay.addLayout(buttons)
        if self._rows:
            self._table.selectRow(0)

    def _cells(self, row) -> list:
        classes = ", ".join(self._labels.get(c, c) for c in row["classes"]) or "—"
        if row["depth_from"] is None or row["depth_to"] is None:
            depth = "n/d"
        else:
            depth = f"{row['depth_from']:.2f}–{row['depth_to']:.2f}"
        contrast = "—" if row["contrast_db"] is None else f"{row['contrast_db']:+.1f} dB"
        return [row["code"], classes, str(row["n"]), str(row["profiles"]), depth, contrast]

    def _go(self, r: int) -> None:
        if not (0 <= r < len(self._rows)):
            return
        first = self._rows[r]["first"]
        self.goto.emit(first["profile_stem"], int(first["channel"]), first["id"])

    def export_csv_to(self, path: str) -> int:
        return anomalies_to_csv(self._rows, path, self._labels)

    def _export_csv(self) -> None:
        path, _flt = QFileDialog.getSaveFileName(self, "Esporta anomalie", "anomalie.csv", "CSV (*.csv)")
        if path:
            self.export_csv_to(path)
