# -*- coding: utf-8 -*-
"""Finestra per gestire le classi di segnale delle annotazioni sul radargramma."""

from __future__ import annotations

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor, QIcon, QPixmap
from qgis.PyQt.QtWidgets import (
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
)

from .gpr_annotation_classes import (
    GEOMETRIES,
    GEOMETRY_LABELS,
    MAX_CLASSES,
    clean_classes,
    default_classes,
    normalize_color,
    unique_code,
)


def color_icon(color: str, size: int = 12) -> QIcon:
    pm = QPixmap(size, size)
    pm.fill(QColor(normalize_color(color)))
    return QIcon(pm)


class AnnotationClassesDialog(QDialog):
    """Tabella modificabile: nome, geometria con cui si disegna, colore.

    Il codice di ogni classe (quello salvato nel layer) non si vede e non cambia
    rinominando: le annotazioni gia' fatte restano collegate alla loro classe.
    """

    def __init__(self, classes, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Classi di segnale")
        self.setMinimumWidth(460)
        self._codes: list = []
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(
            f"Le classi (al massimo {MAX_CLASSES}) si scelgono con i tasti 1-9 mentre si lavora\n"
            "sul radargramma. Il colore le distingue sul radargramma e nella legenda."))
        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Nome", "Si disegna come", "Colore"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setColumnWidth(0, 230)
        self._table.setColumnWidth(1, 110)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setSelectionMode(QTableWidget.SingleSelection)
        lay.addWidget(self._table)

        row = QHBoxLayout()
        for text, slot in (("Aggiungi", self._add), ("Rimuovi", self._remove),
                           ("Su", lambda: self._move(-1)), ("Giù", lambda: self._move(1)),
                           ("Predefinite", self._restore_defaults)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        self._btn_add = row.itemAt(0).widget()
        row.addStretch(1)
        lay.addLayout(row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        for cls in classes:
            self._append(cls)
        self._sync_add_button()

    # -- righe -------------------------------------------------------------------------

    def _append(self, cls: dict) -> None:
        r = self._table.rowCount()
        self._table.insertRow(r)
        self._codes.append(cls.get("code") or "")
        self._table.setCellWidget(r, 0, QLineEdit(cls.get("label", "")))
        geom = QComboBox()
        for g in GEOMETRIES:
            geom.addItem(GEOMETRY_LABELS[g], g)
        geom.setCurrentIndex(max(0, GEOMETRIES.index(cls.get("geometry", "polygon"))))
        self._table.setCellWidget(r, 1, geom)
        btn = QPushButton()
        btn.setProperty("color", normalize_color(cls.get("color")))
        self._paint_color_button(btn)
        btn.clicked.connect(lambda _=False, b=btn: self._pick_color(b))
        self._table.setCellWidget(r, 2, btn)

    @staticmethod
    def _paint_color_button(btn) -> None:
        color = btn.property("color")
        btn.setText(str(color))
        btn.setIcon(color_icon(color))

    def _pick_color(self, btn) -> None:
        chosen = QColorDialog.getColor(QColor(btn.property("color")), self, "Colore della classe")
        if chosen.isValid():
            btn.setProperty("color", chosen.name())
            self._paint_color_button(btn)

    def _add(self) -> None:
        if self._table.rowCount() >= MAX_CLASSES:
            return
        current = self.classes()
        label = "Nuova classe"
        self._append({"code": unique_code(current, label), "label": label,
                      "geometry": "polygon", "color": "#ffa726"})
        self._table.setCurrentCell(self._table.rowCount() - 1, 0)
        self._sync_add_button()

    def _remove(self) -> None:
        r = self._table.currentRow()
        if r < 0 or self._table.rowCount() <= 1:
            return
        self._table.removeRow(r)
        del self._codes[r]
        self._sync_add_button()

    def _move(self, delta: int) -> None:
        r = self._table.currentRow()
        t = r + delta
        if r < 0 or not (0 <= t < self._table.rowCount()):
            return
        items = self.classes_raw()
        items[r], items[t] = items[t], items[r]
        self._reload(items)
        self._table.setCurrentCell(t, 0)

    def _restore_defaults(self) -> None:
        self._reload(default_classes())
        self._sync_add_button()

    def _reload(self, items) -> None:
        self._table.setRowCount(0)
        self._codes = []
        for cls in items:
            self._append(cls)

    def _sync_add_button(self) -> None:
        self._btn_add.setEnabled(self._table.rowCount() < MAX_CLASSES)

    # -- risultato ---------------------------------------------------------------------

    def classes_raw(self) -> list:
        out = []
        for r in range(self._table.rowCount()):
            label = self._table.cellWidget(r, 0).text().strip()
            out.append({
                "code": self._codes[r], "label": label,
                "geometry": self._table.cellWidget(r, 1).currentData(),
                "color": self._table.cellWidget(r, 2).property("color"),
            })
        return out

    def classes(self) -> list:
        """Elenco ripulito: le righe senza nome si scartano, i codici nuovi si assegnano qui."""
        items, taken = [], []
        for raw in self.classes_raw():
            if not raw["label"]:
                continue
            if not raw["code"]:
                raw["code"] = unique_code(taken, raw["label"])
            items.append(raw)
            taken.append(raw)
        return clean_classes(items)
