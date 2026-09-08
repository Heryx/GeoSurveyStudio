# -*- coding: utf-8 -*-
"""
GprManualPositioningDialog
==========================
Dialog interattivo per la definizione manuale della geometria di acquisizione
di profili GPR privi di coordinate GPS/GEC.

Caratteristiche:
  - Definizione griglia bounding box (X0 - X1, Y0 - Y1) in metri.
  - Spaziatura automatica uniforme dei profili lungo l'asse ortogonale.
  - Supporto per profili paralleli all'asse X e profili ortogonali all'asse Y.
  - Tabella profili con rinomina, riordino (Sposta Su / Giu'), offset personalizzato.
  - Gestione reversing / zig-zag (inversione alternata o manuale della direzione di percorrenza).
  - Georeferenziazione opzionale su CRS di progetto (Origine E0, N0 + Azimut rotazione).
  - Anteprima visuale 2D della griglia con frecce direzionali.
"""

from __future__ import annotations

import math
import os
from typing import Any, Callable

import numpy as np

from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QDoubleSpinBox, QSpinBox, QCheckBox, QComboBox,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QGroupBox, QSplitter, QWidget, QMessageBox, QDialogButtonBox,
    QToolButton, QAbstractItemView,
)

try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def check_profiles_have_gps(profiles: list) -> bool:
    """Verifica se almeno un canale dei profili caricati possiede coordinate GPS reali plausibili."""
    if not profiles:
        return False
    for prof in profiles:
        channels = getattr(prof, "channels", []) or []
        for ch in channels:
            e = np.asarray(getattr(ch, "easting", []), dtype=np.float64)
            n = np.asarray(getattr(ch, "northing", []), dtype=np.float64)
            if e.size > 1 and n.size > 1:
                fe = e[np.isfinite(e)]
                fn = n[np.isfinite(n)]
                if fe.size > 1 and fn.size > 1:
                    span_e = float(np.ptp(fe))
                    span_n = float(np.ptp(fn))
                    # Coordinate reali UTM/proiettate o con estensione metrica non nulla
                    if (span_e + span_n) > 0.05 and (float(np.nanmax(fe)) > 1000.0 or float(np.nanmax(fn)) > 1000.0):
                        return True
    return False


class GprManualPositioningDialog(QDialog):
    """Dialog per la configurazione della griglia e l'assegnazione coordinate a profili non-GPS."""

    positioning_applied = pyqtSignal(dict)

    def __init__(self, profiles: list, project_crs=None, canvas=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Posizionamento Manuale Griglia GPR (Profili senza GPS)")
        self.resize(1020, 680)

        self._profiles = list(profiles or [])
        self._project_crs = project_crs
        self._canvas = canvas
        self._syncing_table = False

        # Struttura interna dati per profilo:
        # dict(prof=prof, orig_name=str, name=str, axis="X"|"Y", offset=float, reversed=bool, n_traces=int, length=float)
        self._profile_items = self._init_profile_items()

        self._build_ui()
        self._recalculate_spacing(auto_detect_dims=True)
        self._update_table_from_items()
        self._draw_preview()

    def _init_profile_items(self) -> list[dict]:
        items = []
        for i, prof in enumerate(self._profiles):
            path = getattr(prof, "path", f"profile_{i + 1}") or f"profile_{i + 1}"
            orig_name = os.path.splitext(os.path.basename(path))[0]
            swath = getattr(prof, "swath_name", "") or ""
            name = swath if swath else orig_name

            n_slices = int(getattr(prof, "n_slices", 0) or 0)
            if n_slices <= 0 and getattr(prof, "channels", None):
                n_slices = int(prof.channels[0].data.shape[1]) if prof.channels[0].data.ndim == 2 else 0

            step = float(getattr(prof, "sampling_step_m", 0.05) or 0.05)
            length = float(max(0, n_slices - 1) * step) if n_slices > 1 else 10.0

            items.append({
                "prof": prof,
                "orig_name": orig_name,
                "name": name,
                "axis": "X",
                "offset": float(i * 1.0),
                "reversed": False,
                "n_traces": n_slices,
                "sampling_step": step,
                "length": max(0.1, length),
            })
        return items

    def _build_ui(self):
        root_layout = QVBoxLayout(self)

        # Info header
        lbl_info = QLabel(
            "<b>I profili caricati non contengono coordinate GPS/GEC.</b><br>"
            "Imposta le dimensioni della griglia ($X_0-X_1$, $Y_0-Y_1$) e verifica "
            "la direzione delle linee (zig-zag). Il plugin calcolerà automaticamente le coordinate di ciascuna traccia."
        )
        lbl_info.setWordWrap(True)
        root_layout.addWidget(lbl_info)

        splitter = QSplitter(Qt.Horizontal)

        # -------------------------------------------------------------
        # Pannello Sinistro: Parametri Griglia + Tabella Profili
        # -------------------------------------------------------------
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Group Box Griglia Bounding Box
        grp_grid = QGroupBox("Dimensioni Griglia di Rilievo")
        grid_form = QGridLayout(grp_grid)

        grid_form.addWidget(QLabel("Asse X (Lunghezza): da"), 0, 0)
        self._spin_x0 = QDoubleSpinBox()
        self._spin_x0.setRange(-10000.0, 10000.0)
        self._spin_x0.setValue(0.0)
        self._spin_x0.setSuffix(" m")
        self._spin_x0.valueChanged.connect(self._on_grid_bounds_changed)
        grid_form.addWidget(self._spin_x0, 0, 1)

        grid_form.addWidget(QLabel("a"), 0, 2)
        self._spin_x1 = QDoubleSpinBox()
        self._spin_x1.setRange(-10000.0, 10000.0)
        self._spin_x1.setValue(10.0)
        self._spin_x1.setSuffix(" m")
        self._spin_x1.valueChanged.connect(self._on_grid_bounds_changed)
        grid_form.addWidget(self._spin_x1, 0, 3)

        grid_form.addWidget(QLabel("Asse Y (Larghezza): da"), 1, 0)
        self._spin_y0 = QDoubleSpinBox()
        self._spin_y0.setRange(-10000.0, 10000.0)
        self._spin_y0.setValue(0.0)
        self._spin_y0.setSuffix(" m")
        self._spin_y0.valueChanged.connect(self._on_grid_bounds_changed)
        grid_form.addWidget(self._spin_y0, 1, 1)

        grid_form.addWidget(QLabel("a"), 1, 2)
        self._spin_y1 = QDoubleSpinBox()
        self._spin_y1.setRange(-10000.0, 10000.0)
        self._spin_y1.setValue(max(1.0, float(len(self._profile_items) - 1)))
        self._spin_y1.setSuffix(" m")
        self._spin_y1.valueChanged.connect(self._on_grid_bounds_changed)
        grid_form.addWidget(self._spin_y1, 1, 3)

        btn_recalc = QPushButton("Ricalcola spaziature uniformi")
        btn_recalc.setToolTip("Ripartisce uniformemente i profili all'interno dei limiti X0-X1 e Y0-Y1")
        btn_recalc.clicked.connect(lambda: self._recalculate_spacing(auto_detect_dims=False))
        grid_form.addWidget(btn_recalc, 2, 0, 1, 4)

        left_layout.addWidget(grp_grid)

        # Group Box Georeferenziazione GIS Opzionale
        grp_geo = QGroupBox("Georeferenziazione GIS nel Progetto (Opzionale)")
        grp_geo.setCheckable(True)
        grp_geo.setChecked(False)
        self._grp_geo = grp_geo
        geo_form = QGridLayout(grp_geo)

        geo_form.addWidget(QLabel("Origine Est (E0):"), 0, 0)
        self._spin_e0 = QDoubleSpinBox()
        self._spin_e0.setRange(-1e9, 1e9)
        self._spin_e0.setDecimals(3)
        self._spin_e0.setValue(0.0)
        self._spin_e0.valueChanged.connect(self._draw_preview)
        geo_form.addWidget(self._spin_e0, 0, 1)

        geo_form.addWidget(QLabel("Origine Nord (N0):"), 0, 2)
        self._spin_n0 = QDoubleSpinBox()
        self._spin_n0.setRange(-1e9, 1e9)
        self._spin_n0.setDecimals(3)
        self._spin_n0.setValue(0.0)
        self._spin_n0.valueChanged.connect(self._draw_preview)
        geo_form.addWidget(self._spin_n0, 0, 3)

        geo_form.addWidget(QLabel("Azimut griglia:"), 1, 0)
        self._spin_azimuth = QDoubleSpinBox()
        self._spin_azimuth.setRange(-360.0, 360.0)
        self._spin_azimuth.setValue(0.0)
        self._spin_azimuth.setSuffix(" °")
        self._spin_azimuth.setToolTip("Angolo in gradi rispetto al Nord (0° = asse X verso Nord, 90° = asse X verso Est).")
        self._spin_azimuth.valueChanged.connect(self._draw_preview)
        geo_form.addWidget(self._spin_azimuth, 1, 1)

        btn_pick_origin = QPushButton("Punta su Mappa QGIS")
        btn_pick_origin.setToolTip("Clicca sul canvas di QGIS per impostare E0, N0")
        btn_pick_origin.clicked.connect(self._pick_origin_from_canvas)
        geo_form.addWidget(btn_pick_origin, 1, 2, 1, 2)

        grp_geo.toggled.connect(self._draw_preview)
        left_layout.addWidget(grp_geo)

        # Tabella Profili e Strumenti Rapidi
        grp_table = QGroupBox("Elenco Profili e Sequenza")
        tbl_layout = QVBoxLayout(grp_table)

        # Toolbar bottoni rapidi tabella
        tbl_tools = QHBoxLayout()
        btn_zigzag = QPushButton("Zig-Zag (Alterna)")
        btn_zigzag.setToolTip("Inverte la direzione delle linee alternate (0 avanti, 1 indietro, 2 avanti...)")
        btn_zigzag.clicked.connect(self._apply_zigzag)
        tbl_tools.addWidget(btn_zigzag)

        btn_inv_all = QPushButton("Inverti tutti")
        btn_inv_all.clicked.connect(self._invert_all)
        tbl_tools.addWidget(btn_inv_all)

        btn_reset_inv = QPushButton("Ripristina versi")
        btn_reset_inv.clicked.connect(self._reset_inversions)
        tbl_tools.addWidget(btn_reset_inv)

        tbl_tools.addStretch(1)

        btn_up = QPushButton("▲ Su")
        btn_up.clicked.connect(self._move_selected_up)
        tbl_tools.addWidget(btn_up)

        btn_down = QPushButton("▼ Giù")
        btn_down.clicked.connect(self._move_selected_down)
        tbl_tools.addWidget(btn_down)

        tbl_layout.addLayout(tbl_tools)

        self._table = QTableWidget(0, 7)
        self._table.setHorizontalHeaderLabels([
            "#", "Nome Profilo", "Asse", "Offset (m)", "Invertito (Zig-zag)", "N Tracce", "Lunghezza (m)"
        ])
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.itemChanged.connect(self._on_table_item_changed)
        tbl_layout.addWidget(self._table)

        left_layout.addWidget(grp_table)
        splitter.addWidget(left_widget)

        # -------------------------------------------------------------
        # Pannello Destro: Anteprima Schema 2D
        # -------------------------------------------------------------
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        grp_preview = QGroupBox("Anteprima Schema Geometrico")
        prev_layout = QVBoxLayout(grp_preview)

        if HAS_MPL:
            self._fig = Figure(figsize=(5, 5), dpi=100)
            self._canvas_mpl = FigureCanvasQTAgg(self._fig)
            self._ax = self._fig.add_subplot(111)
            prev_layout.addWidget(self._canvas_mpl)
        else:
            self._canvas_mpl = None
            self._ax = None
            lbl_no_mpl = QLabel("Anteprima non disponibile (matplotlib mancante).")
            prev_layout.addWidget(lbl_no_mpl)

        right_layout.addWidget(grp_preview)
        splitter.addWidget(right_widget)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root_layout.addWidget(splitter, 1)

        # Dialog Buttons
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.button(QDialogButtonBox.Ok).setText("Applica Geometria")
        btn_box.accepted.connect(self._apply_and_accept)
        btn_box.rejected.connect(self.reject)
        root_layout.addWidget(btn_box)

    # ------------------------------------------------------------------
    # Calcoli e Azioni Geometria
    # ------------------------------------------------------------------

    def _on_grid_bounds_changed(self):
        self._recalculate_spacing(auto_detect_dims=False)

    def _recalculate_spacing(self, auto_detect_dims: bool = False):
        """Ricalcola gli offset dei profili in base ai limiti correnti."""
        if not self._profile_items:
            return

        x_items = [it for it in self._profile_items if it["axis"] == "X"]
        y_items = [it for it in self._profile_items if it["axis"] == "Y"]

        if auto_detect_dims:
            # Stima lunghezza max dalle tracce
            max_len = max([it["length"] for it in self._profile_items], default=10.0)
            self._spin_x0.blockSignals(True)
            self._spin_x1.blockSignals(True)
            self._spin_x0.setValue(0.0)
            self._spin_x1.setValue(float(round(max_len, 2)))
            self._spin_x0.blockSignals(False)
            self._spin_x1.blockSignals(False)

            y_span = float(max(1.0, len(x_items) - 1)) if x_items else 10.0
            self._spin_y0.blockSignals(True)
            self._spin_y1.blockSignals(True)
            self._spin_y0.setValue(0.0)
            self._spin_y1.setValue(y_span)
            self._spin_y0.blockSignals(False)
            self._spin_y1.blockSignals(False)

        x0 = float(self._spin_x0.value())
        x1 = float(self._spin_x1.value())
        y0 = float(self._spin_y0.value())
        y1 = float(self._spin_y1.value())

        # Assegna offset per profili X
        if x_items:
            nx = len(x_items)
            step_y = (y1 - y0) / float(max(1, nx - 1)) if nx > 1 else 0.0
            for idx, it in enumerate(x_items):
                it["offset"] = float(round(y0 + idx * step_y, 4))

        # Assegna offset per profili Y (ortogonali)
        if y_items:
            ny = len(y_items)
            step_x = (x1 - x0) / float(max(1, ny - 1)) if ny > 1 else 0.0
            for idx, it in enumerate(y_items):
                it["offset"] = float(round(x0 + idx * step_x, 4))

        self._update_table_from_items()
        self._draw_preview()

    def _apply_zigzag(self):
        """Inverte la direzione delle linee alternate (0 forward, 1 reversed, 2 forward, ...)."""
        for i, it in enumerate(self._profile_items):
            it["reversed"] = (i % 2 == 1)
        self._update_table_from_items()
        self._draw_preview()

    def _invert_all(self):
        for it in self._profile_items:
            it["reversed"] = True
        self._update_table_from_items()
        self._draw_preview()

    def _reset_inversions(self):
        for it in self._profile_items:
            it["reversed"] = False
        self._update_table_from_items()
        self._draw_preview()

    def _move_selected_up(self):
        r = self._table.currentRow()
        if r > 0:
            self._profile_items[r - 1], self._profile_items[r] = self._profile_items[r], self._profile_items[r - 1]
            self._recalculate_spacing(auto_detect_dims=False)
            self._table.selectRow(r - 1)

    def _move_selected_down(self):
        r = self._table.currentRow()
        if 0 <= r < len(self._profile_items) - 1:
            self._profile_items[r + 1], self._profile_items[r] = self._profile_items[r], self._profile_items[r + 1]
            self._recalculate_spacing(auto_detect_dims=False)
            self._table.selectRow(r + 1)

    def _pick_origin_from_canvas(self):
        """Permette di acquisire E0, N0 cliccando sulla mappa QGIS."""
        if not self._canvas:
            QMessageBox.information(self, "Punta Origine", "Map canvas di QGIS non disponibile.")
            return
        try:
            from qgis.gui import QgsMapToolEmitPoint

            def _on_canvas_clicked(pt):
                self._spin_e0.setValue(float(pt.x()))
                self._spin_n0.setValue(float(pt.y()))
                self._grp_geo.setChecked(True)
                if hasattr(self, "_temp_map_tool"):
                    self._canvas.unsetMapTool(self._temp_map_tool)
                self.show()
                self._draw_preview()

            self._temp_map_tool = QgsMapToolEmitPoint(self._canvas)
            self._temp_map_tool.canvasClicked.connect(_on_canvas_clicked)
            self._canvas.setMapTool(self._temp_map_tool)
            self.hide()
        except Exception as exc:
            QMessageBox.warning(self, "Punta Origine", f"Impossibile attivare il tool di puntamento:\n{exc}")

    # ------------------------------------------------------------------
    # Sincronizzazione Tabella
    # ------------------------------------------------------------------

    def _update_table_from_items(self):
        self._syncing_table = True
        try:
            self._table.setRowCount(len(self._profile_items))
            for r, it in enumerate(self._profile_items):
                # # ID
                item_id = QTableWidgetItem(str(r + 1))
                item_id.setFlags(item_id.flags() & ~Qt.ItemIsEditable)
                item_id.setTextAlignment(Qt.AlignCenter)
                self._table.setItem(r, 0, item_id)

                # Nome profilo
                item_name = QTableWidgetItem(str(it["name"]))
                self._table.setItem(r, 1, item_name)

                # Asse combo
                cb_axis = QComboBox()
                cb_axis.addItems(["X", "Y"])
                cb_axis.setCurrentText(it["axis"])
                cb_axis.currentTextChanged.connect(lambda txt, row=r: self._on_axis_changed(row, txt))
                self._table.setCellWidget(r, 2, cb_axis)

                # Offset spin
                spin_off = QDoubleSpinBox()
                spin_off.setRange(-10000.0, 10000.0)
                spin_off.setDecimals(3)
                spin_off.setValue(float(it["offset"]))
                spin_off.valueChanged.connect(lambda val, row=r: self._on_offset_changed(row, val))
                self._table.setCellWidget(r, 3, spin_off)

                # Checkbox Reversing
                chk_rev = QCheckBox()
                chk_rev.setChecked(bool(it["reversed"]))
                chk_rev.toggled.connect(lambda chk, row=r: self._on_reversed_changed(row, chk))
                chk_container = QWidget()
                chk_lay = QHBoxLayout(chk_container)
                chk_lay.addWidget(chk_rev)
                chk_lay.setAlignment(Qt.AlignCenter)
                chk_lay.setContentsMargins(0, 0, 0, 0)
                self._table.setCellWidget(r, 4, chk_container)

                # N Tracce
                item_tr = QTableWidgetItem(str(it["n_traces"]))
                item_tr.setFlags(item_tr.flags() & ~Qt.ItemIsEditable)
                item_tr.setTextAlignment(Qt.AlignCenter)
                self._table.setItem(r, 5, item_tr)

                # Lunghezza
                item_len = QTableWidgetItem(f"{it['length']:.2f}")
                item_len.setFlags(item_len.flags() & ~Qt.ItemIsEditable)
                item_len.setTextAlignment(Qt.AlignCenter)
                self._table.setItem(r, 6, item_len)
        finally:
            self._syncing_table = False

    def _on_table_item_changed(self, item):
        if self._syncing_table or not item:
            return
        r = item.row()
        c = item.column()
        if c == 1 and 0 <= r < len(self._profile_items):
            self._profile_items[r]["name"] = str(item.text()).strip()
            self._draw_preview()

    def _on_axis_changed(self, row: int, axis_text: str):
        if 0 <= row < len(self._profile_items):
            self._profile_items[row]["axis"] = str(axis_text).strip()
            self._draw_preview()

    def _on_offset_changed(self, row: int, val: float):
        if 0 <= row < len(self._profile_items):
            self._profile_items[row]["offset"] = float(val)
            self._draw_preview()

    def _on_reversed_changed(self, row: int, is_reversed: bool):
        if 0 <= row < len(self._profile_items):
            self._profile_items[row]["reversed"] = bool(is_reversed)
            self._draw_preview()

    # ------------------------------------------------------------------
    # Disegno Anteprima 2D
    # ------------------------------------------------------------------

    def _draw_preview(self):
        if not HAS_MPL or not self._ax:
            return

        self._ax.clear()

        x0 = float(self._spin_x0.value())
        x1 = float(self._spin_x1.value())
        y0 = float(self._spin_y0.value())
        y1 = float(self._spin_y1.value())

        geo_on = self._grp_geo.isChecked()
        e0 = float(self._spin_e0.value()) if geo_on else 0.0
        n0 = float(self._spin_n0.value()) if geo_on else 0.0
        az_deg = float(self._spin_azimuth.value()) if geo_on else 0.0
        az_rad = math.radians(az_deg)
        cos_a, sin_a = math.cos(az_rad), math.sin(az_rad)

        def _to_display_xy(lx, ly):
            if not geo_on:
                return lx, ly
            # Rotazione azimutale + traslazione
            de = lx * cos_a + ly * sin_a
            dn = -lx * sin_a + ly * cos_a
            return e0 + de, n0 + dn

        # Disegna contorno griglia
        corners_local = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        corners_disp = [_to_display_xy(cx, cy) for cx, cy in corners_local]
        bx = [p[0] for p in corners_disp]
        by = [p[1] for p in corners_disp]
        self._ax.plot(bx, by, "k--", lw=1.2, alpha=0.6, label="Limite griglia")

        # Disegna linee dei profili con frecce direzionali
        for idx, it in enumerate(self._profile_items):
            axis = it["axis"]
            offset = it["offset"]
            is_rev = it["reversed"]

            if axis == "X":
                p_start_loc = (x1, offset) if is_rev else (x0, offset)
                p_end_loc = (x0, offset) if is_rev else (x1, offset)
            else:
                p_start_loc = (offset, y1) if is_rev else (offset, y0)
                p_end_loc = (offset, y0) if is_rev else (offset, y1)

            ps_disp = _to_display_xy(*p_start_loc)
            pe_disp = _to_display_xy(*p_end_loc)

            color = "#1f77b4" if axis == "X" else "#ff7f0e"
            self._ax.plot([ps_disp[0], pe_disp[0]], [ps_disp[1], pe_disp[1]], color=color, lw=2.0)

            # Freccia direzionale
            dx = pe_disp[0] - ps_disp[0]
            dy = pe_disp[1] - ps_disp[1]
            arrow_pos_x = ps_disp[0] + 0.6 * dx
            arrow_pos_y = ps_disp[1] + 0.6 * dy
            self._ax.annotate(
                "",
                xy=(arrow_pos_x + 0.1 * dx, arrow_pos_y + 0.1 * dy),
                xytext=(arrow_pos_x, arrow_pos_y),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.8),
            )

            # Etichetta linea
            lbl_pos_x = ps_disp[0] - 0.03 * dx
            lbl_pos_y = ps_disp[1] - 0.03 * dy
            self._ax.text(
                lbl_pos_x, lbl_pos_y, f"{idx + 1}",
                color="black", fontsize=8, fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec=color, lw=1.0),
            )

        self._ax.set_aspect("equal", "datalim")
        self._ax.grid(True, linestyle=":", alpha=0.5)
        if geo_on:
            self._ax.set_xlabel("Est (m)")
            self._ax.set_ylabel("Nord (m)")
            self._ax.set_title(f"Griglia orientata (Azimut {az_deg:.1f}°)")
        else:
            self._ax.set_xlabel("X (m)")
            self._ax.set_ylabel("Y (m)")
            self._ax.set_title("Griglia locale (X0-X1, Y0-Y1)")

        self._fig.tight_layout()
        self._canvas_mpl.draw()

    # ------------------------------------------------------------------
    # Applicazione Finale
    # ------------------------------------------------------------------

    def _apply_and_accept(self):
        """Calcola le coordinate finali su ciascun OgprProfile e chiude il dialog."""
        x0 = float(self._spin_x0.value())
        x1 = float(self._spin_x1.value())
        y0 = float(self._spin_y0.value())
        y1 = float(self._spin_y1.value())

        geo_on = self._grp_geo.isChecked()
        e0 = float(self._spin_e0.value()) if geo_on else 0.0
        n0 = float(self._spin_n0.value()) if geo_on else 0.0
        az_deg = float(self._spin_azimuth.value()) if geo_on else 0.0
        az_rad = math.radians(az_deg)
        cos_a, sin_a = math.cos(az_rad), math.sin(az_rad)

        configured_records = []

        for idx, it in enumerate(self._profile_items):
            prof = it["prof"]
            axis = it["axis"]
            offset = it["offset"]
            is_rev = it["reversed"]
            custom_name = it["name"]

            # Aggiorna nome logico del profilo
            if hasattr(prof, "swath_name"):
                prof.swath_name = str(custom_name)

            channels = getattr(prof, "channels", []) or []
            n_t = it["n_traces"]

            if n_t > 0:
                t_ratio = np.linspace(0.0, 1.0, n_t, dtype=np.float64)
                if is_rev:
                    t_ratio = t_ratio[::-1]

                if axis == "X":
                    lx = x0 + t_ratio * (x1 - x0)
                    ly = np.full(n_t, offset, dtype=np.float64)
                else:
                    lx = np.full(n_t, offset, dtype=np.float64)
                    ly = y0 + t_ratio * (y1 - y0)

                if geo_on:
                    de = lx * cos_a + ly * sin_a
                    dn = -lx * sin_a + ly * cos_a
                    final_e = e0 + de
                    final_n = n0 + dn
                else:
                    final_e = lx
                    final_n = ly

                # Distanza cumulativa calcolata dalla linea metrica
                dx = np.diff(lx)
                dy = np.diff(ly)
                dist_seg = np.sqrt(dx ** 2 + dy ** 2)
                final_dist = np.concatenate([[0.0], np.cumsum(dist_seg)])

                # Assegna a tutti i canali del profilo
                for ch in channels:
                    ch.easting = final_e.copy()
                    ch.northing = final_n.copy()
                    ch.distances = final_dist.copy()

            configured_records.append({
                "index": idx,
                "path": getattr(prof, "path", ""),
                "name": custom_name,
                "axis": axis,
                "offset": offset,
                "reversed": is_rev,
                "n_traces": n_t,
            })

        result_payload = {
            "grid_bounds": {"x0": x0, "x1": x1, "y0": y0, "y1": y1},
            "georef": {
                "enabled": geo_on,
                "origin_e": e0,
                "origin_n": n0,
                "azimuth_deg": az_deg,
            },
            "profiles": configured_records,
        }

        self.positioning_applied.emit(result_payload)
        self.accept()
