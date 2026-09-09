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
import re
from typing import Any, Callable

import numpy as np

from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QDoubleSpinBox, QSpinBox, QCheckBox, QComboBox,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QGroupBox, QSplitter, QWidget, QMessageBox, QDialogButtonBox,
    QToolButton, QAbstractItemView, QLineEdit, QFileDialog,
)

try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:  # navigazione: primitive marker / dead-reckoning / stagger
    from .navigation import reverse_along, stagger_shift
except ImportError:  # pragma: no cover
    from navigation import reverse_along, stagger_shift


def compute_positions(
    n_traces: int,
    x0: float, x1: float, y0: float, y1: float,
    axis: str,
    offset: float,
    grid_span_m: float,
    length_m: float,
    use_real_length: bool,
    reversed_: bool,
    stagger_m: float = 0.0,
    geo_on: bool = False,
    e0: float = 0.0, n0: float = 0.0, az_deg: float = 0.0,
    elev_mode: str = "none",
    z_start: float = 0.0, z_end: float = 0.0,
    elev_ref: "np.ndarray | list | None" = None,
    elev_ref_kind: str = "x",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calcola E/N/distanza cumulativa/quota per gli scan di un profilo.

    - ``use_real_length`` : se True il profilo occupa solo ``length_m`` a partire
      dall'inizio della linea (linee ANG / parziali), altrimenti l'intero span
      della griglia lungo l'asse.
    - ``reversed_``        : linea percorsa in direzione opposta (zig-zag).
    - ``stagger_m``        : traslazione applicata SOLO alle linee invertite,
      per correggere lo staggering fra andata e ritorno.
    - ``elev_mode``        : ``"none"`` (quota 0), ``"plane"`` (piano inclinato:
      ``z_start`` all'inizio linea, ``z_end`` alla fine — interpolazione lineare
      sulla frazione di linea, tiene conto del verso zig-zag) o ``"file"``
      (interpola ``elev_ref``, array Nx2, contro la posizione lungo linea in
      metri se ``elev_ref_kind == "x"`` o contro l'indice di traccia se
      ``"trace"``).

    Ritorna ``(east, north, dist, elev)`` (array lunghi ``n_traces``).
    """
    n_t = int(n_traces)
    if n_t <= 0:
        z = np.zeros(0, dtype=np.float64)
        return z, z.copy(), z.copy(), z.copy()

    span = float(grid_span_m) if float(grid_span_m) > 1e-9 else 1.0
    L = float(length_m) if (use_real_length and float(length_m) > 1e-9) else span
    # ascissa metrica lungo la linea, 0..L
    along = np.linspace(0.0, L, n_t, dtype=np.float64)
    if reversed_:
        along = reverse_along(along, L)
    frac = np.clip(along / span, 0.0, 1.0)

    if str(axis).upper() == "X":
        lx = float(x0) + frac * (float(x1) - float(x0))
        ly = np.full(n_t, float(offset), dtype=np.float64)
    else:
        lx = np.full(n_t, float(offset), dtype=np.float64)
        ly = float(y0) + frac * (float(y1) - float(y0))

    # Staggering: trasla l'INTERA linea invertita lungo il suo asse di
    # percorrenza (senza clamp) per allineare andata e ritorno.
    if reversed_ and abs(float(stagger_m)) > 1e-12:
        if str(axis).upper() == "X":
            lx = stagger_shift(lx, float(stagger_m))
        else:
            ly = stagger_shift(ly, float(stagger_m))

    if geo_on:
        az = math.radians(float(az_deg))
        ca, sa = math.cos(az), math.sin(az)
        de = lx * ca + ly * sa
        dn = -lx * sa + ly * ca
        east = float(e0) + de
        north = float(n0) + dn
    else:
        east = lx
        north = ly

    dseg = np.sqrt(np.diff(lx) ** 2 + np.diff(ly) ** 2)
    dist = np.concatenate([[0.0], np.cumsum(dseg)]) if n_t > 1 else np.zeros(1)

    # quota per traccia
    elev = np.zeros(n_t, dtype=np.float64)
    mode = str(elev_mode or "none").lower()
    # ascissa lungo il profilo, 0..L dall'inizio linea (tiene conto dello zig-zag)
    frac_line = (along / L) if L > 1e-9 else np.zeros(n_t, dtype=np.float64)
    if mode == "plane":
        elev = float(z_start) + frac_line * (float(z_end) - float(z_start))
    elif mode == "file" and elev_ref is not None:
        ref = np.asarray(elev_ref, dtype=np.float64)
        if ref.ndim == 2 and ref.shape[0] >= 1 and ref.shape[1] >= 2:
            order = np.argsort(ref[:, 0])
            xr, zr = ref[order, 0], ref[order, 1]
            if str(elev_ref_kind or "x").lower().startswith("trace"):
                query = np.arange(n_t, dtype=np.float64)
            else:
                query = along  # metri dall'inizio del profilo
            if xr.size == 1:
                elev = np.full(n_t, float(zr[0]), dtype=np.float64)
            else:
                elev = np.interp(query, xr, zr)

    return east, north, dist, np.asarray(elev, dtype=np.float64)


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

        # Group Box Navigazione / Marker
        grp_nav = QGroupBox("Navigazione")
        nav_form = QFormLayout(grp_nav)
        self._chk_real_length = QCheckBox("Usa la lunghezza reale del profilo (non lo span griglia)")
        self._chk_real_length.setChecked(False)
        self._chk_real_length.setToolTip(
            "Se attivo, ogni profilo occupa solo la sua 'Lunghezza (m)' a partire\n"
            "dall'inizio della linea (linee ANG / parziali). La colonna Lunghezza\n"
            "in tabella diventa modificabile."
        )
        self._chk_real_length.toggled.connect(self._on_nav_changed)
        nav_form.addRow(self._chk_real_length)

        self._spin_stagger = QDoubleSpinBox()
        self._spin_stagger.setRange(-50.0, 50.0)
        self._spin_stagger.setDecimals(3)
        self._spin_stagger.setSingleStep(0.05)
        self._spin_stagger.setValue(0.0)
        self._spin_stagger.setSuffix(" m")
        self._spin_stagger.setToolTip(
            "Correzione staggering: traslazione applicata SOLO alle linee\n"
            "invertite (zig-zag), per allineare andata e ritorno. 0 = nessuna."
        )
        self._spin_stagger.valueChanged.connect(self._on_nav_changed)
        nav_form.addRow("Stagger (linee invertite):", self._spin_stagger)

        # --- Quota per traccia (per migrazione topografica / correzione statica) ---
        self._cb_elev_mode = QComboBox()
        self._cb_elev_mode.addItem("Nessuna quota", "none")
        self._cb_elev_mode.addItem("Piano inclinato (quota inizio → fine)", "plane")
        self._cb_elev_mode.addItem("Da file (x,z  oppure  traccia,z)", "file")
        self._cb_elev_mode.setToolTip(
            "Assegna una quota (Z) a ogni traccia, usata dalla migrazione\n"
            "Kirchhoff topografica e dalla correzione statica su rilievi senza GPS."
        )
        self._cb_elev_mode.currentIndexChanged.connect(self._on_elev_mode_changed)
        nav_form.addRow("Quota per traccia:", self._cb_elev_mode)

        self._spin_z_start = QDoubleSpinBox()
        self._spin_z_start.setRange(-9999.0, 9999.0)
        self._spin_z_start.setDecimals(3)
        self._spin_z_start.setSingleStep(0.05)
        self._spin_z_start.setValue(0.0)
        self._spin_z_start.setSuffix(" m")
        self._spin_z_start.valueChanged.connect(self._on_nav_changed)
        self._lbl_z_start = QLabel("Quota inizio linea:")
        nav_form.addRow(self._lbl_z_start, self._spin_z_start)

        self._spin_z_end = QDoubleSpinBox()
        self._spin_z_end.setRange(-9999.0, 9999.0)
        self._spin_z_end.setDecimals(3)
        self._spin_z_end.setSingleStep(0.05)
        self._spin_z_end.setValue(0.0)
        self._spin_z_end.setSuffix(" m")
        self._spin_z_end.valueChanged.connect(self._on_nav_changed)
        self._lbl_z_end = QLabel("Quota fine linea:")
        nav_form.addRow(self._lbl_z_end, self._spin_z_end)

        elev_file_row = QHBoxLayout()
        self._le_elev_file = QLineEdit()
        self._le_elev_file.setPlaceholderText("file .csv/.txt  —  due colonne: posizione, quota")
        self._le_elev_file.textChanged.connect(self._on_nav_changed)
        self._btn_elev_file = QPushButton("Sfoglia…")
        self._btn_elev_file.clicked.connect(self._pick_elev_file)
        elev_file_row.addWidget(self._le_elev_file, 1)
        elev_file_row.addWidget(self._btn_elev_file, 0)
        self._elev_file_widget = QWidget()
        self._elev_file_widget.setLayout(elev_file_row)
        self._lbl_elev_file = QLabel("File quote:")
        nav_form.addRow(self._lbl_elev_file, self._elev_file_widget)

        self._cb_elev_kind = QComboBox()
        self._cb_elev_kind.addItem("Colonna 1 = metri lungo la linea", "x")
        self._cb_elev_kind.addItem("Colonna 1 = indice di traccia", "trace")
        self._cb_elev_kind.currentIndexChanged.connect(self._on_nav_changed)
        self._lbl_elev_kind = QLabel("Interpretazione colonna 1:")
        nav_form.addRow(self._lbl_elev_kind, self._cb_elev_kind)

        left_layout.addWidget(grp_nav)
        self._on_elev_mode_changed()

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

                # Lunghezza (modificabile solo se "usa lunghezza reale")
                item_len = QTableWidgetItem(f"{it['length']:.2f}")
                editable = bool(
                    getattr(self, "_chk_real_length", None)
                    and self._chk_real_length.isChecked()
                )
                if editable:
                    item_len.setFlags(item_len.flags() | Qt.ItemIsEditable)
                else:
                    item_len.setFlags(item_len.flags() & ~Qt.ItemIsEditable)
                item_len.setTextAlignment(Qt.AlignCenter)
                self._table.setItem(r, 6, item_len)
        finally:
            self._syncing_table = False

    def _on_nav_changed(self, *_a):
        self._update_table_from_items()
        self._draw_preview()

    def _elev_mode(self) -> str:
        cb = getattr(self, "_cb_elev_mode", None)
        return str(cb.currentData()) if cb is not None else "none"

    def _on_elev_mode_changed(self, *_a):
        mode = self._elev_mode()
        is_plane = mode == "plane"
        is_file = mode == "file"
        for w in (getattr(self, "_lbl_z_start", None), getattr(self, "_spin_z_start", None),
                  getattr(self, "_lbl_z_end", None), getattr(self, "_spin_z_end", None)):
            if w is not None:
                w.setVisible(is_plane)
        for w in (getattr(self, "_lbl_elev_file", None), getattr(self, "_elev_file_widget", None),
                  getattr(self, "_lbl_elev_kind", None), getattr(self, "_cb_elev_kind", None)):
            if w is not None:
                w.setVisible(is_file)
        if getattr(self, "_table", None) is not None:
            self._on_nav_changed()

    def _pick_elev_file(self):
        start = os.path.dirname(str(self._le_elev_file.text()).strip() or "") or os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(
            self, "File quote per traccia", start,
            "Tabelle (*.csv *.txt *.dat *.xyz);;Tutti i file (*)",
        )
        if path:
            self._le_elev_file.setText(path)

    @staticmethod
    def _parse_elev_file(path: str) -> "np.ndarray | None":
        """Legge un file a due colonne (posizione, quota). Accetta separatori
        virgola / punto e virgola / tab / spazi, salta righe non numeriche
        (header inclusi). Ritorna un array Nx2 o None."""
        try:
            rows = []
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    parts = [p for p in re.split(r"[,;\t ]+", s) if p != ""]
                    if len(parts) < 2:
                        continue
                    try:
                        a = float(parts[0].replace(",", "."))
                        b = float(parts[1].replace(",", "."))
                    except ValueError:
                        continue
                    rows.append((a, b))
            if len(rows) < 1:
                return None
            return np.asarray(rows, dtype=np.float64)
        except OSError:
            return None

    def _on_table_item_changed(self, item):
        if self._syncing_table or not item:
            return
        r = item.row()
        c = item.column()
        if not (0 <= r < len(self._profile_items)):
            return
        if c == 1:
            self._profile_items[r]["name"] = str(item.text()).strip()
            self._draw_preview()
        elif c == 6:
            try:
                val = float(str(item.text()).replace(",", "."))
                if val > 0:
                    self._profile_items[r]["length"] = val
            except ValueError:
                pass
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

        use_real_len = bool(
            getattr(self, "_chk_real_length", None) and self._chk_real_length.isChecked()
        )
        stagger_m = float(getattr(self, "_spin_stagger", None).value()) if getattr(self, "_spin_stagger", None) else 0.0

        # Disegna linee dei profili con frecce direzionali
        for idx, it in enumerate(self._profile_items):
            axis = it["axis"]
            offset = it["offset"]
            is_rev = it["reversed"]
            sh = stagger_m if is_rev else 0.0

            if axis == "X":
                span = (x1 - x0)
                frac = 1.0
                if use_real_len and abs(span) > 1e-9:
                    frac = min(1.0, float(it.get("length", abs(span))) / abs(span))
                xa, xb = x0 + sh, x0 + frac * span + sh
                p_start_loc = (xb, offset) if is_rev else (xa, offset)
                p_end_loc = (xa, offset) if is_rev else (xb, offset)
            else:
                span = (y1 - y0)
                frac = 1.0
                if use_real_len and abs(span) > 1e-9:
                    frac = min(1.0, float(it.get("length", abs(span))) / abs(span))
                ya, yb = y0 + sh, y0 + frac * span + sh
                p_start_loc = (offset, yb) if is_rev else (offset, ya)
                p_end_loc = (offset, ya) if is_rev else (offset, yb)

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

        use_real_length = bool(
            getattr(self, "_chk_real_length", None) and self._chk_real_length.isChecked()
        )
        stagger_m = float(getattr(self, "_spin_stagger", None).value()) if getattr(self, "_spin_stagger", None) else 0.0

        elev_mode = self._elev_mode()
        z_start = float(self._spin_z_start.value())
        z_end = float(self._spin_z_end.value())
        elev_ref_kind = str(self._cb_elev_kind.currentData() or "x")
        elev_file = str(self._le_elev_file.text()).strip()
        elev_ref = None
        if elev_mode == "file":
            if not elev_file or not os.path.isfile(elev_file):
                QMessageBox.warning(self, "Quota per traccia",
                                    "Modo 'Da file' selezionato ma il file quote non esiste.")
                return
            elev_ref = self._parse_elev_file(elev_file)
            if elev_ref is None:
                QMessageBox.warning(self, "Quota per traccia",
                                    "Impossibile leggere due colonne numeriche dal file quote.")
                return

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
                grid_span = (x1 - x0) if axis == "X" else (y1 - y0)
                final_e, final_n, final_dist, final_elev = compute_positions(
                    n_t, x0, x1, y0, y1, axis, offset,
                    grid_span_m=grid_span,
                    length_m=float(it.get("length", grid_span) or grid_span),
                    use_real_length=use_real_length,
                    reversed_=is_rev,
                    stagger_m=stagger_m,
                    geo_on=geo_on, e0=e0, n0=n0, az_deg=az_deg,
                    elev_mode=elev_mode, z_start=z_start, z_end=z_end,
                    elev_ref=elev_ref, elev_ref_kind=elev_ref_kind,
                )
                for ch in channels:
                    ch.easting = final_e.copy()
                    ch.northing = final_n.copy()
                    ch.distances = final_dist.copy()
                    if elev_mode != "none":
                        ch.altitude = final_elev.copy()

            configured_records.append({
                "index": idx,
                "path": getattr(prof, "path", ""),
                "name": custom_name,
                "axis": axis,
                "offset": offset,
                "reversed": is_rev,
                "n_traces": n_t,
                "length": float(it.get("length", 0.0) or 0.0),
            })

        result_payload = {
            "grid_bounds": {"x0": x0, "x1": x1, "y0": y0, "y1": y1},
            "georef": {
                "enabled": geo_on,
                "origin_e": e0,
                "origin_n": n0,
                "azimuth_deg": az_deg,
            },
            "navigation": {
                "use_real_length": use_real_length,
                "stagger_m": stagger_m,
                "elev_mode": elev_mode,
                "z_start": z_start,
                "z_end": z_end,
                "elev_file": elev_file,
                "elev_ref_kind": elev_ref_kind,
            },
            "profiles": configured_records,
        }

        self.positioning_applied.emit(result_payload)
        self.accept()
