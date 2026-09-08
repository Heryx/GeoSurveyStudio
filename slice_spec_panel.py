# -*- coding: utf-8 -*-
"""``SliceSpecPanel`` — the single time-slice parameter form (Fase D).

A ~13-control widget bound 1:1 to :class:`slice_spec.SliceSpec`. It replaces
the old Timeslice panel block, ``GprProfileViewer.get_slice_params()`` and the
hidden "Esperto" flag: every field is always visible and every value is used.

The widget carries no survey knowledge — ``get_spec()`` returns a *declarative*
``SliceSpec`` (``z_max_m`` / ``search_radius_m`` / ``crs_epsg`` may be ``None``);
the caller runs ``spec.resolve(profiles)`` before generation.
"""

from __future__ import annotations

from qgis.PyQt.QtCore import pyqtSignal
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QWidget,
)

from .slice_spec import (
    AMPLITUDE_MODES,
    INTERP_METHODS,
    SMOOTH_LOWPASS_VALUES,
    SliceSpec,
    SliceSpecError,
)

_AMPLITUDE_LABELS = {
    "abs_mean": "Media |ampiezza| (abs_mean)",
    "rms": "RMS (rms)",
    "env_peak": "Picco inviluppo (env_peak)",
    "signed_mean": "Media con segno (signed_mean)",
    "max_abs": "Max |ampiezza| (max_abs)",
}
_INTERP_LABELS = {"idw": "IDW", "kriging": "Kriging (non implementato)"}

# velocity is entered in cm/ns (typical GPR unit); SliceSpec stores m/s.
_CM_NS_TO_M_S = 1.0e7
_M_S_TO_CM_NS = 1.0e-7


class SliceSpecPanel(QWidget):
    """Form widget whose state is a :class:`slice_spec.SliceSpec`."""

    spec_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._building = True
        form = QFormLayout(self)
        form.setContentsMargins(6, 6, 6, 6)
        form.setSpacing(4)

        # --- depth window -------------------------------------------------
        self.sp_z_min = self._dspin(0.0, 0.0, 500.0, 0.05, " m")
        form.addRow("Profondità min:", self.sp_z_min)

        self.sp_z_max = self._dspin(2.0, 0.01, 1000.0, 0.05, " m")
        self.chk_z_max_auto = QCheckBox("auto (max profili)")
        self.chk_z_max_auto.setChecked(True)
        form.addRow("Profondità max:", self._with_auto(self.sp_z_max, self.chk_z_max_auto))

        self.sp_thickness = self._dspin(0.10, 0.01, 50.0, 0.01, " m")
        form.addRow("Spessore slice:", self.sp_thickness)

        self.sp_overlap = self._dspin(50.0, 0.0, 90.0, 5.0, " %")
        form.addRow("Sovrapposizione:", self.sp_overlap)

        # --- amplitude extraction --------------------------------------------
        self.cb_amplitude = QComboBox()
        for m in AMPLITUDE_MODES:
            self.cb_amplitude.addItem(_AMPLITUDE_LABELS.get(m, m), m)
        form.addRow("Ampiezza:", self.cb_amplitude)

        self.le_source_variant = QLineEdit("raw")
        self.le_source_variant.setToolTip(
            "id della variante di pipeline da affettare, oppure 'raw'"
        )
        form.addRow("Variante sorgente:", self.le_source_variant)

        self.sp_velocity = self._dspin(10.0, 1.0, 30.0, 0.5, " cm/ns")
        self.sp_velocity.setToolTip("velocità onda EM (per conversione tempo→profondità)")
        form.addRow("Velocità:", self.sp_velocity)

        # --- gridding ---------------------------------------------------
        self.sp_resolution = self._dspin(0.10, 0.01, 10.0, 0.01, " m")
        form.addRow("Risoluzione griglia:", self.sp_resolution)

        self.cb_interp = QComboBox()
        for m in INTERP_METHODS:
            self.cb_interp.addItem(_INTERP_LABELS.get(m, m), m)
        idx_krig = self.cb_interp.findData("kriging")
        if idx_krig >= 0:
            self.cb_interp.model().item(idx_krig).setEnabled(False)
        form.addRow("Interpolazione:", self.cb_interp)

        self.sp_search_radius = self._dspin(1.0, 0.01, 500.0, 0.05, " m")
        self.chk_radius_auto = QCheckBox("auto (spaziatura interlinea)")
        self.chk_radius_auto.setChecked(True)
        form.addRow("Raggio ricerca:", self._with_auto(self.sp_search_radius, self.chk_radius_auto))

        self.sp_blanking = self._dspin(0.0, 0.0, 100.0, 0.05, " m")
        self.sp_blanking.setToolTip("0 = nessun blanking; celle oltre questa distanza dai dati -> NoData")
        form.addRow("Raggio blanking:", self.sp_blanking)

        self.cb_smooth = QComboBox()
        for v in SMOOTH_LOWPASS_VALUES:
            self.cb_smooth.addItem("nessuno" if v == 0 else f"{v}×{v}", v)
        form.addRow("Low-pass finale:", self.cb_smooth)

        # --- crs ------------------------------------------------------------
        self.le_crs_epsg = QLineEdit("")
        self.le_crs_epsg.setPlaceholderText("auto (da profili)")
        self.le_crs_epsg.setToolTip("codice EPSG, vuoto = dedotto dai profili")
        form.addRow("CRS (EPSG):", self.le_crs_epsg)

        self._wire_signals()
        self._sync_enabled()
        self._building = False

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _dspin(value, lo, hi, step, suffix=""):
        sp = QDoubleSpinBox()
        sp.setRange(float(lo), float(hi))
        sp.setSingleStep(float(step))
        sp.setDecimals(3)
        if suffix:
            sp.setSuffix(suffix)
        sp.setValue(float(value))
        return sp

    @staticmethod
    def _with_auto(spin, chk):
        host = QWidget()
        lay = QHBoxLayout(host)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lay.addWidget(spin, 1)
        lay.addWidget(chk, 0)
        return host

    def _wire_signals(self):
        for sp in (self.sp_z_min, self.sp_z_max, self.sp_thickness, self.sp_overlap,
                   self.sp_velocity, self.sp_resolution, self.sp_search_radius,
                   self.sp_blanking):
            sp.valueChanged.connect(self._on_changed)
        for cb in (self.cb_amplitude, self.cb_interp, self.cb_smooth):
            cb.currentIndexChanged.connect(self._on_changed)
        for chk in (self.chk_z_max_auto, self.chk_radius_auto):
            chk.toggled.connect(self._on_changed)
        self.le_source_variant.textChanged.connect(self._on_changed)
        self.le_crs_epsg.textChanged.connect(self._on_changed)

    # ------------------------------------------------------------------
    # behaviour
    # ------------------------------------------------------------------
    def _sync_enabled(self):
        self.sp_z_max.setEnabled(not self.chk_z_max_auto.isChecked())
        self.sp_search_radius.setEnabled(not self.chk_radius_auto.isChecked())

    def _on_changed(self, *_):
        self._sync_enabled()
        if not self._building:
            self.spec_changed.emit()

    # ------------------------------------------------------------------
    # spec <-> widgets
    # ------------------------------------------------------------------
    def get_spec(self) -> SliceSpec:
        """Build a (declarative, possibly unresolved) SliceSpec from the form."""
        crs_txt = self.le_crs_epsg.text().strip()
        try:
            crs_epsg = int(crs_txt) if crs_txt else None
        except ValueError:
            crs_epsg = None

        spec = SliceSpec(
            z_min_m=float(self.sp_z_min.value()),
            z_max_m=(None if self.chk_z_max_auto.isChecked() else float(self.sp_z_max.value())),
            thickness_m=float(self.sp_thickness.value()),
            overlap=round(float(self.sp_overlap.value()) / 100.0, 4),
            amplitude_mode=str(self.cb_amplitude.currentData() or "abs_mean"),
            source_variant=(self.le_source_variant.text().strip() or "raw"),
            velocity_m_s=float(self.sp_velocity.value()) * _CM_NS_TO_M_S,
            resolution_m=float(self.sp_resolution.value()),
            interp=str(self.cb_interp.currentData() or "idw"),
            search_radius_m=(None if self.chk_radius_auto.isChecked()
                             else float(self.sp_search_radius.value())),
            blanking_radius_m=float(self.sp_blanking.value()),
            smooth_lowpass=int(self.cb_smooth.currentData() or 0),
            crs_epsg=crs_epsg,
        )
        spec.validate()
        return spec

    def set_spec(self, spec: SliceSpec) -> None:
        """Populate the form from a SliceSpec (resolved or not)."""
        if not isinstance(spec, SliceSpec):
            raise TypeError("set_spec expects a slice_spec.SliceSpec")
        self._building = True
        try:
            self.sp_z_min.setValue(float(spec.z_min_m))
            self.chk_z_max_auto.setChecked(spec.z_max_m is None)
            if spec.z_max_m is not None:
                self.sp_z_max.setValue(float(spec.z_max_m))
            self.sp_thickness.setValue(float(spec.thickness_m))
            self.sp_overlap.setValue(float(spec.overlap) * 100.0)
            self._set_combo(self.cb_amplitude, spec.amplitude_mode)
            self.le_source_variant.setText(str(spec.source_variant or "raw"))
            self.sp_velocity.setValue(float(spec.velocity_m_s) * _M_S_TO_CM_NS)
            self.sp_resolution.setValue(float(spec.resolution_m))
            self._set_combo(self.cb_interp, spec.interp)
            self.chk_radius_auto.setChecked(spec.search_radius_m is None)
            if spec.search_radius_m is not None:
                self.sp_search_radius.setValue(float(spec.search_radius_m))
            self.sp_blanking.setValue(float(spec.blanking_radius_m))
            self._set_combo(self.cb_smooth, int(spec.smooth_lowpass))
            self.le_crs_epsg.setText("" if spec.crs_epsg is None else str(int(spec.crs_epsg)))
            self._sync_enabled()
        finally:
            self._building = False
        self.spec_changed.emit()

    @staticmethod
    def _set_combo(cb: QComboBox, data) -> None:
        idx = cb.findData(data)
        if idx >= 0:
            cb.setCurrentIndex(idx)

    # ------------------------------------------------------------------
    def validate(self) -> "tuple[bool, str]":
        try:
            self.get_spec()
            return True, ""
        except SliceSpecError as exc:
            return False, str(exc)
