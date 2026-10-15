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
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QWidget,
)

from .slice_spec import (
    AMPLITUDE_MODES,
    AMPLITUDE_SCALES,
    BALANCE_MODES,
    DESPECKLE_SIZES,
    INTERP_METHODS,
    NORMALIZE_MODES,
    SMOOTH_LOWPASS_VALUES,
    SliceSpec,
    SliceSpecError,
)

_AMPLITUDE_LABELS = {
    "abs_mean": "Media |ampiezza| (abs_mean)",
    "rms": "RMS (rms)",
    "env_peak": "Picco inviluppo (env_peak)",
    "env_mean": "Media inviluppo (env_mean)",
    "signed_mean": "Media con segno (signed_mean)",
    "max_abs": "Max |ampiezza| (max_abs)",
}
_INTERP_LABELS = {
    "bin": "Binning + gaussiana (consigliata)",
    "idw": "IDW (classico, lascia puntini)",
    "kriging": "Kriging (non implementato)",
}
_NORMALIZE_LABELS = {
    "none": "Nessuna",
    "median": "÷ mediana (1 = fondo)",
}
_SCALE_LABELS = {
    "linear": "Lineare",
    "sqrt": "Radice quadrata (√)",
    "db": "Decibel (20·log10)",
}
_BALANCE_LABELS = {
    "channel": "Per canale (toglie le strisce)",
    "profile": "Per profilo",
    "none": "Nessuna",
}

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
        self.cb_amplitude.setToolTip(
            "Come si riassume l'onda dentro lo spessore della slice, per ogni traccia:\n"
            " • Media / Picco inviluppo: inviluppo di Hilbert, non dipende dalla fase\n"
            "   dell'onda. La media è più stabile con campioni isolati anomali, il\n"
            "   picco mette in risalto i bersagli puntuali.\n"
            " • Media |ampiezza| / RMS: dipendono un po' dalla fase dell'onda.\n"
            " • Media con segno: mantiene il segno (polarità del riflettore).\n"
            "Sul rumore di fondo la differenza tra i modi è piccola: per pulire\n"
            "l'immagine contano soprattutto la lisciatura e la scala colore."
        )
        form.addRow("Ampiezza:", self.cb_amplitude)

        self.cb_source_variant = QComboBox()
        self.cb_source_variant.setEditable(True)
        self.cb_source_variant.addItem("raw")
        self.cb_source_variant.setToolTip(
            "Radargrammi da affettare:\n"
            " • 'raw' = i profili attualmente caricati (grezzi o con la pipeline\n"
            "   di anteprima), come prima;\n"
            " • una variante salvata (Processing/processed/<nome>) = usa i\n"
            "   suoi .npz gia' filtrati, senza riapplicare la pipeline.\n"
            "L'elenco si aggiorna con le varianti presenti nel progetto."
        )
        form.addRow("Variante sorgente:", self.cb_source_variant)

        self.sp_velocity = self._dspin(10.0, 1.0, 30.0, 0.5, " cm/ns")
        self.sp_velocity.setToolTip(
            "Velocità onda EM per la conversione tempo→profondità.\n"
            "Con 'auto' si usa quella di ogni profilo, cioè la stessa dell'asse\n"
            "profondità del radargramma."
        )
        self.chk_velocity_auto = QCheckBox("auto (dal profilo)")
        self.chk_velocity_auto.setChecked(True)
        form.addRow("Velocità:", self._with_auto(self.sp_velocity, self.chk_velocity_auto))

        self.cb_balance = QComboBox()
        for m in BALANCE_MODES:
            self.cb_balance.addItem(_BALANCE_LABELS.get(m, m), m)
        self.cb_balance.setToolTip(
            "Riporta allo stesso livello i canali dell'array prima dell'interpolazione.\n"
            "Canali con guadagno/accoppiamento diverso producono strisce lungo le\n"
            "tracce: 'Per canale' le elimina, 'Per profilo' equalizza solo gli swath."
        )
        form.addRow("Bilanciamento:", self.cb_balance)

        # --- gridding ---------------------------------------------------
        self.sp_resolution = self._dspin(0.10, 0.01, 10.0, 0.01, " m")
        form.addRow("Risoluzione griglia:", self.sp_resolution)

        self.cb_interp = QComboBox()
        for m in INTERP_METHODS:
            self.cb_interp.addItem(_INTERP_LABELS.get(m, m), m)
        idx_krig = self.cb_interp.findData("kriging")
        if idx_krig >= 0:
            self.cb_interp.model().item(idx_krig).setEnabled(False)
        self.cb_interp.setToolTip(
            "Binning + gaussiana: ogni traccia di ogni canale alla sua posizione,\n"
            "media per cella e riempimento dei vuoti. Niente puntini, molto veloce.\n"
            "IDW: metodo classico sulla linea centrale dello swath; lascia una\n"
            "collana di puntini lungo le linee di rilievo."
        )
        form.addRow("Interpolazione:", self.cb_interp)

        self.sp_search_radius = self._dspin(1.0, 0.01, 500.0, 0.05, " m")
        self.sp_search_radius.setToolTip(
            "Distanza massima su cui si colmano i vuoti fra le linee\n"
            "(si chiudono buchi larghi fino al doppio del raggio)."
        )
        self.chk_radius_auto = QCheckBox("auto (spaziatura interlinea)")
        self.chk_radius_auto.setChecked(True)
        form.addRow("Raggio ricerca:", self._with_auto(self.sp_search_radius, self.chk_radius_auto))

        self.sp_blanking = self._dspin(0.0, 0.0, 100.0, 0.05, " m")
        self.sp_blanking.setToolTip("0 = nessun blanking; celle oltre questa distanza dai dati -> NoData")
        form.addRow("Raggio blanking:", self.sp_blanking)

        # --- amplitude tools: the value conditioning, in the order it is applied ---
        self.grp_tools = QGroupBox("Rifinitura ampiezze")
        self.grp_tools.setToolTip(
            "Le voci sono elencate nell'ordine in cui vengono applicate:\n"
            "taglia picchi (sulle tracce, prima dell'interpolazione), poi sulla slice:\n"
            "normalizzazione, scala, despeckle, background, low-pass, sfondo locale."
        )
        tl = QFormLayout(self.grp_tools)
        tl.setContentsMargins(6, 6, 6, 6)
        tl.setSpacing(4)

        self.sp_despike = self._dspin(99.5, 90.0, 99.99, 0.1, " %")
        self.sp_despike.setToolTip(
            "Prima dell'interpolazione: le tracce con ampiezza sopra questo percentile\n"
            "della slice vengono riportate al valore del percentile. Toglie i punti\n"
            "isolati (una traccia rumorosa) che altrimenti diventano macchie luminose."
        )
        self.chk_despike = QCheckBox("attivo")
        tl.addRow("Taglia picchi:", self._with_auto(self.sp_despike, self.chk_despike))

        self.cb_normalize = QComboBox()
        for m in NORMALIZE_MODES:
            self.cb_normalize.addItem(_NORMALIZE_LABELS.get(m, m), m)
        self.cb_normalize.setToolTip(
            "Divide ogni slice per la propria mediana: 1 = ampiezza tipica del fondo,\n"
            "2 = doppia. L'ampiezza cala con la profondità; così tutte le slice sono\n"
            "confrontabili e si può usare un'unica scala colore fissa per l'intero set."
        )
        tl.addRow("Normalizza slice:", self.cb_normalize)

        self.cb_scale = QComboBox()
        for m in AMPLITUDE_SCALES:
            self.cb_scale.addItem(_SCALE_LABELS.get(m, m), m)
        self.cb_scale.setToolTip(
            "Scala non lineare dei valori (solo ampiezze positive): comprime i valori\n"
            "alti e fa emergere i segnali deboli. In dB, 6 dB = ampiezza doppia."
        )
        tl.addRow("Scala valori:", self.cb_scale)

        self.cb_despeckle = QComboBox()
        for v in DESPECKLE_SIZES:
            self.cb_despeckle.addItem("nessuno" if v == 0 else f"{v}×{v}", v)
        self.cb_despeckle.setToolTip(
            "Filtro mediano sulla slice finita: elimina i pixel isolati senza sfocare\n"
            "i bordi come farebbe una gaussiana."
        )
        tl.addRow("Despeckle:", self.cb_despeckle)

        self.chk_bg_h = QCheckBox("orizzontale")
        self.chk_bg_h.setToolTip("Toglie le bande orizzontali (sottrae la media di ogni riga della slice)")
        self.chk_bg_v = QCheckBox("verticale")
        self.chk_bg_v.setToolTip("Toglie le bande verticali (sottrae la media di ogni colonna della slice)")
        self.sp_bg_clip = self._dspin(95.0, 50.0, 100.0, 1.0, " %")
        self.sp_bg_clip.setToolTip("Percentile: i valori piu' alti di ogni linea non entrano nel background (protegge i target forti)")
        _bgw = QWidget()
        _bgl = QHBoxLayout(_bgw)
        _bgl.setContentsMargins(0, 0, 0, 0)
        for _w in (self.chk_bg_h, self.chk_bg_v, QLabel("clip"), self.sp_bg_clip):
            _bgl.addWidget(_w)
        tl.addRow("Background slice:", _bgw)

        self.cb_smooth = QComboBox()
        for v in SMOOTH_LOWPASS_VALUES:
            self.cb_smooth.addItem("nessuno" if v == 0 else f"{v}×{v}", v)
        self.cb_smooth.setCurrentIndex(max(0, self.cb_smooth.findData(SliceSpec().smooth_lowpass)))
        self.cb_smooth.setToolTip(
            "Lisciatura gaussiana della slice finita (sigma = lato/3 celle). Con la\n"
            "risoluzione piena dell'array (4 cm) il rumore incoerente si vede: 3×3 è\n"
            "un buon compromesso; 7×7 – 11×11 danno un aspetto più liscio."
        )
        tl.addRow("Low-pass finale:", self.cb_smooth)

        self.sp_highpass = self._dspin(0.0, 0.0, 50.0, 0.1, " m")
        self.sp_highpass.setSpecialValueText("off")
        self.sp_highpass.setToolTip(
            "Sottrae lo sfondo che varia lentamente (gaussiana di questo sigma, in metri)\n"
            "e lascia le anomalie locali: utile contro gradienti di umidità/terreno.\n"
            "Valori tipici 0.5–2 m; più piccolo del bersaglio che cerchi non va usato."
        )
        tl.addRow("Sfondo locale:", self.sp_highpass)

        form.addRow(self.grp_tools)

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
                   self.sp_blanking, self.sp_bg_clip, self.sp_despike, self.sp_highpass):
            sp.valueChanged.connect(self._on_changed)
        for cb in (self.cb_amplitude, self.cb_interp, self.cb_smooth, self.cb_balance,
                   self.cb_normalize, self.cb_scale, self.cb_despeckle):
            cb.currentIndexChanged.connect(self._on_changed)
        for chk in (self.chk_z_max_auto, self.chk_radius_auto, self.chk_velocity_auto,
                    self.chk_bg_h, self.chk_bg_v, self.chk_despike):
            chk.toggled.connect(self._on_changed)
        self.cb_source_variant.currentTextChanged.connect(self._on_changed)
        self.le_crs_epsg.textChanged.connect(self._on_changed)

    def set_variant_choices(self, names) -> None:
        """Ripopola la tendina 'Variante sorgente' con ``raw`` + le varianti
        date, preservando la selezione corrente."""
        cb = self.cb_source_variant
        cur = cb.currentText().strip() or "raw"
        items = ["raw"]
        for n in names or []:
            n = str(n).strip()
            if n and n != "raw" and n not in items:
                items.append(n)
        if cur not in items:
            items.append(cur)
        block = cb.blockSignals(True)
        try:
            cb.clear()
            cb.addItems(items)
            i = cb.findText(cur)
            cb.setCurrentIndex(i if i >= 0 else 0)
        finally:
            cb.blockSignals(block)

    # ------------------------------------------------------------------
    # behaviour
    # ------------------------------------------------------------------
    def _sync_enabled(self):
        self.sp_z_max.setEnabled(not self.chk_z_max_auto.isChecked())
        self.sp_search_radius.setEnabled(not self.chk_radius_auto.isChecked())
        self.sp_velocity.setEnabled(not self.chk_velocity_auto.isChecked())
        # The amplitude tools live in the "bin" engine; the legacy IDW path has
        # its own (older) chain and ignores them, so do not offer what would lie.
        binned = str(self.cb_interp.currentData() or "bin") == "bin"
        # dB / sqrt / ratio need non-negative amplitudes.
        positive = str(self.cb_amplitude.currentData() or "") != "signed_mean"
        self.chk_despike.setEnabled(binned)
        self.sp_despike.setEnabled(binned and self.chk_despike.isChecked())
        self.cb_despeckle.setEnabled(binned)
        self.sp_highpass.setEnabled(binned)
        self.cb_normalize.setEnabled(binned and positive)
        self.cb_scale.setEnabled(binned and positive)
        note = "" if binned else "Disponibile solo con l'interpolazione «Binning + gaussiana»."
        for w in (self.chk_despike, self.sp_despike, self.cb_despeckle, self.sp_highpass):
            w.setStatusTip(note)
        pos_note = "" if positive else "Non applicabile alla media con segno (valori negativi)."
        self.cb_normalize.setStatusTip(pos_note)
        self.cb_scale.setStatusTip(pos_note)

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
            source_variant=(self.cb_source_variant.currentText().strip() or "raw"),
            velocity_m_s=(None if self.chk_velocity_auto.isChecked()
                          else float(self.sp_velocity.value()) * _CM_NS_TO_M_S),
            balance=str(self.cb_balance.currentData() or "channel"),
            resolution_m=float(self.sp_resolution.value()),
            interp=str(self.cb_interp.currentData() or "idw"),
            search_radius_m=(None if self.chk_radius_auto.isChecked()
                             else float(self.sp_search_radius.value())),
            blanking_radius_m=float(self.sp_blanking.value()),
            smooth_lowpass=int(self.cb_smooth.currentData() or 0),
            bg_horizontal=self.chk_bg_h.isChecked(),
            bg_vertical=self.chk_bg_v.isChecked(),
            bg_clip_pct=float(self.sp_bg_clip.value()),
            despike_pct=(float(self.sp_despike.value()) if self.chk_despike.isChecked() else 100.0),
            normalize=str(self.cb_normalize.currentData() or "none"),
            amp_scale=str(self.cb_scale.currentData() or "linear"),
            despeckle=int(self.cb_despeckle.currentData() or 0),
            highpass_m=float(self.sp_highpass.value()),
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
            self._set_source_variant(str(spec.source_variant or "raw"))
            self.chk_velocity_auto.setChecked(spec.velocity_m_s is None)
            if spec.velocity_m_s is not None:
                self.sp_velocity.setValue(float(spec.velocity_m_s) * _M_S_TO_CM_NS)
            self._set_combo(self.cb_balance, spec.balance)
            self.sp_resolution.setValue(float(spec.resolution_m))
            self._set_combo(self.cb_interp, spec.interp)
            self.chk_radius_auto.setChecked(spec.search_radius_m is None)
            if spec.search_radius_m is not None:
                self.sp_search_radius.setValue(float(spec.search_radius_m))
            self.sp_blanking.setValue(float(spec.blanking_radius_m))
            self._set_combo(self.cb_smooth, int(spec.smooth_lowpass))
            self.chk_bg_h.setChecked(bool(spec.bg_horizontal))
            self.chk_bg_v.setChecked(bool(spec.bg_vertical))
            self.sp_bg_clip.setValue(float(spec.bg_clip_pct))
            self.chk_despike.setChecked(float(spec.despike_pct) < 100.0)
            if float(spec.despike_pct) < 100.0:
                self.sp_despike.setValue(float(spec.despike_pct))
            self._set_combo(self.cb_normalize, spec.normalize)
            self._set_combo(self.cb_scale, spec.amp_scale)
            self._set_combo(self.cb_despeckle, int(spec.despeckle))
            self.sp_highpass.setValue(float(spec.highpass_m))
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

    def _set_source_variant(self, name: str) -> None:
        cb = self.cb_source_variant
        i = cb.findText(name)
        if i < 0:
            cb.addItem(name)
            i = cb.findText(name)
        cb.setCurrentIndex(i if i >= 0 else 0)

    # ------------------------------------------------------------------
    def validate(self) -> "tuple[bool, str]":
        try:
            self.get_spec()
            return True, ""
        except SliceSpecError as exc:
            return False, str(exc)
