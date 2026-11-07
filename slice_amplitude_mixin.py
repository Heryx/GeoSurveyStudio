# -*- coding: utf-8 -*-
"""Scheda "Ampiezze" del pannello principale: rifinitura dei valori per singola
time-slice, senza sovrascrivere le immagini.

Le impostazioni di ogni immagine stanno nel file allegato al progetto
(``timeslice_amplitude.json``, vedi ``slice_amplitude``); i filtri che cambiano i
pixel producono un raster derivato accanto al progetto e il layer QGIS punta a
quello. All'apertura del progetto ``_amp_reconcile_project_layers`` rimette ogni
immagine nello stato salvato (rigenerando il derivato se manca).
"""

from __future__ import annotations

import os

import numpy as np
from qgis.core import Qgis, QgsDataProvider, QgsMessageLog, QgsProject, QgsRasterLayer
from qgis.PyQt.QtCore import Qt, QTimer
from qgis.PyQt.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QMessageBox, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from . import slice_amplitude as amp
from .amplitude_histogram import AmplitudeHistogram
from .amplitude_stats import format_stats, value_stats

_LOG = "GeoSurvey Studio"


def _norm(p) -> str:
    return os.path.normcase(os.path.normpath(str(p).split("|")[0]))


def _safe(fn):
    """Slot a prova di eccezioni: in PyQt un'eccezione non gestita in uno slot
    abbatte l'intero QGIS."""
    import functools

    @functools.wraps(fn)
    def wrapper(self, *a, **kw):
        try:
            return fn(self, *a, **kw)
        except Exception as exc:
            QgsMessageLog.logMessage(f"Ampiezze ({fn.__name__}): {exc}", _LOG, Qgis.Warning)
    return wrapper


class SliceAmplitudeMixin:
    # ------------------------------------------------------------------ UI
    def _build_amplitude_tab(self, parent) -> QWidget:
        """Scheda scorrevole; i controlli restano in ``self._amp_*``."""
        self._amp_orig_cache = (None, None, None)     # (path, mtime, (arr, res))
        self._amp_loading = False
        self._amp_timer = QTimer(parent)
        self._amp_timer.setSingleShot(True)
        self._amp_timer.setInterval(250)
        self._amp_timer.timeout.connect(self._amp_refresh_histogram)

        outer = QWidget(parent)
        outer_l = QVBoxLayout(outer)
        outer_l.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(outer)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        outer_l.addWidget(scroll)
        body = QWidget()
        scroll.setWidget(body)
        lay = QVBoxLayout(body)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)

        self._amp_title = QLabel("Seleziona un'immagine nella lista raster.", body)
        self._amp_title.setWordWrap(True)
        lay.addWidget(self._amp_title)
        self._amp_state = QLabel("", body)
        self._amp_state.setStyleSheet("color: palette(mid); font-size: 8pt;")
        self._amp_state.setWordWrap(True)
        lay.addWidget(self._amp_state)

        gv = QGroupBox("Valori", body)
        fv = QFormLayout(gv)
        fv.setContentsMargins(6, 8, 6, 6)
        self._amp_chk_despike = QCheckBox("Taglia i picchi sopra il percentile", gv)
        self._amp_sp_despike = QDoubleSpinBox(gv)
        self._amp_sp_despike.setRange(50.0, 99.9)
        self._amp_sp_despike.setDecimals(1)
        self._amp_sp_despike.setValue(99.0)
        self._amp_sp_despike.setSuffix(" %")
        row = QHBoxLayout()
        row.addWidget(self._amp_chk_despike, 1)
        row.addWidget(self._amp_sp_despike)
        fv.addRow(row)
        self._amp_cb_norm = QComboBox(gv)
        self._amp_cb_norm.addItem("nessuna", "none")
        self._amp_cb_norm.addItem("sulla mediana", "median")
        fv.addRow("Normalizzazione:", self._amp_cb_norm)
        self._amp_cb_scale = QComboBox(gv)
        self._amp_cb_scale.addItem("lineare", "linear")
        self._amp_cb_scale.addItem("radice", "sqrt")
        self._amp_cb_scale.addItem("dB", "db")
        self._amp_cb_scale.setToolTip("Solo per ampiezze non negative (le ampiezze con segno restano lineari).")
        fv.addRow("Scala:", self._amp_cb_scale)
        self._amp_cb_despeckle = QComboBox(gv)
        for v in amp.DESPECKLE_SIZES:
            self._amp_cb_despeckle.addItem("nessuno" if v == 0 else f"{v}×{v}", v)
        fv.addRow("Despeckle:", self._amp_cb_despeckle)
        self._amp_cb_lowpass = QComboBox(gv)
        for v in amp.LOWPASS_SIZES:
            self._amp_cb_lowpass.addItem("nessuno" if v == 0 else f"{v}×{v}", v)
        fv.addRow("Low-pass:", self._amp_cb_lowpass)
        self._amp_sp_highpass = QDoubleSpinBox(gv)
        self._amp_sp_highpass.setRange(0.0, 50.0)
        self._amp_sp_highpass.setDecimals(2)
        self._amp_sp_highpass.setSingleStep(0.25)
        self._amp_sp_highpass.setSuffix(" m")
        self._amp_sp_highpass.setSpecialValueText("nessuno")
        self._amp_sp_highpass.setToolTip("Toglie il fondo lento (scala in metri) e lascia le anomalie locali.")
        fv.addRow("Alto-passa:", self._amp_sp_highpass)
        lay.addWidget(gv)

        gl = QGroupBox("Scala colore", body)
        fl = QFormLayout(gl)
        fl.setContentsMargins(6, 8, 6, 6)
        self._amp_cb_limits = QComboBox(gl)
        self._amp_cb_limits.addItem("Min / Max", "auto")
        self._amp_cb_limits.addItem("Percentili", "percentile")
        self._amp_cb_limits.addItem("Valori fissi", "fixed")
        fl.addRow("Limiti:", self._amp_cb_limits)
        self._amp_sp_plo = QDoubleSpinBox(gl)
        self._amp_sp_plo.setRange(0.0, 49.9)
        self._amp_sp_plo.setDecimals(1)
        self._amp_sp_plo.setValue(2.0)
        self._amp_sp_phi = QDoubleSpinBox(gl)
        self._amp_sp_phi.setRange(50.1, 100.0)
        self._amp_sp_phi.setDecimals(1)
        self._amp_sp_phi.setValue(98.0)
        prow = QHBoxLayout()
        prow.addWidget(self._amp_sp_plo)
        prow.addWidget(QLabel("–", gl))
        prow.addWidget(self._amp_sp_phi)
        prow.addWidget(QLabel("%", gl))
        fl.addRow("Percentili:", prow)
        self._amp_sp_vmin = QDoubleSpinBox(gl)
        self._amp_sp_vmax = QDoubleSpinBox(gl)
        for sp in (self._amp_sp_vmin, self._amp_sp_vmax):
            sp.setRange(-1e12, 1e12)
            sp.setDecimals(5)
        self._amp_sp_vmax.setValue(1.0)
        frow = QHBoxLayout()
        frow.addWidget(self._amp_sp_vmin)
        frow.addWidget(QLabel("–", gl))
        frow.addWidget(self._amp_sp_vmax)
        fl.addRow("Valori:", frow)
        self._amp_hist = AmplitudeHistogram(gl)
        fl.addRow(self._amp_hist)
        self._amp_stats = QLabel("", gl)
        self._amp_stats.setWordWrap(True)
        self._amp_stats.setStyleSheet("font-size: 8pt;")
        fl.addRow(self._amp_stats)
        lay.addWidget(gl)

        self._amp_chk_all = QCheckBox("Applica a tutte le immagini selezionate nella lista", body)
        self._amp_chk_all.setToolTip(
            "Spento: solo l'immagine corrente. Acceso: le stesse impostazioni a tutte le "
            "immagini selezionate (selezione multipla nella lista raster).")
        lay.addWidget(self._amp_chk_all)
        brow = QHBoxLayout()
        self._amp_btn_apply = QPushButton("Applica e salva", body)
        self._amp_btn_apply.setToolTip(
            "Applica le impostazioni all'immagine e le salva nel file allegato del progetto.\n"
            "L'immagine originale non viene modificata.")
        self._amp_btn_reset = QPushButton("Ripristina originale", body)
        self._amp_btn_reset.setToolTip("Toglie le impostazioni salvate e rimostra l'immagine originale.")
        brow.addWidget(self._amp_btn_apply)
        brow.addWidget(self._amp_btn_reset)
        lay.addLayout(brow)
        lay.addStretch(1)

        self._amp_btn_apply.clicked.connect(self._amp_on_apply)
        self._amp_btn_reset.clicked.connect(self._amp_on_reset)
        for w in (self._amp_chk_despike, ):
            w.toggled.connect(self._amp_on_controls_changed)
        for w in (self._amp_sp_despike, self._amp_sp_highpass, self._amp_sp_plo, self._amp_sp_phi,
                  self._amp_sp_vmin, self._amp_sp_vmax):
            w.valueChanged.connect(self._amp_on_controls_changed)
        for w in (self._amp_cb_norm, self._amp_cb_scale, self._amp_cb_despeckle,
                  self._amp_cb_lowpass, self._amp_cb_limits):
            w.currentIndexChanged.connect(self._amp_on_controls_changed)
        self._amp_hist.limits_changed.connect(self._amp_on_hist_limits)
        self._amp_sync_enabled()

        lw = getattr(getattr(self, "dlg", None), "rasterListWidget", None)
        if lw is not None:
            lw.currentRowChanged.connect(self._amp_on_selection_changed)
            lw.itemSelectionChanged.connect(self._amp_on_selection_changed)
        return outer

    # ------------------------------------------------------------ plumbing
    def _amp_root(self) -> str:
        try:
            return str(self._active_project_root() or "")
        except Exception:
            return ""

    def _amp_current_src(self) -> str:
        lw = getattr(getattr(self, "dlg", None), "rasterListWidget", None)
        item = lw.currentItem() if lw is not None else None
        payload = (item.data(Qt.UserRole) or {}) if item is not None else {}
        return str(payload.get("project_path") or "")

    def _amp_target_srcs(self) -> list:
        cur = self._amp_current_src()
        if not self._amp_chk_all.isChecked():
            return [cur] if cur else []
        lw = self.dlg.rasterListWidget
        out = []
        for it in lw.selectedItems():
            p = str((it.data(Qt.UserRole) or {}).get("project_path") or "")
            if p and p not in out:
                out.append(p)
        if cur and cur not in out:
            out.append(cur)
        return out

    def _amp_find_layer(self, src: str):
        root = self._amp_root()
        want = {_norm(src)}
        if root:
            want.add(_norm(amp.derived_path(root, src)))
        for lyr in QgsProject.instance().mapLayers().values():
            if isinstance(lyr, QgsRasterLayer) and _norm(lyr.source()) in want:
                return lyr
        return None

    def _amp_read_original(self, src: str):
        """Originale in cache (si rilegge solo se cambia il file)."""
        try:
            mt = os.path.getmtime(src)
        except OSError:
            return None
        path, mtime, val = self._amp_orig_cache
        if path == src and mtime == mt and val is not None:
            return val
        try:
            val = amp.read_band(src)
        except Exception:
            return None
        self._amp_orig_cache = (src, mt, val)
        return val

    def _amp_read_controls(self) -> amp.AmplitudeSettings:
        mode = self._amp_cb_limits.currentData() or "auto"
        return amp.AmplitudeSettings(
            despike_pct=(float(self._amp_sp_despike.value()) if self._amp_chk_despike.isChecked() else 100.0),
            normalize=self._amp_cb_norm.currentData() or "none",
            amp_scale=self._amp_cb_scale.currentData() or "linear",
            despeckle=int(self._amp_cb_despeckle.currentData() or 0),
            smooth_lowpass=int(self._amp_cb_lowpass.currentData() or 0),
            highpass_m=float(self._amp_sp_highpass.value()),
            limits_mode=mode,
            p_lo=float(self._amp_sp_plo.value()),
            p_hi=float(self._amp_sp_phi.value()),
            vmin=(float(self._amp_sp_vmin.value()) if mode == "fixed" else None),
            vmax=(float(self._amp_sp_vmax.value()) if mode == "fixed" else None),
        )

    def _amp_write_controls(self, s: amp.AmplitudeSettings) -> None:
        self._amp_loading = True
        try:
            self._amp_chk_despike.setChecked(float(s.despike_pct) < 100.0)
            if float(s.despike_pct) < 100.0:
                self._amp_sp_despike.setValue(float(s.despike_pct))
            for cb, val in ((self._amp_cb_norm, s.normalize), (self._amp_cb_scale, s.amp_scale),
                            (self._amp_cb_despeckle, int(s.despeckle)),
                            (self._amp_cb_lowpass, int(s.smooth_lowpass)),
                            (self._amp_cb_limits, s.limits_mode)):
                i = cb.findData(val)
                cb.setCurrentIndex(max(i, 0))
            self._amp_sp_highpass.setValue(float(s.highpass_m))
            self._amp_sp_plo.setValue(float(s.p_lo))
            self._amp_sp_phi.setValue(float(s.p_hi))
            if s.vmin is not None:
                self._amp_sp_vmin.setValue(float(s.vmin))
            if s.vmax is not None:
                self._amp_sp_vmax.setValue(float(s.vmax))
        finally:
            self._amp_loading = False
        self._amp_sync_enabled()

    def _amp_sync_enabled(self) -> None:
        mode = self._amp_cb_limits.currentData()
        self._amp_sp_plo.setEnabled(mode == "percentile")
        self._amp_sp_phi.setEnabled(mode == "percentile")
        self._amp_sp_vmin.setEnabled(mode == "fixed")
        self._amp_sp_vmax.setEnabled(mode == "fixed")
        self._amp_sp_despike.setEnabled(self._amp_chk_despike.isChecked())

    # ----------------------------------------------------- selection / live
    def _amp_on_selection_changed(self, *_):
        """Differita: la lista raster emette molti segnali mentre si popola."""
        try:
            timer = getattr(self, "_amp_sel_timer", None)
            if timer is None:
                timer = QTimer()
                timer.setSingleShot(True)
                timer.setInterval(300)
                timer.timeout.connect(self._amp_do_selection)
                self._amp_sel_timer = timer
            timer.start()
        except Exception as exc:
            QgsMessageLog.logMessage(f"Ampiezze (selezione): {exc}", _LOG, Qgis.Warning)

    def _amp_do_selection(self):
        try:
            self._amp_do_selection_impl()
        except Exception as exc:
            QgsMessageLog.logMessage(f"Ampiezze (selezione): {exc}", _LOG, Qgis.Warning)

    def _amp_do_selection_impl(self):
        src = self._amp_current_src()
        root = self._amp_root()
        if not src or not root:
            self._amp_title.setText("Seleziona un'immagine nella lista raster.")
            self._amp_state.setText("")
            return
        saved = amp.get_settings(root, src)
        self._amp_title.setText(f"<b>{os.path.basename(src)}</b>")
        self._amp_state.setText(
            "Impostazioni salvate nel file allegato del progetto." if saved
            else "Nessuna impostazione salvata: si vede l'originale.")
        self._amp_write_controls(saved or amp.AmplitudeSettings())
        self._amp_refresh_histogram()

    @_safe
    def _amp_on_controls_changed(self, *_):
        if getattr(self, "_amp_loading", False):
            return
        self._amp_sync_enabled()
        self._amp_timer.start()

    def _amp_refresh_histogram(self) -> None:
        try:
            self._amp_refresh_histogram_impl()
        except Exception as exc:
            QgsMessageLog.logMessage(f"Ampiezze (istogramma): {exc}", _LOG, Qgis.Warning)

    def _amp_refresh_histogram_impl(self) -> None:
        """Istogramma e statistiche dei valori che l'immagine avra' con i controlli."""
        src = self._amp_current_src()
        orig = self._amp_read_original(src) if src else None
        if orig is None:
            self._amp_hist.set_data(None, 0.0, 1.0)
            self._amp_stats.setText("")
            return
        arr, res = orig
        try:
            s = self._amp_read_controls().validate()
        except amp.AmplitudeSettingsError as exc:
            self._amp_stats.setText(str(exc))
            return
        try:
            grid = amp.process_grid(arr, s, res) if s.pixel_tools_active else arr
        except Exception as exc:                       # filtri non applicabili
            self._amp_stats.setText(f"Anteprima non disponibile: {exc}")
            return
        lim = amp.display_limits(s, grid)
        if lim is None:
            self._amp_hist.set_data(None, 0.0, 1.0)
            self._amp_stats.setText("nessun dato")
            return
        cols = self._amp_layer_colors(self._amp_find_layer(src))
        self._amp_hist.set_colors(cols)
        self._amp_hist.set_data(grid, lim[0], lim[1])
        self._amp_stats.setText(format_stats(value_stats(grid)))
        # NB: qui non si tocca MAI il layer: solo istogramma e statistiche.

    def _amp_shown_matches(self, src: str, s: amp.AmplitudeSettings) -> bool:
        """True se il layer mostra gia' i valori che i filtri dei controlli darebbero."""
        saved = amp.get_settings(self._amp_root(), src) or amp.AmplitudeSettings()

        def px(x):
            d = x.to_dict()
            return {k: d[k] for k in ("despike_pct", "normalize", "amp_scale", "despeckle",
                                      "smooth_lowpass", "highpass_m")}
        return px(saved) == px(s)

    @_safe
    def _amp_on_hist_limits(self, vmin: float, vmax: float) -> None:
        self._amp_loading = True
        try:
            self._amp_cb_limits.setCurrentIndex(self._amp_cb_limits.findData("fixed"))
            self._amp_sp_vmin.setValue(float(vmin))
            self._amp_sp_vmax.setValue(float(vmax))
        finally:
            self._amp_loading = False
        self._amp_sync_enabled()
        # il layer NON si tocca durante il trascinamento: un solo aggiornamento
        # a marcatore fermo (vedi _amp_apply_pending_limits)
        timer = getattr(self, "_amp_limits_timer", None)
        if timer is None:
            timer = QTimer()
            timer.setSingleShot(True)
            timer.setInterval(500)
            timer.timeout.connect(self._amp_apply_pending_limits)
            self._amp_limits_timer = timer
        self._amp_pending_limits = (float(vmin), float(vmax))
        timer.start()

    @_safe
    def _amp_apply_pending_limits(self):
        lim = getattr(self, "_amp_pending_limits", None)
        src = self._amp_current_src()
        if lim is None or not src:
            return
        layer = self._amp_find_layer(src)
        if layer is not None and self._amp_shown_matches(src, self._amp_read_controls()):
            self._amp_set_limits(layer, lim[0], lim[1])

    def _amp_set_limits(self, layer, lo: float, hi: float) -> bool:
        """Limiti colore di un layer a banda singola con un renderer nuovo (stessa
        costruzione di ``raster_io.apply_embedded_style``), tenendo i colori della
        rampa attuale. Non usa i metodi sul renderer vivo."""
        if layer is None or not hi > lo:
            return False
        try:
            from qgis.core import (QgsColorRampShader, QgsRasterShader,
                                   QgsSingleBandPseudoColorRenderer)
            from qgis.PyQt.QtGui import QColor
            cols = self._amp_layer_colors(layer) or ["#000000", "#ffffff"]
            n = len(cols)
            items = [
                QgsColorRampShader.ColorRampItem(
                    float(lo + (hi - lo) * i / max(n - 1, 1)), QColor(c),
                    "{:.4g}".format(lo + (hi - lo) * i / max(n - 1, 1)))
                for i, c in enumerate(cols)
            ]
            ramp = QgsColorRampShader(lo, hi)
            ramp.setColorRampType(QgsColorRampShader.Interpolated)
            ramp.setClip(False)
            ramp.setColorRampItemList(items)
            shader = QgsRasterShader()
            shader.setRasterShaderFunction(ramp)
            renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader)
            renderer.setClassificationMin(lo)
            renderer.setClassificationMax(hi)
            layer.setRenderer(renderer)
            layer.triggerRepaint()
            return True
        except Exception as exc:
            QgsMessageLog.logMessage(f"Ampiezze (limiti): {exc}", _LOG, Qgis.Warning)
            return False

    @staticmethod
    def _amp_layer_colors(layer):
        try:
            fn = layer.renderer().shader().rasterShaderFunction()
            return [it.color.name() for it in fn.colorRampItemList()]
        except Exception:
            return None

    # --------------------------------------------------------------- apply
    def _amp_show_source(self, layer, new_path: str, src: str):
        """Layer che mostra ``new_path``: se serve ne crea uno nuovo al posto del
        vecchio (stesso gruppo, posizione, nome, CRS, stile, visibilita'); niente
        ``setDataSource`` sul layer vivo. Ritorna il layer da usare o None."""
        if _norm(layer.source()) == _norm(new_path):
            layer.setCustomProperty("rl_amp_src", src)
            return layer
        new = QgsRasterLayer(new_path, layer.name(), "gdal")
        if not new.isValid():
            return None
        try:
            if layer.crs().isValid():
                new.setCrs(layer.crs())
        except Exception:
            pass
        try:
            r = layer.renderer().clone() if layer.renderer() is not None else None
            if r is not None:
                r.setInput(new.dataProvider())
                new.setRenderer(r)
        except Exception:
            from .raster_io import apply_embedded_style
            apply_embedded_style(new)
        new.setCustomProperty("rl_amp_src", src)
        project = QgsProject.instance()
        node = project.layerTreeRoot().findLayer(layer.id())
        parent = node.parent() if node is not None else None
        if parent is None:
            return None
        visible = node.itemVisibilityChecked()
        idx = parent.children().index(node)
        project.addMapLayer(new, False)
        new_node = parent.insertLayer(idx, new)
        try:
            new_node.setItemVisibilityChecked(visible)
        except Exception:
            pass
        old_id = layer.id()
        parent.removeChildNode(node)
        project.removeMapLayer(old_id)
        return new

    def _amp_apply_to_image(self, src: str, s: amp.AmplitudeSettings, *, persist: bool) -> bool:
        root = self._amp_root()
        shown = amp.ensure_derived(root, src, s, force=True)
        layer = self._amp_find_layer(src)
        if layer is not None:
            layer = self._amp_show_source(layer, shown, src)
            if layer is None:
                QgsMessageLog.logMessage(f"Ampiezze: impossibile aprire {shown}", _LOG, Qgis.Warning)
                return False
            grid, _res = amp.read_band(shown)
            lim = amp.display_limits(s, grid)
            if lim is not None:
                self._amp_set_limits(layer, lim[0], lim[1])
            layer.triggerRepaint()
        if persist:
            amp.set_settings(root, src, s)
            if s.is_default or not s.pixel_tools_active:
                amp.remove_derived(root, src)
        return True

    @_safe
    def _amp_on_apply(self):
        root = self._amp_root()
        srcs = self._amp_target_srcs()
        if not root or not srcs:
            QMessageBox.information(self.dlg, "Ampiezze", "Seleziona un'immagine nella lista raster.")
            return
        try:
            s = self._amp_read_controls().validate()
        except amp.AmplitudeSettingsError as exc:
            QMessageBox.warning(self.dlg, "Ampiezze", str(exc))
            return
        done, failed = 0, []
        for src in srcs:
            try:
                if self._amp_apply_to_image(src, s, persist=True):
                    done += 1
                else:
                    failed.append(os.path.basename(src))
            except Exception as exc:
                failed.append(f"{os.path.basename(src)}: {exc}")
                QgsMessageLog.logMessage(f"Ampiezze {src}: {exc}", _LOG, Qgis.Warning)
        self.iface.messageBar().pushInfo(
            "GeoSurvey Studio",
            f"Ampiezze applicate e salvate: {done}/{len(srcs)} immagini "
            f"({amp.STORE_NAME} nella cartella del progetto).")
        if failed:
            QMessageBox.warning(self.dlg, "Ampiezze", "Non applicate:\n" + "\n".join(failed[:8]))
        self._amp_do_selection()

    @_safe
    def _amp_on_reset(self):
        root = self._amp_root()
        srcs = self._amp_target_srcs()
        if not root or not srcs:
            return
        for src in srcs:
            try:
                self._amp_apply_to_image(src, amp.AmplitudeSettings(), persist=True)
            except Exception as exc:
                QgsMessageLog.logMessage(f"Ampiezze reset {src}: {exc}", _LOG, Qgis.Warning)
        self._amp_do_selection()

    # --------------------------------------------------- project (re)open
    def _amp_reconcile_project_layers(self) -> int:
        """Rimette ogni immagine con impostazioni salvate nel suo stato: derivato
        rigenerato se manca o e' vecchio, limiti colore riapplicati. Idempotente."""
        root = self._amp_root()
        if not root:
            return 0
        n = 0
        for key in list(amp.load_store(root).keys()):
            src = key if os.path.isabs(key) else os.path.normpath(os.path.join(root, key))
            if not os.path.exists(src):
                continue
            s = amp.get_settings(root, src)
            layer = self._amp_find_layer(src)
            if s is None or layer is None:
                continue
            try:
                shown = amp.ensure_derived(root, src, s)
                layer = self._amp_show_source(layer, shown, src)
                if layer is None:
                    continue
                grid, _res = amp.read_band(shown)
                lim = amp.display_limits(s, grid)
                if lim is not None:
                    self._amp_set_limits(layer, lim[0], lim[1])
                n += 1
            except Exception as exc:
                QgsMessageLog.logMessage(f"Ampiezze (riapertura) {src}: {exc}", _LOG, Qgis.Warning)
        return n
