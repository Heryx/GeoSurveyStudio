# -*- coding: utf-8 -*-
"""Dialogo per importare una sezione ERT invertita e georiferirla.

Tre modi di dare la posizione (il calcolo sta in ``gpr_ert_georef`` / ``gpr_ert_store``):

- **KML**: primo e ultimo punto di un profilo del KML (retta tra i due);
- **mappa**: due punti cliccati sulla mappa di QGIS, o scritti a mano (retta tra i due);
- **layer di punti**: N punti; con un campo "distanza" ogni punto e' ancorato a quella distanza
  lungo la sezione (fino a un punto per elettrodo), senza campo la distanza e' proporzionale
  alla lunghezza sulla mappa e i punti seguono l'ordine degli elementi.

Le coordinate sono sempre quelle del CRS del progetto.
"""

from __future__ import annotations

import os

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsMapLayerType,
    QgsPointXY,
    QgsProject,
    QgsWkbTypes,
)
from qgis.gui import QgsMapToolEmitPoint

from .gpr_ert_georef import (
    ErtGeoref,
    ErtGeorefError,
    georef_from_kml_profile,
    georef_from_xy_points,
    lonlat_transformer,
    parse_xy,
    read_kml_profiles,
    suggested_length,
)
from .gpr_ert_import import ErtImportError, load_ert_section

MODE_KML, MODE_MAP, MODE_LAYER = 0, 1, 2
_NO_FIELD = "(nessuno: distanza proporzionale)"


def point_layers() -> list:
    """Layer vettoriali di punti del progetto, in ordine di nome."""
    out = []
    for lyr in QgsProject.instance().mapLayers().values():
        try:
            if (lyr.type() == QgsMapLayerType.VectorLayer
                    and QgsWkbTypes.geometryType(lyr.wkbType()) == QgsWkbTypes.PointGeometry):
                out.append(lyr)
        except Exception:
            continue
    return sorted(out, key=lambda lyr: lyr.name().lower())


def points_from_layer(layer, dist_field: str, selected_only: bool, dest_authid: str) -> list:
    """``[(distanza | None, x, y), ...]`` nel CRS di destinazione, in ordine di id dell'elemento.

    ``dist_field`` vuoto = nessuna distanza (``None``). Con ``selected_only`` solo gli elementi
    selezionati.
    """
    feats = layer.selectedFeatures() if selected_only else list(layer.getFeatures())
    tr = None
    if layer.crs().isValid() and layer.crs().authid() != dest_authid:
        tr = QgsCoordinateTransform(layer.crs(), QgsCoordinateReferenceSystem(dest_authid),
                                    QgsProject.instance().transformContext())
    out = []
    for f in sorted(feats, key=lambda x: x.id()):
        g = f.geometry()
        if g is None or g.isNull():
            continue
        p = g.asMultiPoint()[0] if g.isMultipart() else g.asPoint()
        if tr is not None:
            p = tr.transform(QgsPointXY(p))
        d = None
        if dist_field:
            try:
                d = float(f[dist_field])
            except (TypeError, ValueError, KeyError) as exc:
                raise ErtGeorefError(f"campo '{dist_field}' non numerico in un elemento") from exc
        out.append((d, float(p.x()), float(p.y())))
    return out


class ErtImportDialog(QDialog):
    """Raccoglie file invertito, posizione, lunghezza e direzione; costruisce il georiferimento."""

    def __init__(self, crs_authid: str, parent=None, start_dir: str = "", canvas=None):
        super().__init__(parent)
        self.setWindowTitle("Importa sezione ERT")
        self.setMinimumWidth(540)
        self._crs = crs_authid
        self._start_dir = start_dir
        self._canvas = canvas
        self._tool = None
        self._prev_tool = None
        self._pick_index = -1
        self._profiles: list = []
        self._section = None
        self._georef = None
        self._error = ""

        form = QFormLayout()
        self._ed_csv = QLineEdit()
        self._ed_csv.setPlaceholderText("CSV/XYZ invertito esportato da ResIPY")
        form.addRow("Sezione invertita", self._row(self._ed_csv, self._browse_csv))

        self._cb_mode = QComboBox()
        self._cb_mode.addItem("Profilo di un KML (primo e ultimo punto)", MODE_KML)
        self._cb_mode.addItem("Due punti sulla mappa", MODE_MAP)
        self._cb_mode.addItem("Punti da un layer (anche uno per elettrodo)", MODE_LAYER)
        self._cb_mode.currentIndexChanged.connect(self._on_mode)
        form.addRow("Posizione", self._cb_mode)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._page_kml())
        self._stack.addWidget(self._page_map())
        self._stack.addWidget(self._page_layer())
        form.addRow("", self._stack)

        self._spin_len = QDoubleSpinBox()
        self._spin_len.setRange(0.5, 5000.0)
        self._spin_len.setDecimals(2)
        self._spin_len.setSuffix(" m")
        self._spin_len.setValue(71.0)
        self._spin_len.setToolTip("Distanza tra primo e ultimo elettrodo. Proposta dalla sezione; correggila "
                                  "se conosci la lunghezza esatta.")
        self._spin_len.valueChanged.connect(self._update)
        form.addRow("Lunghezza linea", self._spin_len)
        self._chk_invert = QCheckBox("Inverti la direzione (il primo elettrodo è l'ultimo punto)")
        self._chk_invert.toggled.connect(self._update)
        form.addRow("", self._chk_invert)
        self._ed_name = QLineEdit()
        form.addRow("Nome nel progetto", self._ed_name)

        self._lbl_info = QLabel("")
        self._lbl_info.setWordWrap(True)
        self._buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(self._lbl_info)
        lay.addWidget(self._buttons)
        self._ed_csv.editingFinished.connect(self._on_csv)
        self._fill_layers()                 # a costruzione finita: riempie i combo e chiama _update

    # --- pagine -------------------------------------------------------------------------------
    def _page_kml(self) -> QWidget:
        page = QWidget()
        f = QFormLayout(page)
        f.setContentsMargins(0, 0, 0, 0)
        self._ed_kml = QLineEdit()
        self._ed_kml.setPlaceholderText("KML con primo e ultimo punto del profilo")
        self._ed_kml.editingFinished.connect(self._on_kml)
        f.addRow("KML", self._row(self._ed_kml, self._browse_kml))
        self._cb_profile = QComboBox()
        self._cb_profile.setToolTip("Profilo del KML che corrisponde a questa sezione (stesso nome in linea e punti).")
        self._cb_profile.currentIndexChanged.connect(self._update)
        f.addRow("Profilo", self._cb_profile)
        return page

    def _page_map(self) -> QWidget:
        page = QWidget()
        f = QFormLayout(page)
        f.setContentsMargins(0, 0, 0, 0)
        self._ed_p = []
        self._btn_pick = []
        for i, label in enumerate(("Primo elettrodo", "Ultimo elettrodo")):
            ed = QLineEdit()
            ed.setPlaceholderText(f"x; y nel CRS {self._crs or 'del progetto'}")
            ed.editingFinished.connect(self._update)
            btn = QPushButton("Clic sulla mappa")
            btn.setEnabled(self._canvas is not None)
            btn.setToolTip("Chiude per un momento il dialogo: clicca il punto sulla mappa. "
                           "Le coordinate si possono anche scrivere o incollare a mano.")
            btn.clicked.connect(lambda _c=False, k=i: self._start_pick(k))
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(ed, 1)
            h.addWidget(btn)
            f.addRow(label, w)
            self._ed_p.append(ed)
            self._btn_pick.append(btn)
        return page

    def _page_layer(self) -> QWidget:
        page = QWidget()
        f = QFormLayout(page)
        f.setContentsMargins(0, 0, 0, 0)
        self._cb_layer = QComboBox()
        self._cb_layer.setToolTip("Layer di punti del progetto, nell'ordine di percorrenza della linea.")
        self._cb_layer.currentIndexChanged.connect(self._on_layer)
        f.addRow("Layer", self._cb_layer)
        self._cb_dist = QComboBox()
        self._cb_dist.setToolTip("Campo numerico con la distanza (m) di ogni punto dal primo elettrodo. "
                                 "Senza campo, la distanza è proporzionale alla lunghezza sulla mappa.")
        self._cb_dist.currentIndexChanged.connect(self._update)
        f.addRow("Campo distanza", self._cb_dist)
        self._chk_selected = QCheckBox("Solo gli elementi selezionati")
        self._chk_selected.toggled.connect(self._update)
        f.addRow("", self._chk_selected)
        return page

    @staticmethod
    def _row(edit: QLineEdit, slot) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(edit, 1)
        b = QPushButton("…")
        b.setMaximumWidth(32)
        b.clicked.connect(slot)
        h.addWidget(b)
        return w

    # --- modo ----------------------------------------------------------------------------------
    def mode(self) -> int:
        return int(self._cb_mode.currentData())

    def _on_mode(self, _i: int) -> None:
        self._stack.setCurrentIndex(self._cb_mode.currentIndex())
        if self.mode() == MODE_LAYER:
            self._fill_layers()
        self._update()

    # --- file ----------------------------------------------------------------------------------
    def _browse_csv(self) -> None:
        path, _f = QFileDialog.getOpenFileName(self, "Sezione ERT invertita", self._start_dir,
                                               "ERT (*.csv *.xyz *.dat *.txt);;Tutti i file (*)")
        if path:
            self._ed_csv.setText(path)
            self._on_csv()

    def _browse_kml(self) -> None:
        start = os.path.dirname(self._ed_csv.text()) or self._start_dir
        path, _f = QFileDialog.getOpenFileName(self, "KML con la posizione", start,
                                               "KML (*.kml);;Tutti i file (*)")
        if path:
            self._ed_kml.setText(path)
            self._on_kml()

    def _on_csv(self) -> None:
        path = self._ed_csv.text().strip()
        self._section = None
        self._error = ""
        if path and os.path.isfile(path):
            try:
                self._section = load_ert_section(path)
            except ErtImportError as exc:
                self._error = f"File non valido: {exc}"
                self._update()
                return
            # x = 0 e' il primo elettrodo: l'ultimo sta a x massimo (non all'estensione dei centri-cella)
            self._spin_len.setValue(suggested_length(float(self._section.x.max())))
            if not self._ed_name.text().strip():
                self._ed_name.setText(os.path.splitext(os.path.basename(path))[0])
            if not self._ed_kml.text().strip():              # un doc.kml accanto al file e' il caso comune
                near = os.path.join(os.path.dirname(path), "doc.kml")
                if os.path.isfile(near):
                    self._ed_kml.setText(near)
                    self._on_kml()
        self._update()

    def _on_kml(self) -> None:
        path = self._ed_kml.text().strip()
        self._profiles = []
        self._cb_profile.blockSignals(True)
        self._cb_profile.clear()
        if path and os.path.isfile(path):
            try:
                self._profiles = [p for p in read_kml_profiles(path) if len(p.line) >= 2 or len(p.points) >= 2]
            except Exception as exc:
                self._error = f"KML non leggibile: {exc}"
            stem = self._ed_name.text().strip().lower()
            guess = 0
            for i, p in enumerate(self._profiles):
                self._cb_profile.addItem(p.name, i)
                if p.name.lower() in stem:
                    guess = i
            if self._profiles:
                self._cb_profile.setCurrentIndex(guess)
        self._cb_profile.blockSignals(False)
        self._update()

    # --- punti sulla mappa -----------------------------------------------------------------------
    def _start_pick(self, index: int) -> None:
        if self._canvas is None:
            return
        self._pick_index = index
        self._prev_tool = self._canvas.mapTool()
        self._tool = QgsMapToolEmitPoint(self._canvas)
        self._tool.canvasClicked.connect(lambda pt, _btn: self._on_map_click(index, pt))
        self._tool.deactivated.connect(self._end_pick)
        self._canvas.setMapTool(self._tool)
        self.hide()

    def _on_map_click(self, index: int, point) -> None:
        """Punto cliccato (CRS della mappa) -> campo ``index`` nel CRS del progetto."""
        x, y = float(point.x()), float(point.y())
        if self._canvas is not None:
            src = self._canvas.mapSettings().destinationCrs()
            dst = QgsCoordinateReferenceSystem(self._crs)
            if src.isValid() and dst.isValid() and src.authid() != dst.authid():
                p = QgsCoordinateTransform(src, dst, QgsProject.instance().transformContext()).transform(
                    QgsPointXY(x, y))
                x, y = float(p.x()), float(p.y())
        self._ed_p[index].setText(f"{x:.3f}; {y:.3f}")
        self._end_pick()
        self._update()

    def _end_pick(self) -> None:
        """Rimette lo strumento precedente e riporta il dialogo."""
        tool, self._tool = self._tool, None
        if tool is not None and self._canvas is not None:
            try:
                tool.deactivated.disconnect(self._end_pick)
            except TypeError:
                pass
            if self._prev_tool is not None:
                self._canvas.setMapTool(self._prev_tool)
            else:
                self._canvas.unsetMapTool(tool)
        self._prev_tool = None
        if not self.isVisible():
            self.show()
            self.raise_()
            self.activateWindow()

    # --- layer ---------------------------------------------------------------------------------
    def _fill_layers(self) -> None:
        keep = self._cb_layer.currentData()
        self._cb_layer.blockSignals(True)
        self._cb_layer.clear()
        for lyr in point_layers():
            self._cb_layer.addItem(lyr.name(), lyr.id())
        i = self._cb_layer.findData(keep)
        self._cb_layer.setCurrentIndex(i if i >= 0 else 0)
        self._cb_layer.blockSignals(False)
        self._on_layer()

    def _current_layer(self):
        lid = self._cb_layer.currentData()
        return QgsProject.instance().mapLayer(lid) if lid else None

    def _on_layer(self, *_a) -> None:
        lyr = self._current_layer()
        self._cb_dist.blockSignals(True)
        self._cb_dist.clear()
        self._cb_dist.addItem(_NO_FIELD, "")
        if lyr is not None:
            for fld in lyr.fields():
                if fld.isNumeric():
                    self._cb_dist.addItem(fld.name(), fld.name())
        self._cb_dist.blockSignals(False)
        self._update()

    # --- calcolo -------------------------------------------------------------------------------
    def _georef_for_mode(self):
        """``(georef, nota)`` per il modo scelto; solleva ``ErtGeorefError`` se mancano dati."""
        length = self._spin_len.value()
        invert = self._chk_invert.isChecked()
        mode = self.mode()
        if mode == MODE_KML:
            if not self._profiles:
                raise ErtGeorefError("Scegli un KML con il profilo (primo e ultimo punto).")
            prof = self._profiles[int(self._cb_profile.currentData() or 0)]
            return georef_from_kml_profile(prof, length, lonlat_transformer(self._crs), self._crs, invert), ""
        if mode == MODE_MAP:
            pts = [parse_xy(ed.text()) for ed in self._ed_p]
            if None in pts:
                raise ErtGeorefError("Dai i due punti: clicca sulla mappa o scrivi «x; y».")
            return georef_from_xy_points(pts, length, self._crs, invert), ""
        lyr = self._current_layer()
        if lyr is None:
            raise ErtGeorefError("Nel progetto non ci sono layer di punti.")
        field = str(self._cb_dist.currentData() or "")
        pts = points_from_layer(lyr, field, self._chk_selected.isChecked(), self._crs)
        if len(pts) < 2:
            raise ErtGeorefError(f"Il layer «{lyr.name()}» ha {len(pts)} punti: ne servono almeno 2"
                                 + (" (solo selezionati)." if self._chk_selected.isChecked() else "."))
        if field:
            return ErtGeoref(pts, self._crs), f"{len(pts)} punti ancorati al campo «{field}»."
        return (georef_from_xy_points([(x, y) for _d, x, y in pts], length, self._crs, invert),
                f"{len(pts)} punti, distanza proporzionale all'ordine sulla mappa.")

    def _update(self, *_a) -> None:
        self._georef = None
        ok = False
        msg = ""
        # con distanze esplicite dal layer la direzione e' gia' decisa dai dati
        anchored = self.mode() == MODE_LAYER and bool(self._cb_dist.currentData())
        self._chk_invert.setEnabled(not anchored)
        self._spin_len.setEnabled(not anchored)
        if self._section is None:
            msg = self._error or "Scegli il file invertito."
        else:
            try:
                self._georef, note = self._georef_for_mode()
                msg = (f"Sulla mappa la linea misura {self._georef.map_length():.1f} m "
                       f"(sezione {self._georef.points[-1].distance - self._georef.points[0].distance:.1f} m), "
                       f"CRS {self._crs}." + (f"\n{note}" if note else ""))
                warns = self._georef.check()
                if warns:
                    msg += "\n⚠ " + "\n⚠ ".join(warns)
                ok = bool(self._ed_name.text().strip())
                if not ok:
                    msg += "\nDai un nome alla sezione."
            except (ErtGeorefError, ErtImportError) as exc:
                msg = str(exc)
        self._lbl_info.setText(msg)
        self._buttons.button(QDialogButtonBox.Ok).setEnabled(ok)

    def result_data(self) -> dict:
        """``{"name", "csv", "georef"}`` dopo ``accept``."""
        return {"name": self._ed_name.text().strip(), "csv": self._ed_csv.text().strip(),
                "georef": self._georef}
