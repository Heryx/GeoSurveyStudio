# -*- coding: utf-8 -*-
"""
Test unitari per GprManualPositioningDialog e check_profiles_have_gps.

Testa:
  - check_profiles_have_gps() con dati GPS plausibili e non.
  - Calcolo automatico spaziatura uniforme profili asse X.
  - Calcolo coordinate con inversione (reversing).
  - Zig-zag automatico (inversione alternata).
  - Rotazione azimutale e traslazione geografica.
  - Profili ortogonali (asse Y).
"""

import math
import unittest
import sys
import os
import types

import numpy as np

# Stub leggero dei moduli Qt non disponibili nell'ambiente di test.
for mod_name in [
    "qgis", "qgis.PyQt", "qgis.PyQt.QtCore", "qgis.PyQt.QtWidgets",
    "matplotlib", "matplotlib.backends", "matplotlib.backends.backend_qt5agg",
    "matplotlib.figure",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

# Stub minimi per Qt e matplotlib usati nella dialog
QtCore = sys.modules["qgis.PyQt.QtCore"]
QtCore.Qt = type("Qt", (), {"Horizontal": 1, "AlignCenter": 132, "ItemIsEditable": 2})()
QtCore.pyqtSignal = lambda *a, **kw: None

QtWidgets = sys.modules["qgis.PyQt.QtWidgets"]
for cls in ["QDialog", "QVBoxLayout", "QHBoxLayout", "QGridLayout", "QFormLayout",
            "QLabel", "QDoubleSpinBox", "QSpinBox", "QCheckBox", "QComboBox",
            "QPushButton", "QTableWidget", "QTableWidgetItem", "QHeaderView",
            "QGroupBox", "QSplitter", "QWidget", "QMessageBox", "QDialogButtonBox",
            "QToolButton", "QAbstractItemView", "QLineEdit", "QFileDialog"]:
    setattr(QtWidgets, cls, type(cls, (), {
        "__init__": lambda self, *a, **kw: None,
    }))

mpl_mods = {
    "matplotlib.backends.backend_qt5agg": ["FigureCanvasQTAgg"],
    "matplotlib.figure": ["Figure"],
}
for mod, attrs in mpl_mods.items():
    for attr in attrs:
        setattr(sys.modules[mod], attr, None)

# Importa il modulo reale (senza UI Qt)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# -- Helper per creare profili mock --

def _make_profile(easting=None, northing=None, n_slices=100, step=0.1):
    """Crea un OgprProfile-like mock con i campi minimi necessari."""
    ch = types.SimpleNamespace(
        data=np.zeros((10, n_slices), dtype=np.float32),
        easting=np.asarray(easting if easting is not None else np.zeros(n_slices), dtype=np.float64),
        northing=np.asarray(northing if northing is not None else np.zeros(n_slices), dtype=np.float64),
        altitude=np.zeros(n_slices, dtype=np.float64),
        distances=np.arange(n_slices, dtype=np.float64) * step,
        heading=np.zeros(n_slices, dtype=np.float64),
    )
    prof = types.SimpleNamespace(
        path=f"/tmp/profile_{id(ch)}.ogpr",
        swath_name="",
        n_slices=n_slices,
        n_channels=1,
        n_samples=10,
        sampling_step_m=step,
        depth_max_m=1.0,
        channels=[ch],
        header_raw={},
        parse_warnings=[],
    )
    return prof


# -- Test check_profiles_have_gps --

class TestCheckProfilesHaveGps(unittest.TestCase):

    def test_no_profiles_returns_false(self):
        from gpr_manual_positioning_dialog import check_profiles_have_gps
        self.assertFalse(check_profiles_have_gps([]))

    def test_zero_coords_returns_false(self):
        from gpr_manual_positioning_dialog import check_profiles_have_gps
        prof = _make_profile(easting=np.zeros(50), northing=np.zeros(50))
        self.assertFalse(check_profiles_have_gps([prof]))

    def test_small_local_coords_returns_false(self):
        from gpr_manual_positioning_dialog import check_profiles_have_gps
        # Coordinate locali (es. da posizionamento sintetico) con span piccolo
        e = np.linspace(0.0, 10.0, 50)
        n = np.zeros(50)
        prof = _make_profile(easting=e, northing=n)
        self.assertFalse(check_profiles_have_gps([prof]))

    def test_utm_coords_returns_true(self):
        from gpr_manual_positioning_dialog import check_profiles_have_gps
        # Coordinate UTM plausibili
        e = np.linspace(350_000.0, 350_100.0, 50)
        n = np.linspace(4_500_000.0, 4_500_050.0, 50)
        prof = _make_profile(easting=e, northing=n)
        self.assertTrue(check_profiles_have_gps([prof]))


# -- Helper di calcolo indipendente dalla UI --

def _compute_profile_coordinates(
    x0, x1, y0, y1,
    profile_items,
    e0=0.0, n0=0.0, az_deg=0.0, geo_on=False,
):
    """Replica la logica di _apply_and_accept senza UI Qt."""
    az_rad = math.radians(az_deg)
    cos_a, sin_a = math.cos(az_rad), math.sin(az_rad)
    results = []
    for it in profile_items:
        axis = it["axis"]
        offset = it["offset"]
        is_rev = it["reversed"]
        n_t = it["n_traces"]
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
            final_e = e0 + lx * cos_a + ly * sin_a
            final_n = n0 - lx * sin_a + ly * cos_a
        else:
            final_e = lx
            final_n = ly
        results.append((final_e, final_n))
    return results


class TestSpacingCalculation(unittest.TestCase):

    def test_10_profiles_0_to_10_spacing_1m(self):
        """10 profili in Y=[0,10] devono essere spaziati di 1 m."""
        n = 10
        y0, y1 = 0.0, 9.0  # span = 9 m → step = 1.0 m tra 10 profili
        step = (y1 - y0) / (n - 1)
        offsets = [round(y0 + i * step, 6) for i in range(n)]
        self.assertAlmostEqual(offsets[0], 0.0)
        self.assertAlmostEqual(offsets[-1], 9.0)
        self.assertAlmostEqual(step, 1.0)

    def test_5_profiles_0_to_4_spacing_1m(self):
        n = 5
        y0, y1 = 0.0, 4.0
        step = (y1 - y0) / (n - 1)
        offsets = [round(y0 + i * step, 6) for i in range(n)]
        self.assertAlmostEqual(step, 1.0)
        self.assertAlmostEqual(offsets[2], 2.0)


class TestCoordinateCalculation(unittest.TestCase):

    def _item(self, axis="X", offset=0.0, reversed_=False, n_traces=100):
        return {"axis": axis, "offset": offset, "reversed": reversed_, "n_traces": n_traces}

    def test_single_x_profile_not_reversed(self):
        it = self._item(axis="X", offset=0.0, reversed_=False, n_traces=10)
        (e, n) = _compute_profile_coordinates(0.0, 10.0, 0.0, 5.0, [it])[0]
        self.assertAlmostEqual(float(e[0]), 0.0, places=6)
        self.assertAlmostEqual(float(e[-1]), 10.0, places=6)
        np.testing.assert_array_almost_equal(n, 0.0)

    def test_single_x_profile_reversed(self):
        it = self._item(axis="X", offset=2.0, reversed_=True, n_traces=10)
        (e, n) = _compute_profile_coordinates(0.0, 10.0, 0.0, 5.0, [it])[0]
        # Reversed: starts at X=10, ends at X=0
        self.assertAlmostEqual(float(e[0]), 10.0, places=6)
        self.assertAlmostEqual(float(e[-1]), 0.0, places=6)
        np.testing.assert_array_almost_equal(n, 2.0)

    def test_single_y_profile(self):
        it = self._item(axis="Y", offset=3.0, reversed_=False, n_traces=5)
        (e, n) = _compute_profile_coordinates(0.0, 10.0, 0.0, 8.0, [it])[0]
        # Y-axis profile: X is constant = offset, Y varies from y0 to y1
        np.testing.assert_array_almost_equal(e, 3.0)
        self.assertAlmostEqual(float(n[0]), 0.0, places=6)
        self.assertAlmostEqual(float(n[-1]), 8.0, places=6)

    def test_zigzag_alternating(self):
        """Profili zig-zag: pari avanti, dispari indietro."""
        items = [
            self._item(axis="X", offset=0.0, reversed_=(i % 2 == 1), n_traces=5)
            for i in range(4)
        ]
        results = _compute_profile_coordinates(0.0, 10.0, 0.0, 3.0, items)
        # Profilo 0 (avanti): e[0]=0, e[-1]=10
        self.assertAlmostEqual(float(results[0][0][0]), 0.0)
        self.assertAlmostEqual(float(results[0][0][-1]), 10.0)
        # Profilo 1 (indietro): e[0]=10, e[-1]=0
        self.assertAlmostEqual(float(results[1][0][0]), 10.0)
        self.assertAlmostEqual(float(results[1][0][-1]), 0.0)
        # Profilo 2 (avanti): e[0]=0
        self.assertAlmostEqual(float(results[2][0][0]), 0.0)
        # Profilo 3 (indietro): e[0]=10
        self.assertAlmostEqual(float(results[3][0][0]), 10.0)


class TestGeorefRotation(unittest.TestCase):

    def _item(self, axis="X", offset=0.0, reversed_=False, n_traces=5):
        return {"axis": axis, "offset": offset, "reversed": reversed_, "n_traces": n_traces}

    def test_no_rotation_adds_origin(self):
        """Az=0°, geo_on=True: risultato è lx + e0, ly + n0."""
        it = self._item(axis="X", offset=0.0, n_traces=5)
        (e, n) = _compute_profile_coordinates(
            0.0, 10.0, 0.0, 5.0, [it],
            e0=100.0, n0=200.0, az_deg=0.0, geo_on=True
        )[0]
        # Az=0: cos=1, sin=0 → e = e0 + lx, n = n0 + ly
        self.assertAlmostEqual(float(e[0]), 100.0, places=5)
        self.assertAlmostEqual(float(e[-1]), 110.0, places=5)
        np.testing.assert_array_almost_equal(n, 200.0, decimal=5)

    def test_90deg_rotation(self):
        """Az=90°: lx→Nord, ly→Est (scambio assi con cos/sin)."""
        it = self._item(axis="X", offset=0.0, n_traces=3)
        (e, n) = _compute_profile_coordinates(
            0.0, 10.0, 0.0, 5.0, [it],
            e0=0.0, n0=0.0, az_deg=90.0, geo_on=True
        )[0]
        # Az=90: cos_a=0, sin_a=1 → de = lx*0 + ly*1 = ly=0, dn = -lx*1 + ly*0 = -lx
        # e = 0 + 0 = 0, n = 0 + (-lx)
        np.testing.assert_array_almost_equal(e, 0.0, decimal=5)
        self.assertAlmostEqual(float(n[0]), 0.0, places=4)
        self.assertAlmostEqual(float(n[-1]), -10.0, places=4)


class TestComputePositions(unittest.TestCase):
    """Test della funzione reale `compute_positions` (navigazione: lunghezza
    reale + stagger)."""

    def _fn(self):
        from gpr_manual_positioning_dialog import compute_positions
        return compute_positions

    def test_uniform_matches_grid_span(self):
        cp = self._fn()
        e, n, d, z = cp(11, 0.0, 10.0, 0.0, 5.0, "X", 2.0,
                        grid_span_m=10.0, length_m=10.0, use_real_length=False,
                        reversed_=False)
        np.testing.assert_allclose(e, np.linspace(0.0, 10.0, 11))
        np.testing.assert_allclose(n, 2.0)
        self.assertAlmostEqual(float(d[-1]), 10.0, places=6)
        np.testing.assert_allclose(z, 0.0)  # nessuna quota per default

    def test_reversed_flips(self):
        cp = self._fn()
        e, _, _, _ = cp(11, 0.0, 10.0, 0.0, 5.0, "X", 0.0,
                        grid_span_m=10.0, length_m=10.0, use_real_length=False,
                        reversed_=True)
        np.testing.assert_allclose(e, np.linspace(10.0, 0.0, 11))

    def test_real_length_occupies_partial_segment(self):
        cp = self._fn()
        # profilo lungo 4 m su una linea (span griglia) di 10 m -> occupa 0..4 m
        e, n, d, _ = cp(9, 0.0, 10.0, 0.0, 5.0, "X", 1.0,
                        grid_span_m=10.0, length_m=4.0, use_real_length=True,
                        reversed_=False)
        self.assertAlmostEqual(float(e[0]), 0.0, places=6)
        self.assertAlmostEqual(float(e[-1]), 4.0, places=6)
        self.assertAlmostEqual(float(d[-1]), 4.0, places=6)

    def test_stagger_translates_whole_reversed_line(self):
        cp = self._fn()
        # forward line: stagger ignored entirely
        fwd = cp(11, 0.0, 10.0, 0.0, 5.0, "X", 0.0, grid_span_m=10.0,
                 length_m=10.0, use_real_length=False, reversed_=False,
                 stagger_m=0.5)
        np.testing.assert_allclose(fwd[0], np.linspace(0.0, 10.0, 11))
        # reversed line: the WHOLE line translates by +0.5 m along X
        rev = cp(11, 0.0, 10.0, 0.0, 5.0, "X", 0.0, grid_span_m=10.0,
                 length_m=10.0, use_real_length=False, reversed_=True,
                 stagger_m=0.5)
        rev0 = cp(11, 0.0, 10.0, 0.0, 5.0, "X", 0.0, grid_span_m=10.0,
                  length_m=10.0, use_real_length=False, reversed_=True,
                  stagger_m=0.0)
        np.testing.assert_allclose(rev[0], rev0[0] + 0.5, atol=1e-9)
        np.testing.assert_allclose(rev[1], rev0[1], atol=1e-9)  # Y unchanged

    def test_stagger_axis_y(self):
        cp = self._fn()
        rev = cp(6, 0.0, 5.0, 0.0, 10.0, "Y", 2.0, grid_span_m=10.0,
                 length_m=10.0, use_real_length=False, reversed_=True,
                 stagger_m=-0.3)
        rev0 = cp(6, 0.0, 5.0, 0.0, 10.0, "Y", 2.0, grid_span_m=10.0,
                  length_m=10.0, use_real_length=False, reversed_=True,
                  stagger_m=0.0)
        np.testing.assert_allclose(rev[1], rev0[1] - 0.3, atol=1e-9)
        np.testing.assert_allclose(rev[0], rev0[0], atol=1e-9)  # X unchanged

    def test_georef_rotation_90(self):
        cp = self._fn()
        e, n, _, _ = cp(3, 0.0, 10.0, 0.0, 5.0, "X", 0.0, grid_span_m=10.0,
                        length_m=10.0, use_real_length=False, reversed_=False,
                        geo_on=True, e0=0.0, n0=0.0, az_deg=90.0)
        np.testing.assert_allclose(e, 0.0, atol=1e-6)
        self.assertAlmostEqual(float(n[0]), 0.0, places=4)
        self.assertAlmostEqual(float(n[-1]), -10.0, places=4)

    def test_elev_plane_linear_along_line(self):
        cp = self._fn()
        _, _, _, z = cp(11, 0.0, 10.0, 0.0, 5.0, "X", 0.0, grid_span_m=10.0,
                        length_m=10.0, use_real_length=False, reversed_=False,
                        elev_mode="plane", z_start=100.0, z_end=105.0)
        np.testing.assert_allclose(z, np.linspace(100.0, 105.0, 11))

    def test_elev_plane_follows_zigzag(self):
        cp = self._fn()
        # linea invertita: la prima traccia acquisita e' all'estremo "fine"
        # della linea di griglia -> quota z_end; l'ultima -> z_start.
        _, _, _, z = cp(11, 0.0, 10.0, 0.0, 5.0, "X", 0.0, grid_span_m=10.0,
                        length_m=10.0, use_real_length=False, reversed_=True,
                        elev_mode="plane", z_start=100.0, z_end=105.0)
        self.assertAlmostEqual(float(z[0]), 105.0, places=6)
        self.assertAlmostEqual(float(z[-1]), 100.0, places=6)

    def test_elev_file_x_column(self):
        cp = self._fn()
        ref = np.array([[0.0, 10.0], [5.0, 20.0], [10.0, 10.0]])
        _, _, _, z = cp(11, 0.0, 10.0, 0.0, 5.0, "X", 0.0, grid_span_m=10.0,
                        length_m=10.0, use_real_length=False, reversed_=False,
                        elev_mode="file", elev_ref=ref, elev_ref_kind="x")
        self.assertAlmostEqual(float(z[0]), 10.0, places=6)
        self.assertAlmostEqual(float(z[5]), 20.0, places=6)   # a meta' linea
        self.assertAlmostEqual(float(z[-1]), 10.0, places=6)

    def test_elev_file_trace_index(self):
        cp = self._fn()
        ref = np.array([[0.0, 0.0], [10.0, 100.0]])
        _, _, _, z = cp(11, 0.0, 10.0, 0.0, 5.0, "X", 0.0, grid_span_m=10.0,
                        length_m=10.0, use_real_length=False, reversed_=False,
                        elev_mode="file", elev_ref=ref, elev_ref_kind="trace")
        np.testing.assert_allclose(z, np.linspace(0.0, 100.0, 11))

    def test_parse_elev_file_mixed_separators(self, ):
        import tempfile, os as _os
        from gpr_manual_positioning_dialog import GprManualPositioningDialog
        txt = "pos,quota\n0 100\n5;102.5\n10\t101\nbad line\n"
        fd, path = tempfile.mkstemp(suffix=".csv")
        _os.close(fd)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(txt)
            arr = GprManualPositioningDialog._parse_elev_file(path)
        finally:
            _os.remove(path)
        self.assertEqual(arr.shape, (3, 2))
        np.testing.assert_allclose(arr[:, 0], [0.0, 5.0, 10.0])
        np.testing.assert_allclose(arr[:, 1], [100.0, 102.5, 101.0])


if __name__ == "__main__":
    unittest.main()
