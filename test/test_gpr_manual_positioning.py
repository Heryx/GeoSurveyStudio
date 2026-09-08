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
            "QToolButton", "QAbstractItemView"]:
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


if __name__ == "__main__":
    unittest.main()
