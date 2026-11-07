# -*- coding: utf-8 -*-
"""Limite della griglia delle time-slice (poligoni di ritaglio): maschera
punto-in-poligono, spec, motore "bin" e strumento di disegno nel viewer."""

import importlib
import pathlib
import sys
import time
import unittest
from types import SimpleNamespace

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_PLUGIN_PARENT = _HERE.parents[1].parent
_TEST_DIR = _HERE.parent
for _p in (str(_PLUGIN_PARENT), str(_TEST_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PKG = "rasterlinker_feature_interpolation"
clip = importlib.import_module(PKG + ".slice_clip")
slice_spec = importlib.import_module(PKG + ".slice_spec")
SliceSpec = slice_spec.SliceSpec
SliceSpecError = slice_spec.SliceSpecError

try:
    import scipy  # noqa: F401
    HAS_SCIPY = True
except Exception:  # pragma: no cover
    HAS_SCIPY = False

import test_viewer_timeslice as _vts  # noqa: E402

HAS_MPL, HAS_VIEWER = _vts.HAS_MPL, _vts.HAS_VIEWER


def _brute_inside(ring, x, y):
    """Regola pari-dispari, un punto alla volta (riferimento lento)."""
    inside = False
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            if x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
                inside = not inside
    return inside


class MaskTest(unittest.TestCase):
    X0, Y0, RES, NX, NY = 100.0, 200.0, 0.5, 40, 30

    def _mask(self, rings):
        return clip.polygons_mask(rings, self.X0, self.Y0, self.NX, self.NY, self.RES)

    def test_rectangle_selects_the_cells_whose_centre_is_inside(self):
        # cell (r, c) has its centre at (X0 + c*RES, Y0 + r*RES)
        # (edges off the cell centres: a centre exactly on an edge is a coin toss)
        ring = [[102.1, 202.1], [109.9, 202.1], [109.9, 207.9], [102.1, 207.9]]
        m = self._mask([ring])
        self.assertEqual(m.shape, (self.NY, self.NX))
        cols = np.where(m.any(axis=0))[0]
        rows = np.where(m.any(axis=1))[0]
        self.assertEqual((cols.min(), cols.max()), (5, 19))     # 102.1..109.9 m -> c 5..19
        self.assertEqual((rows.min(), rows.max()), (5, 15))     # 202.1..207.9 m -> r 5..15
        self.assertEqual(int(m.sum()), 15 * 11)

    def test_matches_a_brute_force_test_on_a_concave_polygon(self):
        ring = [[101.13, 201.17], [118.21, 203.09], [112.07, 210.23], [119.11, 214.19],
                [104.17, 213.53], [108.03, 207.11]]
        m = self._mask([ring])
        xs = self.X0 + np.arange(self.NX) * self.RES
        ys = self.Y0 + np.arange(self.NY) * self.RES
        want = np.array([[_brute_inside(ring, x, y) for x in xs] for y in ys])
        # no vertex sits on a cell row/centre here, so the two must agree
        self.assertTrue(np.array_equal(m, want))
        self.assertGreater(int(m.sum()), 50)

    def test_polygons_are_united(self):
        a = [[101.0, 201.0], [104.0, 201.0], [104.0, 204.0], [101.0, 204.0]]
        b = [[112.0, 208.0], [116.0, 208.0], [116.0, 212.0], [112.0, 212.0]]
        both = self._mask([a, b])
        self.assertTrue(np.array_equal(both, self._mask([a]) | self._mask([b])))

    def test_polygon_outside_the_grid_or_degenerate_selects_nothing(self):
        far = [[1000.0, 1000.0], [1010.0, 1000.0], [1010.0, 1010.0]]
        self.assertEqual(int(self._mask([far]).sum()), 0)
        self.assertEqual(int(self._mask([[[1.0, 1.0], [2.0, 2.0]]]).sum()), 0)
        self.assertEqual(int(self._mask([]).sum()), 0)

    def test_normalize_rings(self):
        ring = [[0, 0], [4, 0], [4, 4], [0, 4], [0, 0], [float("nan"), 1]]
        self.assertEqual(clip.normalize_rings([ring]), [[[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]]])
        self.assertEqual(clip.normalize_rings([[[0, 0], [1, 1]]]), [])
        self.assertEqual(clip.normalize_rings(None), [])

    def test_apply_clip_without_polygons_returns_the_same_object(self):
        g = np.ones((3, 3), dtype=np.float32)
        self.assertIs(clip.apply_clip(g, [], 0, 0, 1.0), g)


class SpecTest(unittest.TestCase):
    def test_default_has_no_clip_and_roundtrips(self):
        spec = SliceSpec()
        self.assertFalse(spec.has_clip)
        ring = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
        back = SliceSpec.from_json(SliceSpec(clip_polygons=[ring]).to_json())
        self.assertTrue(back.has_clip)
        self.assertEqual(back.clip_polygons, [ring])

    def test_old_spec_without_the_field_still_loads(self):
        d = {"format": "rasterlinker_slicespec_v1", "balance": "channel", "thickness_m": 0.2}
        self.assertEqual(SliceSpec.from_dict(d).clip_polygons, [])

    def test_two_vertex_polygon_is_refused(self):
        with self.assertRaises(SliceSpecError):
            SliceSpec(clip_polygons=[[[0.0, 0.0], [1.0, 1.0]]]).validate()

    def test_instances_do_not_share_the_list(self):
        a, b = SliceSpec(), SliceSpec()
        a.clip_polygons.append([[0, 0], [1, 0], [1, 1]])
        self.assertEqual(b.clip_polygons, [])


@unittest.skipUnless(HAS_SCIPY, "SciPy required")
class EngineTest(unittest.TestCase):
    def setUp(self):
        import test_slice_engine as base
        self.engine = importlib.import_module(PKG + ".slice_engine")
        self.profs = base._array_survey(
            n_prof=5, n_ch=6,
            fill=lambda p, c, n_s, n_t: np.random.default_rng(p * 10 + c).normal(1.0, 0.2, (n_s, n_t)),
        )

    def _spec(self, **kw):
        base = dict(z_min_m=0.9, z_max_m=1.3, thickness_m=0.1, resolution_m=0.1,
                    search_radius_m=0.5, smooth_lowpass=3, balance="none")
        base.update(kw)
        return SliceSpec(**base).resolve(self.profs)

    def _grid(self, **kw):
        grids, meta = self.engine.compute_grids(self.profs, self._spec(**kw),
                                                single_window=(1.0, 1.1), emit_diagnostics=False)
        return grids[0], meta

    def _central_ring(self, meta):
        x0, x1 = meta["x_min"], meta["x_max"]
        y0, y1 = meta["y_min"], meta["y_max"]
        # keep the middle 60 % of the grid in both directions
        mx, my = 0.2 * (x1 - x0), 0.2 * (y1 - y0)
        return [[x0 + mx, y0 + my], [x1 - mx, y0 + my], [x1 - mx, y1 - my], [x0 + mx, y1 - my]]

    def test_outside_is_nodata_and_inside_is_untouched(self):
        full, meta = self._grid()
        ring = self._central_ring(meta)
        cut, meta2 = self._grid(clip_polygons=[ring])
        g0, g1 = full["grid"], cut["grid"]
        self.assertEqual(g0.shape, g1.shape)
        mask = clip.polygons_mask([ring], meta["x_min"], meta["y_min"], g0.shape[1], g0.shape[0],
                                  meta["resolution"])
        self.assertTrue(np.all(np.isnan(g1[~mask])))
        # clipping is the last step: the cells it keeps are bit-identical
        self.assertTrue(np.array_equal(g1[mask], g0[mask], equal_nan=True))
        self.assertLess(cut["n_valid_cells"], full["n_valid_cells"])
        self.assertEqual(cut["n_valid_cells"], int(np.isfinite(g1).sum()))
        self.assertTrue(meta2["clip_applied"])
        self.assertFalse(meta["clip_applied"])

    def test_no_polygon_gives_exactly_the_old_result(self):
        a, _ = self._grid()
        b, _ = self._grid(clip_polygons=[])
        self.assertTrue(np.array_equal(a["grid"], b["grid"], equal_nan=True))

    def test_clip_is_saved_in_the_manifest_spec(self):
        _, meta = self._grid()
        ring = self._central_ring(meta)
        _, meta2 = self._grid(clip_polygons=[ring])
        self.assertEqual(meta2["slice_spec"]["clip_polygons"], [[list(p) for p in ring]])

    def test_preview_and_batch_agree(self):
        _, meta = self._grid()
        ring = self._central_ring(meta)
        spec = self._spec(clip_polygons=[ring], z_min_m=1.0, z_max_m=1.1)
        one, _ = self.engine.compute_grids(self.profs, spec, single_window=(1.0, 1.1), emit_diagnostics=False)
        batch, _ = self.engine.compute_grids(self.profs, spec, emit_diagnostics=False)
        self.assertEqual(len(batch), 1)
        self.assertTrue(np.array_equal(one[0]["grid"], batch[0]["grid"], equal_nan=True))

    def test_legacy_idw_path_is_clipped_too(self):
        slicer = importlib.import_module(PKG + ".gpr_ogpr_slicer")
        _, meta = self._grid()
        ring = self._central_ring(meta)
        spec = self._spec(interp="idw", clip_polygons=[ring])
        grids, m = slicer.compute_slice(self.profs, spec, single_window=(1.0, 1.1), emit_diagnostics=False)
        g = np.asarray(grids[0]["grid"])
        mask = clip.polygons_mask([ring], m["x_min"], m["y_min"], g.shape[1], g.shape[0], m["resolution"])
        self.assertTrue(np.all(np.isnan(g[~mask])))
        self.assertGreater(int(np.isfinite(g).sum()), 0)


@unittest.skipUnless(HAS_MPL and HAS_VIEWER, "matplotlib/viewer non disponibili")
class ViewerToolTest(_vts.SlicingTabTest):
    """Riusa il banco del tab Slicing (setUp con 4 profili sintetici)."""

    # i test ereditati girano solo nella classe base: qui si saltano
    for _name in [n for n in dir(_vts.SlicingTabTest) if n.startswith("test_")]:
        locals()[_name] = None
    del _name

    def _ev(self, x, y, button=1, dbl=False):
        return SimpleNamespace(inaxes=self.v._ax_slice, xdata=x, ydata=y, button=button,
                               dblclick=dbl, x=0.0, y=0.0)

    def _preview(self):
        self.v._btn_preview_slice.setChecked(True)
        self.assertTrue(self._wait(lambda: self.v._preview_slice_grid is not None))
        self._pump(50)
        return self.v._preview_slice_extent

    def _central_ring(self, ext):
        x0, x1, y0, y1 = ext
        mx, my = 0.25 * (x1 - x0), 0.25 * (y1 - y0)
        return [(x0 + mx, y0 + my), (x1 - mx, y0 + my), (x1 - mx, y1 - my), (x0 + mx, y1 - my)]

    def _draw(self, ring):
        self.panel.btn_clip_draw.setChecked(True)
        self.assertTrue(self.v._clip_is_drawing())
        for x, y in ring:
            self.assertTrue(self.v._clip_handle_press(self._ev(x, y)))
        self.assertTrue(self.v._clip_handle_press(self._ev(0, 0, button=3)))

    def test_form_roundtrip_and_label(self):
        ring = [[0.0, 0.0], [5.0, 0.0], [5.0, 5.0]]
        self.panel.set_clip_polygons([ring])
        self.assertEqual(self.panel.get_spec().clip_polygons, [ring])
        self.assertIn("1 poligono", self.panel.lbl_clip.text())
        self.assertTrue(self.panel.btn_clip_clear.isEnabled())
        self.panel.btn_clip_clear.click()
        self.assertEqual(self.panel.get_spec().clip_polygons, [])
        self.assertFalse(self.panel.btn_clip_clear.isEnabled())

    def test_set_spec_restores_the_polygon(self):
        ring = [[1.0, 1.0], [6.0, 1.0], [6.0, 6.0]]
        self.panel.set_spec(SliceSpec(clip_polygons=[ring]))
        self.assertEqual(self.panel.clip_polygons(), [ring])
        self.panel.set_spec(SliceSpec())
        self.assertEqual(self.panel.clip_polygons(), [])

    def test_drawing_needs_a_slice_on_screen(self):
        self.panel.btn_clip_draw.setChecked(True)
        self.assertFalse(self.v._clip_is_drawing())
        self.assertFalse(self.panel.btn_clip_draw.isChecked())

    def test_draw_a_polygon_clips_the_preview(self):
        ext = self._preview()
        full = self.v._preview_slice_grid.copy()
        ring = self._central_ring(ext)
        self.v._preview_slice_grid = None
        self._draw(ring)
        self.assertFalse(self.v._clip_is_drawing())            # closing ends the tool
        self.assertFalse(self.panel.btn_clip_draw.isChecked())
        self.assertEqual(len(self.panel.get_spec().clip_polygons), 1)
        self.assertTrue(self._wait(lambda: self.v._preview_slice_grid is not None))
        cut = self.v._preview_slice_grid
        self.assertEqual(cut.shape, full.shape)
        self.assertLess(int(np.isfinite(cut).sum()), int(np.isfinite(full).sum()))
        # the polygon is drawn over the slice
        self.assertTrue(len(self.v._clip_state()["artists"]) >= 1)

    def test_preview_is_not_clipped_while_drawing(self):
        ext = self._preview()
        full = self.v._preview_slice_grid.copy()
        self.panel.set_clip_polygons([self._central_ring(ext)])
        self.assertTrue(self._wait(
            lambda: np.isfinite(self.v._preview_slice_grid).sum() < np.isfinite(full).sum()))
        self.v._preview_slice_grid = None
        self.panel.btn_clip_draw.setChecked(True)
        self.assertTrue(self._wait(lambda: self.v._preview_slice_grid is not None))
        self.assertEqual(int(np.isfinite(self.v._preview_slice_grid).sum()), int(np.isfinite(full).sum()))

    def test_undo_removes_the_last_vertex_then_the_last_polygon(self):
        ext = self._preview()
        self.panel.set_clip_polygons([self._central_ring(ext)])
        self.panel.btn_clip_draw.setChecked(True)
        x0, x1, y0, y1 = ext
        for x, y in ((x0, y0), (x1, y0), (x1, y1)):
            self.v._clip_handle_press(self._ev(x, y))
        self.assertEqual(len(self.v._clip_state()["pending"]), 3)
        self.panel.btn_clip_undo.click()
        self.assertEqual(len(self.v._clip_state()["pending"]), 2)
        self.panel.btn_clip_undo.click()
        self.panel.btn_clip_undo.click()
        self.assertEqual(self.v._clip_state()["pending"], [])
        self.assertEqual(len(self.panel.get_spec().clip_polygons), 1)   # still there
        self.panel.btn_clip_undo.click()
        self.assertEqual(self.panel.get_spec().clip_polygons, [])

    def test_fewer_than_three_vertices_is_discarded_and_escape_cancels(self):
        ext = self._preview()
        self.panel.btn_clip_draw.setChecked(True)
        x0, x1, y0, y1 = ext
        self.v._clip_handle_press(self._ev(x0, y0))
        self.v._clip_handle_press(self._ev(x1, y1))
        self.v._clip_handle_press(self._ev(0, 0, dbl=True))
        self.assertEqual(self.panel.get_spec().clip_polygons, [])
        self.assertTrue(self.v._clip_is_drawing())
        self.v._clip_handle_press(self._ev(x0, y0))
        self.v._on_clip_key(SimpleNamespace(key="escape"))
        self.assertEqual(self.v._clip_state()["pending"], [])
        self.assertTrue(self.v._clip_is_drawing())
        self.v._on_clip_key(SimpleNamespace(key="escape"))
        self.assertFalse(self.v._clip_is_drawing())
        self.assertFalse(self.panel.btn_clip_draw.isChecked())

    def test_tool_does_not_eat_clicks_when_off(self):
        self.assertFalse(self.v._clip_handle_press(self._ev(1.0, 1.0)))
        self.assertFalse(self.v._clip_handle_move(self._ev(1.0, 1.0)))

    def test_clip_is_part_of_the_project_state(self):
        ring = [[1.0, 1.0], [6.0, 1.0], [6.0, 6.0]]
        self.panel.set_clip_polygons([ring])
        d = self.panel.get_spec().to_dict()
        self.panel.set_clip_polygons([])
        self.panel.set_spec(SliceSpec.from_dict(d))
        self.assertEqual(self.panel.clip_polygons(), [ring])


if __name__ == "__main__":
    unittest.main()
