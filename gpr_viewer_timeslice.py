# -*- coding: utf-8 -*-
"""Time-slice del GPR Profile Viewer: vista, catalogo, anteprima, navigazione, generazione e strumenti sui set.

Estratto da gpr_profile_viewer.py senza modifiche ai metodi."""

from __future__ import annotations

from . import project_zones
from .amplitude_stats import finite_values, format_stats, value_stats
from .gpr_viewer_common import (
    DEFAULT_CHAIN_ORDER,
    DEFAULT_CMAP,
    HAS_MPL,
    Optional,
    QCheckBox,
    QColor,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QInputDialog,
    QLineEdit,
    QMessageBox,
    QSpinBox,
    QThread,
    QgsProject,
    QgsRasterLayer,
    QgsRubberBand,
    QgsWkbTypes,
    Qt,
    _PreviewSliceWorker,
    _dbg,
    apply_geometry_from_profile,
    datetime,
    glob,
    json,
    load_processed_profile_npz,
    np,
    os,
    re,
    shutil,
)


class TimesliceViewMixin:
    """Time-slice del GPR Profile Viewer: vista, catalogo, anteprima, navigazione, generazione e strumenti sui set.

    Mixin di GprProfileViewer: i metodi usano `self` della finestra.
    """


    def _clear_timeslice_crosshair(self):
        for rb in (self._rb_crosshair_h, self._rb_crosshair_v):
            if rb is not None:
                try:
                    rb.setVisible(False)
                except Exception:
                    pass
                try:
                    rb.reset(QgsWkbTypes.LineGeometry)
                except Exception:
                    pass

    def _get_cached_catalog(self, project_root: str):
        if not project_root:
            return None
        cat_path = os.path.join(project_root, "metadata", "project_catalog.json")
        try:
            mtime = float(os.path.getmtime(cat_path))
        except Exception:
            mtime = None
        needs_reload = (
            self._cached_catalog is None
            or self._cached_catalog_pr != project_root
            or self._cached_catalog_mtime != mtime
        )
        if needs_reload:
            from .project_catalog import load_catalog

            self._cached_catalog = load_catalog(project_root)
            self._cached_catalog_pr = project_root
            self._cached_catalog_mtime = mtime
        return self._cached_catalog

    def _active_group_timeslices_context(self):
        if self.plugin is None:
            return None, None
        settings = getattr(self.plugin, "settings", None)
        settings_key = getattr(self.plugin, "settings_key_active_project", None)
        if settings is None or not settings_key:
            return None, None
        try:
            pr = (settings.value(settings_key, "", type=str) or "").strip()
        except Exception:
            pr = ""
        if not pr:
            return None, None
        catalog = self._get_cached_catalog(pr)
        if not isinstance(catalog, dict):
            return None, None

        groups = [g for g in catalog.get("raster_groups", []) if isinstance(g, dict)]
        if not groups:
            return None, None

        active_group = None
        active_group_id = str(getattr(self.plugin, "_active_group_id", "") or "").strip()
        if active_group_id:
            active_group = next((g for g in groups if str(g.get("id") or "").strip() == active_group_id), None)

        if active_group is None:
            dlg = getattr(self.plugin, "dlg", None)
            glw = getattr(dlg, "groupListWidget", None)
            current_item = None
            try:
                current_item = glw.currentItem() if glw is not None else None
                if current_item is None and glw is not None:
                    selected_items = list(glw.selectedItems() or [])
                    current_item = selected_items[0] if selected_items else None
            except Exception:
                current_item = None
            if current_item is not None:
                gid = str(current_item.data(Qt.UserRole) or "").strip()
                gname = str(current_item.text() or "").strip()
                if gid:
                    active_group = next((g for g in groups if str(g.get("id") or "").strip() == gid), None)
                if active_group is None and gname:
                    active_group = next(
                        (g for g in groups if str(g.get("name") or "").strip().lower() == gname.lower()),
                        None,
                    )

        if active_group is None and hasattr(self.plugin, "_selected_group_names"):
            try:
                selected_names = list(getattr(self.plugin, "_selected_group_names")() or [])
            except Exception:
                selected_names = []
            for name in selected_names:
                key = str(name or "").strip().lower()
                if not key:
                    continue
                active_group = next(
                    (g for g in groups if str(g.get("name") or "").strip().lower() == key),
                    None,
                )
                if active_group is not None:
                    break

        if active_group is None:
            active_group = next((g for g in groups if g.get("timeslice_ids")), None)
        if active_group is None:
            return None, None

        tids = [str(tid).strip() for tid in (active_group.get("timeslice_ids") or []) if str(tid).strip()]
        if not tids:
            return active_group, []
        ts_by_id = {
            str(t.get("id") or "").strip(): t
            for t in catalog.get("timeslices", [])
            if isinstance(t, dict) and str(t.get("id") or "").strip()
        }
        slices = [ts_by_id[tid] for tid in tids if tid in ts_by_id]
        if not slices:
            tids_set = set(tids)
            slices = [
                t for t in catalog.get("timeslices", [])
                if isinstance(t, dict) and str(t.get("id") or "").strip() in tids_set
            ]
        return active_group, slices

    def _nearest_timeslice_index(self, slices, depth_m: float):
        if not slices:
            return None
        centers = []
        for t in slices:
            try:
                d0 = float(t.get("depth_from", 0.0) or 0.0)
                d1 = float(t.get("depth_to", d0) or d0)
            except Exception:
                d0 = 0.0
                d1 = 0.0
            centers.append((d0 + d1) * 0.5)
        depth_arr = np.asarray(centers, dtype=np.float64)
        if depth_arr.size <= 0 or not np.isfinite(depth_arr).any():
            return None
        try:
            target = float(depth_m)
        except Exception:
            return None
        if not np.isfinite(target):
            return None
        return int(np.nanargmin(np.abs(depth_arr - target)))

    def _resolve_timeslice_raster_path(self, ts: dict) -> str:
        if not isinstance(ts, dict):
            return ""
        project_root = self._active_project_root()
        candidates = []
        for key in ("raster_path", "project_path", "path", "file_path", "source_path"):
            raw = str(ts.get(key) or "").strip()
            if raw:
                candidates.append(raw)
        for raw in candidates:
            p = raw
            if not os.path.isabs(p) and project_root:
                p = os.path.normpath(os.path.join(project_root, p))
            p = os.path.normpath(p)
            if os.path.exists(p):
                return p
        if candidates:
            fallback = candidates[0]
            if not os.path.isabs(fallback) and project_root:
                fallback = os.path.normpath(os.path.join(project_root, fallback))
            return os.path.normpath(fallback)
        return ""

    def _load_slice_raster_cached(self, raster_path: str):
        p = str(raster_path or "").strip()
        if not p:
            return None, None
        try:
            mtime = float(os.path.getmtime(p))
        except Exception:
            mtime = None

        cache_hit = (
            self._slice_cache_path == p
            and self._slice_cache_arr is not None
            and self._slice_cache_extent is not None
            and self._slice_cache_mtime == mtime
        )
        if cache_hit:
            return self._slice_cache_arr, self._slice_cache_extent

        from osgeo import gdal

        ds = gdal.Open(p)
        if ds is None:
            return None, None
        band = ds.GetRasterBand(1)
        if band is None:
            ds = None
            return None, None
        arr = band.ReadAsArray()
        if arr is None:
            ds = None
            return None, None
        arr = np.asarray(arr, dtype=np.float32)
        nodata = band.GetNoDataValue()
        if nodata is not None and np.isfinite(float(nodata)):
            arr[arr == float(nodata)] = np.nan
        arr[~np.isfinite(arr)] = np.nan

        gt = ds.GetGeoTransform(can_return_null=True)
        if gt:
            xmin = float(gt[0])
            xmax = float(gt[0] + gt[1] * ds.RasterXSize)
            ymax = float(gt[3])
            ymin = float(gt[3] + gt[5] * ds.RasterYSize)
        else:
            xmin, xmax = 0.0, float(arr.shape[1])
            ymin, ymax = 0.0, float(arr.shape[0])
        ds = None

        extent = (xmin, xmax, ymin, ymax)
        self._slice_cache_path = p
        self._slice_cache_mtime = mtime
        self._slice_cache_arr = arr
        self._slice_cache_extent = extent
        return arr, extent

    def _slice_world_to_pixel(self, east: float, north: float) -> tuple[float, float] | tuple[None, None]:
        if self._slice_current_extent is None:
            return None, None
        xmin, xmax, ymin, ymax = [float(v) for v in self._slice_current_extent]
        if not (np.isfinite(xmin) and np.isfinite(xmax) and np.isfinite(ymin) and np.isfinite(ymax)):
            return None, None
        # Ora la slice e' visualizzata direttamente in coordinate geografiche (extent su imshow),
        # quindi il crosshair puo' usare world-coordinates senza conversione in pixel.
        if abs(float(xmax - xmin)) < 1e-12 or abs(float(ymax - ymin)) < 1e-12:
            return None, None
        ex = float(east)
        ny = float(north)
        if not (np.isfinite(ex) and np.isfinite(ny)):
            return None, None
        return ex, ny

    def _update_slice_crosshair_overlay(self, draw: bool = True) -> bool:
        if self._ax_slice is None or self._slice_im is None:
            return False
        self._update_slice_track_overlay()
        m = self._cursor_metrics()
        if not m:
            return False
        east = m.get("east")
        north = m.get("north")
        try:
            east_f = float(east) if east is not None else np.nan
            north_f = float(north) if north is not None else np.nan
        except Exception:
            return False
        if not (np.isfinite(east_f) and np.isfinite(north_f)):
            return False
        xw, yw = self._slice_world_to_pixel(east_f, north_f)
        if xw is None or yw is None:
            return False
        if self._slice_vline is None:
            self._slice_vline = self._ax_slice.axvline(xw, color="yellow", lw=1.0, alpha=0.85)
        else:
            self._slice_vline.set_xdata([xw, xw])
        if self._slice_hline is None:
            self._slice_hline = self._ax_slice.axhline(yw, color="cyan", lw=1.0, ls="--", alpha=0.8)
        else:
            self._slice_hline.set_ydata([yw, yw])
        if draw and self._canvas_slice is not None:
            self._safe_draw_idle(self._canvas_slice)
        return True

    def _on_slice_clip_changed(self, _value):
        if not hasattr(self, "_spin_slice_vmin_pct") or not hasattr(self, "_spin_slice_vmax_pct"):
            return
        lo = float(self._spin_slice_vmin_pct.value())
        hi = float(self._spin_slice_vmax_pct.value())
        if hi <= lo + 0.1:
            sender = self.sender()
            if sender is self._spin_slice_vmin_pct:
                hi = min(100.0, lo + 0.1)
                self._spin_slice_vmax_pct.blockSignals(True)
                try:
                    self._spin_slice_vmax_pct.setValue(hi)
                finally:
                    self._spin_slice_vmax_pct.blockSignals(False)
            else:
                lo = max(0.0, hi - 0.1)
                self._spin_slice_vmin_pct.blockSignals(True)
                try:
                    self._spin_slice_vmin_pct.setValue(lo)
                finally:
                    self._spin_slice_vmin_pct.blockSignals(False)
        self._redraw_slice_view(force=True)
        self._request_preview_update(immediate=False)

    def _slice_display_style(self) -> dict:
        """Colormap + clip percentili scelti per le timeslice.

        Sono gli stessi valori usati dalla preview; finiscono anche nel .qml
        scritto accanto ai GeoTIFF, cosi' in QGIS le slice hanno lo stesso
        aspetto che vedi nel viewer."""
        style = {}
        cb = getattr(self, "_cb_slice_cmap", None)
        if cb is not None and self._is_qt_alive(cb):
            name = str(cb.currentText() or "").strip()
            if name:
                style["cmap"] = name
        for key, attr in (("clip_lo", "_spin_slice_vmin_pct"), ("clip_hi", "_spin_slice_vmax_pct")):
            sp = getattr(self, attr, None)
            if sp is not None and self._is_qt_alive(sp):
                style[key] = float(sp.value())
        if self._slice_scale_mode() == "fixed":
            # one scale for every slice: also written into the .qml / GeoTIFF
            lo, hi = self._fixed_slice_limits()
            style["vmin"], style["vmax"] = lo, hi
        return style

    # ------------------------------------------------------------------
    # Scala colore: percentili per slice oppure limiti fissi + istogramma
    # ------------------------------------------------------------------
    def _slice_scale_mode(self) -> str:
        cb = getattr(self, "_cb_slice_scale_mode", None)
        if cb is None or not self._is_qt_alive(cb):
            return "percentile"
        return "fixed" if str(cb.currentData() or "") == "fixed" else "percentile"

    def _fixed_slice_limits(self) -> tuple[float, float]:
        lo = float(self._spin_slice_vmin_abs.value())
        hi = float(self._spin_slice_vmax_abs.value())
        if not hi > lo:
            hi = lo + 1e-6
        return lo, hi

    def _sync_slice_scale_widgets(self) -> None:
        """Percentile spins and fixed-value spins are mutually exclusive."""
        fixed = self._slice_scale_mode() == "fixed"
        for attr, on in (("_spin_slice_vmin_pct", not fixed), ("_spin_slice_vmax_pct", not fixed),
                         ("_spin_slice_vmin_abs", fixed), ("_spin_slice_vmax_abs", fixed)):
            w = getattr(self, attr, None)
            if w is not None and self._is_qt_alive(w):
                w.setEnabled(on)

    def _set_slice_fixed_spins(self, lo: float, hi: float) -> None:
        for sp, val in ((self._spin_slice_vmin_abs, lo), (self._spin_slice_vmax_abs, hi)):
            sp.blockSignals(True)
            try:
                sp.setValue(float(val))
            finally:
                sp.blockSignals(False)

    def _set_slice_scale_mode(self, mode: str) -> None:
        cb = self._cb_slice_scale_mode
        idx = cb.findData(mode)
        if idx >= 0 and idx != cb.currentIndex():
            cb.blockSignals(True)
            try:
                cb.setCurrentIndex(idx)
            finally:
                cb.blockSignals(False)
        self._sync_slice_scale_widgets()

    def _on_slice_scale_mode_changed(self, *_):
        self._sync_slice_scale_widgets()
        if self._slice_scale_mode() == "fixed":
            # start from what is on screen, so switching does not change the picture
            im = getattr(self, "_slice_im", None)
            if im is not None:
                try:
                    lo, hi = (float(v) for v in im.get_clim())
                    self._set_slice_fixed_spins(lo, hi)
                except Exception:
                    pass
        self._redraw_slice_view(force=True)

    def _on_slice_fixed_limits_changed(self, *_):
        if self._slice_scale_mode() != "fixed":
            return
        self._apply_slice_clim(*self._fixed_slice_limits())

    def _on_slice_hist_limits(self, vmin: float, vmax: float) -> None:
        """A histogram marker was dragged: from now on the limits are fixed."""
        self._set_slice_scale_mode("fixed")
        self._set_slice_fixed_spins(vmin, vmax)
        self._apply_slice_clim(vmin, vmax)

    def _apply_slice_clim(self, vmin: float, vmax: float) -> None:
        """Change the colour limits of the picture on screen without rebuilding it."""
        im = getattr(self, "_slice_im", None)
        if im is None:
            self._redraw_slice_view(force=True)
            return
        try:
            im.set_clim(float(vmin), float(vmax))
        except Exception:
            self._redraw_slice_view(force=True)
            return
        hist = getattr(self, "_hist_slice", None)
        if hist is not None and self._is_qt_alive(hist):
            hist.set_limits(vmin, vmax)
        self._safe_draw_idle(self._canvas_slice)

    def _slice_color_limits(self, arr) -> tuple[float, float]:
        """``(vmin, vmax)`` of the colour scale for ``arr``: the fixed values, or
        the ``Contrasto`` percentile range of this very slice."""
        if self._slice_scale_mode() == "fixed":
            return self._fixed_slice_limits()
        finite = arr[np.isfinite(arr)]
        p_lo = 2.0
        p_hi = 98.0
        if hasattr(self, "_spin_slice_vmin_pct") and self._spin_slice_vmin_pct is not None:
            try:
                p_lo = float(self._spin_slice_vmin_pct.value())
            except Exception:
                p_lo = 2.0
        if hasattr(self, "_spin_slice_vmax_pct") and self._spin_slice_vmax_pct is not None:
            try:
                p_hi = float(self._spin_slice_vmax_pct.value())
            except Exception:
                p_hi = 98.0
        if p_hi <= p_lo + 0.1:
            p_hi = min(100.0, p_lo + 0.1)
        if finite.size > 0:
            vmin = float(np.nanpercentile(finite, p_lo))
            vmax = float(np.nanpercentile(finite, p_hi))
            if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax <= vmin):
                vmin = float(np.nanmin(finite))
                vmax = float(np.nanmax(finite))
        else:
            vmin, vmax = 0.0, 1.0
        if not np.isfinite(vmin):
            vmin = 0.0
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1e-6
        return vmin, vmax

    def _refresh_slice_histogram(self, arr, vmin: float, vmax: float) -> None:
        """Feed the histogram + statistics with the slice now on screen."""
        hist = getattr(self, "_hist_slice", None)
        if hist is None or not self._is_qt_alive(hist):
            return
        try:
            vals = finite_values(arr)
            hist.set_data(vals, vmin, vmax)
            self._safe_set_text(getattr(self, "_lbl_slice_stats", None), format_stats(value_stats(vals)))
        except Exception:
            pass

    def _refresh_slice_hist_colors(self, *_):
        hist = getattr(self, "_hist_slice", None)
        cb = getattr(self, "_cb_slice_cmap", None)
        if hist is None or cb is None or not (self._is_qt_alive(hist) and self._is_qt_alive(cb)):
            return
        try:
            from .raster_io import colormap_hex_stops

            hist.set_colors(colormap_hex_stops(str(cb.currentText()), 32) or [])
        except Exception:
            hist.set_colors([])

    def _slice_value_at(self, x: float, y: float):
        """Amplitude under the map position ``(x, y)`` in the picture on screen, or None."""
        im = getattr(self, "_slice_im", None)
        if im is None:
            return None
        try:
            arr = im.get_array()
            if arr is None or arr.ndim != 2:
                return None
            x0, x1, y0, y1 = (float(v) for v in im.get_extent())
            if x1 == x0 or y1 == y0:
                return None
            n_r, n_c = arr.shape
            col = int(np.floor((float(x) - x0) / (x1 - x0) * n_c))
            if str(getattr(im, "origin", "upper")) == "lower":
                row = int(np.floor((float(y) - y0) / (y1 - y0) * n_r))
            else:
                row = int(np.floor((y1 - float(y)) / (y1 - y0) * n_r))
            if not (0 <= row < n_r and 0 <= col < n_c):
                return None
            v = float(np.ma.filled(arr[row, col], np.nan))
            return v if np.isfinite(v) else None
        except Exception:
            return None

    def _sync_thickness_from_panel(self, *_):
        """Lo spessore vive nel pannello parametri: il vecchio spinbox nascosto
        (usato da navigator e finestra di preview) lo segue, cosi' preview e
        "Salva gruppo" usano sempre lo stesso spessore."""
        panel = getattr(self, "_slice_spec_panel", None)
        spin = getattr(self, "_spin_slice_thickness", None)
        if panel is None or spin is None:
            return
        if not (self._is_qt_alive(panel) and self._is_qt_alive(spin)):
            return
        try:
            value = float(panel.sp_thickness.value())
        except Exception:
            return
        if abs(float(spin.value()) - value) < 1e-9:
            return
        spin.blockSignals(True)
        try:
            spin.setValue(value)
        finally:
            spin.blockSignals(False)

    def _preview_resolution_m(self) -> float:
        quality = "medium"
        try:
            quality = str(self._cb_preview_quality.currentData() or "medium").strip().lower()
        except Exception:
            quality = "medium"
        mapping = {
            "low": 0.20,
            "medium": 0.10,
            "high": 0.05,
        }
        return float(mapping.get(quality, 0.10))

    def _preview_depth_window(self) -> tuple[float, float]:
        try:
            step = float(self._spin_slice_thickness.value())
        except Exception:
            step = 0.10
        if (not np.isfinite(step)) or step <= 0.0:
            step = 0.10
        z0 = float(self._preview_depth_steps()[2])
        if self._cursor_z is not None:
            try:
                z0 = float(self._cursor_z)
            except Exception:
                z0 = 0.0
        if not np.isfinite(z0):
            z0 = 0.0
        z0 = float(max(0.0, z0))
        z1 = float(z0 + step)
        return z0, z1

    def _preview_depth_steps(self) -> tuple[float, int, float]:
        """(spessore finestra, numero di posizioni, quota iniziale) per lo slider
        in preview: copre l'intervallo profondita' min-max del pannello slice."""
        try:
            step = float(self._spin_slice_thickness.value())
        except Exception:
            step = 0.10
        if (not np.isfinite(step)) or step <= 0.0:
            step = 0.10
        z_lo, z_hi = 0.0, 0.0
        panel = getattr(self, "_slice_spec_panel", None)
        if panel is not None and self._is_qt_alive(panel):
            try:
                sp = panel.get_spec().resolve(self._profiles)
                z_lo = float(sp.z_min_m or 0.0)
                z_hi = float(sp.z_max_m or 0.0)
            except Exception:
                z_lo, z_hi = 0.0, 0.0
        if not (np.isfinite(z_hi) and z_hi > z_lo):
            z_hi = 0.0
            for p in (self._profiles or []):
                try:
                    z_hi = max(z_hi, float(p.depth_max_m))
                except Exception:
                    pass
            if not np.isfinite(z_hi) or z_hi <= z_lo:
                z_hi = z_lo + 5.0
        return step, max(1, int(np.ceil((z_hi - z_lo) / step))), max(0.0, z_lo)

    def _load_variant_profiles(self, npz_paths) -> list:
        """UNICO punto in cui i profili di una variante vengono letti dai .npz.

        I dati radar vengono dal disco, ma la GEOMETRIA (easting/northing/quota/
        distanze) e' sempre quella corrente dei profili in memoria, cioe' quella
        gestita dalla pagina di posizionamento: le coordinate scritte nel .npz
        sono solo una fotografia del momento del salvataggio e non fanno fede."""
        profs = [load_processed_profile_npz(f) for f in npz_paths]

        def _key(path):
            return os.path.normcase(os.path.normpath(str(path or "")))

        by_path = {_key(getattr(p, "path", "")): p for p in (self._profiles or [])}
        by_stem = {os.path.splitext(os.path.basename(str(getattr(p, "path", "") or "")))[0]: p
                   for p in (self._profiles or [])}
        for prof in profs:
            src = by_path.get(_key(getattr(prof, "path", "")))
            if src is None:
                stem = os.path.splitext(os.path.basename(str(getattr(prof, "path", "") or "")))[0]
                src = by_stem.get(stem)
            if src is None:
                _dbg.log("VARIANT-GEOM", f"{os.path.basename(str(getattr(prof, 'path', '?')))}: "
                         "nessun profilo corrispondente in memoria, uso la geometria del .npz")
                continue
            try:
                apply_geometry_from_profile(prof, src)
            except Exception as exc:
                _dbg.log("VARIANT-GEOM", f"riallineamento geometria fallito: {exc}")
        return profs

    def _preview_source_profiles(self):
        """Profili per la preview: la variante sorgente del pannello (come la
        generazione finale), con cache; altrimenti i profili grezzi correnti.

        Ritorna ``(profili, nome_variante, loader, chiave_cache)``. Se la variante
        non e' ancora in cache ``profili`` e' ``None`` e ``loader`` e' una
        funzione che la legge da disco: va eseguita nel thread di lavoro (sono
        ~1 GB di .npz: sul thread GUI la finestra resterebbe bloccata)."""
        panel = getattr(self, "_slice_spec_panel", None)
        name = ""
        if panel is not None and self._is_qt_alive(panel):
            try:
                name = str(panel.get_spec().source_variant or "").strip()
            except Exception:
                name = ""
        d = self._variant_dir_for_name(name)
        if not d:
            return list(self._profiles or []), "", None, None
        npzs = sorted(glob.glob(os.path.join(d, "*.npz")))
        try:
            key = (d, len(npzs), max(os.path.getmtime(f) for f in npzs),
                   int(getattr(self, "_geometry_rev", 0)))
        except Exception:
            return list(self._profiles or []), "", None, None
        cache = getattr(self, "_preview_variant_cache", None)
        if cache is not None and cache[0] == key:
            return cache[1], name, None, key
        return None, name, (lambda: self._load_variant_profiles(npzs)), key

    def _on_toggle_preview_mode(self, checked: bool):
        self._preview_mode_active = bool(checked)
        self._update_slice_navigator()
        if not self._preview_mode_active:
            try:
                self._preview_timer.stop()
            except Exception:
                pass
            self._preview_request_id += 1  # scarta un eventuale risultato ancora in volo
            self._set_preview_busy(False)
            self._preview_slice_grid = None
            self._preview_slice_extent = None
            self._preview_slice_z_from = None
            self._preview_slice_z_to = None
            self._preview_signature = None
            self._redraw_slice_view(force=True)
            return
        if not self._profiles:
            if self._is_qt_alive(getattr(self, "_btn_preview_slice", None)):
                self._btn_preview_slice.blockSignals(True)
                try:
                    self._btn_preview_slice.setChecked(False)
                finally:
                    self._btn_preview_slice.blockSignals(False)
            self._preview_mode_active = False
            QMessageBox.information(self, "Preview", "Importa almeno un profilo .ogpr.")
            return
        self._request_preview_update(immediate=True)

    def _request_preview_update(self, immediate: bool = False):
        if not bool(getattr(self, "_preview_mode_active", False)):
            return
        if not self._profiles:
            return
        if not self._is_qt_alive(getattr(self, "_preview_timer", None)):
            return
        try:
            self._preview_timer.start(1 if immediate else 300)
        except Exception:
            pass

    def _stop_preview_worker_blocking(self, timeout_ms: int = 3000):
        """Attende la fine di un eventuale thread di preview in corso.

        Chiamato solo alla chiusura reale della finestra, per evitare di
        distruggere il QThread mentre e' ancora attivo.
        """
        self._preview_request_id += 1  # scarta comunque il risultato
        thread = getattr(self, "_preview_thread", None)
        if self._is_qt_alive(thread) and thread.isRunning():
            try:
                thread.quit()
                thread.wait(int(timeout_ms))
            except Exception:
                pass

    def _set_preview_busy(self, busy: bool, message: str = ""):
        """Mostra/nasconde l'indicatore di avanzamento del preview.

        Non esiste una % reale (compute_timeslice_preview non riporta stati
        intermedi), quindi la QProgressBar resta in modalita' indeterminata
        (range 0,0): comunica "sta lavorando" senza promettere una % finta.
        """
        self._preview_busy = bool(busy)
        if self._is_qt_alive(getattr(self, "_progress_preview", None)):
            self._progress_preview.setVisible(bool(busy))
        if self._is_qt_alive(getattr(self, "_lbl_preview_status", None)):
            if busy:
                self._lbl_preview_status.setText(message or "Preview: elaborazione in corso...")
                self._lbl_preview_status.setVisible(True)
            elif message:
                self._lbl_preview_status.setText(message)
                self._lbl_preview_status.setVisible(True)
            else:
                self._lbl_preview_status.setVisible(False)

    @staticmethod
    def _stable_preview_signature(value):
        """Converte ricorsivamente un valore (dict/list/float inclusi) in una
        tupla hashabile e stabile, usata per capire se i parametri di preview
        sono davvero cambiati rispetto all'ultimo calcolo."""
        if isinstance(value, float):
            return round(value, 6)
        if isinstance(value, dict):
            return tuple(sorted(
                (k, TimesliceViewMixin._stable_preview_signature(v)) for k, v in value.items()
            ))
        if isinstance(value, (list, tuple)):
            return tuple(TimesliceViewMixin._stable_preview_signature(v) for v in value)
        return value

    def _compute_and_show_preview(self):
        if not bool(getattr(self, "_preview_mode_active", False)):
            return
        if not self._profiles:
            return
        try:
            from .gpr_ogpr_slicer import compute_timeslice_preview  # noqa: F401 (verifica disponibilita')
        except Exception as exc:
            self._set_preview_busy(False, f"Preview non disponibile: {exc}")
            return

        z_from, z_to = self._preview_depth_window()
        res = self._preview_resolution_m()
        prev_profiles, prev_variant, prev_loader, prev_key = self._preview_source_profiles()
        if prev_profiles is None and prev_loader is not None:
            # Variante non ancora in memoria: la legge il thread di lavoro. Una
            # sola lettura alla volta (sono ~1 GB): se e' gia' in corso, si
            # riparte appena finisce invece di lanciarne un'altra.
            if getattr(self, "_preview_loading_key", None) == prev_key:
                self._preview_reload_pending = True
                return
            resolve_profiles = list(self._profiles or [])  # stessa geometria della variante
        else:
            resolve_profiles = prev_profiles
        if not resolve_profiles:
            return
        slice_params = dict(self.get_slice_params() or {})
        chain_order = list(self._current_chain_order() or [])

        # Fase D: spec-native preview when the SliceSpec panel is active.
        preview_spec = None
        _panel = getattr(self, "_slice_spec_panel", None)
        if _panel is not None and self._is_qt_alive(_panel):
            try:
                preview_spec = _panel.get_spec().resolve(resolve_profiles)
            except Exception as exc:
                self._set_preview_busy(False, f"Parametri slice non validi: {exc}")
                return
            # Never render the preview finer than the final output; the
            # "Qualita' preview" combo only makes it coarser.
            res = max(float(res), float(preview_spec.resolution_m))

        kwargs = {
            "channel": int(self._ch_idx),
            "combine_method": "mean",
            "pipeline_params": dict(slice_params.get("pipeline_params") or {}),
            "normalize_channels": bool(slice_params.get("normalize_channels", False)),
            "extraction_mode": str(slice_params.get("extraction_mode", "las_like") or "las_like"),
            "use_hilbert": bool(slice_params.get("use_hilbert", True)),
            "use_processing": bool(slice_params.get("use_processing", False)),
            "amplitude_sigma": slice_params.get("amplitude_sigma"),
            "use_anisotropic_idw": bool(slice_params.get("use_anisotropic_idw", False)),
            "auto_radius": bool(slice_params.get("auto_radius", True)),
            "idw_mode": str(slice_params.get("idw_mode", "quality") or "quality"),
            "radius": slice_params.get("search_radius"),
            "idw_power": float(slice_params.get("idw_power", 2) or 2),
            "min_points": int(slice_params.get("min_points", 1) or 1),
            "fill_nodata": bool(slice_params.get("fill_nodata", True)),
            "fill_nodata_max_distance": 0.0,
            "blanking_distance": float(slice_params.get("blanking_distance", 0.0) or 0.0),
            "smooth_sigma": float(slice_params.get("smooth_sigma", 0.0) or 0.0),
            "depth_radius_factor": float(slice_params.get("depth_radius_factor", 0.6) or 0.0),
            "balance_profiles": bool(slice_params.get("balance_profiles", True)),
            "pre_slice_bg_removal": bool(slice_params.get("pre_slice_bg_removal", False)),
            "pre_slice_bg_mode": str(slice_params.get("pre_slice_bg_mode", "line_by_line") or "line_by_line"),
            "pre_slice_bg_window": int(slice_params.get("pre_slice_bg_window", 0) or 0),
            "pre_slice_bg_sample_start": int(slice_params.get("pre_slice_bg_sample_start", 0) or 0),
            "pre_slice_bg_sample_end": int(slice_params.get("pre_slice_bg_sample_end", 0) or 0),
            "stack_n": int(slice_params.get("stack_n", 1) or 1),
            "stack_kernel": str(slice_params.get("stack_kernel", "boxcar") or "boxcar"),
            "flip_traces_mode": str(slice_params.get("flip_traces_mode", "none") or "none"),
            "topographic_correction": bool(slice_params.get("topographic_correction", False)),
            "topo_reference_mode": str(slice_params.get("topo_reference_mode", "median") or "median"),
            "topo_reference_elevation": slice_params.get("topo_reference_elevation"),
            "parallel_profiles": bool(slice_params.get("parallel_profiles", False)),
            "profile_workers": int(slice_params.get("profile_workers", 0) or 0),
        }

        if prev_variant:
            # dati gia' filtrati su disco: niente ri-processing della pipeline
            kwargs["use_processing"] = False
            kwargs["pipeline_params"] = {}
            kwargs["pre_slice_bg_removal"] = False

        # La firma di cache e' derivata direttamente da kwargs (invece di una
        # lista di campi mantenuta a mano): cosi' un parametro come Blanking
        # o Search radius che cambia non puo' piu' passare inosservato e
        # lasciare il preview "congelato" sul risultato precedente.
        sig = (
            round(float(z_from), 4),
            round(float(z_to), 4),
            round(float(res), 4),
            str(prev_variant),
            tuple(chain_order),
            (self._stable_preview_signature(preview_spec.to_dict())
             if preview_spec is not None
             else self._stable_preview_signature(kwargs)),
        )
        if sig == self._preview_signature:
            return

        # Calcolo su thread separato: il thread GUI resta reattivo e Windows
        # non mostra piu' il cursore "in caricamento" per l'intera durata.
        self._preview_request_id += 1
        request_id = self._preview_request_id
        self._preview_signature = sig
        self._preview_pending_zwindow = (float(z_from), float(z_to))
        if prev_profiles is None:
            self._preview_loading_key = prev_key
            self._preview_reload_pending = False
            self._set_preview_busy(
                True, f"Carico la variante '{prev_variant}' (solo la prima volta)..."
            )
        else:
            self._set_preview_busy(True)

        old_thread = getattr(self, "_preview_thread", None)
        if self._is_qt_alive(old_thread):
            try:
                old_thread.finished.disconnect()
            except Exception:
                pass

        thread = QThread(self)
        worker = _PreviewSliceWorker(
            request_id, prev_profiles, float(z_from), float(z_to), float(res), chain_order, kwargs,
            spec=preview_spec, loader=prev_loader,
            points_cache=(self._preview_points_cache_for(prev_variant, prev_key)
                          if preview_spec is not None else None),
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_preview_worker_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._preview_thread = thread
        self._preview_worker = worker
        thread.start()

    def _preview_points_cache_for(self, prev_variant, prev_key) -> dict:
        """Cache of the extracted amplitudes for the live preview.

        Moving the amplitude tools / smoothing / colour limits at a fixed depth
        must not repeat the expensive extraction (a Hilbert transform per channel
        for the envelope modes): the engine keeps it here, keyed by window and
        extraction settings. It is dropped as soon as the data behind it can have
        changed (another variant / file mtime, other profiles, new geometry)."""
        rev = int(getattr(self, "_geometry_rev", 0))
        if prev_variant:
            sig, pin = ("variant", str(prev_variant), prev_key, rev), None
        else:
            pin = list(self._profiles or [])      # keeps the ids below from being reused
            sig = ("raw", tuple(id(p) for p in pin), rev)
        holder = getattr(self, "_preview_points_holder", None)
        if holder is None or holder["sig"] != sig:
            holder = {"sig": sig, "pin": pin, "cache": {}}
            self._preview_points_holder = holder
        return holder["cache"]

    def _on_preview_worker_finished(self, request_id: int, grid, extent, error: str,
                                    load_result=None):
        if load_result is not None:
            # Il thread ha letto la variante da disco: va in cache anche se la
            # richiesta e' stata superata, altrimenti la prossima la rileggerebbe.
            status, profs = load_result
            key = getattr(self, "_preview_loading_key", None)
            self._preview_loading_key = None
            if status == "ok" and profs is not None and key is not None:
                self._preview_variant_cache = (key, profs)
            if getattr(self, "_preview_reload_pending", False):
                self._preview_reload_pending = False
                self._request_preview_update(immediate=True)
        if int(request_id) != int(self._preview_request_id):
            return  # superato da una richiesta di preview piu' recente
        self._set_preview_busy(False)
        if error:
            self._safe_set_text(self._lbl_status, f"Preview errore: {error}")
            return
        if grid is None or extent is None:
            self._safe_set_text(self._lbl_status, "Preview: nessun dato nella finestra selezionata.")
            return

        z_from, z_to = getattr(self, "_preview_pending_zwindow", (0.0, 0.0))
        self._preview_slice_grid = np.asarray(grid, dtype=np.float32)
        self._preview_slice_extent = tuple(float(v) for v in extent)
        self._preview_slice_z_from = float(z_from)
        self._preview_slice_z_to = float(z_to)
        self._redraw_slice_view(force=True)

    def _redraw_slice_view(self, force: bool = False):
        if not HAS_MPL or self._ax_slice is None or self._canvas_slice is None:
            return

        if bool(getattr(self, "_preview_mode_active", False)):
            arr = np.asarray(getattr(self, "_preview_slice_grid", None)) if getattr(self, "_preview_slice_grid", None) is not None else None
            extent = getattr(self, "_preview_slice_extent", None)
            if arr is None or extent is None or arr.ndim != 2 or arr.size <= 0:
                self._ax_slice.clear()
                self._slice_im = None
                self._slice_vline = None
                self._slice_hline = None
                self._ax_slice.text(
                    0.5,
                    0.5,
                    "Preview in attesa...\nMuovi il cursore nel radargramma.",
                    transform=self._ax_slice.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="#666666",
                )
                self._ax_slice.set_title("Timeslice preview")
                self._ax_slice.set_xticks([])
                self._ax_slice.set_yticks([])
                self._safe_draw_idle(self._canvas_slice)
                return

            vmin, vmax = self._slice_color_limits(arr)
            self._refresh_slice_histogram(arr, vmin, vmax)

            cmap = (
                self._cb_slice_cmap.currentText()
                if hasattr(self, "_cb_slice_cmap") and self._cb_slice_cmap is not None
                else DEFAULT_CMAP
            )
            arr_masked = np.ma.masked_invalid(arr)
            slice_cmap = cmap
            try:
                from matplotlib import colormaps

                cm_obj = colormaps.get_cmap(str(cmap)).copy()
                cm_obj.set_bad(color="#e6e6e6", alpha=1.0)
                slice_cmap = cm_obj
            except Exception:
                slice_cmap = cmap
            self._ax_slice.clear()
            self._slice_im = self._ax_slice.imshow(
                arr_masked,
                cmap=slice_cmap,
                vmin=vmin,
                vmax=vmax,
                extent=[float(extent[0]), float(extent[1]), float(extent[2]), float(extent[3])],
                # la griglia della preview e' ordinata dal sud al nord (riga 0 =
                # y minima, convenzione interna dello slicer): "lower", altrimenti
                # l'immagine risulta ribaltata in verticale rispetto ai GeoTIFF
                # (che vengono scritti nord-in-alto, quindi vanno con "upper").
                origin="lower",
                aspect="equal",
                interpolation="bilinear",
            )
            xmin, xmax, ymin, ymax = [float(v) for v in extent]
            x_full = (min(xmin, xmax), max(xmin, xmax))
            y_full = (min(ymin, ymax), max(ymin, ymax))
            if self._slice_view_xlim is None:
                self._slice_view_xlim = x_full
            if self._slice_view_ylim is None:
                self._slice_view_ylim = y_full
            self._slice_view_xlim = self._clamp_axis_limits(
                self._slice_view_xlim[0], self._slice_view_xlim[1], x_full[0], x_full[1]
            )
            self._slice_view_ylim = self._clamp_axis_limits(
                self._slice_view_ylim[0], self._slice_view_ylim[1], y_full[0], y_full[1]
            )
            self._ax_slice.set_xlim(*self._slice_view_xlim)
            self._ax_slice.set_ylim(*self._slice_view_ylim)
            self._ax_slice.set_aspect("equal", adjustable="datalim")
            self._ax_slice.grid(False)
            self._ax_slice.set_xticks([])
            self._ax_slice.set_yticks([])
            self._ax_slice.set_xlabel("")
            self._ax_slice.set_ylabel("")
            self._ax_slice.set_facecolor("#e6e6e6")
            for spine in self._ax_slice.spines.values():
                spine.set_visible(False)
            z0 = float(getattr(self, "_preview_slice_z_from", 0.0) or 0.0)
            z1 = float(getattr(self, "_preview_slice_z_to", z0) or z0)
            self._ax_slice.set_title(f"Timeslice preview {z0:.2f}-{z1:.2f} m")
            self._slice_vline = None
            self._slice_hline = None
            self._update_slice_crosshair_overlay(draw=False)
            self._safe_draw_idle(self._canvas_slice)
            return

        local_catalog = list(getattr(self, "_slice_catalog", []) or [])
        using_local_catalog = len(local_catalog) > 0
        slices = local_catalog if using_local_catalog else []

        if not slices:
            self._ax_slice.clear()
            self._slice_im = None
            self._slice_vline = None
            self._slice_hline = None
            self._slice_current_idx = None
            self._slice_current_path = None
            self._slice_current_extent = None
            self._slice_current_shape = None
            self._slice_current_cmap = ""
            self._ax_slice.text(
                0.5,
                0.5,
                "Nessuna timeslice calcolata.\nUsa 'Crea Timeslice...' nel pannello Timeslice.",
                transform=self._ax_slice.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color="#666666",
            )
            self._ax_slice.set_title("Timeslice locale")
            self._ax_slice.set_xticks([])
            self._ax_slice.set_yticks([])
            self._safe_draw_idle(self._canvas_slice)
            return

        idx = 0 if self._slice_current_idx is None else int(self._slice_current_idx)
        idx = int(np.clip(idx, 0, len(slices) - 1))
        ts = dict(slices[idx] or {})
        raster_path = str(ts.get("path") or "").strip()
        cmap = (
            self._cb_slice_cmap.currentText()
            if hasattr(self, "_cb_slice_cmap") and self._cb_slice_cmap is not None
            else (self._cb_cmap.currentText() if hasattr(self, "_cb_cmap") else DEFAULT_CMAP)
        )

        if not raster_path:
            self._ax_slice.clear()
            self._slice_im = None
            self._ax_slice.set_title(f"Slice {idx} - percorso raster non disponibile")
            self._ax_slice.set_xticks([])
            self._ax_slice.set_yticks([])
            self._safe_draw_idle(self._canvas_slice)
            return
        if not os.path.exists(raster_path):
            self._ax_slice.clear()
            self._slice_im = None
            self._ax_slice.set_title(f"Slice {idx} - file non trovato")
            self._ax_slice.set_xticks([])
            self._ax_slice.set_yticks([])
            self._safe_draw_idle(self._canvas_slice)
            return

        needs_full_redraw = bool(
            force
            or self._slice_im is None
            or self._slice_current_idx != idx
            or str(self._slice_current_path or "") != str(raster_path)
        )

        if needs_full_redraw:
            try:
                arr, extent = self._load_slice_raster_cached(raster_path)
            except Exception as exc:
                self._ax_slice.clear()
                self._slice_im = None
                self._ax_slice.set_title(f"Errore lettura raster: {exc}")
                self._ax_slice.set_xticks([])
                self._ax_slice.set_yticks([])
                self._safe_draw_idle(self._canvas_slice)
                return
            if arr is None or extent is None:
                self._ax_slice.clear()
                self._slice_im = None
                self._ax_slice.set_title("Errore lettura raster")
                self._ax_slice.set_xticks([])
                self._ax_slice.set_yticks([])
                self._safe_draw_idle(self._canvas_slice)
                return

            vmin, vmax = self._slice_color_limits(arr)
            self._refresh_slice_histogram(arr, vmin, vmax)

            self._ax_slice.clear()
            self._slice_vline = None
            self._slice_hline = None
            arr_masked = np.ma.masked_invalid(arr)
            slice_cmap = cmap
            try:
                from matplotlib import colormaps

                cm_obj = colormaps.get_cmap(str(cmap)).copy()
                cm_obj.set_bad(color="#e6e6e6", alpha=1.0)
                slice_cmap = cm_obj
            except Exception:
                slice_cmap = cmap
            self._slice_im = self._ax_slice.imshow(
                arr_masked,
                cmap=slice_cmap,
                vmin=vmin,
                vmax=vmax,
                extent=[float(extent[0]), float(extent[1]), float(extent[2]), float(extent[3])],
                origin="upper",
                aspect="equal",
                interpolation="bilinear",
            )

            n_rows, n_cols = int(arr.shape[0]), int(arr.shape[1])
            xmin, xmax, ymin, ymax = [float(v) for v in extent]
            x_full = (min(xmin, xmax), max(xmin, xmax))
            y_full = (min(ymin, ymax), max(ymin, ymax))
            if self._slice_view_xlim is None:
                self._slice_view_xlim = x_full
            if self._slice_view_ylim is None:
                self._slice_view_ylim = y_full
            self._slice_view_xlim = self._clamp_axis_limits(
                self._slice_view_xlim[0], self._slice_view_xlim[1], x_full[0], x_full[1]
            )
            self._slice_view_ylim = self._clamp_axis_limits(
                self._slice_view_ylim[0], self._slice_view_ylim[1], y_full[0], y_full[1]
            )
            self._ax_slice.set_xlim(*self._slice_view_xlim)
            self._ax_slice.set_ylim(*self._slice_view_ylim)
            self._ax_slice.set_aspect("equal", adjustable="datalim")

            self._ax_slice.grid(False)
            self._ax_slice.set_xticks([])
            self._ax_slice.set_yticks([])
            self._ax_slice.set_xlabel("")
            self._ax_slice.set_ylabel("")
            self._ax_slice.set_facecolor("#e6e6e6")
            for spine in self._ax_slice.spines.values():
                spine.set_visible(False)

            try:
                d0 = float(ts.get("z_top", 0.0) or 0.0)
                d1 = float(ts.get("z_bot", d0) or d0)
            except Exception:
                d0 = 0.0
                d1 = 0.0
            source_name = os.path.basename(str(self._slice_catalog_source_dir or "")) or "local"
            title_left = f"{source_name} | "
            base = os.path.basename(raster_path)
            self._ax_slice.set_title(f"{title_left}Timeslice {d0:.2f}-{d1:.2f} m  |  {base}")

            self._slice_current_idx = idx
            self._slice_current_path = raster_path
            self._slice_current_extent = tuple(float(v) for v in extent)
            self._slice_current_shape = (n_rows, n_cols)
            self._slice_current_cmap = str(cmap)
        else:
            if self._slice_im is not None and str(self._slice_current_cmap) != str(cmap):
                cm_to_set = cmap
                try:
                    from matplotlib import colormaps

                    cm_obj = colormaps.get_cmap(str(cmap)).copy()
                    cm_obj.set_bad(color="#e6e6e6", alpha=1.0)
                    cm_to_set = cm_obj
                except Exception:
                    cm_to_set = cmap
                self._slice_im.set_cmap(cm_to_set)
                self._slice_current_cmap = str(cmap)

        changed = self._update_slice_crosshair_overlay(draw=False)
        if changed or needs_full_redraw:
            self._safe_draw_idle(self._canvas_slice)

    def _on_slice_press(self, event):
        if event is None or event.inaxes != self._ax_slice:
            return
        if int(getattr(event, "button", 0) or 0) != 1:
            return
        if self._slice_im is None or event.xdata is None or event.ydata is None:
            return
        try:
            xlim0 = tuple(self._ax_slice.get_xlim())
            ylim0 = tuple(self._ax_slice.get_ylim())
            self._slice_pan_anchor = (
                float(event.xdata),
                float(event.ydata),
                (float(xlim0[0]), float(xlim0[1])),
                (float(ylim0[0]), float(ylim0[1])),
            )
        except Exception:
            self._slice_pan_anchor = None
        self._slice_press_px = (float(event.x), float(event.y))

    def _on_slice_release(self, event):
        self._slice_pan_anchor = None
        press, self._slice_press_px = self._slice_press_px, None
        if (
            press is None or event is None or event.inaxes != self._ax_slice
            or int(getattr(event, "button", 0) or 0) != 1
            or event.xdata is None or event.ydata is None
        ):
            return
        # Clic senza trascinare (il trascinamento e' il pan): puntatore inverso.
        if abs(float(event.x) - press[0]) <= 4.0 and abs(float(event.y) - press[1]) <= 4.0:
            self._on_slice_click(float(event.xdata), float(event.ydata))

    def _on_slice_mouse_move(self, event):
        if event is None or event.inaxes != self._ax_slice:
            return
        if event.xdata is None or event.ydata is None:
            return
        if self._slice_pan_anchor is not None and self._slice_im is not None:
            try:
                px, py, xlim0, ylim0 = self._slice_pan_anchor
                dx = float(px) - float(event.xdata)
                dy = float(py) - float(event.ydata)
                nx0 = float(xlim0[0]) + dx
                nx1 = float(xlim0[1]) + dx
                ny0 = float(ylim0[0]) + dy
                ny1 = float(ylim0[1]) + dy
                ex0, ex1, ey0, ey1 = self._slice_im.get_extent()
                x_min, x_max = float(min(ex0, ex1)), float(max(ex0, ex1))
                y_min, y_max = float(min(ey0, ey1)), float(max(ey0, ey1))
                self._slice_view_xlim = self._clamp_axis_limits(nx0, nx1, x_min, x_max)
                self._slice_view_ylim = self._clamp_axis_limits(ny0, ny1, y_min, y_max)
                self._ax_slice.set_xlim(*self._slice_view_xlim)
                self._ax_slice.set_ylim(*self._slice_view_ylim)
                self._safe_draw_idle(self._canvas_slice)
            except Exception:
                pass
            return
        try:
            txt = f"Slice cursor: x {float(event.xdata):.1f}  y {float(event.ydata):.1f}"
            val = self._slice_value_at(float(event.xdata), float(event.ydata))
            if val is not None:
                txt += f"  |  valore {val:.4g}"
            self._safe_set_text(self._lbl_status, txt)
        except Exception:
            pass

    def _on_slice_scroll_zoom(self, event):
        if event is None or event.inaxes != self._ax_slice or self._slice_im is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        step = float(getattr(event, "step", 0.0) or 0.0)
        btn = str(getattr(event, "button", "") or "").lower()
        if step > 0 or btn == "up":
            scale = 1.0 / 1.2
        elif step < 0 or btn == "down":
            scale = 1.2
        else:
            return

        x0, x1 = self._ax_slice.get_xlim()
        y0, y1 = self._ax_slice.get_ylim()
        x = float(event.xdata)
        y = float(event.ydata)

        nx0 = x - (x - x0) * scale
        nx1 = x + (x1 - x) * scale
        ny0 = y - (y - y0) * scale
        ny1 = y + (y1 - y) * scale

        ex0, ex1, ey0, ey1 = self._slice_im.get_extent()
        x_min, x_max = float(min(ex0, ex1)), float(max(ex0, ex1))
        y_min, y_max = float(min(ey0, ey1)), float(max(ey0, ey1))

        self._slice_view_xlim = self._clamp_axis_limits(nx0, nx1, x_min, x_max)
        self._slice_view_ylim = self._clamp_axis_limits(ny0, ny1, y_min, y_max)
        self._ax_slice.set_xlim(*self._slice_view_xlim)
        self._ax_slice.set_ylim(*self._slice_view_ylim)
        self._safe_draw_idle(self._canvas_slice)

    def _update_dial(self):
        if self._cursor_z is None or self.plugin is None:
            return
        settings = getattr(self.plugin, "settings", None)
        settings_key = getattr(self.plugin, "settings_key_active_project", None)
        if settings is None or not settings_key:
            return
        dlg = getattr(self.plugin, "dlg", None)
        if dlg is None:
            return
        dial = getattr(dlg, "Dial", None) or getattr(dlg, "dial", None)
        if dial is None:
            return
        try:
            slices = list(getattr(self, "_slice_catalog", []) or [])
            if not slices:
                return
            centers = []
            for i, s in enumerate(slices):
                try:
                    zc = s.get("z_center", None)
                    if zc is None:
                        z0 = float(s.get("z_top", 0.0) or 0.0)
                        z1 = float(s.get("z_bot", z0) or z0)
                        zc = 0.5 * (z0 + z1)
                    centers.append((i, float(zc)))
                except Exception:
                    continue
            if not centers:
                idx = int(self._slice_current_idx if self._slice_current_idx is not None else 0)
                idx = int(np.clip(idx, 0, len(slices) - 1))
            else:
                target = float(self._cursor_z)
                idx = min(centers, key=lambda p: abs(p[1] - target))[0]
            dial_val = int(np.clip(idx, 0, int(dial.maximum())))
            if dial.value() != dial_val:
                dial.setValue(dial_val)
        except Exception:
            pass

    def _imported_slice_qgis_group(self):
        """Sottogruppo di "GeoSurvey Studio" creato dall'ultimo Import to Canvas."""
        name = str(getattr(self, "_imported_slice_group_name", "") or "").strip()
        if not name:
            return None
        root = QgsProject.instance().layerTreeRoot()
        for top in root.children():
            if hasattr(top, "name") and top.name() == "GeoSurvey Studio":
                for g in top.children():
                    if hasattr(g, "findGroup") and hasattr(g, "name") and g.name() == name:
                        return g
        return None

    def _update_timeslice_visibility_in_canvas(self):
        if not bool(getattr(self, "_chk_timeslice_sync", None) and self._chk_timeslice_sync.isChecked()):
            return
        if self._cursor_z is None or self.plugin is None:
            return

        slices = list(getattr(self, "_slice_catalog", []) or [])
        if not slices:
            return

        centers = []
        for i, s in enumerate(slices):
            try:
                zc = s.get("z_center", None)
                if zc is None:
                    z0 = float(s.get("z_top", 0.0) or 0.0)
                    z1 = float(s.get("z_bot", z0) or z0)
                    zc = 0.5 * (z0 + z1)
                centers.append((i, float(zc)))
            except Exception:
                continue
        if not centers:
            return

        target = float(self._cursor_z)
        idx = min(centers, key=lambda p: abs(p[1] - target))[0]

        try:
            qgis_group = self._imported_slice_qgis_group()
            if qgis_group is None:
                return

            raster_nodes = []
            for child in qgis_group.children():
                if not hasattr(child, "setItemVisibilityChecked"):
                    continue
                layer = child.layer() if hasattr(child, "layer") else None
                if layer is None:
                    continue
                if not hasattr(layer, "bandCount"):
                    continue
                raster_nodes.append(child)
            if not raster_nodes:
                return
            index = int(np.clip(idx, 0, len(raster_nodes) - 1))
            for i, node in enumerate(raster_nodes):
                node.setItemVisibilityChecked(i == index)
        except Exception:
            pass

    def _update_timeslice_crosshair(self):
        if not bool(getattr(self, "_chk_timeslice_sync", None) and self._chk_timeslice_sync.isChecked()):
            self._clear_timeslice_crosshair()
            return
        if self.iface is None or not self._profiles:
            self._clear_timeslice_crosshair()
            return
        m = self._cursor_metrics()
        if not m:
            self._clear_timeslice_crosshair()
            return
        east = m.get("east")
        north = m.get("north")
        if east is None or north is None:
            self._clear_timeslice_crosshair()
            return
        try:
            east_f = float(east)
            north_f = float(north)
        except Exception:
            self._clear_timeslice_crosshair()
            return
        if not (np.isfinite(east_f) and np.isfinite(north_f)):
            self._clear_timeslice_crosshair()
            return
        prof = self._profiles[self._prof_idx]
        canvas = self.iface.mapCanvas()
        if canvas is None:
            self._clear_timeslice_crosshair()
            return
        try:
            size = float(getattr(self, "_spin_crosshair_size").value())
        except Exception:
            size = float(getattr(self, "_crosshair_size_m", 5.0) or 5.0)
        size = float(np.clip(size, 0.5, 1000.0))

        if self._rb_crosshair_h is None:
            self._rb_crosshair_h = QgsRubberBand(canvas, QgsWkbTypes.LineGeometry)
            self._rb_crosshair_h.setColor(QColor(255, 50, 50))
            self._rb_crosshair_h.setWidth(2)
        if self._rb_crosshair_v is None:
            self._rb_crosshair_v = QgsRubberBand(canvas, QgsWkbTypes.LineGeometry)
            self._rb_crosshair_v.setColor(QColor(255, 50, 50))
            self._rb_crosshair_v.setWidth(2)

        try:
            self._rb_crosshair_h.reset(QgsWkbTypes.LineGeometry)
            p_left = self._to_canvas_point(prof, east_f - size, north_f)
            p_right = self._to_canvas_point(prof, east_f + size, north_f)
            self._rb_crosshair_h.addPoint(p_left, False)
            self._rb_crosshair_h.addPoint(p_right, True)
            self._rb_crosshair_h.setVisible(True)
        except Exception:
            pass
        try:
            self._rb_crosshair_v.reset(QgsWkbTypes.LineGeometry)
            p_bottom = self._to_canvas_point(prof, east_f, north_f - size)
            p_top = self._to_canvas_point(prof, east_f, north_f + size)
            self._rb_crosshair_v.addPoint(p_bottom, False)
            self._rb_crosshair_v.addPoint(p_top, True)
            self._rb_crosshair_v.setVisible(True)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Timeslice
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_slice_depth_label(label: str) -> Optional[float]:
        tok = str(label or "").strip().lower()
        if not tok:
            return None
        sign = -1.0 if tok.startswith("m") else 1.0
        if tok.startswith("m"):
            tok = tok[1:]
        if "_" in tok:
            tok = tok.replace("_", ".", 1)
        try:
            return sign * float(tok)
        except Exception:
            return None

    def _scan_slice_catalog(self, outdir: str, dz: float) -> list[dict]:
        base = str(outdir or "").strip()
        if not base or not os.path.isdir(base):
            return []

        # Manifesto autorevole: se c'e' un sliceset.json valido leggi da li'
        # profondita'/ordine/nomi invece di dedurli dai nomi file.
        try:
            from . import slice_manifest as _sm
            if _sm.has_sliceset(base):
                rows = _sm.slice_catalog_from_sliceset(base)
                if rows:
                    rows.sort(key=lambda it: (float(it.get("z_center", 0.0)),
                                              int(it.get("index", 0))))
                    self._slice_catalog_from_manifest = True
                    return rows
        except Exception:
            pass
        self._slice_catalog_from_manifest = False

        paths = sorted(glob.glob(os.path.join(base, "slice_*.tif")))
        if not paths:
            paths = sorted(glob.glob(os.path.join(base, "slice_*.tiff")))
        if not paths:
            return []

        catalog = []
        centers = []
        for fallback_idx, path in enumerate(paths):
            name = os.path.basename(path)
            idx = fallback_idx
            z_center = None
            m_win = re.search(r"^slice_(\d+)_(n?\d+\.\d+)-(n?\d+\.\d+)m\.tiff?$", name, flags=re.IGNORECASE)
            m = re.search(r"^slice_(\d+)_z([A-Za-z0-9_]+)\.tiff?$", name, flags=re.IGNORECASE)
            if m_win:
                try:
                    idx = int(m_win.group(1))
                    lo, hi = (float(g.replace("n", "-")) for g in m_win.group(2, 3))
                    z_center = 0.5 * (lo + hi)
                except Exception:
                    idx = fallback_idx
            elif m:
                try:
                    idx = int(m.group(1))
                except Exception:
                    idx = fallback_idx
                z_center = self._parse_slice_depth_label(m.group(2))
            if z_center is not None and np.isfinite(z_center):
                centers.append(float(z_center))
            catalog.append(
                {
                    "path": os.path.normpath(path),
                    "index": int(idx),
                    "z_center": float(z_center) if z_center is not None else None,
                    "name": os.path.splitext(name)[0],
                }
            )

        catalog.sort(key=lambda item: (int(item.get("index", 0)), str(item.get("path", ""))))

        dz_use = float(dz) if np.isfinite(dz) and float(dz) > 0.0 else 0.0
        if dz_use <= 0.0:
            z_vals = [
                float(item["z_center"])
                for item in catalog
                if item.get("z_center") is not None and np.isfinite(float(item.get("z_center")))
            ]
            if len(z_vals) >= 2:
                z_vals = sorted(z_vals)
                diffs = np.diff(np.asarray(z_vals, dtype=np.float64))
                diffs = diffs[np.isfinite(diffs) & (diffs > 1e-9)]
                if diffs.size > 0:
                    dz_use = float(np.nanmedian(diffs))
        if dz_use <= 0.0:
            dz_use = 0.10

        for i, item in enumerate(catalog):
            zc = item.get("z_center")
            if zc is None or not np.isfinite(float(zc)):
                z_top = float(i) * dz_use
                z_center = z_top + 0.5 * dz_use
            else:
                z_center = float(zc)
                z_top = z_center - 0.5 * dz_use
            item["z_top"] = float(z_top)
            item["z_bot"] = float(z_top + dz_use)
            item["z_center"] = float(z_center)
        return catalog

    def _refresh_slice_catalog(
        self,
        outdir: str,
        dz: float,
        reset_index: bool = False,
        redraw: bool = True,
    ):
        base = os.path.normpath(str(outdir or "").strip()) if str(outdir or "").strip() else ""
        catalog = self._scan_slice_catalog(base, dz)
        self._slice_catalog = list(catalog or [])
        self._slice_catalog_source_dir = base
        try:
            self._slice_catalog_dz = float(dz) if np.isfinite(float(dz)) else 0.0
        except Exception:
            self._slice_catalog_dz = 0.0
        if not self._slice_catalog:
            self._slice_current_idx = None
        else:
            if reset_index or self._slice_current_idx is None:
                self._slice_current_idx = 0
            else:
                self._slice_current_idx = int(
                    np.clip(int(self._slice_current_idx), 0, len(self._slice_catalog) - 1)
                )
        self._update_slice_navigator()
        if redraw:
            self._slice_view_xlim = None
            self._slice_view_ylim = None
            self._redraw_slice_view(force=True)

    def _on_slice_depth_slider(self, value: int):
        if self._updating_slice_nav:
            return
        if bool(getattr(self, "_preview_mode_active", False)) and self._profiles:
            # In preview lo slider sposta sempre la finestra di profondita'
            # (anche se esistono slice generate: quelle non sono mostrate).
            step, n, z_lo = self._preview_depth_steps()
            self._cursor_z = z_lo + float(np.clip(int(value), 0, n - 1)) * step
            self._update_cursor_lines()
            self._update_status()
            self._canvas_timer.start()
            self._request_preview_update(immediate=False)
            return
        if not self._slice_catalog:
            return
        idx = int(np.clip(int(value), 0, len(self._slice_catalog) - 1))
        if self._slice_current_idx == idx:
            return
        self._slice_current_idx = idx
        try:
            center = float(self._slice_catalog[idx].get("z_center"))
            if np.isfinite(center):
                self._cursor_z = center
        except Exception:
            pass
        self._update_slice_navigator()
        self._redraw_slice_view(force=True)
        self._update_timeslice_visibility_in_canvas()

    def _step_slice(self, delta: int):
        if bool(getattr(self, "_preview_mode_active", False)) and self._profiles:
            if self._slider_slice_depth is not None:
                self._slider_slice_depth.setValue(
                    int(self._slider_slice_depth.value()) + int(delta)
                )
            return
        if not self._slice_catalog:
            return
        curr = int(self._slice_current_idx if self._slice_current_idx is not None else 0)
        nxt = int(np.clip(curr + int(delta), 0, len(self._slice_catalog) - 1))
        if self._slider_slice_depth is not None:
            self._slider_slice_depth.setValue(nxt)
        else:
            self._on_slice_depth_slider(nxt)

    def _update_slice_navigator(self):
        has_slider = hasattr(self, "_slider_slice_depth") and self._slider_slice_depth is not None
        if (not has_slider) or (not hasattr(self, "_lbl_slice_nav")):
            return
        preview_on = bool(getattr(self, "_preview_mode_active", False)) and bool(self._profiles)
        if preview_on or not self._slice_catalog:
            self._updating_slice_nav = True
            try:
                if preview_on:
                    step, n, z_lo = self._preview_depth_steps()
                    z = float(self._cursor_z) if self._cursor_z is not None else z_lo
                    self._slider_slice_depth.setRange(0, n - 1)
                    self._slider_slice_depth.setValue(int(np.clip(round((z - z_lo) / step), 0, n - 1)))
                    self._slider_slice_depth.setEnabled(True)
                else:
                    self._slider_slice_depth.setRange(0, 0)
                    self._slider_slice_depth.setValue(0)
                    self._slider_slice_depth.setEnabled(False)
            finally:
                self._updating_slice_nav = False
            if preview_on:
                for _b in ("_btn_slice_up", "_btn_slice_down"):
                    if self._is_qt_alive(getattr(self, _b, None)):
                        getattr(self, _b).setEnabled(True)
                _st, _n, _zl = self._preview_depth_steps()
                _z = float(self._cursor_z) if self._cursor_z is not None else _zl
                self._safe_set_text(
                    self._lbl_slice_nav,
                    f"Preview  |  {_z:.2f}-{_z + _st:.2f} m",
                )
                return
            if self._is_qt_alive(getattr(self, "_btn_slice_up", None)):
                self._btn_slice_up.setEnabled(False)
            if self._is_qt_alive(getattr(self, "_btn_slice_down", None)):
                self._btn_slice_down.setEnabled(False)
            self._safe_set_text(self._lbl_slice_nav, "\u2014 nessuna slice \u2014")
            if hasattr(self, "_lbl_slice_depth_top"):
                self._safe_set_text(self._lbl_slice_depth_top, "0.00 m")
            if hasattr(self, "_lbl_slice_depth_bottom"):
                self._safe_set_text(self._lbl_slice_depth_bottom, "\u2014")
            return

        idx = int(self._slice_current_idx if self._slice_current_idx is not None else 0)
        idx = int(np.clip(idx, 0, len(self._slice_catalog) - 1))
        self._slice_current_idx = idx
        n = int(len(self._slice_catalog))
        entry = self._slice_catalog[idx]
        z_top = float(entry.get("z_top", 0.0) or 0.0)
        z_bot = float(entry.get("z_bot", z_top) or z_top)

        self._updating_slice_nav = True
        try:
            self._slider_slice_depth.setRange(0, n - 1)
            self._slider_slice_depth.setValue(idx)
            self._slider_slice_depth.setEnabled(True)
        finally:
            self._updating_slice_nav = False
        if self._is_qt_alive(getattr(self, "_btn_slice_up", None)):
            self._btn_slice_up.setEnabled(idx > 0)
        if self._is_qt_alive(getattr(self, "_btn_slice_down", None)):
            self._btn_slice_down.setEnabled(idx < (n - 1))

        self._safe_set_text(self._lbl_slice_nav, f"Slice {idx + 1}/{n}  |  {z_top:.2f}-{z_bot:.2f} m")
        if hasattr(self, "_lbl_slice_depth_top"):
            self._safe_set_text(self._lbl_slice_depth_top, "0.00 m")
        if hasattr(self, "_lbl_slice_depth_bottom"):
            z_max = float(self._slice_catalog[-1].get("z_bot", z_bot) or z_bot)
            self._safe_set_text(self._lbl_slice_depth_bottom, f"{z_max:.2f} m")

    def _on_timeslice_build_finished(self, ok: bool, payload: dict | None = None):
        if not bool(ok):
            return
        data = dict(payload or {})
        outdir = str(data.get("output_dir") or "").strip()
        if not outdir:
            outdir = str(getattr(self, "_le_slice_outdir", None).text() if hasattr(self, "_le_slice_outdir") else "").strip()
        if outdir and hasattr(self, "_le_slice_outdir") and self._le_slice_outdir is not None:
            try:
                self._le_slice_outdir.setText(os.path.normpath(outdir))
            except Exception:
                pass
        dz = data.get("z_step")
        if dz is None:
            try:
                dz = float(self._spin_slice_thickness.value())
            except Exception:
                dz = 0.10
        try:
            dz_f = float(dz)
        except Exception:
            dz_f = 0.10
        if hasattr(self, "_chk_slice_thickness_locked") and self._chk_slice_thickness_locked is not None:
            self._chk_slice_thickness_locked.setChecked(True)
        self._refresh_slice_catalog(outdir, dz=dz_f, reset_index=True, redraw=True)
        self._save_gpr_processing_project_meta(
            last_timeslice=self._path_to_gpr_processing_store(outdir),
        )

    def _default_slice_output_dir(self) -> str:
        paths = self._ensure_gpr_processing_layout()
        if paths:
            return os.path.normpath(paths.get("timeslices", ""))
        project_root = self._active_project_root()
        if project_root and os.path.isdir(project_root):
            return os.path.normpath(os.path.join(project_root, "timeslices_2d"))
        return ""

    def _choose_slice_outdir(self):
        start = str(getattr(self, "_le_slice_outdir", None).text() if hasattr(self, "_le_slice_outdir") else "").strip()
        if not start:
            start = self._default_slice_output_dir() or os.path.expanduser("~")
        folder = QFileDialog.getExistingDirectory(self, "Output folder Timeslice", start)
        if folder:
            try:
                norm = os.path.normpath(folder)
                self._le_slice_outdir.setText(norm)
                try:
                    dz = float(self._spin_slice_thickness.value())
                except Exception:
                    dz = 0.10
                self._refresh_slice_catalog(norm, dz=dz, reset_index=True, redraw=True)
                self._save_gpr_processing_project_meta(
                    last_timeslice=self._path_to_gpr_processing_store(norm),
                )
            except Exception:
                pass

    def _update_slice_profile_info(self):
        if not hasattr(self, "_lbl_slice_profiles") or self._lbl_slice_profiles is None:
            return
        n_prof = int(len(self._profiles or []))
        if n_prof <= 0:
            self._safe_set_text(self._lbl_slice_profiles, "Nessun profilo caricato")
            return
        total_traces = 0
        for prof in self._profiles:
            try:
                if int(getattr(prof, "n_channels", 0) or 0) <= 0:
                    continue
                total_traces += int(prof.channel(0).data.shape[1])
            except Exception:
                continue
        self._safe_set_text(self._lbl_slice_profiles, f"{n_prof} profili  |  ~{total_traces} tracce")

    def _restamp_slice_styles(self, tifs) -> None:
        """Riscrive il .qml accanto a ogni GeoTIFF con colormap e percentili
        attuali del viewer, cosi' in QGIS le slice hanno lo stesso aspetto
        della preview anche se il set e' stato generato con un altro stile."""
        style = self._slice_display_style()
        if not style.get("cmap"):
            return
        try:
            from .raster_io import read_tif_singleband, write_qml_singleband
        except Exception:
            return
        for tif in tifs:
            try:
                grid = read_tif_singleband(tif)[0]
                write_qml_singleband(tif, grid, style)
            except Exception:
                continue

    def _import_slices_to_canvas(self):
        outdir = str(getattr(self, "_le_slice_outdir", None).text() if hasattr(self, "_le_slice_outdir") else "").strip()
        if not outdir:
            QMessageBox.warning(self, "Import Timeslice", "Seleziona prima la cartella output.")
            return
        if not os.path.isdir(outdir):
            QMessageBox.warning(self, "Import Timeslice", "Cartella output non valida.")
            return

        tifs = sorted(glob.glob(os.path.join(outdir, "*.tif"))) + sorted(glob.glob(os.path.join(outdir, "*.tiff")))
        if not tifs:
            QMessageBox.information(self, "Import Timeslice", "Nessun GeoTIFF trovato nella cartella output.")
            return

        plugin = self.plugin
        if plugin is None or not hasattr(plugin, "import_slice_files_as_group"):
            QMessageBox.warning(self, "Import Timeslice", "Plugin GeoSurvey Studio non disponibile.")
            return
        project_root = str(plugin._active_project_root() or "").strip()
        if not project_root:
            ans = QMessageBox.question(
                self, "Import Timeslice",
                "Nessun progetto GeoSurvey Studio collegato.\n"
                "Aprire il Project Manager per aprire o creare un progetto?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
            )
            if ans == QMessageBox.Yes:
                self._open_project_manager()
            return

        default_name = os.path.basename(os.path.normpath(outdir)) or "gpr_slices"
        zone = self._current_zone_name()
        if zone and zone != project_zones.DEFAULT_ZONE:
            # Stesso nome di variante in aree diverse (es. "Lavoro") non deve
            # finire nello stesso gruppo del catalogo.
            default_name = f"{zone} - {default_name}"
        group_name, ok = QInputDialog.getText(
            self, "Import Timeslice",
            "Nome del sottogruppo da creare in GeoSurvey Studio:",
            text=default_name,
        )
        group_name = str(group_name or "").strip()
        if not ok or not group_name:
            return

        self._restamp_slice_styles(tifs)

        def _done(result):
            added = int(result.get("imported", 0))
            if added > 0:
                self._imported_slice_group_name = group_name
            self._safe_set_text(
                self._lbl_status,
                f"Import completato: {added} layer nel sottogruppo '{group_name}' di GeoSurvey Studio.",
            )
            if added <= 0:
                if not result.get("cancelled"):
                    QMessageBox.warning(self, "Import Timeslice", "Nessun layer importato.")
                return
            try:
                dz = float(self._spin_slice_thickness.value())
            except Exception:
                dz = 0.10
            self._refresh_slice_catalog(outdir, dz=dz, reset_index=False, redraw=True)
            self._save_gpr_processing_project_meta(
                last_timeslice=self._path_to_gpr_processing_store(outdir),
            )

        plugin.import_slice_files_as_group(project_root, group_name, tifs, on_done=_done)

    @staticmethod
    def _set_combo_to_data(combo: QComboBox, value) -> None:
        if combo is None:
            return
        idx = combo.findData(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _filter_timeslices_grid(self):
        """Applica un filtro nel dominio griglia a tutte le GeoTIFF di un set di
        time-slice, scrivendo un nuovo set filtrato accanto."""
        start = str(self._le_slice_outdir.text() or "").strip() if self._is_qt_alive(
            getattr(self, "_le_slice_outdir", None)) else ""
        if not start or not os.path.isdir(start):
            paths = self._gpr_processing_paths()
            start = (paths.get("timeslices", "") if paths else "") or os.path.expanduser("~")
        folder = QFileDialog.getExistingDirectory(
            self, "Cartella time-slice (GeoTIFF) da filtrare", start
        )
        if not folder:
            return
        tifs = sorted(
            glob.glob(os.path.join(folder, "*.tif"))
            + glob.glob(os.path.join(folder, "*.tiff"))
        )
        if not tifs:
            QMessageBox.warning(self, "Filtra time-slice",
                                "Nessuna GeoTIFF nella cartella selezionata.")
            return
        try:
            from .raster_io import (
                read_tif_singleband, write_tif_singleband, write_qml_singleband,
            )
            from . import grid_filters as gf
        except Exception as exc:
            QMessageBox.critical(self, "Filtra time-slice",
                                 f"Modulo non disponibile:\n{exc}")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Filtra time-slice (dominio griglia)")
        fl = QFormLayout(dlg)
        cb = QComboBox()
        cb.addItem("Passa-basso 3x3", ("lowpass", 3))
        cb.addItem("Passa-basso 5x5", ("lowpass", 5))
        cb.addItem("Passa-basso 7x7", ("lowpass", 7))
        cb.addItem("Passa-alto 3x3", ("hipass", 3))
        cb.addItem("Laplaciano (enfasi bordi)", ("laplacian", 0))
        cb.addItem("Background di linea — X (righe)", ("bg_x", 0))
        cb.addItem("Background di linea — Y (colonne)", ("bg_y", 0))
        cb.addItem("Notch FFT direzionale", ("fftnotch", 0))
        fl.addRow("Filtro:", cb)
        sp_len = QSpinBox(); sp_len.setRange(0, 99999); sp_len.setValue(0)
        sp_len.setToolTip("Lunghezza del filtro di linea in celle (0 = intera linea).")
        sp_clip = QDoubleSpinBox(); sp_clip.setRange(50.0, 100.0); sp_clip.setValue(100.0)
        sp_clip.setToolTip("Background di linea: esclude dal calcolo i valori sopra\n"
                           "questo percentile della linea (protegge i target forti).")
        sp_ang = QDoubleSpinBox(); sp_ang.setRange(0.0, 180.0); sp_ang.setValue(0.0)
        sp_ang.setSuffix(" °")
        sp_hw = QDoubleSpinBox(); sp_hw.setRange(1.0, 45.0); sp_hw.setValue(10.0)
        sp_hw.setSuffix(" °")
        le_suffix = QLineEdit("filt")
        fl.addRow("Lunghezza filtro linea:", sp_len)
        fl.addRow("Clip percentile (bg linea):", sp_clip)
        fl.addRow("Angolo (notch FFT):", sp_ang)
        fl.addRow("Semi-larghezza (notch FFT):", sp_hw)
        fl.addRow("Suffisso cartella output:", le_suffix)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        fl.addRow(bb)
        if dlg.exec_() != QDialog.Accepted:
            return

        kind, param = cb.currentData()
        suffix = "".join(c if c.isalnum() or c in "_-" else "_"
                         for c in str(le_suffix.text()).strip()).strip("_") or "filt"
        out_dir = os.path.normpath(folder).rstrip("\\/") + "_" + suffix
        os.makedirs(out_dir, exist_ok=True)

        from PyQt5.QtWidgets import QProgressDialog
        from PyQt5.QtCore import Qt as _Qt
        prog = QProgressDialog("Filtraggio time-slice...", "Annulla", 0, len(tifs), self)
        prog.setWindowModality(_Qt.WindowModal)
        prog.setMinimumDuration(0)

        n_ok, errors = 0, []
        for i, tif in enumerate(tifs):
            if prog.wasCanceled():
                break
            try:
                grid, x_min, y_min, y_max, res, epsg = read_tif_singleband(tif)
                if kind == "lowpass":
                    out = gf.lowpass(grid, int(param))
                elif kind == "hipass":
                    out = gf.hipass(grid, int(param))
                elif kind == "laplacian":
                    out = gf.laplacian(grid)
                elif kind == "bg_x":
                    out = gf.line_background(grid, "x", int(sp_len.value()),
                                            0.0, float(sp_clip.value()))
                elif kind == "bg_y":
                    out = gf.line_background(grid, "y", int(sp_len.value()),
                                            0.0, float(sp_clip.value()))
                elif kind == "fftnotch":
                    out = gf.fft2d_directional_notch(
                        grid, float(sp_ang.value()), float(sp_hw.value()), "reject"
                    )
                else:
                    out = grid
                out_path = os.path.join(out_dir, os.path.basename(tif))
                write_tif_singleband(out, out_path, x_min, y_min, y_max, res, epsg or 0)
                try:
                    write_qml_singleband(out_path, out, self._slice_display_style())
                except Exception:
                    pass
                n_ok += 1
            except Exception as e:
                errors.append(f"{os.path.basename(tif)}: {e}")
            prog.setValue(i + 1)
        prog.setValue(len(tifs))

        # porta con se' il manifesto (i nomi file combaciano)
        src_manifest = os.path.join(folder, "sliceset.json")
        if os.path.isfile(src_manifest):
            try:
                shutil.copy2(src_manifest, os.path.join(out_dir, "sliceset.json"))
            except Exception:
                pass

        if n_ok:
            msg = f"{n_ok}/{len(tifs)} time-slice filtrate in\n{out_dir}"
            if errors:
                msg += "\n\nErrori:\n" + "\n".join(errors[:5])
            QMessageBox.information(self, "Filtra time-slice", msg)
            if self._is_qt_alive(getattr(self, "_le_slice_outdir", None)):
                self._le_slice_outdir.setText(out_dir)
                try:
                    dz0 = float(self._spin_slice_thickness.value())
                except Exception:
                    dz0 = 0.10
                try:
                    self._refresh_slice_catalog(out_dir, dz=dz0, reset_index=True, redraw=True)
                except Exception:
                    pass
        else:
            QMessageBox.warning(self, "Filtra time-slice",
                                "Nessuna slice filtrata.\n" + "\n".join(errors[:5]))

    # ------------------------------------------------------------------
    # grid_geometry in UI — unione / rototraslazione di set di time-slice
    # ------------------------------------------------------------------
    @staticmethod
    def _gg_list_tifs(folder):
        return sorted(
            glob.glob(os.path.join(folder, "*.tif"))
            + glob.glob(os.path.join(folder, "*.tiff"))
        )

    @staticmethod
    def _gg_read_manifest(folder):
        """Raw sliceset.json (dict) o None; niente rehydrate della spec."""
        path = os.path.join(folder, "sliceset.json")
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def _gg_slice_start_dir(self):
        start = str(self._le_slice_outdir.text() or "").strip() if self._is_qt_alive(
            getattr(self, "_le_slice_outdir", None)) else ""
        if not start or not os.path.isdir(start):
            paths = self._gpr_processing_paths()
            start = (paths.get("timeslices", "") if paths else "") or os.path.expanduser("~")
        return start

    def _gg_repoint_slice_view(self, out_dir):
        if not self._is_qt_alive(getattr(self, "_le_slice_outdir", None)):
            return
        self._le_slice_outdir.setText(out_dir)
        try:
            dz0 = float(self._spin_slice_thickness.value())
        except Exception:
            dz0 = 0.10
        try:
            self._refresh_slice_catalog(out_dir, dz=dz0, reset_index=True, redraw=True)
        except Exception:
            pass

    def _merge_timeslice_sets(self):
        """Unisce 2+ cartelle di GeoTIFF (stessa cella, stessi z) in un set unico
        via ``grid_geometry.append_grids`` (edge-match opzionale sulla giunzione)."""
        try:
            from .raster_io import read_tif_singleband, write_tif_singleband, write_qml_singleband
            from . import grid_geometry as gg
            from . import slice_manifest as sm
        except Exception as exc:
            QMessageBox.critical(self, "Unisci set di time-slice",
                                 f"Modulo non disponibile:\n{exc}")
            return

        start = self._gg_slice_start_dir()
        folders = []
        while True:
            title = ("Cartella time-slice #%d (Annulla per terminare)"
                     % (len(folders) + 1)) if folders else "Prima cartella time-slice"
            d = QFileDialog.getExistingDirectory(self, title, start)
            if not d:
                break
            d = os.path.normpath(d)
            if d in folders:
                continue
            if not self._gg_list_tifs(d):
                QMessageBox.warning(self, "Unisci set di time-slice",
                                    "Nessuna GeoTIFF in:\n" + d)
                continue
            folders.append(d)
            start = os.path.dirname(d)
            if len(folders) >= 2:
                if QMessageBox.question(
                    self, "Unisci set di time-slice",
                    "Aggiungere un'altra cartella?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No
                ) != QMessageBox.Yes:
                    break
        if len(folders) < 2:
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Unisci set di time-slice")
        fl = QFormLayout(dlg)
        chk_em = QCheckBox("Edge-match sulla giunzione (pareggia la media)")
        chk_em.setChecked(True)
        cb_axis = QComboBox()
        cb_axis.addItem("Affiancate lungo X (bordo verticale)", "x")
        cb_axis.addItem("Impilate lungo Y (bordo orizzontale)", "y")
        sp_ov = QSpinBox(); sp_ov.setRange(1, 999); sp_ov.setValue(3)
        cb_overlap = QComboBox()
        for lbl, val in (("Media", "mean"), ("Prima griglia", "first"), ("Ultima griglia", "last")):
            cb_overlap.addItem(lbl, val)
        le_suffix = QLineEdit("merged")
        fl.addRow(chk_em)
        fl.addRow("Asse giunzione (edge-match):", cb_axis)
        fl.addRow("Celle di sovrapposizione:", sp_ov)
        fl.addRow("Celle coperte da piu' griglie:", cb_overlap)
        fl.addRow("Suffisso cartella output:", le_suffix)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        fl.addRow(bb)
        if dlg.exec_() != QDialog.Accepted:
            return

        do_em = bool(chk_em.isChecked())
        em_axis = cb_axis.currentData()
        em_ov = int(sp_ov.value())
        overlap = cb_overlap.currentData()
        suffix = "".join(c if c.isalnum() or c in "_-" else "_"
                         for c in str(le_suffix.text()).strip()).strip("_") or "merged"
        out_dir = os.path.normpath(folders[0]).rstrip("\\/") + "_" + suffix
        os.makedirs(out_dir, exist_ok=True)

        # indicizza ogni cartella per chiave z (manifesto se c'e', altrimenti
        # posizione ordinata dei file)
        manifests = [self._gg_read_manifest(f) for f in folders]
        per_folder = []           # list[dict[zkey] -> (tif_path, z_from, z_to, z_center)]
        for f, man in zip(folders, manifests):
            tifs = self._gg_list_tifs(f)
            entries = {}
            if man and isinstance(man.get("slices"), list) and len(man["slices"]) == len(tifs):
                for s in man["slices"]:
                    gp = os.path.join(f, str(s.get("grid", "")))
                    if not os.path.isfile(gp):
                        entries = {}
                        break
                    zc = round(float(s.get("z_center", 0.0)), 4)
                    entries[zc] = (gp, float(s.get("z_from", zc)),
                                   float(s.get("z_to", zc)), float(s.get("z_center", zc)))
            if not entries:
                for i, gp in enumerate(tifs):
                    entries[float(i)] = (gp, float(i), float(i), float(i))
            per_folder.append(entries)

        all_keys = sorted(set().union(*[set(d.keys()) for d in per_folder]))
        if not all_keys:
            QMessageBox.warning(self, "Unisci set di time-slice", "Nessuna slice da unire.")
            return

        from PyQt5.QtWidgets import QProgressDialog
        from PyQt5.QtCore import Qt as _Qt
        prog = QProgressDialog("Unione time-slice...", "Annulla", 0, len(all_keys), self)
        prog.setWindowModality(_Qt.WindowModal)
        prog.setMinimumDuration(0)

        epsg_out = 0
        for man in manifests:
            if man and man.get("crs_epsg"):
                epsg_out = int(man["crs_epsg"]); break

        out_slices = []
        n_ok, errors = 0, []
        ext_xmin = ext_ymin = ext_xmax = ext_ymax = None
        for i, zk in enumerate(all_keys):
            if prog.wasCanceled():
                break
            try:
                grids, meta = [], None
                for entries in per_folder:
                    if zk not in entries:
                        continue
                    gp, zf, zt, zc = entries[zk]
                    arr, x_min, y_min, y_max, res, epsg = read_tif_singleband(gp)
                    grids.append({"array": arr, "x0": x_min, "y0": y_min, "cell": res})
                    if meta is None:
                        meta = (zf, zt, zc)
                    if not epsg_out and epsg:
                        epsg_out = int(epsg)
                if not grids:
                    continue
                if do_em and len(grids) >= 2:
                    base = grids[0]
                    for k in range(1, len(grids)):
                        grids[k] = gg.edge_match(base, grids[k], axis=em_axis,
                                                 overlap_cells=em_ov)
                merged = grids[0] if len(grids) == 1 else gg.append_grids(grids, overlap=overlap)
                arr = merged["array"]
                x0, y0, cell = float(merged["x0"]), float(merged["y0"]), float(merged["cell"])
                n_y, n_x = arr.shape
                y_top = y0 + (n_y - 1) * cell
                x_right = x0 + (n_x - 1) * cell
                zf, zt, zc = meta if meta else (float(i), float(i), float(i))
                name = sm.slice_file_name(i, zf, zt)
                out_path = os.path.join(out_dir, name)
                write_tif_singleband(arr, out_path, x0, y0, y_top, cell, epsg_out or 0)
                try:
                    write_qml_singleband(out_path, arr, self._slice_display_style())
                except Exception:
                    pass
                out_slices.append({
                    "index": i, "z_from": zf, "z_to": zt, "z_center": zc,
                    "grid": name, "stats": sm.compute_slice_stats(arr),
                })
                hw = cell / 2.0
                ext_xmin = (x0 - hw) if ext_xmin is None else min(ext_xmin, x0 - hw)
                ext_ymin = (y0 - hw) if ext_ymin is None else min(ext_ymin, y0 - hw)
                ext_xmax = (x_right + hw) if ext_xmax is None else max(ext_xmax, x_right + hw)
                ext_ymax = (y_top + hw) if ext_ymax is None else max(ext_ymax, y_top + hw)
                n_ok += 1
            except Exception as e:
                errors.append("z=%s: %s" % (zk, e))
            prog.setValue(i + 1)
        prog.setValue(len(all_keys))

        # manifesto unito (se almeno una sorgente ne aveva uno)
        base_man = next((m for m in manifests if m), None)
        if base_man is not None and out_slices:
            merged_man = dict(base_man)
            merged_man["slices"] = out_slices
            merged_man["extent"] = [ext_xmin, ext_ymin, ext_xmax, ext_ymax]
            if epsg_out:
                merged_man["crs_epsg"] = int(epsg_out)
            src = dict(merged_man.get("source") or {})
            src["merged_from"] = [os.path.basename(f) for f in folders]
            merged_man["source"] = src
            try:
                with open(os.path.join(out_dir, "sliceset.json"), "w", encoding="utf-8") as f:
                    json.dump(merged_man, f, indent=2, ensure_ascii=True, sort_keys=True)
            except Exception:
                pass

        if n_ok:
            msg = "%d/%d slice unite in\n%s" % (n_ok, len(all_keys), out_dir)
            if errors:
                msg += "\n\nErrori:\n" + "\n".join(errors[:5])
            QMessageBox.information(self, "Unisci set di time-slice", msg)
            self._gg_repoint_slice_view(out_dir)
        else:
            QMessageBox.warning(self, "Unisci set di time-slice",
                                "Nessuna slice unita.\n" + "\n".join(errors[:5]))

    def _rotate_translate_timeslice_set(self):
        """Ruota/trasla ogni GeoTIFF di un set via
        ``grid_geometry.rotate_translate`` e scrive un nuovo set ``_rt``."""
        try:
            from .raster_io import read_tif_singleband, write_tif_singleband, write_qml_singleband
            from . import grid_geometry as gg
            from . import slice_manifest as sm
        except Exception as exc:
            QMessageBox.critical(self, "Ruota/trasla set", f"Modulo non disponibile:\n{exc}")
            return

        folder = QFileDialog.getExistingDirectory(
            self, "Cartella time-slice (GeoTIFF) da ruotare/traslare",
            self._gg_slice_start_dir()
        )
        if not folder:
            return
        tifs = self._gg_list_tifs(folder)
        if not tifs:
            QMessageBox.warning(self, "Ruota/trasla set",
                                "Nessuna GeoTIFF nella cartella selezionata.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Ruota/trasla set di time-slice")
        fl = QFormLayout(dlg)
        sp_ang = QDoubleSpinBox(); sp_ang.setRange(-360.0, 360.0); sp_ang.setDecimals(4)
        sp_ang.setValue(0.0); sp_ang.setSuffix(" ° CCW")
        sp_e = QDoubleSpinBox(); sp_e.setRange(-1e9, 1e9); sp_e.setDecimals(3); sp_e.setValue(0.0)
        sp_n = QDoubleSpinBox(); sp_n.setRange(-1e9, 1e9); sp_n.setDecimals(3); sp_n.setValue(0.0)
        cb_mode = QComboBox()
        cb_mode.addItem("Angolo SW sorgente -> origine (fix_origin)", "fix_origin")
        cb_mode.addItem("Angolo SW inviluppo ruotato -> origine (translate_origin)",
                        "translate_origin")
        le_suffix = QLineEdit("rt")
        fl.addRow("Angolo:", sp_ang)
        fl.addRow("Origine E:", sp_e)
        fl.addRow("Origine N:", sp_n)
        fl.addRow("Modo:", cb_mode)
        fl.addRow("Suffisso cartella output:", le_suffix)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        fl.addRow(bb)
        if dlg.exec_() != QDialog.Accepted:
            return

        angle = float(sp_ang.value())
        origin = (float(sp_e.value()), float(sp_n.value()))
        mode = cb_mode.currentData()
        suffix = "".join(c if c.isalnum() or c in "_-" else "_"
                         for c in str(le_suffix.text()).strip()).strip("_") or "rt"
        out_dir = os.path.normpath(folder).rstrip("\\/") + "_" + suffix
        os.makedirs(out_dir, exist_ok=True)

        man = self._gg_read_manifest(folder)
        stats_by_name = {}

        from PyQt5.QtWidgets import QProgressDialog
        from PyQt5.QtCore import Qt as _Qt
        prog = QProgressDialog("Rototraslazione time-slice...", "Annulla", 0, len(tifs), self)
        prog.setWindowModality(_Qt.WindowModal)
        prog.setMinimumDuration(0)

        n_ok, errors = 0, []
        ext_xmin = ext_ymin = ext_xmax = ext_ymax = None
        for i, tif in enumerate(tifs):
            if prog.wasCanceled():
                break
            try:
                arr, x_min, y_min, y_max, res, epsg = read_tif_singleband(tif)
                res_g = gg.rotate_translate(
                    {"array": arr, "x0": x_min, "y0": y_min, "cell": res},
                    angle, origin, mode=mode,
                )
                oarr = res_g["array"]
                x0, y0, cell = float(res_g["x0"]), float(res_g["y0"]), float(res_g["cell"])
                n_y, n_x = oarr.shape
                y_top = y0 + (n_y - 1) * cell
                x_right = x0 + (n_x - 1) * cell
                out_path = os.path.join(out_dir, os.path.basename(tif))
                write_tif_singleband(oarr, out_path, x0, y0, y_top, cell, epsg or 0)
                try:
                    write_qml_singleband(out_path, oarr, self._slice_display_style())
                except Exception:
                    pass
                stats_by_name[os.path.basename(tif)] = sm.compute_slice_stats(oarr)
                hw = cell / 2.0
                ext_xmin = (x0 - hw) if ext_xmin is None else min(ext_xmin, x0 - hw)
                ext_ymin = (y0 - hw) if ext_ymin is None else min(ext_ymin, y0 - hw)
                ext_xmax = (x_right + hw) if ext_xmax is None else max(ext_xmax, x_right + hw)
                ext_ymax = (y_top + hw) if ext_ymax is None else max(ext_ymax, y_top + hw)
                n_ok += 1
            except Exception as e:
                errors.append("%s: %s" % (os.path.basename(tif), e))
            prog.setValue(i + 1)
        prog.setValue(len(tifs))

        if man is not None and n_ok:
            new_man = dict(man)
            slices = []
            for s in (man.get("slices") or []):
                s2 = dict(s)
                nm = str(s.get("grid", ""))
                if nm in stats_by_name:
                    s2["stats"] = stats_by_name[nm]
                slices.append(s2)
            if slices:
                new_man["slices"] = slices
            new_man["extent"] = [ext_xmin, ext_ymin, ext_xmax, ext_ymax]
            src = dict(new_man.get("source") or {})
            src["rotate_translate"] = {"angle_deg": angle, "origin": list(origin), "mode": mode}
            new_man["source"] = src
            try:
                with open(os.path.join(out_dir, "sliceset.json"), "w", encoding="utf-8") as f:
                    json.dump(new_man, f, indent=2, ensure_ascii=True, sort_keys=True)
            except Exception:
                pass

        if n_ok:
            msg = "%d/%d slice rototraslate in\n%s" % (n_ok, len(tifs), out_dir)
            if errors:
                msg += "\n\nErrori:\n" + "\n".join(errors[:5])
            QMessageBox.information(self, "Ruota/trasla set", msg)
            self._gg_repoint_slice_view(out_dir)
        else:
            QMessageBox.warning(self, "Ruota/trasla set",
                                "Nessuna slice scritta.\n" + "\n".join(errors[:5]))

    def _profiles_for_slicing(self):
        """(profiles, variant_name) da usare per la generazione time-slice.

        Se il pannello ha una 'Variante sorgente' diversa da 'raw' e la cartella
        esiste con .npz, ritorna i radargrammi gia' filtrati caricati da disco
        (il chiamante deve disabilitare la pipeline). Altrimenti i profili
        correnti e ``""``.
        """
        panel = getattr(self, "_slice_spec_panel", None)
        name = ""
        if panel is not None and self._is_qt_alive(panel):
            try:
                name = str(panel.get_spec().source_variant or "").strip()
            except Exception:
                name = ""
        d = self._variant_dir_for_name(name)
        if not d:
            return list(self._profiles or []), ""
        npzs = sorted(glob.glob(os.path.join(d, "*.npz")))
        try:
            profs = self._load_variant_profiles(npzs)
        except Exception as exc:
            QMessageBox.warning(
                self, "Variante sorgente",
                f"Impossibile leggere i .npz della variante '{name}':\n{exc}\n\n"
                "Uso i profili correnti.",
            )
            return list(self._profiles or []), ""
        return profs, name

    def _slice_from_variant_folder(self, folder: str):
        """Carica i .npz di una cartella-variante e lancia la generazione slice."""
        paths = self._gpr_processing_paths()
        folder = os.path.normpath(str(folder or ""))
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "Time-slice da variante", "Cartella non valida.")
            return
        npzs = sorted(glob.glob(os.path.join(folder, "*.npz")))
        if not npzs:
            QMessageBox.warning(
                self, "Time-slice da variante",
                "Nessun file .npz nella cartella selezionata.",
            )
            return
        try:
            profiles = self._load_variant_profiles(npzs)
        except Exception as e:
            QMessageBox.critical(
                self, "Time-slice da variante", f"Errore lettura .npz: {e}"
            )
            return
        if self.plugin is None or not hasattr(self.plugin, "import_ogpr_as_slices"):
            QMessageBox.information(
                self, "Time-slice da variante",
                "Funzione disponibile dal plugin principale.",
            )
            return
        try:
            dz = float(self._spin_slice_thickness.value())
        except Exception:
            dz = 0.10
        slice_params = self.get_slice_params()
        # I dati sono gia' filtrati su disco: nessun ri-processing pipeline.
        slice_params["use_processing"] = False
        slice_params["pipeline_params"] = {}
        slice_params["pre_slice_bg_removal"] = False
        slice_params["z_step_override"] = float(dz)
        slice_params["thickness_m"] = float(dz)
        variant_name = os.path.basename(os.path.normpath(folder)) or "variante"
        if paths.get("timeslices"):
            slice_params["output_dir"] = os.path.join(paths["timeslices"], variant_name)
        try:
            self.plugin.import_ogpr_as_slices(
                profiles, slice_params=slice_params,
                completion_callback=self._on_timeslice_build_finished,
            )
        except TypeError:
            self.plugin.import_ogpr_as_slices(profiles, slice_params=slice_params)

    def _slice_pipeline_enabled_map(self) -> dict:
        enabled = dict(self._pipeline_enabled_map_from_controls() or {})
        if self._is_qt_alive(self._filter_chain_widget):
            try:
                from_widget = dict(self._filter_chain_widget.get_enabled_map() or {})
                for sid, on in from_widget.items():
                    enabled[str(sid)] = bool(on)
            except Exception:
                pass
        return enabled

    def _sync_slice_source_mode(self) -> None:
        if not self._is_qt_alive(getattr(self, "_cb_slice_pipeline_source", None)):
            return
        mode = str(self._cb_slice_pipeline_source.currentData() or "auto").strip().lower()
        manual = (mode == "manual")
        for w in (
            getattr(self, "_cb_slice_extraction", None),
            getattr(self, "_chk_slice_use_hilbert", None),
            getattr(self, "_chk_slice_use_processing", None),
        ):
            if self._is_qt_alive(w):
                try:
                    w.setEnabled(manual)
                except Exception:
                    pass

    def _resolve_slice_source(
        self,
        proc_params: dict,
        extraction_mode_manual: str,
        use_hilbert_manual: bool,
        use_processing_manual: bool,
    ) -> tuple[dict, bool, str, bool, str]:
        params = dict(proc_params or {})
        chain_order = [str(s) for s in list(params.get("chain_order") or self._current_chain_order() or DEFAULT_CHAIN_ORDER)]
        enabled_map = self._slice_pipeline_enabled_map()

        def _is_enabled(step_id: str) -> bool:
            if step_id in enabled_map:
                return bool(enabled_map.get(step_id))
            return bool(params.get(step_id, False))

        mode = "auto"
        if self._is_qt_alive(getattr(self, "_cb_slice_pipeline_source", None)):
            mode = str(self._cb_slice_pipeline_source.currentData() or "auto").strip().lower()
        if mode == "manual":
            return params, bool(use_processing_manual), str(extraction_mode_manual), bool(use_hilbert_manual), "manual"

        # Auto mode: prefer Hilbert/envelope when active, otherwise use last active step.
        if mode == "auto":
            if ("envelope" in chain_order) and _is_enabled("envelope"):
                target = "envelope"
            else:
                target = next((sid for sid in reversed(chain_order) if _is_enabled(sid)), "raw")
        elif mode == "raw":
            target = "raw"
        else:
            target = str(mode)
            if (target not in chain_order) or (not _is_enabled(target)):
                target = next((sid for sid in reversed(chain_order) if _is_enabled(sid)), "raw")

        if target == "raw":
            return params, False, "las_like", False, "raw"

        if target in chain_order:
            truncated = chain_order[: chain_order.index(target) + 1]
        else:
            truncated = [sid for sid in chain_order if _is_enabled(sid)]
        if not truncated:
            return params, False, "las_like", False, "raw"

        resolved = dict(params)
        for sid in DEFAULT_CHAIN_ORDER:
            resolved[sid] = bool(_is_enabled(sid) and (sid in truncated))
        resolved["chain_order"] = list(truncated)

        if target == "envelope":
            extraction_mode = "envelope"
            use_hilbert = True
        else:
            extraction_mode = "las_like"
            use_hilbert = False
        return resolved, True, extraction_mode, use_hilbert, target

    def _apply_selected_timeslice_preset(self):
        key = str(self._cb_slice_preset.currentData() or "base").strip().lower()
        self._apply_timeslice_preset(key)

    def _apply_timeslice_preset(self, preset_key: str):
        key = str(preset_key or "base").strip().lower()
        if key not in {"base", "quality", "aggressive"}:
            key = "base"

        if key == "quality":
            cfg = {
                "extraction_mode": "envelope",
                "use_hilbert": True,
                "use_processing": True,
                "pre_slice_bg_removal": True,
                "pre_slice_bg_mode": "line_by_line",
                "pre_slice_bg_auto": True,
                "pre_slice_bg_window": 200,
                "pre_slice_bg_sample_start": 10,
                "pre_slice_bg_sample_end": 0,
                "stack_n": 3,
                "stack_kernel": "triangular",
                "flip_traces_mode": "none",
                "topographic_correction": False,
                "topo_reference_mode": "median",
                "topo_reference_elevation": 0.0,
                "normalize_channels": False,
                "amplitude_filter": True,
                "amplitude_sigma": 3.0,
                "idw_mode": "quality",
                "slice_source": "auto",
                "search_radius": 0.0,
                "idw_power": 2,
                "overlap_fraction": 0.5,
                "blanking_distance": 0.30,
                "use_anisotropic_idw": True,
                "auto_radius": True,
                "balance_profiles": True,
                "depth_radius_factor": 0.8,
                "min_points": 2,
                "parallel_profiles": True,
                "profile_workers": 0,
                "fill_nodata": True,
                "smooth": True,
                "smooth_sigma": 1.0,
            }
            lbl = "Preset timeslice: Qualita' applicato."
        elif key == "aggressive":
            cfg = {
                "extraction_mode": "envelope",
                "use_hilbert": True,
                "use_processing": True,
                "pre_slice_bg_removal": True,
                "pre_slice_bg_mode": "grid_by_grid",
                "pre_slice_bg_auto": False,
                "pre_slice_bg_window": 300,
                "pre_slice_bg_sample_start": 10,
                "pre_slice_bg_sample_end": 0,
                "stack_n": 5,
                "stack_kernel": "triangular",
                "flip_traces_mode": "none",
                "topographic_correction": False,
                "topo_reference_mode": "median",
                "topo_reference_elevation": 0.0,
                "normalize_channels": False,
                "amplitude_filter": True,
                "amplitude_sigma": 2.5,
                "idw_mode": "quality",
                "slice_source": "auto",
                "search_radius": 0.0,
                "idw_power": 3,
                "overlap_fraction": 0.5,
                "blanking_distance": 0.30,
                "use_anisotropic_idw": True,
                "auto_radius": True,
                "balance_profiles": True,
                "depth_radius_factor": 1.0,
                "min_points": 3,
                "parallel_profiles": True,
                "profile_workers": 0,
                "fill_nodata": True,
                "smooth": True,
                "smooth_sigma": 1.5,
            }
            lbl = "Preset timeslice: Aggressivo applicato."
        else:
            cfg = {
                "extraction_mode": "las_like",
                "use_hilbert": True,
                "use_processing": True,
                "pre_slice_bg_removal": True,
                "pre_slice_bg_mode": "line_by_line",
                "pre_slice_bg_auto": True,
                "pre_slice_bg_window": 200,
                "pre_slice_bg_sample_start": 10,
                "pre_slice_bg_sample_end": 0,
                "stack_n": 1,
                "stack_kernel": "boxcar",
                "flip_traces_mode": "none",
                "topographic_correction": False,
                "topo_reference_mode": "median",
                "topo_reference_elevation": 0.0,
                "normalize_channels": False,
                "amplitude_filter": False,
                "amplitude_sigma": 3.0,
                "idw_mode": "quality",
                "slice_source": "auto",
                "search_radius": 0.0,
                "idw_power": 2,
                "overlap_fraction": 0.5,
                "blanking_distance": 0.30,
                "use_anisotropic_idw": False,
                "auto_radius": True,
                "balance_profiles": True,
                "depth_radius_factor": 0.6,
                "min_points": 1,
                "parallel_profiles": True,
                "profile_workers": 0,
                "fill_nodata": True,
                "smooth": True,
                "smooth_sigma": 0.8,
            }
            lbl = "Preset timeslice: Base applicato."

        self._set_combo_to_data(self._cb_slice_extraction, cfg["extraction_mode"])
        self._chk_slice_use_hilbert.setChecked(bool(cfg.get("use_hilbert", True)))
        self._chk_slice_use_processing.setChecked(bool(cfg["use_processing"]))
        self._chk_slice_bg.setChecked(bool(cfg["pre_slice_bg_removal"]))
        self._set_combo_to_data(self._cb_slice_bg_mode, cfg["pre_slice_bg_mode"])
        self._chk_slice_bg_auto.setChecked(bool(cfg["pre_slice_bg_auto"]))
        self._spin_slice_bg_window.setValue(int(cfg["pre_slice_bg_window"]))
        self._spin_slice_bg_sample_start.setValue(int(cfg["pre_slice_bg_sample_start"]))
        self._spin_slice_bg_sample_end.setValue(int(cfg["pre_slice_bg_sample_end"]))
        self._spin_slice_stack_n.setValue(int(cfg["stack_n"]))
        self._set_combo_to_data(self._cb_slice_stack_kernel, cfg["stack_kernel"])
        self._set_combo_to_data(self._cb_slice_flip_mode, cfg["flip_traces_mode"])
        self._chk_slice_topographic.setChecked(bool(cfg["topographic_correction"]))
        self._set_combo_to_data(self._cb_slice_topo_ref_mode, cfg["topo_reference_mode"])
        self._spin_slice_topo_ref_custom.setValue(float(cfg["topo_reference_elevation"]))
        self._chk_normalize_ch.setChecked(bool(cfg["normalize_channels"]))
        self._chk_amplitude_filter.setChecked(bool(cfg["amplitude_filter"]))
        self._spin_amplitude_sigma.setValue(float(cfg["amplitude_sigma"]))
        self._set_combo_to_data(self._cb_slice_idw_mode, cfg["idw_mode"])
        self._spin_slice_search_radius.setValue(float(cfg.get("search_radius", 0.0) or 0.0))
        self._set_combo_to_data(self._cb_slice_pipeline_source, str(cfg.get("slice_source", "auto") or "auto"))
        self._spin_slice_idw_power.setValue(int(cfg.get("idw_power", 2)))
        self._spin_slice_overlap_pct.setValue(int(round(float(cfg.get("overlap_fraction", 0.5)) * 100.0)))
        self._spin_slice_blanking_m.setValue(float(cfg.get("blanking_distance", 0.30)))
        self._chk_anisotropic_idw.setChecked(bool(cfg["use_anisotropic_idw"]))
        self._chk_auto_radius.setChecked(bool(cfg["auto_radius"]))
        self._chk_slice_balance_profiles.setChecked(bool(cfg["balance_profiles"]))
        self._spin_slice_depth_radius_factor.setValue(float(cfg["depth_radius_factor"]))
        self._spin_slice_min_points.setValue(int(cfg["min_points"]))
        self._chk_slice_parallel_profiles.setChecked(bool(cfg["parallel_profiles"]))
        self._spin_slice_profile_workers.setValue(int(cfg["profile_workers"]))
        self._chk_fill_nodata.setChecked(bool(cfg["fill_nodata"]))
        self._chk_smooth.setChecked(bool(cfg["smooth"]))
        self._spin_smooth_sigma.setValue(float(cfg["smooth_sigma"]))

        # Sync dependent controls if state has not emitted toggles.
        self._spin_slice_bg_window.setEnabled(
            bool(self._chk_slice_bg.isChecked()) and (not bool(self._chk_slice_bg_auto.isChecked()))
        )
        self._cb_slice_bg_mode.setEnabled(bool(self._chk_slice_bg.isChecked()))
        self._chk_slice_bg_auto.setEnabled(bool(self._chk_slice_bg.isChecked()))
        self._spin_slice_bg_sample_start.setEnabled(bool(self._chk_slice_bg.isChecked()))
        self._spin_slice_bg_sample_end.setEnabled(bool(self._chk_slice_bg.isChecked()))
        topo_on = bool(self._chk_slice_topographic.isChecked())
        self._cb_slice_topo_ref_mode.setEnabled(topo_on)
        self._spin_slice_topo_ref_custom.setEnabled(
            topo_on and (str(self._cb_slice_topo_ref_mode.currentData() or "") == "custom")
        )
        self._spin_amplitude_sigma.setEnabled(bool(self._chk_amplitude_filter.isChecked()))
        self._spin_smooth_sigma.setEnabled(bool(self._chk_smooth.isChecked()))
        self._spin_slice_profile_workers.setEnabled(bool(self._chk_slice_parallel_profiles.isChecked()))
        self._sync_slice_source_mode()

        # I preset "Qualita'"/"Aggressivo" impostano parametri della sezione
        # Esperto: senza abilitarla esplicitamente quei valori verrebbero
        # ignorati da get_slice_params(). "Base" li riporta ai default e
        # ricollassa la sezione per tornare alla vista semplice.
        if self._is_qt_alive(getattr(self, "_btn_slice_expert", None)):
            self._btn_slice_expert.setChecked(key != "base")

        self._safe_set_text(self._lbl_status, lbl)

    def get_slice_params(self) -> dict:
        # Fase D: quando il pannello SliceSpec e' attivo e' l'unica sorgente
        # dei parametri di generazione. Il dict restituito ha la stessa forma
        # legacy (piu' la chiave "_spec") cosi' il resto della pipeline non
        # cambia; il ramo Esperto sotto resta solo come fallback.
        panel = getattr(self, "_slice_spec_panel", None)
        if panel is not None and self._is_qt_alive(panel):
            try:
                from .slice_params_compat import spec_to_legacy_params

                spec = panel.get_spec()
                if self._profiles:
                    spec = spec.resolve(self._profiles)
                elif spec.z_max_m is None:
                    from dataclasses import replace as _replace
                    spec = _replace(spec, z_max_m=5.0)  # placeholder finche' non ci sono profili
                open_3d = bool(
                    self._is_qt_alive(getattr(self, "_chk_slice_open_3d_on_complete", None))
                    and self._chk_slice_open_3d_on_complete.isChecked()
                )
                out = spec_to_legacy_params(
                    spec, channel=-1, combine_method="mean",
                    open_3d_on_complete=open_3d,
                    style=self._slice_display_style(),
                )
                return out
            except Exception as exc:
                print(f"[GPR viewer][WARN] get_slice_params via SliceSpec fallito: {exc}")

        # "Esperto" e' un flag: finche' non e' esplicitamente abilitato, questi
        # campi vengono ignorati e si usa il comportamento di default, anche
        # se i widget (nascosti) contengono un valore diverso.
        expert_on = bool(
            self._is_qt_alive(getattr(self, "_btn_slice_expert", None))
            and self._btn_slice_expert.isChecked()
        )
        slice_bg_auto = bool(self._chk_slice_bg_auto.isChecked()) if expert_on else True
        topo_mode = str(self._cb_slice_topo_ref_mode.currentData() or "median") if expert_on else "median"
        topo_on = bool(self._chk_slice_topographic.isChecked()) if expert_on else False
        topo_custom = None
        if expert_on and topo_mode == "custom":
            topo_custom = float(self._spin_slice_topo_ref_custom.value())
        proc_params = dict(self._current_processing_params() or {})
        proc_params, use_processing, extraction_mode, use_hilbert, source_step = self._resolve_slice_source(
            proc_params,
            extraction_mode_manual=(
                str(self._cb_slice_extraction.currentData() or "las_like") if expert_on else "las_like"
            ),
            use_hilbert_manual=(bool(self._chk_slice_use_hilbert.isChecked()) if expert_on else True),
            use_processing_manual=(bool(self._chk_slice_use_processing.isChecked()) if expert_on else True),
        )
        try:
            search_radius = float(self._spin_slice_search_radius.value())
        except Exception:
            search_radius = 0.0
        radius_override = None
        if np.isfinite(search_radius) and search_radius > 0.0:
            radius_override = float(search_radius)
        auto_radius = radius_override is None
        return {
            "normalize_channels":  (self._chk_normalize_ch.isChecked() if expert_on else False),
            "slice_source":        str(source_step or "auto"),
            "extraction_mode":     str(extraction_mode or "las_like"),
            "use_hilbert":         bool(use_hilbert),
            "use_processing":      bool(use_processing),
            "pre_slice_bg_removal": self._chk_slice_bg.isChecked(),
            "pre_slice_bg_mode": (
                str(self._cb_slice_bg_mode.currentData() or "line_by_line") if expert_on else "line_by_line"
            ),
            "pre_slice_bg_window": (0 if slice_bg_auto else int(self._spin_slice_bg_window.value())),
            "pre_slice_bg_sample_start": (int(self._spin_slice_bg_sample_start.value()) if expert_on else 0),
            "pre_slice_bg_sample_end": (int(self._spin_slice_bg_sample_end.value()) if expert_on else 0),
            "stack_n": (int(self._spin_slice_stack_n.value()) if expert_on else 1),
            "stack_kernel": (str(self._cb_slice_stack_kernel.currentData() or "boxcar") if expert_on else "boxcar"),
            "flip_traces_mode": (str(self._cb_slice_flip_mode.currentData() or "none") if expert_on else "none"),
            "pipeline_params": proc_params,
            "topographic_correction": topo_on,
            "topo_reference_mode": topo_mode,
            "topo_reference_elevation": topo_custom,
            "amplitude_sigma":     (
                float(self._spin_amplitude_sigma.value())
                if (expert_on and self._chk_amplitude_filter.isChecked()) else None
            ),
            "idw_mode":            str(self._cb_slice_idw_mode.currentData() or "quality"),
            "radius":              radius_override,
            "search_radius":       radius_override,
            "idw_power":           int(self._spin_slice_idw_power.value()),
            "overlap_fraction":    float(self._spin_slice_overlap_pct.value()) / 100.0,
            "blanking_distance":   float(self._spin_slice_blanking_m.value()),
            "use_anisotropic_idw": (self._chk_anisotropic_idw.isChecked() if expert_on else False),
            "auto_radius":         bool(auto_radius),
            "balance_profiles":    self._chk_slice_balance_profiles.isChecked(),
            "depth_radius_factor": (float(self._spin_slice_depth_radius_factor.value()) if expert_on else 0.6),
            "min_points":          (int(self._spin_slice_min_points.value()) if expert_on else 1),
            "parallel_profiles":   (self._chk_slice_parallel_profiles.isChecked() if expert_on else True),
            "profile_workers":     (int(self._spin_slice_profile_workers.value()) if expert_on else 0),
            "fill_nodata":         self._chk_fill_nodata.isChecked(),
            "smooth_sigma":        (
                float(self._spin_smooth_sigma.value())
                if self._chk_smooth.isChecked() else 0.0
            ),
            "open_3d_on_complete": bool(
                self._is_qt_alive(self._chk_slice_open_3d_on_complete)
                and self._chk_slice_open_3d_on_complete.isChecked()
            ),
        }

    def _offer_variant_before_slicing(self) -> None:
        """Le slice usano i profili grezzi se la variante sorgente e' 'raw': se nel
        viewer ci sono filtri attivi propone di salvarli in una variante e usarla."""
        panel = getattr(self, "_slice_spec_panel", None)
        if panel is None or not self._is_qt_alive(panel):
            return
        try:
            src = str(panel.get_spec().source_variant or "raw").strip() or "raw"
        except Exception:
            return
        if src.lower() != "raw":
            return
        chain = getattr(self, "_filter_chain_widget", None)
        active = []
        if self._is_qt_alive(chain):
            active = [k for k, on in dict(chain.get_enabled_map() or {}).items() if on]
        if not active:
            return
        box = QMessageBox(self)
        box.setWindowTitle("Time-slice dal grezzo")
        box.setText(
            "La variante sorgente e' 'raw': le time-slice verranno generate dai profili "
            "SENZA i filtri attivi nel viewer (" + ", ".join(active[:6]) + ").\n\n"
            "Vuoi salvare prima una variante con i filtri correnti e usarla?"
        )
        b_save = box.addButton("Salva variante e usala", QMessageBox.AcceptRole)
        box.addButton("Genera dal grezzo", QMessageBox.ActionRole)
        box.exec_()
        if box.clickedButton() is not b_save:
            return
        name = "auto_" + datetime.now().strftime("%Y%m%d_%H%M")
        saved = self._save_processed_variant(name, quiet=False)
        if saved:
            try:
                self._refresh_slice_variant_choices()
                panel._set_source_variant(saved)
            except Exception:
                pass

    def _open_slice_dialog(self):
        if not self._profiles:
            QMessageBox.information(self, "Nessun profilo",
                                    "Importa almeno un file .ogpr prima.")
            return
        self._offer_variant_before_slicing()
        slice_params = self.get_slice_params()
        slice_profiles, variant_src = self._profiles_for_slicing()
        if not slice_profiles:
            QMessageBox.warning(self, "Timeslice", "Nessun radargramma da affettare.")
            return
        if variant_src:
            # dati gia' filtrati su disco: niente ri-processing della pipeline
            slice_params["use_processing"] = False
            slice_params["pipeline_params"] = {}
            slice_params["pre_slice_bg_removal"] = False
        panel = getattr(self, "_slice_spec_panel", None)
        if panel is not None and self._is_qt_alive(panel):
            try:
                dz = float(panel.get_spec().thickness_m)
            except Exception:
                dz = float(slice_params.get("z_step", 0.10) or 0.10)
        else:
            try:
                dz = float(self._spin_slice_thickness.value())
            except Exception:
                dz = 0.10
        if (not np.isfinite(dz)) or dz <= 0.0:
            QMessageBox.warning(self, "Timeslice", "Spessore slice non valido.")
            return
        slice_params["z_step_override"] = float(dz)
        slice_params["thickness_m"] = float(dz)
        outdir = str(getattr(self, "_le_slice_outdir", None).text() if hasattr(self, "_le_slice_outdir") else "").strip()
        if not outdir and variant_src:
            paths = self._gpr_processing_paths()
            if paths.get("timeslices"):
                outdir = os.path.join(paths["timeslices"], variant_src)
        if outdir:
            slice_params["output_dir"] = outdir
        if self.plugin and hasattr(self.plugin, "import_ogpr_as_slices"):
            try:
                self.plugin.import_ogpr_as_slices(
                    slice_profiles,
                    slice_params=slice_params,
                    completion_callback=self._on_timeslice_build_finished,
                )
            except TypeError:
                self.plugin.import_ogpr_as_slices(slice_profiles, slice_params=slice_params)
        else:
            QMessageBox.information(
                self, "Timeslice",
                "Funzione disponibile dal plugin principale."
            )
