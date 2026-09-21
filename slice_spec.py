# -*- coding: utf-8 -*-
"""``SliceSpec`` — the single parameter object for time-slice generation.

Fase B of the time-slice redesign. One dataclass with ~13 sensible-default
fields replaces the ~40 keyword arguments of ``compute_ogpr_slice_grids`` and
the parallel ``_ask_ogpr_slice_params`` dialog set.

The spec is *declarative*: ``resolve(profiles)`` fills the fields that depend on
the survey (``z_max_m``, ``search_radius_m``, ``crs_epsg``) and returns a new,
fully-populated ``SliceSpec``. ``to_compute_kwargs()`` maps the spec onto the
legacy ``compute_ogpr_slice_grids`` signature so the two can coexist during the
migration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
import json

FORMAT = "rasterlinker_slicespec_v1"

AMPLITUDE_MODES = ("abs_mean", "rms", "env_peak", "signed_mean", "max_abs")
# "bin"  = every channel at its own position + binned/normalised-convolution
#          gridding (slice_engine, the default; no IDW bull's-eyes, fast);
# "idw"  = the legacy profile-centre-line IDW path (compute_ogpr_slice_grids);
# "kriging" is reserved and not implemented.
INTERP_METHODS = ("bin", "idw", "kriging")
SMOOTH_LOWPASS_VALUES = (0, 3, 5)
# Level equalisation applied before gridding: "channel" removes the per-channel
# gain differences of array radars (stripes along the tracks), "profile" only
# equalises whole swaths, "none" leaves the amplitudes untouched.
BALANCE_MODES = ("channel", "profile", "none")

# amplitude_mode -> (extraction_mode, use_hilbert) for the LEGACY (idw) OGPR
# pipeline only; the "bin" engine implements every mode natively.
# rms / max_abs have no dedicated legacy path and fall back to the rectified
# ("las_like") extraction.
_AMPLITUDE_MAP = {
    "abs_mean": ("las_like", False),
    "rms": ("las_like", False),
    "env_peak": ("envelope", True),
    "signed_mean": ("signed", False),
    "max_abs": ("las_like", False),
}


class SliceSpecError(ValueError):
    """Raised for structurally invalid specs (bad enum value, negative size…)."""


@dataclass
class SliceSpec:
    # --- depth windowing -------------------------------------------------
    z_min_m: float = 0.0
    z_max_m: float | None = None          # None -> max profile depth (resolve)
    thickness_m: float = 0.10
    overlap: float = 0.5                  # 0 .. 0.9, fraction of thickness
    # --- amplitude extraction -----------------------------------------------
    amplitude_mode: str = "abs_mean"
    source_variant: str = "raw"           # pipeline variant id or "raw"
    velocity_m_s: float | None = None     # None -> each profile's own velocity
    balance: str = "channel"              # "channel" | "profile" | "none"
    # --- gridding ---------------------------------------------------------
    resolution_m: float = 0.10
    interp: str = "bin"
    search_radius_m: float | None = None  # None -> auto from interline spacing
    blanking_radius_m: float = 0.0
    smooth_lowpass: int = 3               # 0 | 3 | 5 (low-pass kernel size)
    bg_horizontal: bool = False           # rimuove bande orizzontali (media per riga)
    bg_vertical: bool = False             # rimuove bande verticali (media per colonna)
    bg_clip_pct: float = 95.0             # percentile: i target forti non entrano nel background
    # --- crs ------------------------------------------------------------
    crs_epsg: int | None = None

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------
    def validate(self) -> None:
        if self.amplitude_mode not in AMPLITUDE_MODES:
            raise SliceSpecError(
                f"amplitude_mode {self.amplitude_mode!r} not in {AMPLITUDE_MODES}"
            )
        if self.interp not in INTERP_METHODS:
            raise SliceSpecError(f"interp {self.interp!r} not in {INTERP_METHODS}")
        if self.balance not in BALANCE_MODES:
            raise SliceSpecError(f"balance {self.balance!r} not in {BALANCE_MODES}")
        if self.velocity_m_s is not None and not float(self.velocity_m_s) > 0.0:
            raise SliceSpecError("velocity_m_s must be > 0 or None (auto)")
        if float(self.thickness_m) <= 0.0:
            raise SliceSpecError("thickness_m must be > 0")
        if not (0.0 <= float(self.overlap) <= 0.9):
            raise SliceSpecError("overlap must be in [0.0, 0.9]")
        if float(self.resolution_m) <= 0.0:
            raise SliceSpecError("resolution_m must be > 0")
        if self.z_max_m is not None and float(self.z_max_m) <= float(self.z_min_m):
            raise SliceSpecError("z_max_m must be > z_min_m")
        if self.search_radius_m is not None and float(self.search_radius_m) <= 0.0:
            raise SliceSpecError("search_radius_m must be > 0 or None")
        if float(self.blanking_radius_m) < 0.0:
            raise SliceSpecError("blanking_radius_m must be >= 0")
        if not (50.0 <= float(self.bg_clip_pct) <= 100.0):
            raise SliceSpecError("bg_clip_pct must be in [50, 100]")
        if int(self.smooth_lowpass) not in SMOOTH_LOWPASS_VALUES:
            raise SliceSpecError(
                f"smooth_lowpass {self.smooth_lowpass!r} not in {SMOOTH_LOWPASS_VALUES}"
            )

    # ------------------------------------------------------------------
    # derived geometry
    # ------------------------------------------------------------------
    @property
    def stride_m(self) -> float:
        """Distance between consecutive slice centres."""
        return float(self.thickness_m) * (1.0 - float(self.overlap))

    @property
    def smooth_sigma(self) -> float:
        """Gaussian sigma equivalent of the low-pass kernel size."""
        return float(self.smooth_lowpass) / 3.0 if self.smooth_lowpass else 0.0

    @property
    def grid_bg_spec(self) -> str:
        """Stringa per grid_filters.apply_grid_bg ('' = nessun background)."""
        flags = ("h" if self.bg_horizontal else "") + ("v" if self.bg_vertical else "")
        return f"{flags}@{float(self.bg_clip_pct):g}" if flags else ""

    def is_resolved(self) -> bool:
        return self.z_max_m is not None and (
            self.interp not in ("idw", "bin") or self.search_radius_m is not None
        )

    # ------------------------------------------------------------------
    # resolution against a concrete survey
    # ------------------------------------------------------------------
    def resolve(self, profiles: list | None = None) -> "SliceSpec":
        """Return a copy with survey-dependent fields filled in.

        - ``z_max_m``        -> max ``depth_max_m`` across profiles
        - ``search_radius_m``-> auto interline radius (idw only)
        - ``crs_epsg``       -> first profile EPSG when unset
        """
        self.validate()
        if self.interp == "kriging":
            raise SliceSpecError("interp='kriging' is not implemented yet")

        profiles = list(profiles or [])
        z_max = self.z_max_m
        if z_max is None:
            depths = [
                float(getattr(p, "depth_max_m", 0.0) or 0.0) for p in profiles
            ]
            depths = [d for d in depths if d > 0.0]
            z_max = max(depths) if depths else None
            if z_max is None:
                raise SliceSpecError(
                    "z_max_m is None and no profile exposes a positive depth_max_m"
                )

        radius = self.search_radius_m
        if radius is None and self.interp in ("idw", "bin"):
            radius = _auto_interline_radius(profiles)

        epsg = self.crs_epsg
        if epsg is None:
            for p in profiles:
                e = getattr(p, "epsg", None)
                if e:
                    epsg = int(e)
                    break

        return replace(
            self,
            z_max_m=float(z_max),
            search_radius_m=(float(radius) if radius is not None else None),
            crs_epsg=(int(epsg) if epsg is not None else None),
        )

    # ------------------------------------------------------------------
    # mapping onto the legacy compute signature
    # ------------------------------------------------------------------
    def to_compute_kwargs(self) -> dict:
        """Map the spec onto ``compute_ogpr_slice_grids`` keyword arguments.

        Only the parameters the spec actually governs are returned; every other
        compute kwarg keeps its own default/caller value. Call on a *resolved*
        spec so ``z_max``/``radius`` are concrete.
        """
        self.validate()
        extraction_mode, use_hilbert = _AMPLITUDE_MAP[self.amplitude_mode]
        kwargs: dict = {
            "z_min": float(self.z_min_m),
            "z_step": float(self.thickness_m),
            "overlap_fraction": float(self.overlap),
            "resolution": float(self.resolution_m),
            "extraction_mode": extraction_mode,
            "use_hilbert": use_hilbert,
            "blanking_distance": float(self.blanking_radius_m),
            "smooth_sigma": float(self.smooth_sigma),
            "grid_bg": self.grid_bg_spec,
        }
        if self.z_max_m is not None:
            kwargs["z_max"] = float(self.z_max_m)
        if self.search_radius_m is not None:
            kwargs["radius"] = float(self.search_radius_m)
            kwargs["auto_radius"] = False
        else:
            kwargs["auto_radius"] = True
        return kwargs

    # ------------------------------------------------------------------
    # serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["format"] = FORMAT
        return d

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=True, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict) -> "SliceSpec":
        if not isinstance(data, dict):
            raise SliceSpecError("SliceSpec.from_dict expects a dict")
        fmt = data.get("format")
        if fmt is not None and fmt != FORMAT:
            raise SliceSpecError(f"unknown SliceSpec format {fmt!r} (expected {FORMAT})")
        known = {f.name for f in fields(cls)}
        data = dict(data)
        if "balance" not in data and data.get("interp") == "idw":
            # Spec saved before the "bin" engine existed (no "balance" field):
            # "idw" back then just meant "the default", so move it to the new
            # default instead of pinning old projects to the dotted IDW result.
            data["interp"] = "bin"
        spec = cls(**{k: v for k, v in data.items() if k in known})
        spec.validate()
        return spec

    @classmethod
    def from_json(cls, text: str) -> "SliceSpec":
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_compute_meta(cls, meta: dict, *, crs_epsg: int | None = None) -> "SliceSpec":
        """Best-effort reconstruction of a spec from a ``compute_ogpr_slice_grids``
        ``meta`` dict, for runs that did not pass a ``SliceSpec``.

        Fields the meta does not carry (``smooth_lowpass``, ``source_variant``,
        ``velocity_m_s``) keep their defaults.
        """
        meta = dict(meta or {})
        rev = {
            ("las_like", False): "abs_mean",
            ("envelope", True): "env_peak",
            ("signed", False): "signed_mean",
        }
        ex_mode = str(meta.get("extraction_mode", "las_like"))
        use_hil = bool(meta.get("use_hilbert", False))
        amplitude_mode = rev.get((ex_mode, use_hil), "abs_mean")
        if meta.get("amplitude_mode") in AMPLITUDE_MODES:  # written by slice_engine
            amplitude_mode = str(meta["amplitude_mode"])

        radius = meta.get("radius")
        try:
            radius = float(radius) if radius is not None else None
        except (TypeError, ValueError):
            radius = None

        spec = cls(
            z_min_m=float(meta.get("z_min", 0.0)),
            z_max_m=(float(meta["z_max"]) if meta.get("z_max") is not None else None),
            thickness_m=float(meta.get("z_step", 0.10) or 0.10),
            overlap=float(meta.get("overlap_fraction", 0.5) or 0.0),
            amplitude_mode=amplitude_mode,
            resolution_m=float(meta.get("resolution", 0.10) or 0.10),
            interp=("bin" if meta.get("engine") == "bin" else "idw"),
            search_radius_m=radius,
            blanking_radius_m=float(meta.get("blanking_distance_m", 0.0) or 0.0),
            crs_epsg=(int(crs_epsg) if crs_epsg is not None else None),
        )
        spec.validate()
        return spec


def _auto_interline_radius(profiles: list) -> float | None:
    """Rough auto search radius: ~1.5x the median inter-profile spacing.

    Falls back to a small multiple of the along-trace spacing for a single
    profile, or ``None`` when no geometry is available (caller keeps
    ``auto_radius=True`` and lets the compute stage decide).
    """
    import numpy as np

    centroids = []
    along = []
    for p in profiles:
        x = _first_finite_array(p, ("easting", "x"))
        y = _first_finite_array(p, ("northing", "y"))
        if x is None or y is None or x.size == 0:
            continue
        centroids.append((float(np.nanmean(x)), float(np.nanmean(y))))
        if x.size > 1:
            d = np.hypot(np.diff(x), np.diff(y))
            d = d[np.isfinite(d) & (d > 0)]
            if d.size:
                along.append(float(np.median(d)))

    if len(centroids) >= 2:
        pts = np.asarray(centroids, dtype=np.float64)
        # nearest-neighbour distance between profile centroids
        nn = []
        for i in range(len(pts)):
            dd = np.hypot(pts[:, 0] - pts[i, 0], pts[:, 1] - pts[i, 1])
            dd[i] = np.inf
            nn.append(float(dd.min()))
        spacing = float(np.median(nn))
        if np.isfinite(spacing) and spacing > 0:
            # 1.5x so every grid cell blends at least two survey lines even
            # when the line spacing equals the nominal radius (e.g. 2 m lines).
            return round(spacing * 1.5, 4)

    if along:
        return round(float(np.median(along)) * 8.0, 4)
    return None


def _first_finite_array(obj, names):
    import numpy as np

    for n in names:
        v = getattr(obj, n, None)
        if v is None and hasattr(obj, "channel"):
            try:
                v = getattr(obj.channel(0), n, None)
            except Exception:
                v = None
        if v is None:
            continue
        arr = np.asarray(v, dtype=np.float64).ravel()
        if arr.size and np.isfinite(arr).any():
            return arr
    return None
