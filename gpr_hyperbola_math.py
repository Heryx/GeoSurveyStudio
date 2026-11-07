# -*- coding: utf-8 -*-
"""Matematica pura per il fit di iperboli di diffrazione GPR.

Estratta da ``gpr_profile_viewer.py`` (strumento "Hyperbola Fit" del Profile
Viewer): stesse formule, senza alcuna dipendenza da Qt/matplotlib/istanza, cosi'
da poter essere riusata anche dal ``RadargramDock`` (puntatore di velocita' live).
Lo strumento esistente nel Profile Viewer non viene toccato.
"""

from __future__ import annotations

import math

import numpy as np

_C_M_S = 299792458.0                 # velocita' della luce nel vuoto

# Valore di riserva quando nessuna velocita' e' stata calcolata/impostata per un
# profilo (vedi gpr_velocity_store.py): 0.1 m/ns e' il valore tipico usato "di
# base" nel rilievo GPR su suolo generico, richiesto esplicitamente dall'utente.
DEFAULT_VELOCITY_M_NS = 0.1


def rdp_to_velocity_m_s(rdp: float) -> float:
    """RDP (costante dielettrica relativa) -> velocita' di propagazione (m/s)."""
    rr = max(float(rdp), 1e-6)
    return float(_C_M_S / math.sqrt(rr))


def velocity_to_rdp(v_m_s: float) -> float:
    """Inversa di :func:`rdp_to_velocity_m_s`."""
    v = max(float(v_m_s), 1e-9)
    return float((_C_M_S / v) ** 2)


def velocity_m_s_to_m_ns(v_m_s: float) -> float:
    """m/s -> m/ns (unita' consueta in GPR: m/s * 1e-9)."""
    return float(v_m_s) * 1e-9


def m_ns_to_velocity_m_s(v_m_ns: float) -> float:
    """Inversa di :func:`velocity_m_s_to_m_ns`."""
    return float(v_m_ns) * 1e9


def hyperbola_curve_depth(x_rel, t0: float, v_fit: float, v_ref: float, s: float):
    """Profondita' display della curva iperbolica per l'offset orizzontale ``x_rel``.

    Modello bistatico: riflettore puntiforme a profondita' ``d_fit`` (nel mezzo a
    velocita' ``v_fit``), antenne Tx-Rx separate da ``s``. Con ``s=0`` collassa
    nella forma monostatica ``t = sqrt(t0^2 + (2*dx/v)^2)``.
    """
    x_rel = np.asarray(x_rel, dtype=np.float64)
    d_fit = math.sqrt(max(0.0, (0.5 * float(v_fit) * float(t0)) ** 2 - (0.5 * float(s)) ** 2))
    half_s = 0.5 * float(s)
    t = (
        np.hypot(x_rel - half_s, d_fit) + np.hypot(x_rel + half_s, d_fit)
    ) / max(float(v_fit), 1e-9)
    return 0.5 * float(v_ref) * t


def _solve_velocity_bistatic(dx: float, t0: float, v_ref: float, s: float, z_target: float,
                              v_bounds, iters: int = 40):
    """Bisezione su ``hyperbola_curve_depth`` quando l'offset Tx-Rx non e' zero."""
    v_lo, v_hi = min(v_bounds), max(v_bounds)

    def _f(v: float) -> float:
        return float(hyperbola_curve_depth(np.array([dx]), t0, v, v_ref, s)[0]) - float(z_target)

    f_lo, f_hi = _f(v_lo), _f(v_hi)
    if not (np.isfinite(f_lo) and np.isfinite(f_hi)):
        return None
    if f_lo == 0.0:
        return v_lo
    if f_hi == 0.0:
        return v_hi
    if (f_lo > 0.0) == (f_hi > 0.0):
        return None          # nessuno zero nell'intervallo di RDP ammesso
    for _ in range(iters):
        v_mid = 0.5 * (v_lo + v_hi)
        f_mid = _f(v_mid)
        if not np.isfinite(f_mid):
            return None
        if (f_mid > 0.0) == (f_lo > 0.0):
            v_lo, f_lo = v_mid, f_mid
        else:
            v_hi, f_hi = v_mid, f_mid
    return 0.5 * (v_lo + v_hi)


def velocity_from_two_points(x0: float, z0: float, x1: float, z1: float, v_ref: float,
                              s: float = 0.0, rdp_bounds=(1.0, 40.0)):
    """Velocita'/RDP dell'iperbole con apice (x0,z0) che passa anche per (x1,z1).

    ``x0``/``x1`` = distanza lungo il profilo (m), ``z0``/``z1`` = profondita'
    display (m), ``v_ref`` = velocita' di riferimento del profilo (usata per
    convertire la profondita' dell'apice in tempo doppio ``t0``), ``s`` = offset
    Tx-Rx (m, 0 = monostatico).

    Ritorna ``(v_fit_m_s, rdp)`` oppure ``None`` se il punto non ammette una
    soluzione fisicamente valida (sopra o troppo vicino all'apice, o RDP fuori
    da ``rdp_bounds``).
    """
    dx = float(x1) - float(x0)
    z0 = float(z0)
    z1 = float(z1)
    v_ref = max(float(v_ref), 1e-9)
    if not (np.isfinite(dx) and np.isfinite(z0) and np.isfinite(z1)):
        return None
    if z0 < 0.0 or z1 <= z0:
        return None
    t0 = (2.0 * z0) / v_ref
    if not (np.isfinite(t0) and t0 > 0.0):
        return None
    s = max(0.0, float(s))
    if s <= 0.0:
        denom = math.sqrt(max(z1 * z1 - z0 * z0, 1e-12))
        v_fit = v_ref * abs(dx) / denom
    else:
        v_bounds = (rdp_to_velocity_m_s(max(rdp_bounds)), rdp_to_velocity_m_s(min(rdp_bounds)))
        v_fit = _solve_velocity_bistatic(dx, t0, v_ref, s, z1, v_bounds)
        if v_fit is None:
            return None
    if not (np.isfinite(v_fit) and v_fit > 0.0):
        return None
    rdp = velocity_to_rdp(v_fit)
    lo, hi = min(rdp_bounds), max(rdp_bounds)
    if not (lo <= rdp <= hi):
        return None
    return float(v_fit), float(rdp)
