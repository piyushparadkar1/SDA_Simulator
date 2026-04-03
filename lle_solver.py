"""
lle_solver.py
=============
LLE solver for HPCL PDA Unit (Plant No. 41).

K-values calibrated per SARA class to reproduce plant-range DAO yields:
    Propane (~75°C, S/O=8): DAO ~ 28–38 wt%  (Basra-Kuwait mix)
    Butane  (~140°C, S/O=8): DAO ~ 42–55 wt%

Physical basis:
    K_i = w_i^DAO / w_i^asphalt  (mass-fraction basis, residue only)
    Solvent pre-assigned to Phase I — residue-only Rachford-Rice.

SARA K-value hierarchy at T_ref:
    Saturates:   K >> 1   (propane is paraffinic; strong affinity)
    Aromatics:   K ~ 1–5  (partial extraction; lighter aromatics dissolve)
    Resins:      K ~ 0.1–0.8  (mostly rejected; light resins partially extracted)
    Asphaltenes: K < 0.05  (essentially all to asphalt)
"""

import numpy as np
from scipy.optimize import brentq
from typing import List
from residue_distribution import PseudoComponent

# ── Calibrated K-value parameters (SARA-class, MW-dependent) ─────────────────
# ln(K) = a + b*ln(MW) + c_T*(T - T_ref) + d_solvent
# Re-anchored to plant operating reality (2026-03): 14 vol% DAO at S/O=9, T=82°C top, K_mult=1.0
# Old design targets (28–35%) no longer match current plant; base shifted by ln(K_ref=-0.4387).
# sat MW=400: K≈5  |  aro MW=550: K≈2.3  |  res MW=900: K≈0.17  |  asp MW=1600: K≈0.001

_K_PARAMS = {
    'saturates':   {'a':  9.4293, 'b': -1.30},
    'aromatics':   {'a':  9.0173, 'b': -1.30},
    'resins':      {'a':  8.5976, 'b': -1.50},
    'asphaltenes': {'a':  6.372,  'b': -1.80},
}

_D_SOLV = {'propane': 0.0, 'butane': +0.55}       # butane: ~1.7x stronger than propane
_ALPHA_DENSITY = 3.0   # FIXED — do not make this a free calibration parameter.
                        # Floated to 1.59 in v1 calibration and destroyed live-sim yield.
                        # Physical value for propane SDA: 3.0 (fits NIST rho curve slope).
_RHO_REF = {'propane': 0.44, 'butane': 0.467}    # reference density at Tref, 40 bar


def K_value(component: PseudoComponent, T: float, solvent_name: str,
            solvent_ratio: float = 8.0,
            K_multiplier: float = 1.0,
            delta_crit: float = 2.5,
            P_bar: float = 40.0,
            alpha_density: float = 3.0,
            c_t_params: dict = None) -> float:
    """
    Compute K_i = w_i^DAO / w_i^asphalt for one pseudo-component.

    Uses SARA-class parameters + MW dependence + density-based solvent power
    correction (physically correct T AND P dependence via propane density).

    Calibration hooks
    -----------------
    K_multiplier : float, default 1.0
        Uniform multiplier applied to all K-values after base calculation.
        Values > 1 increase DAO yield; < 1 decrease it.
    delta_crit : float, default 2.5
        Solubility parameter spread modifier. Controls the resin-to-asphaltene
        K-ratio. Higher delta_crit → more resins partition to asphalt phase →
        lower DAO CCR/contamination but lower DAO yield.
        Implemented as a selective penalty on resins and aromatics K-values:
            K_resins   *= exp(-0.10 * (delta_crit - 2.5))
            K_aromatics *= exp(-0.03 * (delta_crit - 2.5))
    P_bar : float, default 40.0
        Operating pressure in bar. Affects propane density → solvent power.
        Higher pressure → denser propane → higher K-values.
    """
    from phct_eos import propane_density

    sara = getattr(component, 'sara_class', 'aromatics')
    p    = _K_PARAMS.get(sara, _K_PARAMS['aromatics'])
    MW   = max(component.MW, 50.0)

    # Power-law S/O correction: higher S/O increases K (dilution drives more extraction)
    so_factor = (max(solvent_ratio, 0.5) / 8.0) ** 0.50

    # Base K — C_T temperature term removed; T/P handled via density below
    lnK = (p['a']
           + p['b']  * np.log(MW)
           + _D_SOLV.get(solvent_name, 0.0))

    K_base = float(np.exp(np.clip(lnK, -9, 5)) * so_factor)

    # Path B: direct temperature sensitivity term (optional)
    # lnK += C_T * (T_C - T_ref_C)  where T_ref_C = 75.0°C
    # C_T < 0: higher T → lower K (propane solvent power falls near critical point)
    # Only applied when c_t_params dict is passed; default None = Path A behaviour.
    T_C = T - 273.15
    if c_t_params is not None:
        T_ref_C = 75.0
        c_t = float(c_t_params.get(sara, 0.0))
        K_base *= float(np.exp(np.clip(c_t * (T_C - T_ref_C), -3.0, 3.0)))

    # Density-based solvent power correction (physically correct T AND P dependence)
    # Reference: 75°C, 40 bar (ROSE extractor design point)
    _alpha = float(alpha_density)
    rho_prop = propane_density(T_C, P_bar)
    rho_ref  = _RHO_REF.get(solvent_name, 0.44)
    density_power = (rho_prop / max(rho_ref, 0.20)) ** _alpha

    # Apply calibration: uniform K multiplier (primary yield control)
    K_cal = K_base * K_multiplier * density_power

    # Apply calibration: delta_crit shifts resin/aromatic partitioning
    # Higher delta_crit → resins less soluble in DAO → cleaner DAO (lower CCR)
    d_shift = delta_crit - 2.5   # deviation from default
    if sara == 'resins':
        K_cal *= np.exp(-0.10 * d_shift)
    elif sara == 'aromatics':
        K_cal *= np.exp(-0.03 * d_shift)
    elif sara == 'asphaltenes':
        K_cal *= np.exp(-0.05 * d_shift)

    return float(max(K_cal, 1e-12))


# ── Rachford-Rice ─────────────────────────────────────────────────────────────

def _rr(psi: float, w: np.ndarray, K: np.ndarray) -> float:
    return float(np.sum(w * (K - 1.0) / (1.0 + psi * (K - 1.0))))


def solve_lle(
    components:    List[PseudoComponent],
    T:             float,
    P:             float,
    solvent_name:  str,
    solvent_ratio: float,
    feed_mass:     np.ndarray,
    K_multiplier:  float = 1.0,
    delta_crit:    float = 2.5,
    alpha_density: float = 3.0,
    c_t_params:    dict  = None,
) -> dict:
    """
    Single-stage Rachford-Rice LLE (residue-only, mass basis).

    Solvent is pre-assigned to Phase I (DAO-rich). Only residue
    pseudo-components participate in the flash calculation.

    Calibration parameters K_multiplier and delta_crit are forwarded
    to K_value() — see that function for physical interpretation.

    Returns dict with psi, DAO_yield, K_values, w_I, w_II,
    mass_I, mass_II, converged, precip_yield.
    """
    n          = len(components)
    total_mass = feed_mass.sum()
    if total_mass < 1e-14:
        return _empty(n)

    w = feed_mass / total_mass
    P_bar = P / 1e5   # P is in Pa; K_value needs bar
    K = np.array([K_value(c, T, solvent_name, solvent_ratio,
                          K_multiplier, delta_crit, P_bar, alpha_density,
                          c_t_params=c_t_params)
                  for c in components])

    K_max, K_min = K.max(), K.min()

    if K_max <= 1.0:
        psi, converged = 0.01, False   # feed prefers Phase II
    elif K_min >= 1.0:
        psi, converged = 0.99, False   # feed prefers Phase I
    else:
        rr0 = _rr(1e-6, w, K)
        if rr0 <= 0:
            # No two-phase split in [0,1] — all goes to Phase II
            psi, converged = 0.01, False
        else:
            try:
                psi       = brentq(_rr, 1e-6, 1 - 1e-6, args=(w, K), xtol=1e-10)
                converged = True
            except Exception:
                psi, converged = 0.50, False

    denom = np.where(np.abs(1.0 + psi * (K - 1.0)) < 1e-15,
                     1e-15, 1.0 + psi * (K - 1.0))

    w_I  = np.maximum(w * K / denom, 1e-20);  w_I  /= w_I.sum()
    w_II = np.maximum(w   / denom,   1e-20);  w_II /= w_II.sum()

    mass_I   = psi         * total_mass * w_I
    mass_II  = (1.0 - psi) * total_mass * w_II

    precip   = np.array([c.precipitable for c in components])
    p_feed   = feed_mass[precip].sum()
    p_II     = mass_II[precip].sum()
    precip_yield = p_II / p_feed * 100.0 if p_feed > 1e-14 else 0.0

    return {
        'psi':          float(psi),
        'DAO_yield':    float(psi * 100.0),
        'K_values':     K,
        'w_I':          w_I,
        'w_II':         w_II,
        'mass_I':       mass_I,
        'mass_II':      mass_II,
        'converged':    converged,
        'precip_yield': float(precip_yield),
    }


def _empty(n: int) -> dict:
    return {
        'psi': 0.0, 'DAO_yield': 0.0, 'K_values': np.ones(n),
        'w_I': np.ones(n)/n, 'w_II': np.ones(n)/n,
        'mass_I': np.zeros(n), 'mass_II': np.zeros(n),
        'converged': True, 'precip_yield': 0.0,
    }


def solve_lle_phct(
    components:    List[PseudoComponent],
    T:             float,
    P:             float,
    solvent_name:  str,
    solvent_ratio: float,
    feed_mass:     np.ndarray,
    K_multiplier:  float = 1.0,
    delta_crit:    float = 2.5,
    alpha_density: float = 3.0,
    max_iter:      int   = 30,
    tol:           float = 1e-4,
    c_t_params:    dict  = None,
) -> dict:
    """
    LLE flash using PHCT fugacity coefficients.

    Method: Successive substitution on K-values derived from fugacity ratios.
    Uses K-value model as initial guess, then refines using PHCT EOS.
    K_i^(n+1) = phi_i^II(n) / phi_i^I(n)
    """
    try:
        from phct_eos import calculate_fugacity_coefficients
    except ImportError:
        # Fall back to standard K-value solver if PHCT not available
        return solve_lle(components, T, P, solvent_name, solvent_ratio, feed_mass,
                         K_multiplier, delta_crit, alpha_density,
                         c_t_params=c_t_params)

    n          = len(components)
    total_mass = feed_mass.sum()
    if total_mass < 1e-14:
        return _empty(n)

    P_bar = P / 1e5
    w = feed_mass / total_mass

    # Initial K from K-value model
    K = np.array([K_value(c, T, solvent_name, solvent_ratio,
                          K_multiplier, delta_crit, P_bar, alpha_density,
                          c_t_params=c_t_params)
                  for c in components])

    # Successive substitution
    for iteration in range(max_iter):
        K_max, K_min = K.max(), K.min()
        if K_max <= 1.0:
            psi = 0.01
        elif K_min >= 1.0:
            psi = 0.99
        else:
            rr0 = _rr(1e-6, w, K)
            if rr0 <= 0:
                psi = 0.01
            else:
                try:
                    psi = brentq(_rr, 1e-6, 1 - 1e-6, args=(w, K), xtol=1e-8)
                except Exception:
                    psi = 0.50

        denom = np.where(np.abs(1.0 + psi * (K - 1.0)) < 1e-15,
                         1e-15, 1.0 + psi * (K - 1.0))
        w_I  = np.maximum(w * K / denom, 1e-20); w_I  /= w_I.sum()
        w_II = np.maximum(w   / denom,   1e-20); w_II /= w_II.sum()

        # Get mole fractions for fugacity calculation
        # Simple approximation: use MW-weighted conversion
        MW_arr = np.array([c.MW for c in components])

        def mass_to_mole_frac(w_mass):
            x = w_mass / np.maximum(MW_arr, 1.0)
            s = x.sum()
            return x / s if s > 1e-14 else np.ones(len(x)) / len(x)

        x_I  = mass_to_mole_frac(w_I)
        x_II = mass_to_mole_frac(w_II)

        try:
            ln_phi_I  = calculate_fugacity_coefficients(
                components, x_I,  T, P, solvent_name, solvent_z=max(psi, 0.01))
            ln_phi_II = calculate_fugacity_coefficients(
                components, x_II, T, P, solvent_name, solvent_z=max(1-psi, 0.01))

            K_new = np.exp(np.clip(ln_phi_II - ln_phi_I, -8, 8))
            K_new = np.clip(K_new, 1e-8, 1e6)
        except Exception:
            # Fall back if PHCT fails
            break

        if np.max(np.abs(K_new / np.maximum(K, 1e-12) - 1.0)) < tol and iteration > 0:
            K = K_new
            break

        # Damped update for stability
        K = 0.6 * K_new + 0.4 * K

    # Final flash with converged K
    K_max, K_min = K.max(), K.min()
    if K_max <= 1.0:
        psi, converged = 0.01, False
    elif K_min >= 1.0:
        psi, converged = 0.99, False
    else:
        rr0 = _rr(1e-6, w, K)
        if rr0 <= 0:
            psi, converged = 0.01, False
        else:
            try:
                psi       = brentq(_rr, 1e-6, 1 - 1e-6, args=(w, K), xtol=1e-10)
                converged = True
            except Exception:
                psi, converged = 0.50, False

    denom = np.where(np.abs(1.0 + psi * (K - 1.0)) < 1e-15,
                     1e-15, 1.0 + psi * (K - 1.0))
    w_I  = np.maximum(w * K / denom, 1e-20); w_I  /= w_I.sum()
    w_II = np.maximum(w   / denom,   1e-20); w_II /= w_II.sum()

    mass_I   = psi         * total_mass * w_I
    mass_II  = (1.0 - psi) * total_mass * w_II

    precip   = np.array([c.precipitable for c in components])
    p_feed   = feed_mass[precip].sum()
    p_II     = mass_II[precip].sum()
    precip_yield = p_II / p_feed * 100.0 if p_feed > 1e-14 else 0.0

    return {
        'psi':          float(psi),
        'DAO_yield':    float(psi * 100.0),
        'K_values':     K,
        'w_I':          w_I,
        'w_II':         w_II,
        'mass_I':       mass_I,
        'mass_II':      mass_II,
        'converged':    converged,
        'precip_yield': float(precip_yield),
        'thermo_mode':  'phct',
    }


if __name__ == '__main__':
    from residue_distribution import build_residue_distribution

    for feed_name in ['basra_kuwait_mix', 'basra_light']:
        for solvent in ['propane', 'butane']:
            T = 348.15 if solvent == 'propane' else 413.15
            comps  = build_residue_distribution(feed_name=feed_name, n_comp=20,
                                                solvent_name=solvent)
            masses = np.array([c.z * c.MW for c in comps])
            r = solve_lle(comps, T=T, P=35e5, solvent_name=solvent,
                          solvent_ratio=8.0, feed_mass=masses)
            print(f"[{feed_name}|{solvent}] DAO={r['DAO_yield']:.1f}%  "
                  f"precip_in_asp={r['precip_yield']:.1f}%  conv={r['converged']}")
            for sara in ['saturates','aromatics','resins','asphaltenes']:
                idxs = [i for i,c in enumerate(comps) if c.sara_class==sara]
                if idxs:
                    k_avg = r['K_values'][idxs].mean()
                    print(f"  {sara:12s}  K_avg={k_avg:.3f}")
