"""
phct_eos.py
===========
Simplified Perturbed Hard-Chain Theory (PHCT) – like equation of state.

Provides:
    calculate_density()            – liquid density from T, P, MW
    calculate_fugacity_coefficients() – ln(φ_i) for each pseudo-component
    chemical_potential()           – residual chemical potential μ_i^r / RT

These are engineering-quality placeholder correlations calibrated to give
physically sensible values for heavy petroleum fractions + light alkane solvents.

Reference frame: Paper 2 (García Cárdenas & Ancheyta, IECR 2022) – PHCT EOS
framework adapted for continuous thermodynamics in SDA simulation.
"""

import numpy as np
from typing import Union, List


# ---------------------------------------------------------------------------
# Solvent physical constants
# ---------------------------------------------------------------------------

SOLVENT_DATA = {
    'propane': {
        'MW':    44.10,
        'Tc':   369.83,   # K
        'Pc':    42.48e5, # Pa
        'omega': 0.153,
        'delta': 13.1,    # Hildebrand solubility parameter at 25°C, MPa^0.5
    },
    'butane': {
        'MW':    58.12,
        'Tc':   425.12,   # K
        'Pc':    37.96e5, # Pa
        'omega': 0.200,
        'delta': 14.1,
    },
}


# ---------------------------------------------------------------------------
# PHCT parameter correlations (MW-based, from Paper 2 Table 2 framework)
# ---------------------------------------------------------------------------

def _phct_params(MW: float) -> dict:
    """
    Return PHCT segment parameters for a pseudo-component of given MW.

    Correlations fitted to reproduce Paper 2 approach:
        c   – external degrees of freedom  ~  1 + 0.01*MW
        r   – segment number              ~  MW / 40
        v*  – segment volume  [cm³/mol]   ~  12 + 0.02*MW
        u/k – segment energy  [K]         ~  300 + 0.12*MW
    """
    c    = 1.0  + 0.010 * MW
    r    = MW   / 40.0
    v_st = 12.0 + 0.020 * MW          # cm³/mol
    u_k  = 300.0 + 0.120 * MW         # K
    return {'c': c, 'r': r, 'v_star': v_st, 'u_k': u_k}


def _solvent_phct_params(solvent_name: str) -> dict:
    """Return PHCT parameters for propane or butane."""
    MW = SOLVENT_DATA[solvent_name]['MW']
    return _phct_params(MW)


# ---------------------------------------------------------------------------
# Propane density correlation (NIST-fitted)
# ---------------------------------------------------------------------------

def propane_density(T_C: float, P_bar: float = 40.0) -> float:
    """
    Compressed liquid propane density [g/cm³].

    Fitted to NIST Webbook data, 35–45 bar, 40–100°C.

    NIST targets @ 40 bar:
        50°C → 0.52,  65°C → 0.47,  75°C → 0.44,
        85°C → 0.39,  90°C → 0.35

    Quadratic polynomial calibrated at 40 bar:
        rho_40 = 0.4175 + 0.00555·T - 7×10⁻⁵·T²

    Pressure correction: +0.002 g/cm³ per bar above 40 bar.
    """
    rho_40 = 0.4175 + 0.00555 * T_C - 7.0e-5 * T_C ** 2
    P_corr = 0.002 * (P_bar - 40.0)
    return float(np.clip(rho_40 + P_corr, 0.20, 0.65))


# ---------------------------------------------------------------------------
# Density calculation
# ---------------------------------------------------------------------------

def calculate_density(
    MW:           float,
    T:            float,
    P:            float,
    is_solvent:   bool  = False,
    solvent_name: str   = 'propane',
) -> float:
    """
    Estimate liquid density [g/cm³] using a modified Rackett equation.

    For residue pseudo-components:
        ρ(T,P) = ρ_20°C * [1 − κ_T*(T − 293) + κ_P*(P − 1e5)]

    For solvents at elevated SDA pressure, uses NIST-correlated polynomial.

    Parameters
    ----------
    MW          : molecular weight [g/mol]
    T           : temperature [K]
    P           : pressure [Pa]
    is_solvent  : True for propane/butane
    solvent_name: 'propane' or 'butane'

    Returns
    -------
    density : float [g/cm³]
    """
    if is_solvent:
        if solvent_name == 'propane':
            P_bar = P / 1.0e5
            return propane_density(T - 273.15, P_bar)
        # Butane: simple table at ~35 bar
        rho_table_butane = {403: 0.492, 413: 0.467, 423: 0.434, 433: 0.383}
        T_pts  = sorted(rho_table_butane.keys())
        rho_pts= [rho_table_butane[t] for t in T_pts]
        return float(np.interp(T, T_pts, rho_pts, left=rho_pts[0], right=rho_pts[-1]))
    else:
        # Base density at 20°C from correlation
        rho_20 = 0.72 + 0.25 * np.tanh((MW - 300.0) / 400.0)
        # Thermal expansion: κ_T ≈ 6e-4 K⁻¹ for heavy oil
        kappa_T = 6.0e-4
        # Pressure compressibility: κ_P ≈ 5e-10 Pa⁻¹
        kappa_P = 5.0e-10
        rho = rho_20 * (1.0 - kappa_T * (T - 293.15) + kappa_P * (P - 1.0e5))
        return max(float(rho), 0.30)


# ---------------------------------------------------------------------------
# Fugacity coefficients (ln φ_i)
# ---------------------------------------------------------------------------

def calculate_fugacity_coefficients(
    components:   list,          # list of PseudoComponent
    x:            np.ndarray,    # mole fractions (same order as components)
    T:            float,         # K
    P:            float,         # Pa
    solvent_name: str = 'propane',
    solvent_z:    float = 0.0,   # mole fraction of solvent in this phase
) -> np.ndarray:
    """
    Compute ln(φ_i) for each residue pseudo-component in a given phase.

    Uses a simplified PHCT framework:
        ln(φ_i) = ln(φ_i^HS) + ln(φ_i^chain) + ln(φ_i^dispersion)

    Hard-sphere term:    Carnahan-Starling
    Chain term:          Flory-entropy of mixing (volume-fraction based)
    Dispersion term:     Regular-solution theory  δ_i vs δ_mix

    Parameters
    ----------
    components  : list of PseudoComponent objects
    x           : mole fraction array for residue components  [n]
    T           : temperature [K]
    P           : pressure [Pa]
    solvent_name: 'propane' or 'butane'
    solvent_z   : mole fraction of solvent in this phase

    Returns
    -------
    ln_phi : np.ndarray  [n]  – one value per component
    """
    n     = len(components)
    MW_arr= np.array([c.MW    for c in components])
    d_arr = np.array([c.delta for c in components])   # solubility parameters

    # Volume fractions (proportional to MW/density)
    rho_arr = np.array([calculate_density(c.MW, T, P) for c in components])
    V_arr   = MW_arr / rho_arr     # molar volume cm³/mol

    # Mixture volume fraction of solvent
    solv  = SOLVENT_DATA[solvent_name]
    rho_s = calculate_density(solv['MW'], T, P, is_solvent=True, solvent_name=solvent_name)
    V_s   = solv['MW'] / rho_s

    # Volume fractions in the phase (include solvent)
    x_safe = np.maximum(x, 1e-20)
    vol_nums = x_safe * V_arr
    vol_total= vol_nums.sum() + solvent_z * V_s
    phi_vol  = vol_nums / max(vol_total, 1e-20)   # volume fractions of residue comps

    # Volume fraction of solvent
    phi_s = solvent_z * V_s / max(vol_total, 1e-20)

    # Phase-averaged solubility parameter
    delta_s   = solv['delta']
    delta_mix = (phi_vol * d_arr).sum() + phi_s * delta_s

    # ---- Hard-sphere contribution (Carnahan-Starling) ----
    # Packing fraction η ∝ rho * V; simplified to a constant for heavy oil
    eta = 0.40   # representative packing fraction for dense liquid
    f_hs = (8*eta - 9*eta**2 + 3*eta**3) / (1 - eta)**3

    # Per-component hard-sphere ln(φ) scales with segment number r_i
    r_arr = MW_arr / 40.0
    ln_phi_hs = r_arr * f_hs / max(r_arr.mean(), 1.0)

    # ---- Chain (Flory) contribution ----
    # ln(φ_i^chain) = ln(φ_vol_i / x_i) + 1 - V_i / V_mix_avg
    V_mix_avg = (phi_vol * V_arr).sum() + phi_s * V_s
    V_mix_avg = max(V_mix_avg, 1.0)
    ln_phi_chain = np.log(np.maximum(phi_vol / x_safe, 1e-20)) + 1.0 - V_arr / V_mix_avg

    # ---- Dispersion / Regular-solution contribution ----
    # ln(φ_i^disp) = V_i * (δ_i − δ_mix)² / RT
    # R = 8.314e-3 kJ/mol, δ in MPa^0.5 → units: cm³/mol * MPa / (kJ/mol) = 1 (since 1 MPa*cm³=1 J=1e-3 kJ)
    R_kJ = 8.314e-3   # kJ/(mol·K)
    ln_phi_disp = V_arr * (d_arr - delta_mix) ** 2 / (R_kJ * T * 1000.0)
    # Note: (MPa^0.5)^2 = MPa; V[cm³/mol]*MPa = 1 J/mol = 1e-3 kJ/mol → divide by RT[kJ/mol]
    ln_phi_disp = V_arr * (d_arr - delta_mix) ** 2 / (R_kJ * T * 1000.0)

    ln_phi = ln_phi_hs + ln_phi_chain + ln_phi_disp
    return ln_phi.astype(float)


# ---------------------------------------------------------------------------
# Chemical potential
# ---------------------------------------------------------------------------

def chemical_potential(
    component,                # single PseudoComponent
    x:           float,       # mole fraction of this component in the phase
    x_all:       np.ndarray,  # full mole fraction array (all residue comps)
    components:  list,        # all PseudoComponent objects
    T:           float,
    P:           float,
    solvent_name:str   = 'propane',
    solvent_z:   float = 0.0,
) -> float:
    """
    Return dimensionless residual chemical potential  μ_i^r / RT  for component i.

    μ_i / RT = ln(x_i) + ln(φ_i)

    Parameters
    ----------
    component   : the specific PseudoComponent
    x           : mole fraction of this component
    x_all       : mole fractions of all residue components in this phase
    components  : list of all PseudoComponent objects
    T, P        : temperature [K], pressure [Pa]
    solvent_name, solvent_z : solvent identity and mole fraction

    Returns
    -------
    mu_over_RT : float
    """
    ln_phi_all = calculate_fugacity_coefficients(
        components, x_all, T, P, solvent_name, solvent_z
    )
    i = component.index
    mu = np.log(max(x, 1e-20)) + ln_phi_all[i]
    return float(mu)


if __name__ == '__main__':
    from residue_distribution import build_residue_distribution
    comps = build_residue_distribution(n_comp=5)
    x = np.array([c.z for c in comps])
    x /= x.sum()
    ln_phi = calculate_fugacity_coefficients(comps, x, T=413.0, P=35e5,
                                              solvent_name='butane', solvent_z=0.7)
    print("ln(φ) for 5-component system:")
    for c, lp in zip(comps, ln_phi):
        print(f"  MW={c.MW:6.0f}  ln_phi={lp:+.3f}")
