"""
quality_model.py
================
DAO quality predictions for HPCL PDA Unit Simulator.

Provides two engineering correlations:
    predict_dao_viscosity()  – kinematic viscosity [cSt] at 100°C
    predict_astm_colour()    – ASTM colour number (0.5–8.0 scale)

Both are calibrated to design manual targets:
    Lube DAO  (K_mult=0.65, MW~470, rho=0.921)  →  ~33 cSt,  colour < 3
    FCC  DAO  (K_mult=1.05, MW~500, rho=0.936)  →  ~45 cSt,  colour < 5
"""

import numpy as np
from math import log10


def predict_dao_viscosity(
    MW_dao:      float,
    density_dao: float,
    SARA_dao:    dict,
    T_eval_C:    float = 100.0,
) -> float:
    """
    Predict DAO kinematic viscosity [cSt] at T_eval_C using a modified
    Walther-type correlation.

    Calibrated targets:
        Lube DAO  MW~470, rho=0.921, aromatic~54%  →  ~33 cSt @100°C
        FCC  DAO  MW~500, rho=0.936, aromatic~61%  →  ~45 cSt @100°C

    Parameters
    ----------
    MW_dao      : number-average MW of DAO [g/mol]
    density_dao : liquid density of DAO [g/cm³]
    SARA_dao    : dict with wt% of each SARA class in DAO
                  keys: 'saturates', 'aromatics', 'resins', 'asphaltenes'
    T_eval_C    : evaluation temperature [°C], default 100

    Returns
    -------
    viscosity : float [cSt]
    """
    f_aro = SARA_dao.get('aromatics', 61.0) / 100.0

    # Base log-viscosity at 100°C (Walther-type, MW and density dependent).
    # Recalibrated to actual simulator DAO outputs:
    #   Lube DAO  MW~594, rho=0.921, aro=57%  ->  33 cSt  (log10=1.519)
    #   FCC  DAO  MW~616, rho=0.931, aro=63%  ->  45 cSt  (log10=1.653)
    log_visc_100 = (
        -0.70
        + 0.0029 * MW_dao
        + 0.80   * f_aro
        + 2.00   * (density_dao - 0.90)
    )

    # Temperature shift below/above 100°C (Walther slope ≈ -3.5 per log decade K)
    if abs(T_eval_C - 100.0) > 0.5:
        T_ref_K  = 373.15
        T_eval_K = T_eval_C + 273.15
        # log(log(v+0.7)) shifts linearly with log(T): slope ≈ -3.5
        # Simplified: Δlog_visc ≈ -3.5 * log10(T_eval / T_ref)
        log_visc = log_visc_100 - 3.5 * log10(T_eval_K / T_ref_K)
    else:
        log_visc = log_visc_100

    visc_cSt = 10.0 ** log_visc
    return float(max(visc_cSt, 1.0))


def predict_astm_colour(
    asphaltene_in_dao_wt_pct: float,
    SARA_dao:                 dict,
) -> float:
    """
    Predict ASTM colour number for DAO product (scale 0.5–8.0).

    Colour is driven by:
      1. Resin content (base colour, brown tint even without asphaltenes)
      2. Asphaltene contamination (logarithmic penalty; < 100 wppm → colour < 6)

    Calibration:
        Typical lube DAO  (~10 wt% resins, <100 wppm asphaltenes)  →  colour ~2–3
        FCC  DAO           (~18 wt% resins, ~200 wppm asphaltenes)  →  colour ~4–5

    Parameters
    ----------
    asphaltene_in_dao_wt_pct : asphaltene contamination in DAO [wt%]
    SARA_dao                  : dict with wt% of each SARA class in DAO

    Returns
    -------
    colour : float (0.5 = water-white, 8.0 = very dark)
    """
    f_res     = SARA_dao.get('resins', 10.0) / 100.0
    asph_wppm = asphaltene_in_dao_wt_pct * 10_000.0   # wt% → wppm

    base_colour  = 1.5 + 4.0 * f_res
    asph_penalty = 1.5 * log10(1.0 + asph_wppm / 20.0)

    colour = base_colour + asph_penalty
    return float(min(max(colour, 0.5), 8.0))
