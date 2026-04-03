"""
downstream_corrections.py
=========================
Post-extractor corrections for DAO quality prediction.

These adjustments account for real-plant phenomena that occur AFTER
the extraction column but BEFORE the final DAO product measurement:

1. Flash precipitation: dissolved asphaltenes precipitate when pressure
   drops in the DAO flash drum (supercritical -> subcritical transition)
2. Residual solvent: trace propane in DAO affects measured density/viscosity
3. Solvent purity: industrial propane contains ethane/butane impurities
   that affect solvent power

RULES:
- These corrections are applied AFTER run_extractor(), never inside it
- They are OPTIONAL (toggled by user in UI)
- They modify DAO yield, density, viscosity, CCR only slightly (1-3%)
"""

import numpy as np


def flash_precipitation_adjustment(
    DAO_yield_pct: float,
    density_DAO: float,
    T_flash_C: float = 180.0,
    P_flash_bar: float = 5.0,
    severity: float = 1.0,
) -> dict:
    """
    Adjust DAO properties for flash drum precipitation.

    When DAO-solvent mixture flashes from column pressure (~40 bar)
    to flash drum pressure (~5 bar), dissolved heavy resins and
    borderline asphaltenes can precipitate out, slightly reducing
    DAO yield and improving quality.

    Parameters
    ----------
    DAO_yield_pct : DAO yield from extractor [wt%]
    density_DAO : DAO density from extractor [g/cm3]
    T_flash_C : flash drum temperature [degC]
    P_flash_bar : flash drum pressure [bar]
    severity : multiplier for correction magnitude (0=off, 1=default)

    Returns
    -------
    dict with corrected values and adjustment magnitudes
    """
    if severity <= 0:
        return {'DAO_yield_adj': DAO_yield_pct, 'density_adj': density_DAO,
                'yield_change': 0.0, 'density_change': 0.0,
                'note': 'Flash correction disabled'}

    # Higher pressure drop -> more precipitation
    dP = max(40.0 - P_flash_bar, 0) / 40.0  # normalised pressure drop

    # Flash precipitation reduces yield by 0.5-2% and improves density
    yield_loss = 0.5 + 1.5 * dP * severity   # wt% absolute
    density_improvement = 0.002 * dP * severity  # g/cm3

    return {
        'DAO_yield_adj': DAO_yield_pct - yield_loss,
        'density_adj': density_DAO - density_improvement,
        'yield_change': -yield_loss,
        'density_change': -density_improvement,
        'note': (f'Flash precipitation at {P_flash_bar:.0f} bar: '
                 f'yield -{yield_loss:.1f}%, density {density_improvement:.4f} lighter'),
    }


def residual_solvent_correction(
    density_DAO: float,
    viscosity_DAO: float,
    residual_solvent_wt_pct: float = 0.5,
) -> dict:
    """
    Correct for trace propane remaining in DAO product.

    Even after solvent recovery, 0.1-1.0 wt% propane remains dissolved.
    This slightly reduces measured density and viscosity.

    Parameters
    ----------
    density_DAO : DAO density [g/cm3]
    viscosity_DAO : DAO viscosity [cSt]
    residual_solvent_wt_pct : propane remaining in DAO [wt%]

    Returns
    -------
    dict with corrected values
    """
    if residual_solvent_wt_pct <= 0:
        return {'density_adj': density_DAO, 'viscosity_adj': viscosity_DAO,
                'density_change': 0.0, 'viscosity_change': 0.0,
                'note': 'Residual solvent correction disabled'}

    frac = residual_solvent_wt_pct / 100.0
    # Propane density ~0.50 g/cm3 vs DAO ~0.92 -> dilution effect
    density_adj = density_DAO * (1.0 - frac) + 0.50 * frac
    # Viscosity: propane acts as diluent -> reduces viscosity
    viscosity_adj = viscosity_DAO * (1.0 - 0.8 * frac)

    return {
        'density_adj': round(density_adj, 4),
        'viscosity_adj': round(viscosity_adj, 1),
        'density_change': round(density_adj - density_DAO, 4),
        'viscosity_change': round(viscosity_adj - viscosity_DAO, 1),
        'note': (f'Residual C3 ({residual_solvent_wt_pct:.1f}%): '
                 f'density {density_adj - density_DAO:+.4f}, '
                 f'visc {viscosity_adj - viscosity_DAO:+.1f} cSt'),
    }


def solvent_purity_adjustment(
    K_multiplier: float,
    ethane_pct: float = 2.0,
    butane_pct: float = 3.0,
) -> dict:
    """
    Adjust effective K-multiplier for impure propane solvent.

    Industrial propane typically contains 1-5% ethane and 2-8% butane.
    Ethane (lighter) weakens solvent power; butane (heavier) strengthens it.
    Net effect depends on the balance.

    Parameters
    ----------
    K_multiplier : base K_multiplier
    ethane_pct : ethane content in propane [vol%]
    butane_pct : butane content in propane [vol%]

    Returns
    -------
    dict with adjusted K_multiplier
    """
    # Ethane weakens (lower density), butane strengthens (higher density)
    ethane_effect = -0.01 * ethane_pct   # each %ethane reduces K by 1%
    butane_effect = +0.015 * butane_pct  # each %butane increases K by 1.5%
    net_factor = 1.0 + ethane_effect + butane_effect

    K_adj = K_multiplier * net_factor

    return {
        'K_multiplier_adj': round(K_adj, 4),
        'net_factor': round(net_factor, 4),
        'note': (f'Solvent purity ({ethane_pct}% C2, {butane_pct}% C4): '
                 f'K_mult {K_multiplier:.3f} -> {K_adj:.3f}'),
    }


def apply_all_downstream(
    sim_result: dict,
    enable_flash: bool = True,
    enable_residual: bool = True,
    flash_P_bar: float = 5.0,
    residual_solvent_pct: float = 0.5,
) -> dict:
    """
    Apply all enabled downstream corrections to simulation results.
    Returns a copy of sim_result with corrections applied.
    """
    result = dict(sim_result)
    corrections = []

    if enable_flash:
        fc = flash_precipitation_adjustment(
            result.get('DAO_yield_net', result.get('DAO_yield', 0)),
            result.get('density_DAO', 0.92),
            P_flash_bar=flash_P_bar)
        # Update whichever yield key is present
        if 'DAO_yield_net' in result:
            result['DAO_yield_net'] = fc['DAO_yield_adj']
        if 'DAO_yield' in result:
            result['DAO_yield'] = fc['DAO_yield_adj']
        result['density_DAO'] = fc['density_adj']
        corrections.append(fc['note'])

    if enable_residual:
        rc = residual_solvent_correction(
            result.get('density_DAO', 0.92),
            result.get('viscosity_dao_cSt', result.get('viscosity_cSt', 33.0)),
            residual_solvent_pct)
        result['density_DAO'] = rc['density_adj']
        if 'viscosity_dao_cSt' in result:
            result['viscosity_dao_cSt'] = rc['viscosity_adj']
        if 'viscosity_cSt' in result:
            result['viscosity_cSt'] = rc['viscosity_adj']
        corrections.append(rc['note'])

    result['downstream_corrections'] = corrections
    return result
