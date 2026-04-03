"""
hunter_nash_extractor.py
========================
Hunter-Nash countercurrent extraction column for HPCL PDA Unit.
Updated to use asphalt-into-DAO entrainment model.
"""

import numpy as np
from typing import List
from residue_distribution import PseudoComponent, build_residue_distribution
from lle_solver import solve_lle
from asphaltene_kinetics import KineticParams, apply_precipitation_kinetics, DEFAULT_KINETICS
from stage_efficiency import StageEfficiency, apply_stage_efficiency, DEFAULT_EFFICIENCY
from entrainment_model import EntrainmentParams, apply_entrainment, DEFAULT_ENTRAINMENT


def run_extractor(
    components:         List[PseudoComponent],
    solvent_name:       str,
    solvent_ratio:      float,
    N_stages:           int,
    T_profile:          list,
    P:                  float = 40e5,
    kinetics:           KineticParams     = DEFAULT_KINETICS,
    efficiency:         StageEfficiency   = DEFAULT_EFFICIENCY,
    entrainment:        EntrainmentParams = DEFAULT_ENTRAINMENT,
    K_multiplier:       float = 1.0,
    delta_crit:         float = 2.5,
    predilution_frac:   float = 0.0,
    alpha_density:      float = 3.0,
    thermo_mode:        str   = 'kvalue',
    feed_basis:         float = 1.0,
    max_outer_iter:     int   = 60,
    outer_tol:          float = 0.3,
    verbose:            bool  = False,
    c_t_params:         dict  = None,
) -> dict:
    """
    Run Hunter-Nash countercurrent SDA extractor.

    Phase I  (DAO-rich / solvent-rich) flows UPWARD:   stage 1 → N → exit top
    Phase II (asphalt-rich)            flows DOWNWARD:  stage N → 1 → exit bottom

    Asphalt-into-DAO entrainment is applied at each stage to track
    asphaltene contamination of the DAO product.
    """
    N_stages    = max(N_stages, 1)   # safety: prevent empty-list IndexError
    n_comp      = len(components)
    MW_arr      = np.array([c.MW for c in components])
    precip_mask = np.array([c.precipitable for c in components])
    total_res   = float(feed_basis)
    solvent_mass= solvent_ratio * total_res

    z_arr     = np.array([c.z for c in components])
    mass_feed = z_arr * MW_arr
    mass_feed *= total_res / mass_feed.sum()

    # Initialise inter-stage streams
    inter_I  = [mass_feed * (i + 1) / N_stages * 0.25       for i in range(N_stages)]
    inter_II = [mass_feed * (1.0 - (i + 1) / N_stages * 0.25) for i in range(N_stages)]
    A_prev   = [np.zeros(n_comp) for _ in range(N_stages)]

    prev_DAO_yield = 0.0
    stage_results  = []
    outer_iter     = 0

    for outer_iter in range(max_outer_iter):
        stage_results  = []
        new_inter_I    = [None] * N_stages
        new_inter_II   = [None] * N_stages

        for s in range(N_stages):
            T_s  = T_profile[s] if s < len(T_profile) else T_profile[-1]
            in_I = inter_I[s - 1] if s > 0 else np.zeros(n_comp)
            in_II= inter_II[s + 1] if s < N_stages - 1 else mass_feed

            # Per-stage S/O: support split-solvent injection (predilution_frac)
            if N_stages == 1 or predilution_frac <= 0.0:
                so_stage = float(solvent_ratio)
            else:
                frac_accumulated = ((1.0 - predilution_frac)
                                    + predilution_frac * (s / (N_stages - 1)))
                so_stage = float(solvent_ratio) * frac_accumulated

            merged = np.maximum(in_I + in_II, 0.0)

            # LLE equilibrium
            if thermo_mode == 'phct':
                from lle_solver import solve_lle_phct
                lle = solve_lle_phct(components, T_s, P, solvent_name,
                                     min(so_stage, 20.0), merged,
                                     K_multiplier=K_multiplier,
                                     delta_crit=delta_crit,
                                     alpha_density=alpha_density,
                                     c_t_params=c_t_params)
            else:
                lle = solve_lle(components, T_s, P, solvent_name,
                                min(so_stage, 20.0), merged,
                                K_multiplier=K_multiplier,
                                delta_crit=delta_crit,
                                alpha_density=alpha_density,
                                c_t_params=c_t_params)
            mass_I_eq  = lle['mass_I']
            mass_II_eq = lle['mass_II']

            # Precipitation kinetics
            A_eq      = mass_II_eq * precip_mask
            A_kinetic = apply_precipitation_kinetics(A_prev[s], A_eq, kinetics)
            delta     = A_kinetic - A_eq
            mass_I_kin  = np.maximum(mass_I_eq  - np.maximum(delta, 0.0), 0.0)
            mass_II_kin = np.maximum(mass_II_eq + np.maximum(delta, 0.0), 0.0)
            A_prev[s]   = A_kinetic.copy()

            # Murphree efficiency
            mass_I_eff, mass_II_eff = apply_stage_efficiency(
                mass_I_kin, mass_II_kin, in_I, efficiency)

            # Asphalt-into-DAO entrainment (REVISED)
            mass_I_fin, mass_II_fin, entrained = apply_entrainment(
                mass_I_eff, mass_II_eff, so_stage, entrainment, precip_mask)

            new_inter_I[s]  = mass_I_fin
            new_inter_II[s] = mass_II_fin

            # Asphaltene contamination in DAO at this stage
            asphal_in_dao = entrained[precip_mask].sum()
            total_dao     = mass_I_fin.sum()
            asphal_pct    = asphal_in_dao / max(total_dao, 1e-12) * 100.0

            stage_results.append({
                'stage':           s + 1,
                'T_C':             T_s - 273.15,
                'lle_yield':       lle['DAO_yield'],
                'precip_yield':    lle['precip_yield'],
                'psi':             lle['psi'],
                'converged':       lle['converged'],
                'entrained_kg':    float(entrained.sum()),
                'asphal_in_dao_pct': float(asphal_pct),
            })

        damp = 0.5
        for i in range(N_stages):
            inter_I[i]  = damp * new_inter_I[i]  + (1 - damp) * inter_I[i]
            inter_II[i] = damp * new_inter_II[i] + (1 - damp) * inter_II[i]

        dao_stream      = new_inter_I[N_stages - 1]
        DAO_yield_gross = dao_stream.sum() / total_res * 100.0

        if verbose:
            print(f"  Iter {outer_iter+1:3d}: DAO = {DAO_yield_gross:.2f}%")

        if abs(DAO_yield_gross - prev_DAO_yield) < outer_tol and outer_iter >= 3:
            break
        prev_DAO_yield = DAO_yield_gross

    mass_DAO     = new_inter_I[N_stages - 1]
    mass_asphalt = new_inter_II[0]

    total_products = mass_DAO.sum() + mass_asphalt.sum()
    if total_products > 1e-14:
        scale        = total_res / total_products
        mass_DAO    *= scale
        mass_asphalt*= scale

    DAO_yield_gross = mass_DAO.sum() / total_res * 100.0

    # Asphaltene contamination in final DAO product
    # Contamination = only asphaltene-CLASS components in DAO
    asp_class_mask = np.array([c.sara_class == "asphaltenes" for c in components])
    asphal_in_dao_kg = mass_DAO[asp_class_mask].sum()
    asphal_contam_pct = asphal_in_dao_kg / max(mass_DAO.sum(), 1e-14) * 100.0

    asphalt_yield   = mass_asphalt.sum() / total_res * 100.0
    DAO_yield_net   = DAO_yield_gross   # no separate post-correction needed

    def _mw_avg(masses):
        t = masses.sum()
        return float(np.dot(masses / t, MW_arr)) if t > 1e-14 else 0.0

    def _density_avg(masses):
        t = masses.sum()
        rho = np.array([c.density for c in components])
        return float(np.dot(masses / t, rho)) if t > 1e-14 else 0.0

    sara_classes = [c.sara_class for c in components]
    def _sara_in_stream(masses):
        t = masses.sum()
        if t < 1e-14:
            return {}
        d = {}
        for c, m in zip(components, masses):
            d[c.sara_class] = d.get(c.sara_class, 0.0) + m
        return {k: v / t * 100 for k, v in d.items()}

    return {
        'DAO_yield_gross':       float(DAO_yield_gross),
        'DAO_yield_net':         float(DAO_yield_net),
        'asphalt_yield':         float(asphalt_yield),
        'DAO_entrainment':       float(sum(s['entrained_kg'] for s in stage_results)),
        'asphal_contam_pct':     float(asphal_contam_pct),
        'MW_DAO_avg':            _mw_avg(mass_DAO),
        'MW_asphalt_avg':        _mw_avg(mass_asphalt),
        'density_DAO':           _density_avg(mass_DAO),
        'mass_DAO':              mass_DAO,
        'mass_asphalt':          mass_asphalt,
        'SARA_DAO':              _sara_in_stream(mass_DAO),
        'SARA_asphalt':          _sara_in_stream(mass_asphalt),
        'stage_results':         stage_results,
        'converged':             abs(DAO_yield_gross - prev_DAO_yield) < outer_tol,
        'outer_iterations':      outer_iter + 1,
        'components':            components,
        'MW_arr':                MW_arr,
    }
