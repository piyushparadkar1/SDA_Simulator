"""
simulator_bridge.py
===================
Bridges the plant_data_loader calibration dataset to the existing
physics simulation engine (run_extractor + predict_dao_viscosity).

Key responsibilities:
  1. Reconstruct feed PseudoComponents from measured lab properties
     (density, CCR, viscosity at 135°C).
  2. Build T-profile from calibrated thermal model or DCS measurements.
  3. Run train A and train B independently through run_extractor().
  4. Blend the two train outputs into a single plant-level prediction.
  5. Return predicted DAO yield, viscosity, density, and convergence flag.
"""

import logging
import math
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default propane pressure for PDA unit
_DEFAULT_P_PA = 40e5

# Fixed kinetics and stage count (HPCL PDA design)
_N_STAGES = 4
_DEFAULT_ALPHA_DENSITY = 3.0   # FIXED — not a free parameter (see STEP13)

# Design feed defaults (basra_kuwait_mix) used when lab data is missing
_DEFAULT_DENSITY  = 1.028   # g/cm³
_DEFAULT_CCR      = 22.8    # wt%
_DEFAULT_VISC135  = 230.0   # cSt


def _visc135_to_visc100(visc_135_cSt: float) -> float:
    """
    Convert kinematic viscosity at 135°C to 100°C using the Walther equation.

    Walther: log(log(v + 0.7)) = A - B * log(T_K)

    Two-point fit anchored on Basra-Kuwait design values:
      visc_135 = 230 cSt  at T = 408.15 K  (135°C)
      visc_100 = 1621 cSt at T = 373.15 K  (100°C)

    The slope B is determined from these two design points.
    The intercept A is then solved, and used to evaluate at T_100 with the
    actual measured visc_135.

    This anchoring keeps the VTI slope physically meaningful for VR-range
    feeds even at sparse calibration data.
    """
    # Design anchor points
    T1_K   = 408.15   # 135°C
    T2_K   = 373.15   # 100°C
    v1_des = 230.0    # cSt at 135°C (design)
    v2_des = 1621.0   # cSt at 100°C (design)

    y1 = math.log10(math.log10(v1_des + 0.7))
    y2 = math.log10(math.log10(v2_des + 0.7))
    lnT1 = math.log10(T1_K)
    lnT2 = math.log10(T2_K)

    # Walther slope — positive value (viscosity DECREASES with rising T).
    # Formula: log10(log10(v+0.7)) = A - B * log10(T), B > 0.
    # B = (y2 - y1) / (lnT1 - lnT2) gives positive B since y2>y1 and lnT1>lnT2.
    B = (y2 - y1) / (lnT1 - lnT2)

    # Use actual measured visc_135 to shift intercept A
    visc_135_safe = max(visc_135_cSt, 0.5)
    y_actual = math.log10(math.log10(visc_135_safe + 0.7))
    A = y_actual + B * lnT1

    # Evaluate at 100°C
    y100 = A - B * lnT2
    # Inverse Walther: v = 10^(10^y) - 0.7
    visc_100 = 10 ** (10 ** y100) - 0.7
    return float(max(visc_100, 1.0))


def _build_feed_components_from_lab(
    feed_density_g_cm3: float,
    feed_ccr_wt_pct: float,
    feed_visc_135_cSt: float,
    n_comp: int = 20,
    solvent: str = 'propane',
) -> list:
    """
    Build List[PseudoComponent] from measured lab properties.

    Strategy:
    1. Convert feed_visc_135 to feed_visc_100 using _visc135_to_visc100().
    2. Estimate SARA fractions from density, CCR, viscosity using
       estimate_sara_from_properties() from residue_distribution.py.
    3. Build a custom feed dict and call build_residue_distribution()
       with custom_feed parameter.

    If any lab value is NaN or zero, falls back to basra_kuwait_mix defaults.
    """
    from residue_distribution import build_residue_distribution, estimate_sara_from_properties

    # --- Fallback to design values if lab data missing ---
    density  = feed_density_g_cm3  if (feed_density_g_cm3 and not math.isnan(feed_density_g_cm3)) else _DEFAULT_DENSITY
    ccr      = feed_ccr_wt_pct     if (feed_ccr_wt_pct    and not math.isnan(feed_ccr_wt_pct))    else _DEFAULT_CCR
    visc135  = feed_visc_135_cSt   if (feed_visc_135_cSt  and not math.isnan(feed_visc_135_cSt))  else _DEFAULT_VISC135

    density  = max(density, 0.85)
    ccr      = max(min(ccr, 45.0), 1.0)
    visc135  = max(visc135, 1.0)

    # Step 1: convert visc135 → visc100
    visc100 = _visc135_to_visc100(visc135)

    # Step 2: estimate SARA from lab properties
    # estimate_sara_from_properties takes density_kg_m3
    sara = estimate_sara_from_properties(
        density_kg_m3=density * 1000.0,
        CCR=ccr,
        visc_100=visc100,
    )

    # Step 3: build custom feed dict
    F_precip = sara['asphaltenes'] / 100.0 * 2.5
    feed_dict = {
        'SG_15':        density,
        'CCR_wt':       ccr,
        'visc_100':     visc100,
        'visc_135':     visc135,
        'SARA':         sara,
        'MW_log_mean':  6.45,
        'MW_log_std':   0.72,
        'MW_heavy_cut': 750.0,
        'F_precip':     float(np.clip(F_precip, 0.05, 0.70)),
    }

    # Step 4: call build_residue_distribution with custom_feed
    components = build_residue_distribution(
        n_comp=n_comp,
        solvent_name=solvent,
        custom_feed=feed_dict,
    )
    return components


def simulate_single_train(
    components: list,
    so_ratio: float,
    t_profile: list,
    predilution_frac: float,
    calibration_params: dict,
    n_stages: int = 4,
    pressure_pa: float = 40e5,
    solvent: str = 'propane',
    feed_basis: float = 1.0,
    c_t_params: dict = None,
) -> dict:
    """
    Run run_extractor() for a single train with calibration parameters.

    t_profile: [T_bot, T_mid, T_steam_coil, T_top] in °C.

    calibration_params keys: K_multiplier, E_murphree, C_entrain, delta_crit

    Returns:
        {
          'dao_yield_mass_frac': float,
          'dao_yield_vol_pct':   float,   (volume basis, using feed/DAO densities)
          'density_DAO':         float,
          'MW_DAO':              float,
          'SARA_DAO':            dict,
          'visc_DAO_100':        float,
          'asph_contam_wt_pct':  float,
          'converged':           bool,
          'dao_mass_flow':       float,
        }
    """
    from hunter_nash_extractor import run_extractor
    from asphaltene_kinetics import KineticParams
    from stage_efficiency import StageEfficiency
    from entrainment_model import EntrainmentParams
    from quality_model import predict_dao_viscosity

    K_mult        = float(calibration_params.get('K_multiplier',  1.0))
    E_murphree    = float(calibration_params.get('E_murphree',    0.70))
    C_entrain     = float(calibration_params.get('C_entrain',     0.015))
    delta_crit    = float(calibration_params.get('delta_crit',    2.5))
    alpha_density = 3.0   # FIXED — never read from calibration_params.
                           # See STEP13 notes: floated alpha destroyed yield prediction.

    # Convert °C T-profile to Kelvin for run_extractor
    # Map 4 bed temperatures to N stages
    t_profile_c = [float(t) for t in t_profile]
    if n_stages == len(t_profile_c):
        T_profile_K = [t + 273.15 for t in t_profile_c]
    else:
        # Interpolate from T_bot to T_top across N stages
        T_bot = t_profile_c[0]
        T_top = t_profile_c[-1]
        fracs = [i / max(n_stages - 1, 1) for i in range(n_stages)]
        fracs = [f ** 0.8 for f in fracs]   # concave-up (matches build_T_profile)
        T_profile_K = [T_bot + (T_top - T_bot) * f + 273.15 for f in fracs]

    try:
        r = run_extractor(
            components       = components,
            solvent_name     = solvent,
            solvent_ratio    = float(np.clip(so_ratio, 1.0, 20.0)),
            N_stages         = n_stages,
            T_profile        = T_profile_K,
            P                = pressure_pa,
            kinetics         = KineticParams(k_precip=0.5, tau=10.0),
            efficiency       = StageEfficiency(E_murphree),
            entrainment      = EntrainmentParams(C_entrain, 1.20),
            K_multiplier     = K_mult,
            delta_crit       = delta_crit,
            predilution_frac = float(np.clip(predilution_frac, 0.0, 0.8)),
            alpha_density    = alpha_density,
            thermo_mode      = 'kvalue',
            feed_basis       = feed_basis,
            c_t_params       = c_t_params,
        )
    except Exception as e:
        logger.debug(f"run_extractor failed: {e}")
        return {
            'dao_yield_mass_frac': 0.0,
            'dao_yield_vol_pct':   0.0,
            'density_DAO':         0.93,
            'MW_DAO':              500.0,
            'SARA_DAO':            {},
            'visc_DAO_100':        999.0,
            'asph_contam_wt_pct':  0.0,
            'converged':           False,
            'dao_mass_flow':       0.0,
        }

    # run_extractor's 'converged' flag can be False due to post-loop mass-balance
    # rescaling shifting the yield by more than outer_tol — check iterations too.
    converged = bool(r.get('converged', False)) or (
        r.get('outer_iterations', 60) < 55 and r.get('DAO_yield_gross', 0) > 0.5
    )

    if not converged or r.get('DAO_yield_gross', 0) < 0.5:
        return {
            'dao_yield_mass_frac': 0.0,
            'dao_yield_vol_pct':   0.0,
            'density_DAO':         0.93,
            'MW_DAO':              500.0,
            'SARA_DAO':            r.get('SARA_DAO', {}),
            'visc_DAO_100':        999.0,
            'asph_contam_wt_pct':  float(r.get('asphal_contam_pct', 0.0)),
            'converged':           False,
            'dao_mass_flow':       0.0,
        }

    dao_yield_mass_pct = float(r['DAO_yield_gross'])
    dao_yield_mass_frac = dao_yield_mass_pct / 100.0
    density_dao = float(r['density_DAO'])
    MW_dao = float(r['MW_DAO_avg'])
    sara_dao = dict(r.get('SARA_DAO', {}))

    # Compute feed density from component weight fractions (harmonic mean = volume basis)
    z_arr = np.array([c.z for c in components])
    mw_arr = np.array([c.MW for c in components])
    rho_arr = np.array([c.density for c in components])
    wt_frac = z_arr * mw_arr
    wt_frac /= wt_frac.sum()
    feed_density = float(1.0 / np.dot(wt_frac, 1.0 / rho_arr))

    # Volume yield: mass_yield * rho_feed / rho_DAO
    dao_yield_vol_pct = dao_yield_mass_frac * (feed_density / density_dao) * 100.0

    # DAO viscosity
    visc_dao_100 = predict_dao_viscosity(MW_dao, density_dao, sara_dao, T_eval_C=100.0)

    return {
        'dao_yield_mass_frac': dao_yield_mass_frac,
        'dao_yield_vol_pct':   float(dao_yield_vol_pct),
        'density_DAO':         density_dao,
        'MW_DAO':              MW_dao,
        'SARA_DAO':            sara_dao,
        'visc_DAO_100':        float(visc_dao_100),
        'asph_contam_wt_pct':  float(r.get('asphal_contam_pct', 0.0)),
        'converged':           True,
        'dao_mass_flow':       float(feed_basis * dao_yield_mass_frac),
    }


def simulate_parallel_trains(
    row: pd.Series,
    calibration_params: dict,
    feed_components_cache: dict,
    thermal_params: dict | None = None,
    use_dcs_temperatures: bool = True,
    c_t_params: dict = None,
    pinn_corrector=None,
    pinn_cluster_id: int = -1,
) -> dict:
    """
    Simulate both trains for one calibration dataset row.

    Parameters
    ----------
    row : one row from the visc_anchored or dcs_hourly DataFrame
    calibration_params : dict with K_multiplier, E_murphree, C_entrain, delta_crit
    feed_components_cache : dict keyed by (density, ccr, visc135) for memoization
    thermal_params : output of thermal_calibration.calibrate_thermal_model()
                     If None, use DCS temperatures directly
    use_dcs_temperatures : if True, use measured DCS temps as T-profile
                           if False, use thermal_params to predict T-profile
    pinn_corrector : PINNCorrector or None
        If provided, applies PINN multiplicative corrections to visc and yield
        after the physics blending step. Falls back to OLS if None or on error.
    pinn_cluster_id : int
        Operating regime cluster ID for this row (-1 = unknown).

    Returns:
        {
          'dao_yield_vol_pct': float,
          'dao_visc_100_cSt':  float,
          'dao_density':       float,
          'converged':         bool,
          'train_a_converged': bool,
          'train_b_converged': bool,
          'correction_mode':   str ('ols' | 'pinn' | 'ols_fallback')
        }
    """
    from thermal_calibration import predict_t_profile_calibrated

    def _safe(val, default):
        """Return val if not NaN/None, else default."""
        try:
            v = float(val)
            return default if math.isnan(v) else v
        except (TypeError, ValueError):
            return default

    # --- Step 1: Build feed components (with cache) ---
    density  = _safe(row.get('feed_density'),   _DEFAULT_DENSITY)
    ccr      = _safe(row.get('feed_ccr'),       _DEFAULT_CCR)
    visc135  = _safe(row.get('feed_visc_135'),  _DEFAULT_VISC135)

    cache_key = (round(density, 3), round(ccr, 2), round(visc135, 0))
    if cache_key not in feed_components_cache:
        feed_components_cache[cache_key] = _build_feed_components_from_lab(
            density, ccr, visc135
        )
    components = feed_components_cache[cache_key]

    results = {}
    for train in ['a', 'b']:
        # --- Step 2: T-profile ---
        if use_dcs_temperatures or thermal_params is None:
            t_bot = _safe(row.get(f't_bot_{train}'), 67.0)
            t_mid = _safe(row.get(f't_mid_{train}'), 72.0)
            t_sc  = _safe(row.get(f't_steam_coil_{train}'), 77.0)
            t_top = _safe(row.get(f't_top_{train}'), 82.0)
            t_profile = [t_bot, t_mid, t_sc, t_top]
        else:
            t_feed   = _safe(row.get(f'feed_temp_{train}'), 90.0)
            t_propane = _safe(row.get('propane_temp'), 54.0)
            so_ratio = _safe(row.get(f'so_ratio_{train}'), 8.0)
            t_profile = predict_t_profile_calibrated(
                thermal_params, t_feed, t_propane, so_ratio, train
            )

        # --- Step 3: Run single train ---
        so_ratio    = _safe(row.get(f'so_ratio_{train}'), 8.0)
        predil_frac = _safe(row.get(f'predilution_frac_{train}'), 0.0)

        results[train] = simulate_single_train(
            components         = components,
            so_ratio           = so_ratio,
            t_profile          = t_profile,
            predilution_frac   = predil_frac,
            calibration_params = calibration_params,
            c_t_params         = c_t_params,
        )

    ra = results['a']
    rb = results['b']

    # --- Step 1 (visc correction): linear post-correction on the Walther prediction.
    # quality_model.py predicts ~14 cSt for this feed; plant measures ~31 cSt.
    # visc_corrected = visc_slope * walther_prediction + visc_bias
    # Default identity (slope=1, bias=0) preserves backward-compatibility when
    # calibration_params does not yet contain these keys.
    # Calibrated simultaneously with K_multiplier/delta_crit by the optimizer so
    # the yield target and viscosity target are decoupled: K_multiplier controls
    # yield; visc_slope+visc_bias absorb the Walther model offset.
    visc_slope = float(calibration_params.get('visc_slope', 1.0))
    visc_bias  = float(calibration_params.get('visc_bias',  0.0))

    def _correct_visc(raw_visc: float) -> float:
        """Apply linear viscosity correction; enforce physical floor of 0.1 cSt."""
        return max(visc_slope * float(raw_visc) + visc_bias, 0.1)

    def _apply_pinn(result: dict) -> dict:
        """
        Apply PINN multiplicative correction if a corrector is available.
        Falls back gracefully to the OLS-corrected result on any error.
        The PINN operates AFTER OLS correction (stacked corrections).
        """
        if pinn_corrector is None or not getattr(pinn_corrector, 'is_trained', False):
            result['correction_mode'] = 'ols'
            return result
        try:
            from pinn_network import extract_features_from_row
            from pinn_calibration_engine import apply_pinn_correction
            n_clusters = getattr(pinn_corrector, 'n_clusters', 0)
            feats = extract_features_from_row(row, n_clusters, pinn_cluster_id)
            return apply_pinn_correction(result, feats, pinn_corrector)
        except Exception as exc:
            logger.debug("PINN correction skipped: %s", exc)
            result['correction_mode'] = 'ols_fallback'
            return result

    # --- Step 4: Blend predictions (volume-weighted) ---
    conv_a = ra['converged']
    conv_b = rb['converged']

    if conv_a and conv_b:
        feed_a = _safe(row.get('feed_flow_a'), 1.0)
        feed_b = _safe(row.get('feed_flow_b'), 1.0)
        dao_vol_a = feed_a * ra['dao_yield_vol_pct'] / 100.0
        dao_vol_b = feed_b * rb['dao_yield_vol_pct'] / 100.0
        dao_vol_total = dao_vol_a + dao_vol_b
        feed_total = _safe(row.get('feed_flow_total'),
                           max(feed_a + feed_b, 1e-6))

        if dao_vol_total < 1e-9:
            dao_vol_total = 1.0
            dao_vol_a = 0.5

        w_a = dao_vol_a / dao_vol_total
        w_b = dao_vol_b / dao_vol_total

        visc_raw      = w_a * ra['visc_DAO_100'] + w_b * rb['visc_DAO_100']
        density_blend = w_a * ra['density_DAO']  + w_b * rb['density_DAO']
        yield_blend   = dao_vol_total / max(feed_total, 1e-9) * 100.0

        result = {
            'dao_yield_vol_pct': float(yield_blend),
            'dao_visc_100_cSt':  _correct_visc(visc_raw),
            'dao_density':       float(density_blend),
            'converged':         True,
            'train_a_converged': True,
            'train_b_converged': True,
        }
        return _apply_pinn(result)
    elif conv_a:
        result = {
            'dao_yield_vol_pct': float(ra['dao_yield_vol_pct']),
            'dao_visc_100_cSt':  _correct_visc(ra['visc_DAO_100']),
            'dao_density':       float(ra['density_DAO']),
            'converged':         True,
            'train_a_converged': True,
            'train_b_converged': False,
        }
        return _apply_pinn(result)
    elif conv_b:
        result = {
            'dao_yield_vol_pct': float(rb['dao_yield_vol_pct']),
            'dao_visc_100_cSt':  _correct_visc(rb['visc_DAO_100']),
            'dao_density':       float(rb['density_DAO']),
            'converged':         True,
            'train_a_converged': False,
            'train_b_converged': True,
        }
        return _apply_pinn(result)
    else:
        return {
            'dao_yield_vol_pct': 0.0,
            'dao_visc_100_cSt':  999.0,
            'dao_density':       0.93,
            'converged':         False,
            'train_a_converged': False,
            'train_b_converged': False,
            'correction_mode':   'ols',
        }
