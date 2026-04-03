"""
run_simulation.py
=================
CLI entry point + Flask web UI for HPCL PDA Unit Simulator (Plant No. 41).

Usage:
    python run_simulation.py                  # propane default, opens web UI
    python run_simulation.py --no-ui          # CLI only, print results
    python run_simulation.py --solvent butane --so 10 --T 140 --stages 4

Web UI: http://localhost:5000
"""

import argparse
import json
import sys
import numpy as np

from residue_distribution  import (build_residue_distribution, distribution_summary,
                                   HPCL_FEEDS, estimate_sara_from_properties)
from asphaltene_kinetics   import KineticParams, precipitation_efficiency
from stage_efficiency      import StageEfficiency
from entrainment_model     import EntrainmentParams
from hunter_nash_extractor import run_extractor
from quality_model         import predict_dao_viscosity, predict_astm_colour
from hydraulics_entrain    import (check_column_hydraulics, hydraulic_metrics,
                                   compute_solvent_flows, stages_from_packing,
                                   propane_saturation_check,
                                   COLUMN_DEFAULTS,
                                   BED_CONFIG, beds_to_stages,
                                   steam_pressure_to_properties)


# ── Stage-count helper (single source of truth) ───────────────────────────────
def _derive_N_from_request(d: dict) -> int:
    """Derive number of stages from bed geometry in request data.

    Used by /api/simulate, /api/sensitivity, /api/operating_margins, /api/tradeoff.
    Falls back to BED_CONFIG defaults if bed params not provided.
    Never returns 0 — minimum 1.
    """
    bed_config = {
        'top':    {'height_mm': float(d.get('top_h', BED_CONFIG['top']['height_mm'])),
                   'ID_mm':     float(d.get('top_d', BED_CONFIG['top']['ID_mm']))},
        'middle': {'height_mm': float(d.get('mid_h', BED_CONFIG['middle']['height_mm'])),
                   'ID_mm':     float(d.get('mid_d', BED_CONFIG['middle']['ID_mm']))},
        'bottom': {'height_mm': float(d.get('bot_h', BED_CONFIG['bottom']['height_mm'])),
                   'ID_mm':     float(d.get('bot_d', BED_CONFIG['bottom']['ID_mm']))},
    }
    HETP = float(d.get('HETP_mm', 2000))
    stage_list = beds_to_stages(HETP_mm=HETP, bed_config=bed_config)
    return max(len(stage_list), 1)


# ── Extractor thermal defaults (Item C) ──────────────────────────────────────
THERMAL_DEFAULTS = {
    'T_feed_C':            90.0,   # VR feed inlet temperature [degC]
    'T_propane_inlet_C':   54.0,   # propane injection temperature [degC]
    'steam_flow_kg_hr':  2830.0,   # steam flow for top-bed heating [kg/hr]
    'bottom_T_blend':      0.35,   # bottom bed T blend fraction
    'middle_T_blend':      0.55,   # middle bed T blend fraction
    'steam_effectiveness': 0.60,   # fraction of steam enthalpy used for heating
}


def estimate_bed_temperatures(
    T_feed_C:            float = 90.0,
    T_propane_inlet_C:   float = 54.0,
    steam_flow_kg_hr:    float = 2830.0,
    feed_flow_kg_hr:     float = 88547.0,
    bottom_T_blend:      float = 0.35,
    middle_T_blend:      float = 0.55,
    steam_effectiveness: float = 0.60,
) -> dict:
    """
    Estimate extractor bed temperatures from inlet stream conditions.

    Thermal model:
        T_bottom = T_propane + bottom_T_blend * (T_feed - T_propane)
        T_middle = T_bottom  + middle_T_blend * (T_feed - T_bottom)
        DeltaT_steam = steam_effectiveness * steam_enthalpy / (feed_flow * Cp_mix)
        T_top    = T_middle + DeltaT_steam

    Returns dict with T_bottom, T_middle, T_top (all degC).
    """
    T_bottom = T_propane_inlet_C + bottom_T_blend * (T_feed_C - T_propane_inlet_C)
    T_middle = T_bottom + middle_T_blend * (T_feed_C - T_bottom)

    # Steam heating: approximate enthalpy = 2000 kJ/kg (low-pressure steam)
    # Cp_mix ~ 2.1 kJ/(kg.K) for heavy oil/propane mixture
    steam_enthalpy_kJ_hr = steam_flow_kg_hr * 2000.0
    feed_cp_kJ_hr_K      = max(feed_flow_kg_hr, 1.0) * 2.1
    delta_T_steam        = steam_effectiveness * steam_enthalpy_kJ_hr / feed_cp_kJ_hr_K
    T_top = T_middle + delta_T_steam

    return {
        'T_bottom_C': round(T_bottom, 1),
        'T_middle_C': round(T_middle, 1),
        'T_top_C':    round(T_top,    1),
        'delta_T_steam_C': round(delta_T_steam, 1),
    }


# ── JSON serialisation helper (handles numpy types) ──────────────────────────
def _to_json_safe(obj):
    """Recursively convert numpy types to Python native types."""
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ── Temperature profile builder ───────────────────────────────────────────────
def build_T_profile(T_bottom_C: float, T_top_C: float, N_stages: int) -> list:
    """
    Build a concave-up temperature profile [K] for N_stages.

    Stage 0 = bottom (feed end, lower T),  stage N-1 = top (solvent end, higher T).
    Interpolation uses a 0.8-power curve to put more curvature near the bottom,
    matching typical SDA extractor thermal profiles.

    If T_bottom_C == T_top_C a flat profile is returned (backward compatible).
    """
    if N_stages == 1 or abs(T_top_C - T_bottom_C) < 0.1:
        return [T_bottom_C + 273.15] * N_stages
    fracs = [i / (N_stages - 1) for i in range(N_stages)]
    fracs = [f ** 0.8 for f in fracs]          # concave-up
    return [T_bottom_C + (T_top_C - T_bottom_C) * f + 273.15 for f in fracs]


# ── Core simulation runner ────────────────────────────────────────────────────
def run_base_case(feed_name='basra_kuwait_mix', solvent='propane',
                  SO=8.0, T_C=75.0,
                  T_bottom=None, T_top=None, predilution_frac=0.0,
                  E=0.70, k_precip=0.5, tau=10.0, C_entrain=0.015,
                  K_multiplier=1.0, delta_crit=2.5,
                  P_bar=40.0, alpha_density=3.0,
                  thermo_mode='kvalue',
                  HETP_mm=2000.0, bed_config=None,
                  verbose=False):
    """Run simulation using 3-bed supercritical extraction model.

    N_stages is DERIVED from bed geometry (bed_config + HETP_mm) — it is
    never a direct parameter. Change HETP_mm or bed heights to change stages.
    """
    if bed_config is None:
        bed_config = BED_CONFIG

    # Derive stage count from bed geometry — this is the single source of truth
    stages = beds_to_stages(HETP_mm=HETP_mm, bed_config=bed_config)
    N = len(stages)

    # Resolve T_bottom / T_top (fall back to legacy T_C if not provided)
    if T_bottom is None:
        T_bottom = T_C
    if T_top is None:
        T_top = T_C + 10.0 if solvent == 'propane' else T_C + 20.0

    comps     = build_residue_distribution(feed_name=feed_name, n_comp=20, solvent_name=solvent)
    T_profile = build_T_profile(T_bottom, T_top, N)
    r = run_extractor(
        components       = comps,
        solvent_name     = solvent,
        solvent_ratio    = SO,
        N_stages         = N,
        T_profile        = T_profile,
        P                = P_bar * 1e5,
        kinetics         = KineticParams(k_precip, tau),
        efficiency       = StageEfficiency(E),
        entrainment      = EntrainmentParams(C_entrain, 1.20),
        K_multiplier     = K_multiplier,
        delta_crit       = delta_crit,
        predilution_frac = predilution_frac,
        alpha_density    = alpha_density,
        thermo_mode      = thermo_mode,
        verbose          = verbose,
    )
    r['distribution_summary'] = distribution_summary(comps, feed_name=feed_name)
    r['feed_name']       = feed_name
    r['solvent']         = solvent
    r['SO_ratio']        = SO
    r['T_C']             = (T_bottom + T_top) / 2.0   # kept for backward compat
    r['T_bottom']        = T_bottom
    r['T_top']           = T_top
    r['predilution_frac']= predilution_frac
    r['N_stages']        = N
    r['HETP_mm']         = HETP_mm
    r['T_profile_C']     = [t - 273.15 for t in T_profile]

    # Quality model
    sara_dao = r.get('SARA_DAO', {})
    r['viscosity_dao_cSt'] = predict_dao_viscosity(
        r['MW_DAO_avg'], r['density_DAO'], sara_dao)
    r['astm_colour_dao']   = predict_astm_colour(
        r['asphal_contam_pct'], sara_dao)
    return r


# ── Pretty printer ────────────────────────────────────────────────────────────
def _sep(c='=', w=64): return c * w
def _kv(l, v, u='', w=38):
    fv = f"{float(v):.4g}" if isinstance(v, (float, np.floating)) else str(v)
    print(f"  {l:<{w}}: {fv}  {u}")


def print_summary(r):
    ds = r['distribution_summary']
    _GENERIC_LABELS = {'basra_kuwait_mix': 'Heavy VR Blend A  (Design)',
                       'basra_light':      'Heavy VR Blend B  (Check)'}
    print(f"\n{_sep('*')}")
    print("  SDA Unit Simulator — Base Case Results")
    print(f"  Feed: {_GENERIC_LABELS.get(r['feed_name'], r['feed_name'])}")
    print(_sep('*'))

    print(f"\n{_sep()}\n  FEED PROPERTIES (Operating Manual, Plant 41)\n{_sep()}")
    if 'API' in ds:
        _kv("API Gravity",              ds['API'])
        _kv("Specific Gravity @15.5°C", ds['SG_15'])
        _kv("Conradson Carbon",         ds['CCR_wt'],        'wt%')
        _kv("Viscosity @100°C",         ds['visc_100_cSt'],  'cSt')
        _kv("Viscosity @135°C",         ds['visc_135_cSt'],  'cSt')
        _kv("Nickel",                   ds['Ni_wppm'],       'wppm')
        _kv("Vanadium",                 ds['V_wppm'],        'wppm')
    sara = ds.get('SARA_wt_pct', {})
    if sara:
        print("\n  SARA Breakdown (back-calculated from CCR/API/metals):")
        for k, v in sara.items():
            _kv(f"    {k.capitalize()}", v, 'wt%')
    _kv("Number-avg MW",   ds['MW_number_avg'], 'g/mol')
    _kv("Weight-avg MW",   ds['MW_weight_avg'], 'g/mol')
    _kv("Mean density",    ds['mean_density'],  'g/cm³')

    print(f"\n{_sep()}\n  OPERATING CONDITIONS\n{_sep()}")
    _kv("Solvent",           r['solvent'])
    _kv("Solvent/Oil ratio", r['SO_ratio'],  'kg/kg')
    _kv("Stages",            r['N_stages'])
    _kv("Temperature profile",
        ", ".join(f"{t:.1f}" for t in r['T_profile_C']), 'degC  (bottom -> top)')

    print(f"\n{_sep()}\n  MAIN RESULTS\n{_sep()}")
    _kv("DAO Yield",                    r['DAO_yield_net'],      'wt%')
    _kv("Asphalt / Pitch Yield",        r['asphalt_yield'],      'wt%')
    _kv("Asphaltene in DAO (contam.)",  r['asphal_contam_pct'],  'wt%  [colour/quality risk]')
    _kv("Mass balance check",
        r['DAO_yield_net'] + r['asphalt_yield'],                 'wt%')
    _kv("DAO avg MW",           r['MW_DAO_avg'],     'g/mol')
    _kv("Asphalt avg MW",       r['MW_asphalt_avg'], 'g/mol')
    _kv("DAO liquid density",   r['density_DAO'],    'g/cm³')
    sara_dao = r.get('SARA_DAO', {})
    if sara_dao:
        print("\n  DAO SARA composition:")
        for k, v in sara_dao.items():
            _kv(f"    {k.capitalize()}", float(v), 'wt%')

    print(f"\n{_sep()}\n  STAGE BREAKDOWN\n{_sep()}")
    print(f"  {'Stage':>5}  {'T[°C]':>6}  {'LLE Yield':>10}  {'Precip%':>8}  {'Asphal in DAO':>14}")
    print(f"  {'-'*5}  {'-'*6}  {'-'*10}  {'-'*8}  {'-'*14}")
    for s in r['stage_results']:
        print(f"  {s['stage']:>5}  {s['T_C']:>6.1f}  "
              f"{s['lle_yield']:>9.1f}%  "
              f"{s['precip_yield']:>7.1f}%  "
              f"{s['asphal_in_dao_pct']:>13.3f}%")

    print(f"\n{_sep()}\n  Simulation complete.\n{_sep()}")


# ── Flask Web UI ──────────────────────────────────────────────────────────────
def launch_web_ui():
    import os
    import sys
    import logging
    import threading
    import webbrowser
    from flask import Flask, render_template_string, request, jsonify

    # ── Configure logging ─────────────────────────────────────────────────────
    # Python's root logger defaults to WARNING — every logger.info() call in
    # calibration_engine, thermal_calibration, plant_data_loader etc. is
    # silently dropped without this. force=True removes any handlers Flask
    # may have already added so our format takes effect.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
        stream=sys.stdout,
        force=True,
    )
    # Keep werkzeug HTTP request logs quiet — they'd drown out calibration logs
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    # ─────────────────────────────────────────────────────────────────────────

    app  = Flask(__name__)
    _tmpl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ui_template.html')

    # ── helper: clean result dict for JSON response ──────────────────────────
    def _clean_result(r, ds):
        stage_results = []
        for s in r.get('stage_results', []):
            stage_results.append({
                'stage':             int(s['stage']),
                'T_C':               float(s['T_C']),
                'lle_yield':         float(s['lle_yield']),
                'precip_yield':      float(s['precip_yield']),
                'asphal_in_dao_pct': float(s['asphal_in_dao_pct']),
                'converged':         bool(s['converged']),
            })
        sara_dao = {k: round(float(v), 1)
                    for k, v in r.get('SARA_DAO', {}).items()}
        feed_props = {}
        for key in ['API','SG_15','CCR_wt','visc_100_cSt','visc_135_cSt','Ni_wppm','V_wppm']:
            val = ds.get(key)
            if val is not None:
                feed_props[key] = float(val) if isinstance(val, (int, float, np.number)) else val

        # Quality model for DAO
        from quality_model import predict_dao_viscosity, predict_astm_colour
        visc_cSt   = predict_dao_viscosity(
            float(r['MW_DAO_avg']), float(r['density_DAO']), sara_dao)
        astm_colour = predict_astm_colour(
            float(r['asphal_contam_pct']), sara_dao)

        return {
            'ok':              True,
            'DAO_yield':       round(float(r['DAO_yield_net']), 2),
            'asphalt_yield':   round(float(r['asphalt_yield']), 2),
            'asphal_contam':   round(float(r['asphal_contam_pct']), 3),
            'MW_DAO':          round(float(r['MW_DAO_avg']), 1),
            'MW_asphalt':      round(float(r['MW_asphalt_avg']), 1),
            'density_DAO':     round(float(r['density_DAO']), 4),
            'viscosity_cSt':   round(float(visc_cSt), 1),
            'astm_colour':     round(float(astm_colour), 2),
            'converged':       bool(r['converged']),
            'iterations':      int(r['outer_iterations']),
            'stage_results':   stage_results,
            'SARA_DAO':        sara_dao,
            'feed_label':      str(HPCL_FEEDS.get(ds.get('feed_name',''), {}).get('label', '')),
            'feed_props':      feed_props,
        }

    # ── Simulation history ring-buffer (last 5, newest first) ────────────────
    _simulation_history = []

    def _record_simulation(params, result):
        """Append a snapshot to the history ring-buffer (max 5 entries)."""
        import time as _time
        entry = {
            'id':        len(_simulation_history) + 1,
            'timestamp': _time.strftime('%H:%M:%S'),
            'params': {
                'feed':       params.get('feed', ''),
                'solvent':    params.get('solvent', ''),
                'SO':         float(params.get('so', 8)),
                'T_bottom':   float(params.get('T_bottom', 75)),
                'T_top':      float(params.get('T_top', 85)),
                'predilution':float(params.get('predilution_frac', 0)),
                'N_stages':   int(result.get('N_stages_used', params.get('stages', 3))),
            },
            'results': {
                'DAO_yield':    float(result.get('DAO_yield',    0)),
                'asphalt_yield':float(result.get('asphalt_yield',0)),
                'density':      float(result.get('density_DAO',  0)),
                'viscosity':    float(result.get('viscosity_cSt',0)),
                'astm_colour':  float(result.get('astm_colour',  0)),
                'asphal_contam':float(result.get('asphal_contam',0)),
                'MW_DAO':       float(result.get('MW_DAO',       0)),
            },
        }
        _simulation_history.insert(0, entry)
        if len(_simulation_history) > 5:
            _simulation_history.pop()

    @app.route('/api/simulation_history', methods=['GET'])
    def api_simulation_history():
        return jsonify({'ok': True, 'history': _simulation_history})

    # ── Feed characterisation ─────────────────────────────────────────────────
    @app.route('/api/characterise_feed', methods=['POST'])
    def api_characterise_feed():
        try:
            from residue_distribution import api_from_density
            d    = request.json or {}
            density = float(d.get('density_kg_m3', d.get('density', 1028.0)))
            CCR  = float(d.get('CCR', 22.8))
            visc = d.get('visc_100')
            asph = d.get('asphaltene_wt')
            api_val = api_from_density(density)
            sara = estimate_sara_from_properties(
                density_kg_m3=density, CCR=CCR,
                visc_100=float(visc) if visc is not None else None,
                asphaltene_wt=float(asph) if asph is not None else None,
            )
            # Build a quick component list to get MW stats
            custom_feed = {
                'SARA': sara,
                'MW_heavy_cut': 750.0,
                'F_precip': float(np.clip(0.28 + (sara['asphaltenes'] - 10) / 80, 0.15, 0.50)),
                'API': api_val, 'CCR_wt': CCR,
            }
            comps = build_residue_distribution(custom_feed=custom_feed, n_comp=20)
            ds    = distribution_summary(comps, feed_name='custom')
            return jsonify({
                'ok':           True,
                'SARA':         sara,
                'API_computed': round(api_val, 2),
                'MW_avg':       round(ds['MW_number_avg'], 1),
                'heavy_frac':   round(ds['z_heavy_fraction'], 3),
                'precip_frac':  round(ds['z_precipitable'],   3),
                'mean_density': round(ds['mean_density'],      4),
            })
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)})

    @app.route('/')
    def index():
        # Re-read template on every request so UI edits are visible without restart
        tmpl = open(_tmpl_path, encoding='utf-8').read()
        return render_template_string(tmpl)

    # ── Temperature pre-calculation (fast, no simulation) ────────────────────
    @app.route('/api/calculate_temperatures', methods=['POST'])
    def api_calculate_temperatures():
        """
        Estimate extractor bed temperatures from inlet stream conditions.
        Does NOT run a full simulation — returns in < 1 ms.

        Input JSON fields:
            T_feed_mixed          float  Feed temp after predilution [degC]  default 90
            T_propane_fresh       float  Fresh propane inlet temp [degC]      default 54
            steam_flow_kg_hr      float  LP steam flow to coils [kg/hr]       default 2830
            lp_steam_pressure     float  LP steam gauge pressure [kg/cm²g]    default 4.5
            P_kgcm2               float  Extractor pressure [kg/cm²g]         default 40
            feed_flow_m3hr        float  Feed flow [m³/hr]                    default 86.1
            so                    float  Solvent/oil ratio                     default 8.0
            n_extractors          int    Extractors in parallel                default 2

        Returns JSON:
            T_bottom, T_middle, T_top  (all degC)
            dT_steam_C, T_sat_steam_C, h_fg_kJ_kg
            T_sat_propane_C, flash_warning, steam_props
        """
        try:
            d = request.get_json(force=True) or {}

            # ── Collect inputs ────────────────────────────────────────────────
            T_feed      = float(d.get('T_feed_mixed',      d.get('T_feed_C', 90.0)))
            T_propane   = float(d.get('T_propane_fresh',   d.get('T_propane_inlet_C', 54.0)))
            stm_flow    = float(d.get('steam_flow_kg_hr',  THERMAL_DEFAULTS['steam_flow_kg_hr']))
            stm_press   = float(d.get('lp_steam_pressure', 4.5))
            P_kgcm2     = float(d.get('P_kgcm2',           40.0))
            P_bar       = P_kgcm2 * 0.980665
            feed_m3hr   = float(d.get('feed_flow_m3hr',    86.1))
            so          = float(d.get('so',                 8.0))
            n_ext       = int(d.get('n_extractors',         2))
            bot_blend   = float(d.get('bottom_T_blend', THERMAL_DEFAULTS['bottom_T_blend']))
            mid_blend   = float(d.get('middle_T_blend', THERMAL_DEFAULTS['middle_T_blend']))
            stm_eff     = float(d.get('steam_effectiveness', THERMAL_DEFAULTS['steam_effectiveness']))

            # Convert feed flow to kg/hr per extractor (assume rho_feed ~ 1028 kg/m³)
            rho_feed        = float(d.get('feed_density_kg_m3', 1028.0))
            total_feed_kg   = feed_m3hr * rho_feed
            feed_per_ext    = total_feed_kg / max(n_ext, 1)
            solvent_per_ext = feed_per_ext * so

            # ── Call estimate_bed_temperatures ────────────────────────────────
            from hydraulics_entrain import estimate_bed_temperatures
            result = estimate_bed_temperatures(
                T_feed_mixed_C           = T_feed,
                T_propane_fresh_C        = T_propane,
                steam_flow_kg_hr         = stm_flow,
                feed_flow_kg_hr          = feed_per_ext,
                solvent_flow_kg_hr       = solvent_per_ext,
                P_bar                    = P_bar,
                bottom_blend             = bot_blend,
                middle_blend             = mid_blend,
                steam_effectiveness      = stm_eff,
                lp_steam_pressure_kgcm2g = stm_press,
            )

            # ── Attach steam props for UI display ─────────────────────────────
            result['steam_props']        = steam_pressure_to_properties(stm_press)
            result['inputs_echo'] = {
                'T_feed_mixed_C':          T_feed,
                'T_propane_fresh_C':       T_propane,
                'steam_flow_kg_hr':        stm_flow,
                'lp_steam_pressure_kgcm2g': stm_press,
                'P_kgcm2':                 P_kgcm2,
                'feed_flow_m3hr':          feed_m3hr,
                'so':                      so,
                'n_extractors':            n_ext,
            }
            return jsonify(result)

        except Exception as exc:
            return jsonify({'error': str(exc)}), 400

    @app.route('/api/simulate', methods=['POST'])
    def api_simulate():
        try:
            d        = request.json or {}
            feed     = d.get('feed',      'basra_kuwait_mix')
            solvent  = d.get('solvent',   'propane')
            so       = float(d.get('so',       8.0))
            # Support both legacy T_C and new T_bottom/T_top
            _T_def   = 75.0 if solvent == 'propane' else 140.0
            T_C      = float(d.get('T_C', _T_def))
            T_bottom = float(d.get('T_bottom', d.get('T_C', _T_def)))
            T_top    = float(d.get('T_top',    T_bottom + (10.0 if solvent == 'propane' else 20.0)))
            pred_frac= float(d.get('predilution_frac', 0.0))
            E        = float(d.get('E',        0.70))
            k_p      = float(d.get('k_precip', 0.5))
            tau      = float(d.get('tau',      10.0))
            C_ent    = float(d.get('C_entrain',0.015))
            K_mult   = float(d.get('K_multiplier', 1.0))
            d_crit      = float(d.get('delta_crit', 2.5))
            thermo_mode = d.get('thermo_mode', 'kvalue')
            alpha_d  = float(d.get('alpha_density', 3.0))

            # Pressure and extractor count
            P_bar    = float(d.get('P_kgcm2', 40.0)) * 0.980665  # kgf/cm2 -> bar
            n_ext    = int(d.get('n_extractors', 2))

            # BED GEOMETRY → N_stages (the ONLY source of stage count)
            # Read per-bed heights and diameters from UI; fall back to BED_CONFIG defaults
            sim_bed_config = {
                'top':    {'height_mm': float(d.get('top_h',  BED_CONFIG['top']['height_mm'])),
                           'ID_mm':     float(d.get('top_d',  BED_CONFIG['top']['ID_mm'])),
                           'role': 'polishing'},
                'middle': {'height_mm': float(d.get('mid_h',  BED_CONFIG['middle']['height_mm'])),
                           'ID_mm':     float(d.get('mid_d',  BED_CONFIG['middle']['ID_mm'])),
                           'role': 'extraction'},
                'bottom': {'height_mm': float(d.get('bot_h',  BED_CONFIG['bottom']['height_mm'])),
                           'ID_mm':     float(d.get('bot_d',  BED_CONFIG['bottom']['ID_mm'])),
                           'role': 'rejection'},
            }
            HETP = float(d.get('HETP_mm', 2000.0))
            stage_list = beds_to_stages(HETP_mm=HETP, bed_config=sim_bed_config)
            N = len(stage_list)

            # Feed density from database (Fix 2: use actual VR density, not DAO density)
            feed_data = HPCL_FEEDS.get(feed, {})
            rho_feed  = float(feed_data.get('SG_15', 1.028))

            # Feed flow: support m³/hr input (operators read from DCS)
            # UI sends feed_flow_m3hr; fallback to kg/hr for legacy
            if d.get('feed_flow_m3hr') is not None:
                feed_flow_m3hr = float(d['feed_flow_m3hr'])
                feed_flow = feed_flow_m3hr * rho_feed * 1000.0  # m³/hr → kg/hr
            else:
                feed_flow = float(d.get('feed_flow_kg_hr', COLUMN_DEFAULTS['design_feed_kg_hr']))

            # Step 3: custom feed (from characterise_feed panel)
            custom_sara = d.get('custom_sara')
            if custom_sara and feed == 'custom':
                custom_feed = {
                    'SARA': custom_sara,
                    'MW_heavy_cut': 750.0,
                    'F_precip': float(np.clip(
                        0.28 + (custom_sara.get('asphaltenes', 12) - 10) / 80,
                        0.15, 0.50)),
                }
                comps = build_residue_distribution(
                    custom_feed=custom_feed, n_comp=20, solvent_name=solvent)
                ds = distribution_summary(comps, feed_name='custom')
                rho_feed = float(d.get('custom_density', 1.028))
            else:
                comps = build_residue_distribution(
                    feed_name=feed, n_comp=20, solvent_name=solvent)
                ds = distribution_summary(comps, feed_name=feed)

            # Build initial extractor params (forwarding P_bar and alpha_density)
            _extractor_kwargs = dict(
                components=comps, solvent_name=solvent, solvent_ratio=so,
                N_stages=N, T_profile=build_T_profile(T_bottom, T_top, N),
                kinetics=KineticParams(k_p, tau), efficiency=StageEfficiency(E),
                entrainment=EntrainmentParams(C_ent, 1.20),
                K_multiplier=K_mult, delta_crit=d_crit,
                predilution_frac=pred_frac,
                P=P_bar * 1e5, alpha_density=alpha_d,
                thermo_mode=thermo_mode,
            )
            r = run_extractor(**_extractor_kwargs)

            # Thermal mode: estimate bed temperatures from inlet conditions
            thermal_mode = d.get('thermal_mode', 'direct')
            thermal_info = {}
            if thermal_mode == 'inlet' or thermal_mode is True or d.get('use_inlet_temps'):
                from hydraulics_entrain import estimate_bed_temperatures as _ebt
                t_feed_mixed = float(d.get('T_feed_mixed',
                                           d.get('T_feed_C', 85.0)))
                t_prop_fresh = float(d.get('T_propane_fresh',
                                           d.get('T_propane_inlet_C', 65.0)))
                stm_flow  = float(d.get('steam_flow_kg_hr', 2830.0))
                bot_blend = float(d.get('bottom_T_blend',   THERMAL_DEFAULTS['bottom_T_blend']))
                mid_blend = float(d.get('middle_T_blend',   THERMAL_DEFAULTS['middle_T_blend']))
                stm_eff   = float(d.get('steam_effectiveness', THERMAL_DEFAULTS['steam_effectiveness']))
                stm_press = float(d.get('lp_steam_pressure', 4.5))  # kg/cm²g
                thermal_info = _ebt(
                    T_feed_mixed_C=t_feed_mixed,
                    T_propane_fresh_C=t_prop_fresh,
                    steam_flow_kg_hr=stm_flow,
                    feed_flow_kg_hr=feed_flow,
                    solvent_flow_kg_hr=feed_flow * so,
                    P_bar=P_bar,
                    bottom_blend=bot_blend,
                    middle_blend=mid_blend,
                    steam_effectiveness=stm_eff,
                    lp_steam_pressure_kgcm2g=stm_press)
                T_bottom = thermal_info['T_bottom_C']
                T_top    = thermal_info['T_top_C']
                # Re-run with thermally estimated bed temperatures
                _extractor_kwargs['T_profile'] = build_T_profile(T_bottom, T_top, N)
                r = run_extractor(**_extractor_kwargs)

            # Propane saturation check (Item F)
            sat_check = propane_saturation_check(T_top, P_bar)

            # Hydraulics + flow calculator (Fix 2: rho_feed from database, not DAO density)
            try:
                flow_data = compute_solvent_flows(
                    feed_flow, so, pred_frac, T_bottom, P_bar, rho_feed)
            except Exception:
                flow_data = {}
            total_kg_hr = feed_flow + flow_data.get('total_solvent_kg_hr', 0)
            rho_lgt = float(r.get('density_DAO',   0.92))
            rho_hvy = float(ds.get('mean_density', 1.02))
            hydr_warnings = check_column_hydraulics(total_kg_hr, rho_lgt, rho_hvy,
                                                    COLUMN_DEFAULTS['column_ID_bot_mm'],
                                                    n_extractors=n_ext)
            if sat_check['status'] != 'ok':
                hydr_warnings.append(sat_check['message'])
            hydr_met = hydraulic_metrics(total_kg_hr, rho_lgt, rho_hvy,
                                         COLUMN_DEFAULTS['column_ID_bot_mm'],
                                         n_extractors=n_ext)

            result = _clean_result(r, ds)
            result['N_stages_used']      = N
            result['hydraulic_warnings'] = hydr_warnings
            result['hydraulic_metrics']  = hydr_met
            result['flow_data']          = flow_data
            result['thermal_info']       = thermal_info
            result['sat_check']          = sat_check
            result['T_bottom']           = T_bottom
            result['T_top']              = T_top
            result['feed_density']       = round(rho_feed, 4)
            result['feed_flow_kg_hr']    = round(feed_flow, 0)

            # Downstream corrections (optional, default OFF)
            enable_downstream = d.get('enable_downstream', False)
            if enable_downstream:
                from downstream_corrections import apply_all_downstream
                result = apply_all_downstream(
                    result,
                    enable_flash=bool(d.get('enable_flash_precip', True)),
                    enable_residual=bool(d.get('enable_residual_solvent', True)),
                    flash_P_bar=float(d.get('flash_P_bar', 5.0)),
                    residual_solvent_pct=float(d.get('residual_solvent_pct', 0.5)),
                )

            _record_simulation(d, result)
            result['history'] = _simulation_history
            return jsonify(result)

        except Exception as e:
            import traceback
            return jsonify({'ok': False, 'error': str(e),
                            'trace': traceback.format_exc()})

    @app.route('/api/sensitivity', methods=['POST'])
    def api_sensitivity():
        try:
            from sensitivity_analysis import (
                sweep_so_ratio, sweep_temperature, sweep_efficiency, sweep_stages,
                sweep_predilution, sweep_gradient, sweep_yield_quality,
                sweep_operating_map, sweep_pressure,
                sweep_temperature_top, sweep_steam_effect,
                plot_so_ratio, plot_temperature, plot_efficiency, plot_stages,
                plot_predilution, plot_gradient, plot_yield_quality,
                plot_operating_map, plot_pressure,
                plot_temperature_top, plot_steam_effect,
            )
            d        = request.json or {}
            feed     = d.get('feed',    'basra_kuwait_mix')
            solvent  = d.get('solvent', 'propane')
            so       = float(d.get('so',           8.0))
            _T_def   = 75.0 if solvent == 'propane' else 140.0
            T_bottom = float(d.get('T_bottom', d.get('T_C', _T_def)))
            T_top    = float(d.get('T_top', T_bottom + (10.0 if solvent == 'propane' else 20.0)))
            N        = _derive_N_from_request(d)
            plot     = d.get('plot',  'so_ratio')
            K_mult   = float(d.get('K_multiplier', 1.0))
            d_crit   = float(d.get('delta_crit',   2.5))
            pred     = float(d.get('predilution_frac', 0.0))
            T_mean   = (T_bottom + T_top) / 2.0

            common = dict(feed_name=feed, solvent=solvent, N=N,
                          K_multiplier=K_mult, delta_crit=d_crit)

            if plot == 'so_ratio':
                data = sweep_so_ratio(**common, SO_range=None,
                                      T_bottom=T_bottom, T_top=T_top,
                                      predilution_frac=pred)
                fig_json = plot_so_ratio(data, solvent, feed, current_SO=so)

            elif plot == 'temperature':
                data = sweep_temperature(**common, SO=so, predilution_frac=pred)
                fig_json = plot_temperature(data, solvent, feed, current_T=T_bottom)

            elif plot == 'predilution':
                data = sweep_predilution(**common, SO=so, T_bottom=T_bottom, T_top=T_top)
                fig_json = plot_predilution(data, solvent, feed, design_pred=pred)

            elif plot == 'gradient':
                data = sweep_gradient(**common, SO=so, T_mean=T_mean,
                                      predilution_frac=pred)
                fig_json = plot_gradient(data, solvent, feed)

            elif plot == 'yield_quality':
                data = sweep_yield_quality(**common,
                                           T_bottom=T_bottom, T_top=T_top,
                                           predilution_frac=pred)
                fig_json = plot_yield_quality(data, solvent, feed, current_SO=so)

            elif plot == 'operating_map':
                data = sweep_operating_map(**common, predilution_frac=pred)
                fig_json = plot_operating_map(data, solvent, feed,
                                              current_SO=so, current_T=T_bottom)

            elif plot == 'pressure':
                data = sweep_pressure(feed_name=feed, solvent=solvent,
                                      T_bottom=T_bottom)
                fig_json = plot_pressure(data, solvent, feed)

            elif plot == 'efficiency':
                data = sweep_efficiency(**common, SO=so,
                                        T_bottom=T_bottom, T_top=T_top,
                                        predilution_frac=pred)
                fig_json = plot_efficiency(data, solvent, feed)

            elif plot == 'stages':
                data = sweep_stages(**common, SO=so,
                                    T_bottom=T_bottom, T_top=T_top,
                                    predilution_frac=pred)
                fig_json = plot_stages(data, solvent, feed)

            elif plot == 'temperature_top':
                data = sweep_temperature_top(**common, SO=so,
                                             T_bottom_fixed=T_bottom,
                                             predilution_frac=pred)
                P_bar_arg = float(d.get('P_kgcm2', 40.0)) * 0.980665
                fig_json = plot_temperature_top(data, solvent, feed,
                                                current_T_top=T_top,
                                                P_bar=P_bar_arg)

            elif plot == 'steam_effect':
                T_feed_mixed  = float(d.get('T_feed_mixed', 85.0))
                T_prop_fresh  = float(d.get('T_propane_fresh', 65.0))
                P_bar_arg     = float(d.get('P_kgcm2', 40.0)) * 0.980665
                data = sweep_steam_effect(T_feed_mixed=T_feed_mixed,
                                          T_propane_fresh=T_prop_fresh,
                                          P_bar=P_bar_arg)
                fig_json = plot_steam_effect(data, P_bar=P_bar_arg)

            else:
                fig_json = '{}'

            return jsonify({'ok': True, 'figure': json.loads(fig_json)})

        except Exception as e:
            import traceback
            return jsonify({'ok': False, 'error': str(e),
                            'trace': traceback.format_exc()})

    # ── Operating margins ─────────────────────────────────────────────────────
    @app.route('/api/operating_margins', methods=['POST'])
    def api_operating_margins():
        try:
            from sensitivity_analysis import compute_operating_margins
            d = request.json or {}
            baseline = {
                'feed_name':        d.get('feed', 'basra_kuwait_mix'),
                'solvent':          d.get('solvent', 'propane'),
                'SO_ratio':         float(d.get('so', 8.0)),
                'T_bottom':         float(d.get('T_bottom', 75.0)),
                'T_top':            float(d.get('T_top', 85.0)),
                'N_stages':         _derive_N_from_request(d),
                'K_multiplier':     float(d.get('K_multiplier', 1.0)),
                'delta_crit':       float(d.get('delta_crit', 2.5)),
                'predilution_frac': float(d.get('predilution_frac', 0.0)),
                'P_bar':            float(d.get('P_kgcm2', 40.0)) * 0.980665,
            }
            constraints = {
                'yield_loss_limit': float(d.get('yield_loss_limit', 1.0)),
                'viscosity_max':    float(d.get('viscosity_max', 200.0)),
                'colour_max':       float(d.get('colour_max', 6.0)),
            }
            margins = compute_operating_margins(baseline, constraints)
            return jsonify({'ok': True, 'margins': margins})
        except Exception as e:
            import traceback
            return jsonify({'ok': False, 'error': str(e), 'trace': traceback.format_exc()})

    # ── Trade-off explorer (Item D) ───────────────────────────────────────────
    @app.route('/api/tradeoff', methods=['POST'])
    def api_tradeoff():
        """Quick S/O sweep ±3 around current point for yield-vs-quality plot."""
        try:
            import plotly.graph_objects as go
            import plotly.utils

            d        = request.json or {}
            feed     = d.get('feed',    'basra_kuwait_mix')
            solvent  = d.get('solvent', 'propane')
            so_cur   = float(d.get('so', 8.0))
            _T_def   = 75.0 if solvent == 'propane' else 140.0
            T_bottom = float(d.get('T_bottom', _T_def))
            T_top    = float(d.get('T_top', T_bottom + (10.0 if solvent == 'propane' else 20.0)))
            N        = _derive_N_from_request(d)
            K_mult   = float(d.get('K_multiplier', 1.0))
            d_crit   = float(d.get('delta_crit', 2.5))
            pred     = float(d.get('predilution_frac', 0.0))

            # 7-point sweep around current S/O
            so_range = np.linspace(max(3, so_cur - 3), min(14, so_cur + 3), 7)
            comps    = build_residue_distribution(
                feed_name=feed if feed != 'custom' else 'basra_kuwait_mix',
                n_comp=20, solvent_name=solvent)

            points = []
            for so in so_range:
                T_profile = build_T_profile(T_bottom, T_top, N)
                r = run_extractor(
                    components=comps, solvent_name=solvent, solvent_ratio=float(so),
                    N_stages=N, T_profile=T_profile,
                    kinetics=KineticParams(), efficiency=StageEfficiency(),
                    entrainment=EntrainmentParams(), K_multiplier=K_mult,
                    delta_crit=d_crit, predilution_frac=pred)
                sara = r.get('SARA_DAO', {})
                visc   = float(predict_dao_viscosity(r['MW_DAO_avg'], r['density_DAO'], sara))
                colour = float(predict_astm_colour(r['asphal_contam_pct'], sara))
                points.append({
                    'SO': round(float(so), 2),
                    'yield':    round(float(r['DAO_yield_net']),  2),
                    'colour':   round(colour, 2),
                    'viscosity':round(visc, 1),
                    'contam':   round(float(r['asphal_contam_pct']), 3),
                })

            yields  = [p['yield']   for p in points]
            colours = [p['colour']  for p in points]
            viscs   = [p['viscosity'] for p in points]
            so_arr  = [p['SO']      for p in points]

            fig = go.Figure()

            # Zone backgrounds
            fig.add_shape(type='rect', x0=0, x1=60, y0=0, y1=4,
                          fillcolor='rgba(52,168,83,0.10)', line=dict(width=0), layer='below')
            fig.add_shape(type='rect', x0=0, x1=60, y0=4, y1=6,
                          fillcolor='rgba(244,180,0,0.08)', line=dict(width=0), layer='below')
            fig.add_shape(type='rect', x0=0, x1=60, y0=6, y1=9,
                          fillcolor='rgba(234,67,53,0.08)', line=dict(width=0), layer='below')
            fig.add_annotation(x=2, y=2.0, text='Lube Spec', showarrow=False,
                               font=dict(color='#34a853', size=9), xanchor='left')
            fig.add_annotation(x=2, y=5.0, text='FCC OK',    showarrow=False,
                               font=dict(color='#f4b400', size=9), xanchor='left')
            fig.add_annotation(x=2, y=7.2, text='Reject',    showarrow=False,
                               font=dict(color='#ea4335', size=9), xanchor='left')

            # Connecting line
            fig.add_trace(go.Scatter(x=yields, y=colours,
                mode='lines', line=dict(color='rgba(255,255,255,0.25)', width=1.5),
                showlegend=False))

            # Points
            fig.add_trace(go.Scatter(
                x=yields, y=colours, mode='markers+text',
                marker=dict(size=10, color=so_arr, colorscale='Viridis',
                            colorbar=dict(title='S/O', len=0.6, thickness=12),
                            line=dict(color='white', width=0.8)),
                text=[f'{s:.1f}' for s in so_arr],
                textposition='top center',
                textfont=dict(size=8, color='#ccc'),
                customdata=[[v] for v in viscs],
                hovertemplate='S/O: %{text}<br>Yield: %{x:.1f}%<br>Colour: %{y:.2f}<br>Visc: %{customdata[0]:.1f} cSt<extra></extra>',
                name='S/O sweep'))

            # Current point star
            cur_idx = int(np.argmin(np.abs(np.array(so_arr) - so_cur)))
            fig.add_trace(go.Scatter(
                x=[yields[cur_idx]], y=[colours[cur_idx]], mode='markers',
                marker=dict(symbol='star', size=16, color='gold',
                            line=dict(color='white', width=1.2)),
                name=f'Current (S/O={so_cur:.1f})', showlegend=True))

            fig.add_hline(y=6.0, line=dict(color='#ea4335', dash='dot', width=1.5),
                          annotation_text='Colour limit 6.0',
                          annotation_font=dict(color='#ea4335', size=9))

            fig.update_layout(
                paper_bgcolor='#0f1117', plot_bgcolor='#1a1d27',
                font=dict(family='Inter, sans-serif', color='#e0e0e0', size=11),
                margin=dict(l=50, r=20, t=35, b=45),
                xaxis=dict(title='DAO Yield  [wt%]', gridcolor='#2a2d3a', range=[0, max(45, max(yields)+3)]),
                yaxis=dict(title='ASTM Colour', gridcolor='#2a2d3a', range=[0, 8.5]),
                legend=dict(bgcolor='rgba(30,34,48,0.8)', bordercolor='#333', borderwidth=1),
                title=dict(text='Yield vs Quality Trade-off  (S/O ±3 sweep)',
                           font=dict(size=13)),
            )
            fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            return jsonify({'ok': True, 'figure': json.loads(fig_json), 'points': points})

        except Exception as e:
            import traceback
            return jsonify({'ok': False, 'error': str(e),
                            'trace': traceback.format_exc()})

    # ── Calibration: list profiles ────────────────────────────────────────────
    @app.route('/api/calibration/profiles', methods=['GET'])
    def api_calib_profiles():
        try:
            from plant_calibration import list_profiles
            return jsonify({'ok': True, 'profiles': list_profiles()})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)})

    # ── Calibration: load a profile ───────────────────────────────────────────
    @app.route('/api/calibration/load', methods=['POST'])
    def api_calib_load():
        try:
            from plant_calibration import load_profile
            name   = (request.json or {}).get('profile', 'hpcl_default')
            params = load_profile(name)
            return jsonify({'ok': True, 'params': params, 'profile': name})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)})

    # ── Calibration: save current params as profile ───────────────────────────
    @app.route('/api/calibration/save', methods=['POST'])
    def api_calib_save():
        try:
            from plant_calibration import save_profile
            d    = request.json or {}
            name = d.get('name') or d.get('profile', 'hpcl_custom')
            # Accept params dict directly or individual keys
            params = d.get('params', {})
            if not params:
                params = {k: float(d.get(k, v))
                          for k, v in {'K_multiplier':1.0,'C_entrain':0.015,
                                       'k_precip':0.5,'E_murphree':0.70,
                                       'delta_crit':2.5}.items()}
            save_profile(name, params, d.get('description', ''))
            return jsonify({'ok': True, 'profile': name})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)})

    # ── Calibration: run from uploaded CSV ───────────────────────────────────
    @app.route('/api/calibration/run', methods=['POST'])
    def api_calib_run():
        try:
            import tempfile
            from plant_calibration import (
                load_plant_data, run_calibration, CalibrationWeights,
                DEFAULT_PARAMS, plot_calibration_results,
            )
            d         = request.json or {}
            csv_data  = d.get('csv_data', '')
            init_p    = d.get('init_params', {})
            w         = d.get('weights', {})
            save_name = d.get('save_profile', None)

            if not csv_data.strip():
                return jsonify({'ok': False, 'error': 'No CSV data provided'})

            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv',
                                             delete=False, encoding='utf-8') as tf:
                tf.write(csv_data)
                tmp_path = tf.name

            dataset = load_plant_data(tmp_path)
            os.unlink(tmp_path)

            init_params = DEFAULT_PARAMS.copy()
            init_params.update({k: float(v) for k, v in init_p.items()
                                if k in init_params})
            weights = CalibrationWeights(
                DAO_yield       = float(w.get('DAO_yield',       1.00)),
                DAO_viscosity   = float(w.get('DAO_viscosity',   0.02)),
                DAO_astm_colour = float(w.get('DAO_astm_colour', 0.80)),
                DAO_CCR         = float(w.get('DAO_CCR',         0.50)),
                DAO_density     = float(w.get('DAO_density',    50.0)),
                asph_contam     = float(w.get('asph_contam',     5.00)),
            )

            result = run_calibration(
                dataset=dataset, init_params=init_params,
                weights=weights, save_profile=save_name,
                max_nfev=150, verbose=True,
            )
            fig_json = plot_calibration_results(result)

            return jsonify({
                'ok':                True,
                'success':           result.success,
                'calibrated_params': result.calibrated_params,
                'cost_initial':      result.cost_initial,
                'cost_final':        result.cost_final,
                'improvement_pct':   result.improvement_pct,
                'n_points':          result.n_operating_points,
                'n_evals':           result.n_function_evals,
                'metrics':           result.metrics,
                'point_results':     result.point_results,
                'message':           result.message,
                'elapsed_s':         result.elapsed_s,
                'figure':            json.loads(fig_json),
            })
        except Exception as e:
            import traceback
            return jsonify({'ok': False, 'error': str(e),
                            'trace': traceback.format_exc()})

    # ── Calibration: load by name via GET (for UI) ───────────────────────────
    @app.route('/api/calibration/load/<name>', methods=['GET'])
    def api_calib_load_by_name(name):
        try:
            from plant_calibration import load_profile
            params = load_profile(name)
            return jsonify({'ok': True, 'params': params, 'profile': name})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)})

    # ── Excel export ──────────────────────────────────────────────────────────
    @app.route('/api/export_excel', methods=['POST'])
    def api_export_excel():
        try:
            import io
            import pandas as pd
            from flask import send_file

            d = request.json or {}

            summary_data = {
                'Parameter': ['Feed', 'Solvent', 'S/O Ratio', 'T Bottom [C]', 'T Top [C]',
                              'Stages', 'Predilution', 'DAO Yield [wt%]', 'Asphalt Yield [wt%]',
                              'DAO Density [g/cm3]', 'DAO Viscosity [cSt]', 'ASTM Colour',
                              'Asph. Contamination [wt%]'],
                'Value': [d.get('feed', ''), d.get('solvent', ''),
                          d.get('so', ''), d.get('T_bottom', ''), d.get('T_top', ''),
                          d.get('stages', ''), d.get('predilution', ''),
                          d.get('DAO_yield', ''), d.get('asphalt_yield', ''),
                          d.get('density', ''), d.get('viscosity', ''),
                          d.get('colour', ''), d.get('asph_contam', '')],
            }

            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                pd.DataFrame(summary_data).to_excel(
                    writer, sheet_name='Summary', index=False)
                stage_data = d.get('stage_results', [])
                if stage_data:
                    pd.DataFrame(stage_data).to_excel(
                        writer, sheet_name='Stage Data', index=False)
                sara_data = d.get('SARA_DAO', {})
                if sara_data:
                    pd.DataFrame([sara_data]).to_excel(
                        writer, sheet_name='SARA', index=False)
                # Sheet 4: Hydraulics
                hw_warnings = d.get('hydraulic_warnings', [])
                hw_metrics  = d.get('hydraulic_metrics', {})
                hyd_rows = []
                if hw_metrics:
                    for k, v in hw_metrics.items():
                        hyd_rows.append({'Item': k, 'Value': v, 'Type': 'metric'})
                for w in hw_warnings:
                    hyd_rows.append({'Item': 'warning', 'Value': w, 'Type': 'warning'})
                if hyd_rows:
                    pd.DataFrame(hyd_rows).to_excel(
                        writer, sheet_name='Hydraulics', index=False)

            buffer.seek(0)
            return send_file(
                buffer, download_name='SDA_Simulation.xlsx',
                as_attachment=True,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        except Exception as e:
            import traceback
            return jsonify({'ok': False, 'error': str(e),
                            'trace': traceback.format_exc()})

    # ── Sample CSV download ───────────────────────────────────────────────────
    @app.route('/api/calibration/sample_csv', methods=['GET'])
    def api_sample_csv():
        try:
            import tempfile
            from plant_calibration import make_sample_csv
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv',
                                             delete=False, encoding='utf-8') as tf:
                path = tf.name
            make_sample_csv(path)
            with open(path) as f:
                csv_text = f.read()
            os.unlink(path)
            return jsonify({'ok': True, 'csv': csv_text})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)})

    # ── Server shutdown ───────────────────────────────────────────────────────
    @app.route('/api/shutdown', methods=['POST'])
    def api_shutdown():
        """
        Gracefully stop the Werkzeug development server.

        Posts a 0.5 s delayed SIGTERM so the HTTP response reaches the browser
        before the process exits.  Safe to call from the UI kill button.
        """
        import signal

        def _deferred_kill():
            import time
            time.sleep(0.5)
            print("\n[SDA] Server stopped by UI kill button.", flush=True)
            os.kill(os.getpid(), signal.SIGTERM)

        t = threading.Thread(target=_deferred_kill, daemon=True)
        t.start()
        return jsonify({'ok': True, 'message': 'Server shutting down in 0.5 s'})

    # ── Workbook calibration: run full two-loop calibration ──────────────────
    @app.route('/api/calibrate_workbooks', methods=['POST'])
    def api_calibrate_workbooks():
        """
        Accept two xlsx files via multipart form upload.
        Run full two-loop calibration. Return JSON result.

        Form fields:
          lims_file    : xlsx file (LIMS workbook)
          dcs_file     : xlsx file (extractor parameters)
          profile_name : str (optional, default 'plant_calibration_v1')
          weight_yield : float (optional, default 0.4)
          weight_visc  : float (optional, default 1.0)
        """
        import tempfile
        import traceback
        lims_tmp = None
        dcs_tmp  = None
        try:
            from calibration_engine import run_full_calibration

            lims_file = request.files.get('lims_file')
            dcs_file  = request.files.get('dcs_file')
            if not lims_file or not dcs_file:
                return jsonify({'ok': False,
                                'error': 'Both lims_file and dcs_file are required'})

            profile_name = request.form.get('profile_name', 'plant_calibration_v1')
            w_yield = float(request.form.get('weight_yield', 0.4))
            w_visc  = float(request.form.get('weight_visc', 1.0))
            enable_pinn = request.form.get('enable_pinn', '0') in ('1', 'true', 'True')
            # Plant operates in DAO mode only — mode selector removed (Issue 2).

            # Save to temp files — never write to project directory
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tf:
                lims_file.save(tf)
                lims_tmp = tf.name
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tf:
                dcs_file.save(tf)
                dcs_tmp = tf.name

            result = run_full_calibration(
                dcs_filepath=dcs_tmp,
                lims_filepath=lims_tmp,
                weights={'yield': w_yield, 'visc': w_visc},
                profile_name=profile_name,
                enable_pinn=enable_pinn,
            )

            # Flatten parity data for JSON response
            mb = result['metrics_before']
            ma = result['metrics_after']
            parity = ma.get('parity_data', {})

            return jsonify(_to_json_safe({
                'ok':                 True,
                'calibrated_params':  result['calibrated_params'],
                'thermal_params':     result['thermal_params'],
                'metrics_before':     {
                    'visc':  mb.get('visc', {}),
                    'yield': mb.get('yield', {}),
                },
                'metrics_after': {
                    'visc':  ma.get('visc', {}),
                    'yield': ma.get('yield', {}),
                },
                'thermal_metrics': {
                    'train_a': result['thermal_params'].get('train_a', {}),
                    'train_b': result['thermal_params'].get('train_b', {}),
                },
                'dataset_info':      result['dataset_info'],
                'split_info':        result.get('split_info', {}),
                'parity_data':       parity,
                'optimizer_success':  result['optimizer_success'],
                'optimizer_message':  result['optimizer_message'],
                'profile_name':       result['profile_name'],
                'calibration_mode':   result.get('calibration_mode', 'dao_only'),
                'correction_mode':    result.get('correction_mode', 'ols'),
                'pinn_result':        result.get('pinn_result'),
            }))

        except Exception as e:
            return jsonify({'ok': False, 'error': str(e),
                            'traceback': traceback.format_exc()}), 500
        finally:
            for path in [lims_tmp, dcs_tmp]:
                if path:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

    # ── Workbook calibration: return thermal metrics ──────────────────────────
    @app.route('/api/thermal_metrics', methods=['GET'])
    def api_thermal_metrics():
        """
        Load and return thermal_profile.json if it exists.
        Returns 404 with clear message if not yet calibrated.
        """
        try:
            from thermal_calibration import load_thermal_profile
            profile = load_thermal_profile()
            return jsonify({'ok': True, 'thermal_profile': profile})
        except FileNotFoundError as e:
            return jsonify({'ok': False, 'error': str(e)}), 404
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)}), 500

    # ── Workbook calibration: dataset info (no calibration) ──────────────────
    @app.route('/api/calibration_dataset_info', methods=['POST'])
    def api_calibration_dataset_info():
        """
        Accept two xlsx files, run only plant_data_loader (no calibration).
        Return dataset_info for the UI to show before committing to full calibration.
        Allows user to verify data quality before running optimisation.
        """
        import tempfile
        import traceback
        lims_tmp = None
        dcs_tmp  = None
        try:
            from plant_data_loader import build_calibration_dataset

            lims_file = request.files.get('lims_file')
            dcs_file  = request.files.get('dcs_file')
            if not lims_file or not dcs_file:
                return jsonify({'ok': False,
                                'error': 'Both lims_file and dcs_file are required'})

            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tf:
                lims_file.save(tf)
                lims_tmp = tf.name
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tf:
                dcs_file.save(tf)
                dcs_tmp = tf.name

            dataset = build_calibration_dataset(dcs_tmp, lims_tmp)
            return jsonify(_to_json_safe({
                'ok':           True,
                'dataset_info': dataset['dataset_info'],
            }))

        except Exception as e:
            return jsonify({'ok': False, 'error': str(e),
                            'traceback': traceback.format_exc()}), 500
        finally:
            for path in [lims_tmp, dcs_tmp]:
                if path:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

    @app.route('/api/run_diagnostic', methods=['POST'])
    def api_run_diagnostic():
        """
        Accept two xlsx files. Run Stages 0-4 only (no optimizer).
        Returns full diagnostic report in ~20 seconds.
        Used by UI 'Run Diagnostic' button before committing to full calibration.

        Form fields: lims_file, dcs_file (same as /api/calibrate_workbooks)
        """
        import tempfile
        import traceback
        lims_tmp = None
        dcs_tmp  = None
        try:
            from plant_data_loader import build_calibration_dataset
            from thermal_calibration import calibrate_thermal_model, save_thermal_profile, load_thermal_profile
            from calibration_engine import _DEFAULT_PARAMS, _prepopulate_feed_cache
            from diagnostic_pipeline import run_diagnostic_pipeline
            import pandas as pd

            lims_file = request.files.get('lims_file')
            dcs_file  = request.files.get('dcs_file')
            if not lims_file or not dcs_file:
                return jsonify({'ok': False,
                                'error': 'Both lims_file and dcs_file are required'})

            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tf:
                lims_file.save(tf)
                lims_tmp = tf.name
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tf:
                dcs_file.save(tf)
                dcs_tmp = tf.name

            dataset = build_calibration_dataset(dcs_tmp, lims_tmp)
            dcs_hourly    = dataset['dcs_hourly']
            visc_anchored = dataset['visc_anchored']
            dataset_info  = dataset['dataset_info']

            # Load or build thermal profile
            thermal_path = os.path.join('calibration_profiles', 'thermal_profile.json')
            try:
                thermal_profile = load_thermal_profile(thermal_path)
            except FileNotFoundError:
                thermal_profile = calibrate_thermal_model(dcs_hourly)
                os.makedirs('calibration_profiles', exist_ok=True)
                save_thermal_profile(thermal_profile, thermal_path)

            # Build feed cache (small subset for diagnostic)
            feed_cache: dict = {}
            sample_rows = visc_anchored.head(100) if len(visc_anchored) > 0 \
                else dcs_hourly.head(100)
            _prepopulate_feed_cache(sample_rows, feed_cache)

            default_params = dict(_DEFAULT_PARAMS)
            diag = run_diagnostic_pipeline(
                dcs_hourly, visc_anchored, thermal_profile, default_params, feed_cache
            )

            # Flatten stages for JSON (convert int keys to strings)
            stages_serializable = {
                str(k): v for k, v in diag['stages'].items()
            }

            return jsonify(_to_json_safe({
                'ok':             True,
                'pipeline_pass':  diag['pipeline_pass'],
                'blocking_stage': diag['blocking_stage'],
                'stages':         stages_serializable,
                'summary':        diag['summary'],
                'total_time_sec': diag['total_time_sec'],
                'dataset_info':   dataset_info,
            }))

        except Exception as e:
            return jsonify({'ok': False, 'error': str(e),
                            'traceback': traceback.format_exc()}), 500
        finally:
            for path in [lims_tmp, dcs_tmp]:
                if path:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

    port = 5000
    print(f"\n{'='*58}")
    print(f"  SDA Unit Simulator Web UI")
    print(f"  ->  http://localhost:{port}")
    print(f"  Press Ctrl+C to stop")
    print(f"{'='*58}\n")
    threading.Timer(1.5, lambda: webbrowser.open(f'http://localhost:{port}')).start()
    app.run(debug=False, port=port, use_reloader=False)


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='SDA Unit Simulator — Propane Deasphalting Digital Twin')
    parser.add_argument('--feed',    default='basra_kuwait_mix',
                        choices=['basra_kuwait_mix', 'basra_light'])
    parser.add_argument('--solvent', default='propane',
                        choices=['propane', 'butane'])
    parser.add_argument('--so',        type=float, default=8.0,
                        help='Solvent/Oil ratio (default: 8.0)')
    parser.add_argument('--T',         type=float, default=None,
                        help='Legacy flat temperature °C (default: 75 propane / 140 butane)')
    parser.add_argument('--T-bottom',  type=float, default=None,
                        help='Bottom (feed-end) temperature °C (default: same as --T)')
    parser.add_argument('--T-top',     type=float, default=None,
                        help='Top (solvent-end) temperature °C (default: T_bottom+10/+20)')
    parser.add_argument('--predilution', type=float, default=0.0,
                        help='Pre-dilution fraction 0–0.5 (split-solvent; default: 0)')
    parser.add_argument('--HETP',      type=float, default=2000.0,
                        help='HETP [mm] for stage count (default: 2000 → 4 stages)')
    parser.add_argument('--no-ui',    action='store_true',
                        help='Print results only, do not start web server')
    parser.add_argument('--verbose',  action='store_true')
    parser.add_argument('--calibrate', metavar='CSV',
                        help='Run calibration against plant CSV then exit')
    parser.add_argument('--profile',   default=None,
                        help='Calibration profile name to save/load')
    parser.add_argument('--weights',   default=None,
                        help='Calibration weights: dao=1.0,ccr=0.5,rho=50,asp=5')
    args = parser.parse_args()

    # ── Calibration-only mode ─────────────────────────────────────────────────
    if args.calibrate:
        from plant_calibration import (load_plant_data, run_calibration,
                                       load_profile, CalibrationWeights,
                                       DEFAULT_PARAMS)
        csv_path = args.calibrate
        profile  = args.profile or 'sda_calibrated'

        # Parse weights from CLI: --weights dao=1.0,ccr=0.5,rho=50,asp=5
        weights = CalibrationWeights()
        if args.weights:
            wmap = {'dao': 'DAO_yield', 'ccr': 'DAO_CCR',
                    'rho': 'DAO_density', 'asp': 'asph_contam'}
            for pair in args.weights.split(','):
                k, v = pair.strip().split('=')
                attr = wmap.get(k.strip().lower())
                if attr:
                    setattr(weights, attr, float(v))

        # Load initial params from profile if specified
        init_params = DEFAULT_PARAMS.copy()
        if args.profile and args.profile != profile:
            init_params = load_profile(args.profile)

        print(f"\n  Calibration mode: '{csv_path}'  ->  profile '{profile}'")
        dataset = load_plant_data(csv_path)
        result  = run_calibration(dataset, init_params=init_params,
                                  weights=weights, save_profile=profile,
                                  verbose=True)
        print(f"\n  Done.  Cost: {result.cost_initial:.3f} -> {result.cost_final:.3f}"
              f"  ({result.improvement_pct:.1f}% improvement)")
        sys.exit(0)

    T_C_default = 75.0 if args.solvent == 'propane' else 140.0
    T_C         = args.T if args.T is not None else T_C_default
    T_bottom    = getattr(args, 'T_bottom', None) or T_C
    T_top       = getattr(args, 'T_top',    None) or (T_bottom + (10.0 if args.solvent == 'propane' else 20.0))

    _n_cli = len(beds_to_stages(HETP_mm=args.HETP, bed_config=BED_CONFIG))
    print(f"\nSimulating: {args.feed} | {args.solvent} | "
          f"S/O={args.so} | T={T_bottom:.0f}-{T_top:.0f}°C | {_n_cli} stages"
          f" (HETP={args.HETP:.0f}mm) | predilution={args.predilution:.3f}")

    # Load calibration profile for runtime if --profile specified
    extra_kw = {}
    if args.profile:
        from plant_calibration import load_profile
        cal_params = load_profile(args.profile)
        extra_kw = {
            'K_multiplier': cal_params.get('K_multiplier', 1.0),
            'delta_crit':   cal_params.get('delta_crit', 2.5),
        }
        print(f"  Using calibration profile: '{args.profile}'  "
              f"(K_mult={extra_kw['K_multiplier']:.3f}, "
              f"delta_crit={extra_kw['delta_crit']:.2f})")

    r = run_base_case(args.feed, args.solvent, args.so, T_C,
                      T_bottom=T_bottom, T_top=T_top,
                      predilution_frac=args.predilution,
                      HETP_mm=args.HETP,
                      verbose=args.verbose, **extra_kw)
    print_summary(r)

    if not args.no_ui:
        launch_web_ui()


if __name__ == '__main__':
    main()


# ── Module-level Flask app (for import / testing) ────────────────────────────
# The full Flask app with all route implementations is created inside
# launch_web_ui(). This module-level instance registers the same URL rules
# as stub no-ops so that `from run_simulation import app; app.url_map`
# works without launching the server (e.g. T-05 verification).
try:
    from flask import Flask as _Flask

    def _noop(*_a, **_kw):
        return ('', 204)

    app = _Flask(__name__)
    _routes = [
        ('/',                                    'index',                 ['GET']),
        ('/api/simulate',                        'api_simulate',          ['POST']),
        ('/api/sensitivity',                     'api_sensitivity',       ['POST']),
        ('/api/operating_margins',               'api_operating_margins', ['POST']),
        ('/api/tradeoff',                        'api_tradeoff',          ['POST']),
        ('/api/calibration/profiles',            'api_calib_profiles',    ['GET']),
        ('/api/calibration/load',                'api_calib_load',        ['POST']),
        ('/api/calibration/save',                'api_calib_save',        ['POST']),
        ('/api/calibration/run',                 'api_calib_run',         ['POST']),
        ('/api/calibration/load/<name>',         'api_calib_load_by_name',['GET']),
        ('/api/calibration/sample_csv',          'api_sample_csv',        ['GET']),
        ('/api/export_excel',                    'api_export_excel',      ['POST']),
        ('/api/calculate_temperatures',          'api_calculate_temps',   ['POST']),
        ('/api/characterise_feed',               'api_characterise_feed', ['POST']),
        ('/api/simulation_history',              'api_sim_history',       ['GET']),
        ('/api/calibrate_workbooks',             'api_calibrate_workbooks',['POST']),
        ('/api/thermal_metrics',                 'api_thermal_metrics',   ['GET']),
        ('/api/calibration_dataset_info',        'api_calib_dataset_info',['POST']),
    ]
    for _rule, _ep, _methods in _routes:
        app.add_url_rule(_rule, endpoint=_ep,
                         view_func=_noop, methods=_methods)
except Exception:
    app = None  # flask not installed — graceful degradation
