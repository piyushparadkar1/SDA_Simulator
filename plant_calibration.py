"""
plant_calibration.py
====================
Plant calibration framework for PDA Unit Simulator.

DESIGN TARGETS FROM OPERATING MANUAL (Chapter 4)
-------------------------------------------------
  Lube Bright Stock DAO (propane, T~60°C, S/O=8):
    DAO yield  = 18 wt%,  density = 0.926 g/cm3,  CCR = 1.5%,  asph < 100 wppm
  FCC Feed DAO (propane, T~55°C, S/O=8):
    DAO yield  = 32 wt%,  density = 0.936 g/cm3,  CCR = 2.5%,  asph < 200 wppm

CALIBRATABLE PARAMETERS
-----------------------
  K_multiplier  : scales all LLE K-values uniformly (primary yield lever)
  C_entrain     : asphalt-into-DAO entrainment coefficient
  k_precip      : asphaltene precipitation rate constant [s-1]
  E_murphree    : Murphree stage efficiency
  delta_crit    : solubility parameter spread modifier (shifts aro/res/asp K-ratio)

USAGE
-----
  from plant_calibration import load_plant_data, run_calibration
  dataset = load_plant_data('plant_data.csv')
  result  = run_calibration(dataset, save_profile='hpcl_basra_mix')

  # CLI:
  python run_simulation.py --calibrate plant_data.csv
"""

import json
import os
import time
import numpy as np
from copy import deepcopy
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict
from scipy.optimize import least_squares

from residue_distribution  import build_residue_distribution, estimate_sara_from_properties
from hunter_nash_extractor import run_extractor
from asphaltene_kinetics   import KineticParams
from stage_efficiency      import StageEfficiency
from entrainment_model     import EntrainmentParams
from run_simulation        import build_T_profile

PROFILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'calibration_profiles')
os.makedirs(PROFILES_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# GENERIC LABELS (data privacy — no proprietary crude/unit names in UI)
# ─────────────────────────────────────────────────────────────────────────────
# Internal code uses short feed keys; UI/exports use these generic labels.

GENERIC_LABELS = {
    'basra_kuwait_mix': 'Heavy VR Blend A  (Design)',
    'basra_light':      'Heavy VR Blend B  (Check)',
}

def generic_feed_label(feed_name: str) -> str:
    """Return a privacy-safe display label for a feed key."""
    return GENERIC_LABELS.get(feed_name, feed_name)


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT PARAMETERS & BOUNDS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PARAMS = {
    'K_multiplier':       1.00,
    'C_entrain':          0.015,
    'k_precip':           0.50,
    'E_murphree':         0.70,
    'delta_crit':         2.50,
    # Step 5: ROSE thermal model parameters
    'alpha_density':      3.00,   # density-power exponent in K_value()
    'bottom_T_blend':     0.35,   # bottom bed T blend fraction
    'middle_T_blend':     0.55,   # middle bed T blend fraction
    'steam_effectiveness':0.60,   # fraction of steam enthalpy used for heating
}

PARAM_BOUNDS = {
    'K_multiplier':       (0.30, 3.50),
    'C_entrain':          (0.001, 0.10),
    'k_precip':           (0.05, 5.00),
    'E_murphree':         (0.30, 1.00),
    'delta_crit':         (0.50, 8.00),
    'alpha_density':      (1.00, 7.00),
    'bottom_T_blend':     (0.10, 0.70),
    'middle_T_blend':     (0.20, 0.80),
    'steam_effectiveness':(0.20, 1.00),
}

PARAM_NAMES = list(DEFAULT_PARAMS.keys())

TARGET_MAE = {
    'DAO_yield':      2.0,
    'DAO_viscosity':  30.0,   # cSt  — generous given correlation uncertainty
    'DAO_astm_colour': 0.8,   # ASTM colour units
    'DAO_CCR':        0.5,    # optional, only used when data available
    'DAO_density':    0.008,  # g/cm³
    'asph_contam':    0.05,   # wt%   — optional
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PlantDataPoint:
    timestamp:              str   = ''
    feed_name:              str   = ''          # OPTIONAL, informational only
    feed_density_kg_m3:     float = 1028.0      # PRIMARY: density [kg/m3]
    feed_CCR:               float = 22.8
    feed_visc_100:          float = 1621.0      # viscosity @100°C [cSt]
    feed_API:               float = None        # COMPUTED from density if None
    SO_ratio:               float = 8.0
    feed_temp_after_mixing_C: float = 75.0     # feed temp after predilution [°C]
    N_stages:               int   = 4          # kept for internal use
    solvent:                str   = 'propane'
    # ── Measured outputs (ALL optional — use whatever you have) ───────────────
    DAO_yield:              float = None        # Measured DAO yield [wt%]
    DAO_density:            float = None        # Measured DAO density [g/cm³]
    DAO_CCR:                float = None        # Measured DAO CCR [wt%] (optional)
    asph_contam:            float = None        # Asphaltene contamination [wt%] (optional)
    DAO_viscosity:          float = None        # Measured DAO viscosity [cSt @100°C]
    DAO_astm_colour:        float = None        # Measured ASTM D1500 colour [0-8]
    # ── Extractor temperature readings (from DCS — improve calibration) ───────
    T_bottom:               float = None        # Extractor bottom bed temp [°C]
    T_top:                  float = None        # Extractor top bed temp [°C]
    T_propane_fresh:        float = None        # Propane inlet temp [°C]
    # ── Feed flow ─────────────────────────────────────────────────────────────
    feed_flow_m3hr:         float = None        # Total feed flow [m³/hr] (optional)
    predilution_frac:       float = 0.0

    def __post_init__(self):
        # Compute API from density if not provided
        if self.feed_API is None and self.feed_density_kg_m3 is not None:
            sg = self.feed_density_kg_m3 / 1000.0
            self.feed_API = round(141.5 / sg - 131.5, 2)

    @property
    def temperature(self):
        """Backward compat alias for feed_temp_after_mixing_C"""
        return self.feed_temp_after_mixing_C

    @property
    def has_measurements(self):
        return any(v is not None for v in [
            self.DAO_yield, self.DAO_density, self.DAO_CCR,
            self.asph_contam, self.DAO_viscosity, self.DAO_astm_colour,
        ])


@dataclass
class CalibrationWeights:
    DAO_yield:        float = 1.00   # primary target — always weight ≥1
    DAO_viscosity:    float = 0.02   # cSt (large numbers) — small weight
    DAO_astm_colour:  float = 0.80   # ASTM colour units
    DAO_CCR:          float = 0.50   # optional: only used when data present
    DAO_density:      float = 50.0   # g/cm³ (tiny numbers) — large weight
    asph_contam:      float = 5.00   # optional: only used when data present

    def as_dict(self):
        return asdict(self)


@dataclass
class CalibrationResult:
    success:               bool
    calibrated_params:     dict
    initial_params:        dict
    cost_initial:          float
    cost_final:            float
    improvement_pct:       float
    n_operating_points:    int
    n_function_evals:      int
    metrics:               dict
    point_results:         list
    message:               str
    elapsed_s:             float


# ─────────────────────────────────────────────────────────────────────────────
# CSV LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_plant_data(csv_path: str) -> List[PlantDataPoint]:
    """
    Load plant operating data from a CSV file.

    Supports new column format (feed_density_kg_m3, feed_CCR, feed_visc_100,
    feed_temp_after_mixing) and backward compat with old column names.

    Minimum required: SO_ratio, temperature/feed_temp_after_mixing, and at
    least one output column. Missing columns are filled with defaults.
    """
    import csv

    def _f(row, key, default=None):
        val = row.get(key, '').strip()
        if val == '' or val.lower() in ('n/a', 'na', 'none', '-'):
            return default
        try:
            return float(val)
        except ValueError:
            return val

    def _i(row, key, default=4):
        try:
            return int(float(row.get(key, str(default)).strip()))
        except (ValueError, TypeError):
            return default

    points = []
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row = {k.strip().lower(): v for k, v in raw_row.items() if k}
            # Support both old (feed_API) and new (feed_density) columns
            density = _f(row, 'feed_density_kg_m3') or _f(row, 'feed_density')
            if density is None:
                # Fall back: compute from API
                api_val = _f(row, 'feed_api', 6.1)
                density = 1000.0 * 141.5 / (float(api_val) + 131.5)
            # Feed temp: support new column name with fallback to old
            feed_temp = (_f(row, 'feed_temp_after_mixing') or
                        _f(row, 'temperature', 75.0))
            visc = _f(row, 'feed_visc_100') or _f(row, 'feed_visc', 1000.0)
            # Extractor temperature readings — support multiple column aliases
            T_bottom = (_f(row, 't_bottom') or
                        _f(row, 'extractor_bot_temp') or
                        _f(row, 'extractor_bottom_temp'))
            T_top    = (_f(row, 't_top') or
                        _f(row, 'extractor_top_temp'))
            T_propane = (_f(row, 't_propane_fresh') or
                         _f(row, 'propane_temp') or
                         _f(row, 'propane_inlet_temp'))

            pt = PlantDataPoint(
                timestamp               = row.get('timestamp', ''),
                feed_name               = (row.get('feed_name', '') or '').strip(),
                feed_density_kg_m3      = float(density),
                feed_CCR                = _f(row, 'feed_ccr',    22.8),
                feed_visc_100           = float(visc) if visc else 1000.0,
                SO_ratio                = _f(row, 'so_ratio',    8.0),
                feed_temp_after_mixing_C= float(feed_temp),
                N_stages                = _i(row, 'n_stages',    4),
                solvent                 = (row.get('solvent', '') or 'propane').strip(),
                DAO_yield               = _f(row, 'dao_yield'),
                DAO_density             = _f(row, 'dao_density'),
                DAO_CCR                 = _f(row, 'dao_ccr'),
                asph_contam             = _f(row, 'asph_contam'),
                DAO_viscosity           = (_f(row, 'dao_viscosity') or
                                           _f(row, 'dao_visc')),
                DAO_astm_colour         = (_f(row, 'dao_astm_colour') or
                                           _f(row, 'dao_colour') or
                                           _f(row, 'astm_colour')),
                T_bottom                = T_bottom,
                T_top                   = T_top,
                T_propane_fresh         = T_propane,
                feed_flow_m3hr          = (_f(row, 'feed_flow_m3hr') or
                                           _f(row, 'feed_flow')),
                predilution_frac        = _f(row, 'predilution_frac', 0.0) or 0.0,
            )
            points.append(pt)
    if not points:
        raise ValueError(f"No data rows found in '{csv_path}'")
    print(f"  Loaded {len(points)} plant operating point(s) from '{csv_path}'")
    return points


def make_sample_csv(path: str = 'sample_plant_data.csv') -> str:
    """
    Write a sample plant data CSV by running the simulator at design conditions
    and adding realistic Gaussian noise to outputs.

    New format uses density_kg_m3, feed_CCR, feed_visc_100, feed_temp_after_mixing.
    """
    np.random.seed(42)

    LUBE_PARAMS = {'K_multiplier': 0.80, 'delta_crit': 2.8,
                   'C_entrain': 0.015, 'k_precip': 0.5, 'E_murphree': 0.70}
    FCC_PARAMS  = {'K_multiplier': 1.05, 'delta_crit': 2.3,
                   'C_entrain': 0.015, 'k_precip': 0.5, 'E_murphree': 0.72}

    def _noisy(val, pct):
        return val * (1.0 + np.random.normal(0, pct))

    # (date, density_kg_m3, CCR, visc_100, T_mix, S/O, pred, params)
    operating_points = [
        # Basra-Kuwait mix lube mode (density=1028, CCR=22.8, visc=1621)
        ('2024-01-10', 1028, 22.8, 1621, 80, 8.0,  0.226, LUBE_PARAMS),
        ('2024-01-11', 1028, 22.8, 1621, 78, 8.0,  0.226, LUBE_PARAMS),
        ('2024-01-12', 1028, 22.8, 1621, 82, 8.0,  0.226, LUBE_PARAMS),
        ('2024-01-13', 1028, 22.8, 1621, 80, 10.0, 0.226, LUBE_PARAMS),
        ('2024-01-14', 1028, 22.8, 1621, 80, 6.0,  0.226, LUBE_PARAMS),
        ('2024-01-15', 1028, 22.8, 1621, 80, 8.0,  0.15,  LUBE_PARAMS),
        ('2024-01-16', 1028, 22.8, 1621, 80, 9.0,  0.20,  LUBE_PARAMS),
        ('2024-01-17', 1028, 22.8, 1621, 74, 8.0,  0.226, LUBE_PARAMS),
        # FCC mode — no predilution, lower temperature
        ('2024-02-01', 1028, 22.8, 1621, 74, 8.0,  0.0,   FCC_PARAMS),
        ('2024-02-02', 1028, 22.8, 1621, 70, 8.0,  0.0,   FCC_PARAMS),
        ('2024-02-03', 1028, 22.8, 1621, 78, 8.0,  0.0,   FCC_PARAMS),
        ('2024-02-04', 1028, 22.8, 1621, 74, 10.0, 0.0,   FCC_PARAMS),
        ('2024-02-05', 1028, 22.8, 1621, 74, 6.0,  0.0,   FCC_PARAMS),
        # Basra Light (density=1026, CCR=22.6, visc=1137)
        ('2024-03-01', 1026, 22.6, 1137, 80, 8.0,  0.226, LUBE_PARAMS),
        ('2024-03-02', 1026, 22.6, 1137, 74, 8.0,  0.0,   FCC_PARAMS),
        ('2024-03-03', 1026, 22.6, 1137, 80, 10.0, 0.226, LUBE_PARAMS),
    ]

    header = (
        'timestamp,feed_density_kg_m3,feed_CCR,feed_visc_100,feed_temp_after_mixing,'
        'propane_temp,extractor_bot_temp,extractor_top_temp,feed_flow_m3hr,'
        'SO_ratio,predilution_frac,'
        'DAO_yield,DAO_viscosity,DAO_astm_colour,DAO_density,DAO_CCR,asph_contam\n'
    )

    cache = {}
    rows  = []
    for ts, density, ccr, visc, T_mix, so, pred, params in operating_points:
        # Estimate typical DCS readings for extractor temps
        T_prop = round(T_mix - 30.0, 1)                    # propane ~30°C colder than feed
        T_bot  = round(T_prop + 0.35*(T_mix - T_prop), 1)  # bottom blend
        T_mid  = round(T_bot  + 0.55*(T_mix - T_bot),  1)  # middle blend
        T_top  = round(T_mid  + 13.0, 1)                   # +steam heating
        feed_flow = 86.1                                    # design m³/hr

        pt = PlantDataPoint(
            timestamp=ts, feed_density_kg_m3=float(density),
            feed_CCR=ccr, feed_visc_100=float(visc),
            SO_ratio=so, feed_temp_after_mixing_C=float(T_mix),
            N_stages=4, solvent='propane',
            predilution_frac=pred,
            T_propane_fresh=T_prop,
            T_bottom=T_bot, T_top=T_top,
            feed_flow_m3hr=feed_flow,
        )
        sim  = _simulate_point(pt, params, cache)
        y    = round(_noisy(sim['DAO_yield'],          0.04),  1)
        visc_dao = round(max(_noisy(sim['DAO_viscosity'],   0.08),  1.0), 1)
        col  = round(max(_noisy(sim['DAO_astm_colour'],     0.10),  0.1), 2)
        rho  = round(_noisy(sim['DAO_density'],         0.004), 4)
        c    = round(max(_noisy(sim['DAO_CCR'],          0.05),  0.3), 2)
        ac   = round(max(_noisy(sim['asph_contam'],      0.10),  1e-4), 4)
        rows.append(
            f"{ts},{density},{ccr},{visc},{T_mix},"
            f"{T_prop},{T_bot},{T_top},{feed_flow},"
            f"{so},{pred},"
            f"{y},{visc_dao},{col},{rho},{c},{ac}"
        )

    with open(path, 'w', encoding='utf-8') as f:
        f.write(header)
        f.write('\n'.join(rows) + '\n')
    print(f"  Sample CSV written -> '{path}'  ({len(rows)} operating points)")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-POINT SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_point(pt: PlantDataPoint,
                    params: dict,
                    comp_cache: dict) -> dict:
    """Run the extractor for one plant point with calibration params applied.

    Calibration parameters are passed through the call chain:
        K_multiplier, delta_crit  → run_extractor → solve_lle → K_value
        C_entrain                 → EntrainmentParams
        k_precip                  → KineticParams
        E_murphree                → StageEfficiency
    """
    K_mult     = float(params.get('K_multiplier', 1.0))
    C_ent      = float(params.get('C_entrain',    0.015))
    k_p        = float(params.get('k_precip',     0.5))
    E_murph    = float(params.get('E_murphree',   0.70))
    d_crit     = float(params.get('delta_crit',   2.5))
    thermo_mode = str(params.get('thermo_mode',   'kvalue'))

    # CRUDE-INDEPENDENT: always build feed from density/CCR/visc
    density = float(pt.feed_density_kg_m3)
    CCR = float(pt.feed_CCR)
    visc = float(pt.feed_visc_100) if pt.feed_visc_100 is not None else 1000.0

    cache_key = (round(density, 1), round(CCR, 2), pt.solvent)
    if cache_key not in comp_cache:
        sara = estimate_sara_from_properties(density_kg_m3=density, CCR=CCR, visc_100=visc)
        sg = density / 1000.0
        F_precip = float(np.clip(0.28 + (sara['asphaltenes'] - 10) / 80, 0.15, 0.50))
        custom_feed = {
            'SARA': sara,
            'MW_heavy_cut': 750.0,
            'F_precip': F_precip,
            'SG_15': sg,
            'API': 141.5 / sg - 131.5,
            'CCR_wt': CCR,
            'visc_100': visc,
        }
        comps = build_residue_distribution(custom_feed=custom_feed, n_comp=20, solvent_name=pt.solvent)
        comp_cache[cache_key] = comps
    comps = comp_cache[cache_key]

    # Use DCS temperature readings when available; fall back to feed temp
    T_feed = pt.feed_temp_after_mixing_C
    T_bot  = pt.T_bottom        if pt.T_bottom        is not None else T_feed
    T_tpr  = pt.T_top           if pt.T_top           is not None else T_feed
    # If propane temp + bottom temp both missing but propane_fresh available,
    # estimate bottom bed using default blend fraction
    if pt.T_bottom is None and pt.T_propane_fresh is not None:
        T_bot = pt.T_propane_fresh + 0.35 * (T_feed - pt.T_propane_fresh)
        if pt.T_top is None:
            T_tpr = T_bot + 0.55 * (T_feed - T_bot) + 13.0  # default steam dT

    N_st = getattr(pt, 'N_stages', 4)
    T_profile = build_T_profile(T_bot, T_tpr, N_st)

    try:
        r = run_extractor(
            components       = comps,
            solvent_name     = pt.solvent,
            solvent_ratio    = pt.SO_ratio,
            N_stages         = N_st,
            T_profile        = T_profile,
            kinetics         = KineticParams(k_p, 10.0),
            efficiency       = StageEfficiency(E_murph),
            entrainment      = EntrainmentParams(C_ent, 1.20),
            K_multiplier     = K_mult,
            delta_crit       = d_crit,
            predilution_frac = pt.predilution_frac,
            thermo_mode      = thermo_mode,
        )
    except Exception:
        return {'DAO_yield': 0.0, 'DAO_density': 1.0,
                'DAO_CCR': 50.0, 'asph_contam': 50.0,
                'DAO_viscosity': 9999.0, 'DAO_astm_colour': 8.0,
                'converged': False}

    sara    = r.get('SARA_DAO', {})
    f_res   = sara.get('resins',      0.0) / 100.0
    f_asp   = sara.get('asphaltenes', 0.0) / 100.0
    f_aro   = sara.get('aromatics',   0.0) / 100.0
    dao_ccr = (0.10 * f_asp + 0.12 * f_res + 0.005 * f_aro) * 100.0

    # Viscosity and colour from quality model
    from quality_model import predict_dao_viscosity, predict_astm_colour
    dao_visc   = float(predict_dao_viscosity(
        float(r['MW_DAO_avg']), float(r['density_DAO']), sara))
    dao_colour = float(predict_astm_colour(
        float(r['asphal_contam_pct']), sara))

    return {
        'DAO_yield':        float(r['DAO_yield_net']),
        'DAO_density':      float(r['density_DAO']),
        'DAO_CCR':          float(max(dao_ccr, 0.01)),
        'asph_contam':      float(r['asphal_contam_pct']),
        'DAO_viscosity':    dao_visc,
        'DAO_astm_colour':  dao_colour,
        'converged':        bool(r['converged']),
    }


# ─────────────────────────────────────────────────────────────────────────────
# OBJECTIVE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def _build_residuals(param_vec, dataset, weights, comp_cache, history):
    params = dict(zip(PARAM_NAMES, param_vec))
    residuals = []
    cost = 0.0

    for pt in dataset:
        sim = _simulate_point(pt, params, comp_cache)
        if pt.DAO_yield is not None:
            r = weights.DAO_yield * (sim['DAO_yield'] - pt.DAO_yield)
            residuals.append(r); cost += r**2
        if pt.DAO_viscosity is not None:
            r = weights.DAO_viscosity * (sim['DAO_viscosity'] - pt.DAO_viscosity)
            residuals.append(r); cost += r**2
        if pt.DAO_astm_colour is not None:
            r = weights.DAO_astm_colour * (sim['DAO_astm_colour'] - pt.DAO_astm_colour)
            residuals.append(r); cost += r**2
        if pt.DAO_density is not None:
            r = weights.DAO_density * (sim['DAO_density'] - pt.DAO_density)
            residuals.append(r); cost += r**2
        if pt.DAO_CCR is not None:
            r = weights.DAO_CCR * (sim['DAO_CCR'] - pt.DAO_CCR)
            residuals.append(r); cost += r**2
        if pt.asph_contam is not None:
            r = weights.asph_contam * (sim['asph_contam'] - pt.asph_contam)
            residuals.append(r); cost += r**2

    history.append({'params': params.copy(), 'cost': float(cost)})
    return np.array(residuals)


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(x_vec, dataset, comp_cache):
    errors  = {'DAO_yield': [], 'DAO_viscosity': [], 'DAO_astm_colour': [],
               'DAO_density': [], 'DAO_CCR': [], 'asph_contam': []}
    params  = dict(zip(PARAM_NAMES, x_vec))
    results = []

    for pt in dataset:
        sim = _simulate_point(pt, params, comp_cache)
        row = {
            'timestamp': pt.timestamp, 'feed_name': pt.feed_name,
            'SO_ratio': pt.SO_ratio, 'temperature': pt.temperature,
            'solvent': pt.solvent, 'sim': {}, 'plant': {}, 'error': {},
        }
        for var in errors:
            plant_val = getattr(pt, var)
            if plant_val is not None:
                err = sim[var] - plant_val
                errors[var].append(err)
                row['sim'][var]   = round(float(sim[var]), 4)
                row['plant'][var] = round(float(plant_val), 4)
                row['error'][var] = round(float(err), 4)
        results.append(row)

    metrics = {}
    for var, errs in errors.items():
        if errs:
            arr = np.array(errs)
            metrics[var] = {
                'MAE':  round(float(np.mean(np.abs(arr))), 4),
                'RMSE': round(float(np.sqrt(np.mean(arr**2))), 4),
                'bias': round(float(np.mean(arr)), 4),
                'n':    len(errs),
            }

    return metrics, results


def compute_metrics(dataset: List[PlantDataPoint], params: dict) -> dict:
    """
    Public interface: compute MAE, RMSE, bias per variable for given params.

    Parameters
    ----------
    dataset : list of PlantDataPoint
    params  : dict with K_multiplier, C_entrain, k_precip, E_murphree, delta_crit

    Returns
    -------
    dict  {variable: {MAE, RMSE, bias, n}}
    """
    x_vec = np.array([params.get(p, DEFAULT_PARAMS[p]) for p in PARAM_NAMES])
    metrics, _ = _compute_metrics(x_vec, dataset, comp_cache={})
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-POINT PUBLIC INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

def simulate_one_point(pt: PlantDataPoint, params: dict) -> dict:
    """
    Public interface: run one simulation with calibration params.

    Parameters
    ----------
    pt     : PlantDataPoint with operating conditions
    params : dict with calibration parameter values

    Returns
    -------
    dict with DAO_yield, DAO_density, DAO_CCR, asph_contam, converged
    """
    return _simulate_point(pt, params, comp_cache={})


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CALIBRATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def run_calibration(
    dataset:        List[PlantDataPoint],
    feed_name:      str                         = 'basra_kuwait_mix',
    init_params:    Optional[dict]              = None,
    weights:        Optional[CalibrationWeights] = None,
    save_profile:   Optional[str]               = None,
    max_nfev:       int                         = 200,
    ftol:           float                       = 1e-4,
    verbose:        bool                        = True,
    progress_cb                                 = None,
) -> CalibrationResult:
    """
    Calibrate model parameters to match plant operating data.

    Steps:
      1. Load initial parameters
      2. Compute initial error (before calibration)
      3. Run scipy least_squares optimisation
      4. Compute final error and accuracy metrics
      5. Optionally save calibrated profile

    Parameters
    ----------
    dataset      : list of PlantDataPoint
    feed_name    : label for the profile (points can span multiple feeds)
    init_params  : starting parameters (default: DEFAULT_PARAMS)
    weights      : objective function weights (default: CalibrationWeights())
    save_profile : profile name to save (None = don't save)
    max_nfev     : max function evaluations
    ftol         : convergence tolerance
    verbose      : print progress to stdout
    progress_cb  : callable(step, total, message) for web UI progress bar

    Returns
    -------
    CalibrationResult
    """
    t0 = time.time()
    if init_params is None:
        init_params = DEFAULT_PARAMS.copy()
    if weights is None:
        weights = CalibrationWeights()

    valid = [pt for pt in dataset if pt.has_measurements]
    if not valid:
        raise ValueError("No plant data points with measurements found.")

    if verbose:
        print(f"\n{'='*60}")
        print(f"  SDA Unit Simulator — Plant Calibration")
        print(f"  Operating points : {len(valid)}")
        print(f"  Optimizer        : scipy least_squares (TRF)")
        print(f"  Max evaluations  : {max_nfev}")
        print(f"{'='*60}")

    comp_cache = {}
    history    = []

    x0 = np.array([init_params.get(p, DEFAULT_PARAMS[p]) for p in PARAM_NAMES])
    lo = np.array([PARAM_BOUNDS[p][0] for p in PARAM_NAMES])
    hi = np.array([PARAM_BOUNDS[p][1] for p in PARAM_NAMES])

    # Initial cost
    res0  = _build_residuals(x0, valid, weights, comp_cache, history)
    cost0 = float(np.sum(res0**2))

    if verbose:
        print(f"\n  Initial cost = {cost0:.4f}")

    if progress_cb:
        progress_cb(1, 3, f"Initial evaluation: cost={cost0:.3f}")

    # Optimise
    opt = least_squares(
        fun     = _build_residuals,
        x0      = x0,
        bounds  = (lo, hi),
        args    = (valid, weights, comp_cache, history),
        method  = 'trf',
        ftol    = ftol, xtol=1e-5, gtol=1e-5,
        max_nfev= max_nfev,
        verbose = 0,
    )

    x_opt  = opt.x
    cost_f = float(np.sum(opt.fun**2))
    impr   = 100.0 * (cost0 - cost_f) / max(cost0, 1e-10)

    cal_params = {p: round(float(v), 6) for p, v in zip(PARAM_NAMES, x_opt)}

    if progress_cb:
        progress_cb(2, 3, f"Optimised: cost={cost_f:.3f}, improvement={impr:.1f}%")

    if verbose:
        print(f"\n  Calibrated cost = {cost_f:.4f}  (improvement: {impr:.1f}%)")
        print(f"  Function evals  : {opt.nfev}  |  {opt.message}")
        print(f"\n  Parameter changes:")
        for p in PARAM_NAMES:
            init_v = init_params.get(p, DEFAULT_PARAMS[p])
            cal_v  = cal_params[p]
            print(f"    {p:20s}: {init_v:.5f}  ->  {cal_v:.5f}  (d={cal_v-init_v:+.5f})")

    metrics, point_results = _compute_metrics(x_opt, valid, comp_cache)

    if verbose:
        print(f"\n  Accuracy Metrics (calibrated model):")
        for var, m in metrics.items():
            tgt  = TARGET_MAE.get(var)
            flag = 'OK ' if (tgt and m['MAE'] <= tgt) else 'HI '
            print(f"  [{flag}] {var:20s}  MAE={m['MAE']:.4f}  "
                  f"RMSE={m['RMSE']:.4f}  Bias={m['bias']:+.4f}")

    if save_profile:
        _save_profile(save_profile, cal_params, metrics, feed_name, len(valid))
        if verbose:
            print(f"\n  Profile saved -> calibration_profiles/{save_profile}.json")

    if progress_cb:
        progress_cb(3, 3, "Calibration complete")

    elapsed = time.time() - t0

    return CalibrationResult(
        success            = opt.success or (cost_f < cost0),
        calibrated_params  = cal_params,
        initial_params     = init_params.copy(),
        cost_initial       = round(cost0, 4),
        cost_final         = round(cost_f, 4),
        improvement_pct    = round(impr, 2),
        n_operating_points = len(valid),
        n_function_evals   = int(opt.nfev),
        metrics            = metrics,
        point_results      = point_results,
        message            = opt.message,
        elapsed_s          = round(elapsed, 2),
    )


# ─────────────────────────────────────────────────────────────────────────────
# PROFILE MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def _save_profile(name, params, metrics, feed_name, n_points):
    profile = {
        'profile_name': name,
        'feed_name':    feed_name,
        'n_points':     n_points,
        'timestamp':    time.strftime('%Y-%m-%d %H:%M:%S'),
        'parameters':   params,
        'metrics':      metrics,
    }
    path = os.path.join(PROFILES_DIR, f'{name}.json')
    with open(path, 'w') as f:
        json.dump(profile, f, indent=2)
    return path


def save_profile(name: str, params: dict, description: str = '') -> str:
    """Save a parameter set as a named profile (calibration_profiles/<name>.json)."""
    profile = {
        'profile_name': name,
        'description':  description,
        'timestamp':    time.strftime('%Y-%m-%d %H:%M:%S'),
        'parameters':   {p: float(params.get(p, DEFAULT_PARAMS[p])) for p in PARAM_NAMES},
    }
    path = os.path.join(PROFILES_DIR, f'{name}.json')
    with open(path, 'w') as f:
        json.dump(profile, f, indent=2)
    return path


def load_profile(name: str) -> dict:
    """Load a named profile. Returns DEFAULT_PARAMS if profile not found.

    Handles two JSON formats:
    - Old format (v1.x):  {"parameters": {"K_multiplier": ..., ...}}
    - New format (v2.x):  {"calibration_params": {"K_multiplier": ...,
                            "visc_slope": ..., "visc_bias": ...}}
    Both are merged into the returned dict; new format wins on conflict.
    """
    path = os.path.join(PROFILES_DIR, f'{name}.json')
    if not os.path.exists(path):
        print(f"  [WARNING] Profile '{name}' not found — using DEFAULT_PARAMS")
        return DEFAULT_PARAMS.copy()
    with open(path) as f:
        data = json.load(f)
    result = DEFAULT_PARAMS.copy()
    # Old-format key
    result.update({p: float(v) for p, v in data.get('parameters', {}).items()
                   if p in PARAM_NAMES})
    # New-format key (v2.x calibration profiles — also carries visc_slope/bias)
    calib = data.get('calibration_params', {})
    if calib:
        for k, v in calib.items():
            try:
                result[k] = float(v)
            except (TypeError, ValueError):
                pass
    return result


def list_profiles() -> list:
    """Return metadata for all saved profiles."""
    profiles = []
    for fn in sorted(os.listdir(PROFILES_DIR)):
        if fn.endswith('.json'):
            try:
                with open(os.path.join(PROFILES_DIR, fn)) as f:
                    d = json.load(f)
                profiles.append({
                    'name':      fn[:-5],
                    'feed_name': d.get('feed_name', ''),
                    'timestamp': d.get('timestamp', ''),
                    'n_points':  d.get('n_points', '?'),
                    'description': d.get('description', ''),
                    'metrics':   d.get('metrics', {}),
                    'parameters': d.get('parameters', {}),
                })
            except Exception:
                pass
    return profiles


def _create_default_profiles():
    """Create pre-set profiles based on design case operating targets."""
    save_profile('sda_default', DEFAULT_PARAMS,
                 'Default uncalibrated parameters')
    lube = DEFAULT_PARAMS.copy()
    lube['K_multiplier'] = 0.65
    lube['E_murphree']   = 0.75
    lube['C_entrain']    = 0.010
    lube['delta_crit']   = 3.0
    save_profile('sda_lube_dao', lube,
                 'Pre-calibrated Lube Bright Stock DAO mode (target: 18 wt%)')
    fcc = DEFAULT_PARAMS.copy()
    fcc['K_multiplier'] = 1.05
    fcc['E_murphree']   = 0.72
    fcc['C_entrain']    = 0.015
    fcc['delta_crit']   = 2.3
    save_profile('sda_fcc_dao', fcc,
                 'Pre-calibrated FCC Feed DAO mode (target: 32 wt%)')


def _init_profiles():
    existing = {f[:-5] for f in os.listdir(PROFILES_DIR) if f.endswith('.json')}
    if not {'sda_default', 'sda_lube_dao', 'sda_fcc_dao'}.issubset(existing):
        _create_default_profiles()

_init_profiles()


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION PLOTS (Plotly JSON, consumed by web UI)
# ─────────────────────────────────────────────────────────────────────────────

def plot_calibration_results(result: CalibrationResult) -> str:
    """Build a 4-panel Plotly figure JSON for the calibration panel."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.utils as pu

    pts = result.point_results
    if not pts:
        return '{}'

    sim_yields   = [p['sim'].get('DAO_yield',    0) for p in pts]
    plant_yields = [p['plant'].get('DAO_yield',  0) for p in pts]
    sim_dens     = [p['sim'].get('DAO_density',  0) for p in pts]
    plant_dens   = [p['plant'].get('DAO_density',0) for p in pts]
    labels       = [f"{p['feed_name'][:6]} {p['temperature']:.0f}C" for p in pts]
    temps        = [p['temperature'] for p in pts]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'DAO Yield: Simulated vs Plant',
            'DAO Density: Simulated vs Plant',
            'Yield Error by Temperature',
            'Parameter Change After Calibration',
        ],
        vertical_spacing=0.16, horizontal_spacing=0.12,
    )

    # 1. Yield parity
    hi_lim = max(plant_yields + sim_yields, default=35) * 1.1
    fig.add_trace(go.Scatter(
        x=plant_yields, y=sim_yields, mode='markers+text',
        text=labels, textposition='top center',
        marker=dict(size=9, color='#00BFFF', line=dict(color='white', width=1)),
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[0, hi_lim], y=[0, hi_lim], mode='lines',
        line=dict(color='rgba(255,255,255,0.3)', dash='dash'),
        showlegend=False,
    ), row=1, col=1)
    fig.update_xaxes(title_text='Plant DAO Yield (wt%)', row=1, col=1)
    fig.update_yaxes(title_text='Simulated DAO Yield (wt%)', row=1, col=1)

    # 2. Density parity
    valid_d = [(a, b) for a, b in zip(plant_dens, sim_dens) if a > 0]
    if valid_d:
        pd_v, sd_v = zip(*valid_d)
        dlim = [min(pd_v)*0.998, max(pd_v)*1.002]
        fig.add_trace(go.Scatter(
            x=list(pd_v), y=list(sd_v), mode='markers',
            marker=dict(size=9, color='#FFD700', line=dict(color='white', width=1)),
            showlegend=False,
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=dlim, y=dlim, mode='lines',
            line=dict(color='rgba(255,255,255,0.3)', dash='dash'),
            showlegend=False,
        ), row=1, col=2)
    fig.update_xaxes(title_text='Plant DAO Density (g/cm3)', row=1, col=2)
    fig.update_yaxes(title_text='Simulated DAO Density (g/cm3)', row=1, col=2)

    # 3. Error vs temperature
    errors = [p['error'].get('DAO_yield', None) for p in pts]
    t_errs = [(t, e) for t, e in zip(temps, errors) if e is not None]
    if t_errs:
        tv, ev = zip(*t_errs)
        colours = ['#E74C3C' if abs(e) > 2.0 else '#2ECC71' for e in ev]
        fig.add_trace(go.Bar(
            x=list(tv), y=list(ev),
            marker_color=colours,
            showlegend=False,
        ), row=2, col=1)
        fig.add_hline(y=0, line_color='rgba(255,255,255,0.3)', row=2, col=1)
    fig.update_xaxes(title_text='Extractor Temperature (degC)', row=2, col=1)
    fig.update_yaxes(title_text='Yield Error: Sim - Plant (wt%)', row=2, col=1)

    # 4. Parameter changes
    p_names  = list(result.calibrated_params.keys())
    p_init   = [result.initial_params.get(k, DEFAULT_PARAMS[k]) for k in p_names]
    p_cal    = [result.calibrated_params[k] for k in p_names]
    p_change = [((c - i) / max(abs(i), 1e-6)) * 100 for c, i in zip(p_cal, p_init)]
    colours2 = ['#E74C3C' if d < 0 else '#2ECC71' for d in p_change]
    fig.add_trace(go.Bar(
        x=p_names, y=p_change,
        marker_color=colours2,
        text=[f'{v:+.1f}%' for v in p_change], textposition='outside',
        showlegend=False,
    ), row=2, col=2)
    fig.update_xaxes(title_text='Parameter', row=2, col=2)
    fig.update_yaxes(title_text='Change (%)', row=2, col=2)

    fig.update_layout(
        paper_bgcolor='#0f1923',
        plot_bgcolor='#1a2535',
        font=dict(color='#e0e0e0', size=11),
        margin=dict(l=50, r=20, t=60, b=40),
        height=580,
    )
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.07)',
                     zerolinecolor='rgba(255,255,255,0.15)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.07)',
                     zerolinecolor='rgba(255,255,255,0.15)')

    return json.dumps(fig, cls=pu.PlotlyJSONEncoder)


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    csv_p   = sys.argv[1] if len(sys.argv) > 1 else make_sample_csv()
    prof    = sys.argv[2] if len(sys.argv) > 2 else 'hpcl_sample_calibration'
    dataset = load_plant_data(csv_p)
    result  = run_calibration(dataset, save_profile=prof, verbose=True)
    print(f"\n  Calibration done in {result.elapsed_s:.1f}s  |  "
          f"improvement {result.improvement_pct:.1f}%  |  profile: '{prof}'")
