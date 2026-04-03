"""
residue_distribution.py
=======================
Continuous mixture representation of vacuum residue for HPCL PDA Unit.

Feed characterisation is based on HPCL Operating Manual (Plant No. 41 - PDA),
Chapter 4, Feed and Product Characteristics.

Two design feeds supported:
    'basra_kuwait_mix'  – 70/30 Basra-Kuwait mix  (Design case)
    'basra_light'       – Basra Light              (Check case, higher DAO yield)

Key feed properties from Operating Manual:
                            Basra-Kuwait Mix    Basra Light
    Specific Gravity @15.5C     1.028               1.026
    API Gravity                 6.1                 6.5
    Nitrogen [wt%]              0.33                0.30
    Sulphur [wt%]               5.0                 4.8
    Conradson Carbon [wt%]      22.8                22.6
    Nickel [wppm]               28                  25
    Vanadium [wppm]             104                 93
    Viscosity @100°C [cSt]      1621                1137
    Viscosity @135°C [cSt]      230                 177
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# HPCL PDA Feed Database  (Plant No.41 Operating Manual, Chapter 4)
# ---------------------------------------------------------------------------

HPCL_FEEDS = {
    'basra_kuwait_mix': {
        'label':        '70/30 Basra-Kuwait Mix',
        'note':         'Design feed for PDA conversion',
        'SG_15':        1.028,
        'API':          6.1,
        'N_wt':         0.33,
        'S_wt':         5.0,
        'CCR_wt':       22.8,
        'Ni_wppm':      28,
        'V_wppm':       104,
        'visc_100':     1621,
        'visc_135':     230,
        # SARA back-calculated from CCR + API + metals (Speight 2020)
        # CCR ≈ 1.1*asphaltenes + 0.4*resins  (Lian et al. 2014)
        # High CCR(22.8) + high V/Ni → high asphaltene + resin fraction
        'SARA': {
            'saturates':    8.5,
            'aromatics':    32.0,
            'resins':       47.5,
            'asphaltenes':  12.0,
        },
        'MW_log_mean':  6.45,   # exp(6.45) ~ 633 g/mol
        'MW_log_std':   0.72,
        'MW_heavy_cut': 750.0,
        'F_precip':     0.35,
    },
    'basra_light': {
        'label':        'Basra Light',
        'note':         'Check case — lower viscosity → higher DAO yield',
        'SG_15':        1.026,
        'API':          6.5,
        'N_wt':         0.30,
        'S_wt':         4.8,
        'CCR_wt':       22.6,
        'Ni_wppm':      25,
        'V_wppm':       93,
        'visc_100':     1137,
        'visc_135':     177,
        'SARA': {
            'saturates':   10.0,
            'aromatics':   33.5,
            'resins':      46.0,
            'asphaltenes': 10.5,
        },
        'MW_log_mean':  6.35,   # exp(6.35) ~ 573 g/mol
        'MW_log_std':   0.70,
        'MW_heavy_cut': 750.0,
        'F_precip':     0.30,
    },
}

# Per-SARA MW distribution parameters [g/mol]
SARA_MW = {
    'saturates':   {'mean': 400,  'std': 120},
    'aromatics':   {'mean': 550,  'std': 180},
    'resins':      {'mean': 900,  'std': 300},
    'asphaltenes': {'mean': 1600, 'std': 500},
}

# Polarity/association energy [K] per SARA class  (PHCT EOS)
EPS_KB = {
    'saturates':   0.0,
    'aromatics':   50.0,
    'resins':      150.0,
    'asphaltenes': 350.0,
}


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class PseudoComponent:
    index:        int
    MW:           float
    z:            float          # mole fraction (feed basis, sum=1)
    density:      float          # g/cm³ at 20°C
    delta:        float          # Hildebrand solubility parameter [MPa^0.5]
    sara_class:   str  = ''
    is_heavy:     bool = False
    precipitable: bool = False
    eps_kb:       float = 0.0


# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------

def _density_from_class(MW: float, sara_class: str) -> float:
    base = {'saturates': 0.895, 'aromatics': 0.985,
            'resins': 1.030, 'asphaltenes': 1.090}.get(sara_class, 0.970)
    correction = -0.06 * np.exp(-(MW - 400) / 400)
    return float(np.clip(base + correction, 0.75, 1.15))


def _solubility_param(MW: float, sara_class: str) -> float:
    # δ = a + b*ln(MW/100)  calibrated to literature ranges
    params = {'saturates': (8.5, 1.10), 'aromatics': (11.0, 1.25),
              'resins': (13.5, 1.40), 'asphaltenes': (15.0, 1.55)}
    a, b = params.get(sara_class, (10.0, 1.20))
    return float(a + b * np.log(max(MW, 50.0) / 100.0))


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_residue_distribution(
    feed_name:    str  = 'basra_kuwait_mix',
    n_comp:       int  = 20,
    solvent_name: str  = 'propane',
    custom_feed:  Optional[dict] = None,
) -> list:
    """
    Build list of PseudoComponent objects from HPCL VR feed specification.

    Each SARA class gets n_comp//4 (min 3) Gauss-Legendre quadrature points.
    Mole fractions preserve the SARA mass-fraction split from the feed data.
    """
    if custom_feed is not None and 'density_kg_m3' in custom_feed and 'API' not in custom_feed:
        custom_feed = dict(custom_feed)
        custom_feed['API'] = api_from_density(custom_feed['density_kg_m3'])
        custom_feed['SG_15'] = custom_feed['density_kg_m3'] / 1000.0

    feed = custom_feed if custom_feed is not None else HPCL_FEEDS[feed_name]

    sara_wt  = feed['SARA']
    total_wt = sum(sara_wt.values())
    sara_frac = {k: v / total_wt for k, v in sara_wt.items()}

    MW_heavy = feed.get('MW_heavy_cut', 750.0)
    F_precip = feed.get('F_precip', 0.30)

    n_per = max(3, n_comp // 4)
    components = []
    global_idx = 0

    for sara_class in ['saturates', 'aromatics', 'resins', 'asphaltenes']:
        wt_frac = sara_frac.get(sara_class, 0.0)
        if wt_frac < 1e-6:
            continue

        mw_info = SARA_MW[sara_class]
        mw_mean = mw_info['mean']
        mw_std  = mw_info['std']
        ln_mean = np.log(mw_mean)
        ln_std  = mw_std / mw_mean

        MW_lo = max(80.0,   mw_mean - 2.5 * mw_std)
        MW_hi = min(3000.0, mw_mean + 3.0 * mw_std)
        ln_lo, ln_hi = np.log(MW_lo), np.log(MW_hi)

        xi, wi = np.polynomial.legendre.leggauss(n_per)
        ln_pts  = 0.5 * (ln_hi - ln_lo) * xi + 0.5 * (ln_hi + ln_lo)
        jac     = 0.5 * (ln_hi - ln_lo)
        MW_pts  = np.exp(ln_pts)

        pdf    = (1 / (ln_std * np.sqrt(2 * np.pi))) * np.exp(
                    -0.5 * ((ln_pts - ln_mean) / ln_std) ** 2)
        raw_z  = np.maximum(pdf * jac * wi, 0.0)
        raw_z /= raw_z.sum()

        avg_MW     = float(np.dot(raw_z, MW_pts))
        mole_scale = wt_frac / max(avg_MW, 1.0)

        for k in range(n_per):
            mw = float(MW_pts[k])
            components.append(PseudoComponent(
                index        = global_idx,
                MW           = mw,
                z            = float(raw_z[k] * mole_scale),
                density      = _density_from_class(mw, sara_class),
                delta        = _solubility_param(mw, sara_class),
                sara_class   = sara_class,
                is_heavy     = mw > MW_heavy,
                precipitable = False,
                eps_kb       = float(EPS_KB.get(sara_class, 0.0)),
            ))
            global_idx += 1

    # Normalise mole fractions
    total_z = sum(c.z for c in components)
    for c in components:
        c.z /= total_z

    # Flag precipitable: top F_precip of heavy + all asphaltenes
    heavy = sorted([c for c in components if c.is_heavy],
                   key=lambda c: c.MW, reverse=True)
    z_heavy_total   = sum(c.z for c in heavy)
    z_precip_target = F_precip * z_heavy_total
    accumulated = 0.0
    for c in heavy:
        if accumulated >= z_precip_target:
            break
        c.precipitable = True
        accumulated += c.z
    for c in components:
        if c.sara_class == 'asphaltenes':
            c.precipitable = True

    components.sort(key=lambda c: c.MW)
    for i, c in enumerate(components):
        c.index = i

    return components


def api_from_density(density_kg_m3: float) -> float:
    """API = 141.5 / SG - 131.5  where SG = density_kg_m3 / 1000"""
    sg = density_kg_m3 / 1000.0
    return 141.5 / sg - 131.5


def density_from_api(API: float) -> float:
    """density [kg/m3] = 1000 * 141.5 / (API + 131.5)"""
    return 1000.0 * 141.5 / (API + 131.5)


def estimate_sara_from_properties(
    density_kg_m3: float = None,
    CCR:           float = 22.8,
    visc_100:      float = None,
    asphaltene_wt: float = None,
    API:           float = None,   # backward compat — computed from density if None
) -> dict:
    """
    Estimate SARA distribution from routine lab properties.
    Primary input is density_kg_m3. API is computed internally from density.
    If density_kg_m3 is None, falls back to API parameter for backward compat.

    Correlations calibrated to HPCL feed database and heavy VR literature
    (Speight 2020, Ancheyta 2005, Lian et al. 2014):

        Asphaltenes ≈  CCR / 1.9          (or measured pentane-insolubles)
        Resins       ≈  3.0 * Asph + 10   (heavy VR: resins dominate)
        Saturates    ≈  3.5 * (API − 2)   (higher API → more paraffins)
        Aromatics    =  100 − Sat − Res − Asph   (by difference)

    All fractions are clipped to physical bounds and normalised to 100 wt%.

    Parameters
    ----------
    density_kg_m3 : density [kg/m3] — PRIMARY input; API computed from this
    CCR           : Conradson Carbon Residue [wt%]
    visc_100      : viscosity at 100°C [cSt] (reserved for future MW estimate)
    asphaltene_wt : measured pentane-insoluble asphaltenes [wt%]; if given,
                    used directly instead of CCR-based estimate.
    API           : API gravity — backward compat; used only if density_kg_m3 is None

    Returns
    -------
    dict : {'saturates': wt%, 'aromatics': wt%, 'resins': wt%, 'asphaltenes': wt%}
           Values sum to 100 wt%.
    """
    # Resolve API from density
    if density_kg_m3 is not None:
        _api = api_from_density(density_kg_m3)
    elif API is not None:
        _api = float(API)
    else:
        _api = 6.1  # default

    # ── Asphaltenes ──────────────────────────────────────────────────────────
    if asphaltene_wt is not None:
        asph = float(np.clip(asphaltene_wt, 0.5, 35.0))
    else:
        asph = float(np.clip(CCR / 1.9, 0.5, 35.0))

    # ── Resins (correlated with asphaltene content for heavy VR) ─────────────
    res = float(np.clip(3.0 * asph + 10.0, 5.0, 65.0))

    # ── Saturates (paraffin index rises with API) ─────────────────────────────
    sat = float(np.clip(3.5 * (_api - 2.0), 2.0, 30.0))

    # ── Aromatics (by difference) ─────────────────────────────────────────────
    aro = max(1.0, 100.0 - sat - res - asph)

    # ── Normalise to 100 wt% ─────────────────────────────────────────────────
    total = sat + aro + res + asph
    scale = 100.0 / total
    return {
        'saturates':   round(sat  * scale, 1),
        'aromatics':   round(aro  * scale, 1),
        'resins':      round(res  * scale, 1),
        'asphaltenes': round(asph * scale, 1),
    }


def distribution_summary(components: list, feed_name: str = '') -> dict:
    MW_arr    = np.array([c.MW      for c in components])
    z_arr     = np.array([c.z       for c in components])
    rho_arr   = np.array([c.density for c in components])
    delta_arr = np.array([c.delta   for c in components])

    sara_wt = {}
    for c in components:
        wm = c.z * c.MW
        sara_wt[c.sara_class] = sara_wt.get(c.sara_class, 0.0) + wm
    total_wm = sum(sara_wt.values())
    sara_wt_pct = {k: v / total_wm * 100 for k, v in sara_wt.items()}

    out = {
        'feed_name':        feed_name or 'custom',
        'n_components':     len(components),
        'MW_number_avg':    float(np.dot(z_arr, MW_arr)),
        'MW_weight_avg':    float(np.dot(z_arr * MW_arr, MW_arr) / max(np.dot(z_arr, MW_arr), 1)),
        'MW_min':           float(MW_arr.min()),
        'MW_max':           float(MW_arr.max()),
        'z_heavy_fraction': float(sum(c.z for c in components if c.is_heavy)),
        'z_precipitable':   float(sum(c.z for c in components if c.precipitable)),
        'mean_density':     float(np.dot(z_arr, rho_arr)),
        'mean_delta':       float(np.dot(z_arr, delta_arr)),
        'SARA_wt_pct':      sara_wt_pct,
    }
    if feed_name and feed_name in HPCL_FEEDS:
        fd = HPCL_FEEDS[feed_name]
        out.update({'API': fd['API'], 'SG_15': fd['SG_15'],
                    'CCR_wt': fd['CCR_wt'], 'visc_100_cSt': fd['visc_100'],
                    'visc_135_cSt': fd['visc_135'],
                    'Ni_wppm': fd['Ni_wppm'], 'V_wppm': fd['V_wppm']})
    return out


if __name__ == '__main__':
    for fn in ['basra_kuwait_mix', 'basra_light']:
        comps = build_residue_distribution(feed_name=fn, n_comp=20)
        stats = distribution_summary(comps, feed_name=fn)
        print(f"\n{HPCL_FEEDS[fn]['label']}")
        for k, v in stats.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    print(f"  SARA {kk:<16s}: {vv:.1f}%")
            elif isinstance(v, float):
                print(f"  {k:<28}: {v:.4g}")
            else:
                print(f"  {k:<28}: {v}")
