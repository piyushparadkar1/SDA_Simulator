"""
sensitivity_analysis.py — PDA Simulator sensitivity sweeps.
Generates Plotly figures (JSON-serialisable) for the web UI.

Step 4: 8 plot types + viscosity/colour in all sweeps.
"""

import numpy as np
import json

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.utils
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from residue_distribution   import build_residue_distribution, HPCL_FEEDS
from asphaltene_kinetics    import KineticParams
from stage_efficiency       import StageEfficiency
from entrainment_model      import EntrainmentParams
from hunter_nash_extractor  import run_extractor
from quality_model          import predict_dao_viscosity, predict_astm_colour
from run_simulation         import build_T_profile
from phct_eos               import propane_density

_GENERIC = {'basra_kuwait_mix': 'Heavy VR Blend A', 'basra_light': 'Heavy VR Blend B'}
def _feed_label(name): return _GENERIC.get(name, name)

# ─── Plant reference data ─────────────────────────────────────────────────────
PLANT_REF = {
    'propane': {'SO': [4, 6, 8, 10, 12], 'DAO_yield': [12, 18, 25, 32, 38]},
    'butane':  {'SO': [4, 6, 8, 10, 12], 'DAO_yield': [30, 40, 48, 54, 58]},
}
PLANT_REF_T = {
    'propane': {'T': [55, 65, 75, 85, 95], 'DAO_yield': [38, 32, 25, 18, 12]},
    'butane':  {'T': [120, 130, 140, 150, 160], 'DAO_yield': [62, 55, 48, 40, 32]},
}

COLORS = {
    'primary':   '#1a73e8',
    'secondary': '#e8711a',
    'green':     '#0f9d58',
    'purple':    '#7b1fa2',
    'red':       '#d32f2f',
    'grey':      '#9e9e9e',
    'plant':     '#f4b400',
    'cyan':      '#0097a7',
    'pink':      '#e91e63',
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor='#0f1117',
    plot_bgcolor='#1a1d27',
    font=dict(family='Inter, sans-serif', color='#e0e0e0', size=12),
    margin=dict(l=55, r=25, t=50, b=50),
    legend=dict(bgcolor='rgba(30,34,48,0.85)', bordercolor='#333', borderwidth=1),
    xaxis=dict(gridcolor='#2a2d3a', zerolinecolor='#333'),
    yaxis=dict(gridcolor='#2a2d3a', zerolinecolor='#333'),
)

def _default_T(solvent):
    return (75.0, 85.0) if solvent == 'propane' else (140.0, 160.0)

def _fig_json(fig):
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


# ─── Core runner ─────────────────────────────────────────────────────────────

def _run(comps, solvent, so, T_bottom, T_top, N,
         kinetics=None, efficiency=None, entrainment=None,
         K_multiplier=1.0, delta_crit=2.5, predilution_frac=0.0):
    if N < 1:
        N = 4  # safety: never run with 0 stages
    T_profile = build_T_profile(T_bottom, T_top, N)
    return run_extractor(
        components       = comps,
        solvent_name     = solvent,
        solvent_ratio    = float(so),
        N_stages         = N,
        T_profile        = T_profile,
        kinetics         = kinetics    or KineticParams(),
        efficiency       = efficiency  or StageEfficiency(),
        entrainment      = entrainment or EntrainmentParams(),
        K_multiplier     = K_multiplier,
        delta_crit       = delta_crit,
        predilution_frac = predilution_frac,
    )


def _quality(r):
    """Extract viscosity + ASTM colour from a raw extractor result."""
    sara = r.get('SARA_DAO', {})
    visc   = predict_dao_viscosity(r['MW_DAO_avg'], r['density_DAO'], sara)
    colour = predict_astm_colour(r['asphal_contam_pct'], sara)
    return float(visc), float(colour)


# ─── Sweep functions ──────────────────────────────────────────────────────────

def sweep_so_ratio(feed_name='basra_kuwait_mix', solvent='propane',
                   T_bottom=None, T_top=None, N=4, SO_range=None,
                   K_multiplier=1.0, delta_crit=2.5, predilution_frac=0.0):
    if SO_range is None:
        SO_range = np.linspace(3, 14, 12)
    _Tbot, _Ttop = _default_T(solvent)
    if T_bottom is None: T_bottom = _Tbot
    if T_top    is None: T_top    = _Ttop
    comps = build_residue_distribution(feed_name=feed_name, n_comp=20, solvent_name=solvent)
    results = [_run(comps, solvent, so, T_bottom, T_top, N,
                    K_multiplier=K_multiplier, delta_crit=delta_crit,
                    predilution_frac=predilution_frac) for so in SO_range]
    visc_col = [_quality(r) for r in results]
    return {
        'SO_range':      SO_range,
        'DAO_yield':     np.array([r['DAO_yield_net']     for r in results]),
        'asphalt_yield': np.array([r['asphalt_yield']     for r in results]),
        'asphal_contam': np.array([r['asphal_contam_pct'] for r in results]),
        'MW_DAO':        np.array([r['MW_DAO_avg']        for r in results]),
        'density_DAO':   np.array([r['density_DAO']       for r in results]),
        'viscosity':     np.array([v for v, c in visc_col]),
        'astm_colour':   np.array([c for v, c in visc_col]),
    }


def sweep_temperature(feed_name='basra_kuwait_mix', solvent='propane',
                      SO=8.0, N=4, T_range=None,
                      K_multiplier=1.0, delta_crit=2.5, predilution_frac=0.0):
    if T_range is None:
        T_range = np.linspace(55, 95, 10) if solvent == 'propane' else np.linspace(120, 165, 10)
    comps = build_residue_distribution(feed_name=feed_name, n_comp=20, solvent_name=solvent)
    dt = 10.0 if solvent == 'propane' else 20.0
    results = [_run(comps, solvent, SO, float(T), float(T) + dt, N,
                    K_multiplier=K_multiplier, delta_crit=delta_crit,
                    predilution_frac=predilution_frac) for T in T_range]
    # Propane density at each T (at 40 bar)
    prop_rho = [propane_density(float(T), 40.0) for T in T_range]
    visc_col = [_quality(r) for r in results]
    return {
        'T_range':       T_range,
        'DAO_yield':     np.array([r['DAO_yield_net']     for r in results]),
        'asphalt_yield': np.array([r['asphalt_yield']     for r in results]),
        'asphal_contam': np.array([r['asphal_contam_pct'] for r in results]),
        'MW_DAO':        np.array([r['MW_DAO_avg']        for r in results]),
        'prop_density':  np.array(prop_rho),
        'viscosity':     np.array([v for v, c in visc_col]),
        'astm_colour':   np.array([c for v, c in visc_col]),
    }


def sweep_efficiency(feed_name='basra_kuwait_mix', solvent='propane',
                     SO=8.0, T_bottom=None, T_top=None, N=4,
                     K_multiplier=1.0, delta_crit=2.5, predilution_frac=0.0):
    _Tbot, _Ttop = _default_T(solvent)
    if T_bottom is None: T_bottom = _Tbot
    if T_top    is None: T_top    = _Ttop
    E_range = np.linspace(0.3, 1.0, 10)
    comps   = build_residue_distribution(feed_name=feed_name, n_comp=20, solvent_name=solvent)
    results = [_run(comps, solvent, SO, T_bottom, T_top, N,
                    efficiency=StageEfficiency(float(E)),
                    K_multiplier=K_multiplier, delta_crit=delta_crit,
                    predilution_frac=predilution_frac)
               for E in E_range]
    visc_col = [_quality(r) for r in results]
    return {
        'E_range':       E_range * 100,
        'DAO_yield':     np.array([r['DAO_yield_net']     for r in results]),
        'asphal_contam': np.array([r['asphal_contam_pct'] for r in results]),
        'viscosity':     np.array([v for v, c in visc_col]),
        'astm_colour':   np.array([c for v, c in visc_col]),
    }


def sweep_stages(feed_name='basra_kuwait_mix', solvent='propane',
                 SO=8.0, T_bottom=None, T_top=None,
                 K_multiplier=1.0, delta_crit=2.5, predilution_frac=0.0):
    _Tbot, _Ttop = _default_T(solvent)
    if T_bottom is None: T_bottom = _Tbot
    if T_top    is None: T_top    = _Ttop
    N_range = [1, 2, 3, 4, 5, 6]
    comps   = build_residue_distribution(feed_name=feed_name, n_comp=20, solvent_name=solvent)
    results = [_run(comps, solvent, SO, T_bottom, T_top, N,
                    K_multiplier=K_multiplier, delta_crit=delta_crit,
                    predilution_frac=predilution_frac) for N in N_range]
    visc_col = [_quality(r) for r in results]
    return {
        'N_range':       N_range,
        'DAO_yield':     [r['DAO_yield_net']     for r in results],
        'asphal_contam': [r['asphal_contam_pct'] for r in results],
        'MW_DAO':        [r['MW_DAO_avg']        for r in results],
        'viscosity':     [v for v, c in visc_col],
        'astm_colour':   [c for v, c in visc_col],
    }


def sweep_predilution(feed_name='basra_kuwait_mix', solvent='propane',
                      SO=8.0, T_bottom=None, T_top=None, N=4,
                      K_multiplier=1.0, delta_crit=2.5,
                      pred_range=None):
    """Sweep predilution_frac from 0 to 0.40 — shows split-solvent selectivity effect."""
    if pred_range is None:
        pred_range = np.linspace(0.0, 0.40, 12)
    _Tbot, _Ttop = _default_T(solvent)
    if T_bottom is None: T_bottom = _Tbot
    if T_top    is None: T_top    = _Ttop
    comps = build_residue_distribution(feed_name=feed_name, n_comp=20, solvent_name=solvent)
    results = [_run(comps, solvent, SO, T_bottom, T_top, N,
                    K_multiplier=K_multiplier, delta_crit=delta_crit,
                    predilution_frac=float(p)) for p in pred_range]
    visc_col = [_quality(r) for r in results]
    return {
        'pred_range':    pred_range,
        'DAO_yield':     np.array([r['DAO_yield_net']     for r in results]),
        'asphal_contam': np.array([r['asphal_contam_pct'] for r in results]),
        'DAO_CCR':       np.array([r.get('DAO_CCR', 0.0)  for r in results]),
        'viscosity':     np.array([v for v, c in visc_col]),
        'astm_colour':   np.array([c for v, c in visc_col]),
    }


def sweep_gradient(feed_name='basra_kuwait_mix', solvent='propane',
                   SO=8.0, T_mean=None, N=4,
                   K_multiplier=1.0, delta_crit=2.5, predilution_frac=0.0,
                   dT_range=None):
    """Sweep T gradient (dT = T_top - T_bottom) at constant T_mean.

    dT=0 → flat profile, dT=30 → steep gradient.
    Illustrates LP-steam reflux: more gradient → better selectivity.
    """
    if dT_range is None:
        dT_range = np.linspace(0, 30, 10)
    if T_mean is None:
        T_mean = 80.0 if solvent == 'propane' else 150.0
    comps = build_residue_distribution(feed_name=feed_name, n_comp=20, solvent_name=solvent)
    results = []
    for dT in dT_range:
        Tbot = T_mean - dT / 2.0
        Ttop = T_mean + dT / 2.0
        results.append(_run(comps, solvent, SO, Tbot, Ttop, N,
                            K_multiplier=K_multiplier, delta_crit=delta_crit,
                            predilution_frac=predilution_frac))
    visc_col = [_quality(r) for r in results]
    return {
        'dT_range':      dT_range,
        'DAO_yield':     np.array([r['DAO_yield_net']     for r in results]),
        'asphal_contam': np.array([r['asphal_contam_pct'] for r in results]),
        'viscosity':     np.array([v for v, c in visc_col]),
        'astm_colour':   np.array([c for v, c in visc_col]),
        'T_mean':        T_mean,
    }


def sweep_yield_quality(feed_name='basra_kuwait_mix', solvent='propane',
                        T_bottom=None, T_top=None, N=4,
                        K_multiplier=1.0, delta_crit=2.5, predilution_frac=0.0,
                        SO_range=None):
    """Generate yield vs quality scatter — each point is a different S/O ratio.

    X = DAO yield, Y = ASTM colour; colour-coded by S/O.
    Reveals the trade-off: more solvent → more yield, but also more contamination.
    """
    if SO_range is None:
        SO_range = np.linspace(3, 14, 12)
    _Tbot, _Ttop = _default_T(solvent)
    if T_bottom is None: T_bottom = _Tbot
    if T_top    is None: T_top    = _Ttop
    comps = build_residue_distribution(feed_name=feed_name, n_comp=20, solvent_name=solvent)
    results = [_run(comps, solvent, so, T_bottom, T_top, N,
                    K_multiplier=K_multiplier, delta_crit=delta_crit,
                    predilution_frac=predilution_frac) for so in SO_range]
    visc_col = [_quality(r) for r in results]
    return {
        'SO_range':      SO_range,
        'DAO_yield':     np.array([r['DAO_yield_net']     for r in results]),
        'asphal_contam': np.array([r['asphal_contam_pct'] for r in results]),
        'viscosity':     np.array([v for v, c in visc_col]),
        'astm_colour':   np.array([c for v, c in visc_col]),
    }


def sweep_operating_map(feed_name='basra_kuwait_mix', solvent='propane',
                        N=4, K_multiplier=1.0, delta_crit=2.5, predilution_frac=0.0,
                        SO_range=None, T_range=None):
    """2D sweep: S/O × T_bottom → yield heatmap + CCR contours.

    Returns 2D arrays (T_range × SO_range) for yield and asphal_contam.
    """
    if SO_range is None:
        SO_range = np.linspace(4, 14, 8)
    if T_range is None:
        T_range = np.linspace(55, 90, 8) if solvent == 'propane' else np.linspace(120, 160, 8)
    dt = 10.0 if solvent == 'propane' else 20.0
    comps = build_residue_distribution(feed_name=feed_name, n_comp=20, solvent_name=solvent)
    yield_map  = np.zeros((len(T_range), len(SO_range)))
    colour_map = np.zeros((len(T_range), len(SO_range)))
    for i, T in enumerate(T_range):
        for j, so in enumerate(SO_range):
            r = _run(comps, solvent, so, float(T), float(T) + dt, N,
                     K_multiplier=K_multiplier, delta_crit=delta_crit,
                     predilution_frac=predilution_frac)
            _, colour = _quality(r)
            yield_map[i, j]  = r['DAO_yield_net']
            colour_map[i, j] = colour
    return {
        'SO_range':   SO_range,
        'T_range':    T_range,
        'yield_map':  yield_map,
        'colour_map': colour_map,
    }


def sweep_pressure(feed_name='basra_kuwait_mix', solvent='propane',
                   T_bottom=None, P_range=None):
    """Informational: propane density vs pressure at current T_bottom.

    Higher P → denser propane → stronger solvent power.
    """
    if P_range is None:
        P_range = np.linspace(34.3, 44.1, 10)  # 35–45 kg/cm²g × 0.980665
    _Tbot, _ = _default_T(solvent)
    if T_bottom is None: T_bottom = _Tbot
    rho = [propane_density(T_bottom, float(P)) for P in P_range]
    P_kgcm2 = P_range / 0.980665   # bar → kg/cm²g
    return {
        'P_kgcm2':    P_kgcm2,
        'P_bar':      P_range,
        'prop_rho':   np.array(rho),
        'T_bottom':   T_bottom,
    }


# ─── Plot builders ────────────────────────────────────────────────────────────

def plot_so_ratio(data, solvent, feed_name, current_SO=None):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('DAO & Asphalt Yield vs S/O Ratio',
                        'DAO Viscosity & ASTM Colour vs S/O'))

    ref = PLANT_REF.get(solvent, {})
    x   = data['SO_range']

    # Panel 1: yields
    fig.add_trace(go.Scatter(x=x, y=data['DAO_yield'],
        name='DAO Yield', line=dict(color=COLORS['primary'], width=2.5),
        marker=dict(size=7)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=data['asphalt_yield'],
        name='Asphalt Yield', line=dict(color=COLORS['red'], width=2.5),
        marker=dict(size=7)), row=1, col=1)
    if ref:
        fig.add_trace(go.Scatter(x=ref['SO'], y=ref['DAO_yield'],
            name='Plant Reference', mode='markers',
            marker=dict(symbol='diamond', size=10, color=COLORS['plant'],
                        line=dict(color='white', width=1))), row=1, col=1)
    if current_SO is not None:
        fig.add_vline(x=current_SO, line=dict(color='white', dash='dot', width=1.5),
                      annotation_text='Current', annotation_position='top right',
                      row=1, col=1)
        fig.add_vline(x=current_SO, line=dict(color='white', dash='dot', width=1.5),
                      row=1, col=2)

    # Panel 2: viscosity + colour (dual y-axis approach using secondary_y)
    fig.add_trace(go.Scatter(x=x, y=data['viscosity'],
        name='Viscosity [cSt]', line=dict(color=COLORS['purple'], width=2.5),
        marker=dict(size=7)), row=1, col=2)
    fig.add_trace(go.Scatter(x=x, y=data['astm_colour'],
        name='ASTM Colour', line=dict(color=COLORS['secondary'], width=2.5, dash='dash'),
        marker=dict(size=7)), row=1, col=2)
    fig.add_hline(y=6.0, line=dict(color=COLORS['red'], dash='dot', width=1.2),
                  annotation_text='Colour limit (6.0)', row=1, col=2,
                  annotation_font=dict(color=COLORS['red'], size=9))

    fig.update_xaxes(title_text='Solvent / Oil Ratio  [kg/kg]')
    fig.update_yaxes(title_text='Yield  [wt%]', row=1, col=1)
    fig.update_yaxes(title_text='Quality Metric', row=1, col=2)
    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(text=f'S/O Ratio — Yield & Quality | {_feed_label(feed_name)} | {solvent.capitalize()}',
                   font=dict(size=14)))
    return _fig_json(fig)


def plot_temperature(data, solvent, feed_name, current_T=None):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('DAO Yield vs T_bottom',
                        'ASTM Colour & Propane Density vs T_bottom'))

    ref = PLANT_REF_T.get(solvent, {})
    x   = data['T_range']

    fig.add_trace(go.Scatter(x=x, y=data['DAO_yield'],
        name='DAO Yield', line=dict(color=COLORS['primary'], width=2.5),
        marker=dict(size=7)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=data['asphalt_yield'],
        name='Asphalt Yield', line=dict(color=COLORS['red'], width=2.5),
        marker=dict(size=7)), row=1, col=1)
    if ref:
        fig.add_trace(go.Scatter(x=ref['T'], y=ref['DAO_yield'],
            name='Plant Reference', mode='markers',
            marker=dict(symbol='diamond', size=10, color=COLORS['plant'],
                        line=dict(color='white', width=1))), row=1, col=1)
    if current_T is not None:
        for col in [1, 2]:
            fig.add_vline(x=current_T, line=dict(color='white', dash='dot', width=1.5),
                          annotation_text='Current T_bot',
                          annotation_font=dict(color='white', size=9),
                          row=1, col=col)

    fig.add_trace(go.Scatter(x=x, y=data['astm_colour'],
        name='ASTM Colour', line=dict(color=COLORS['secondary'], width=2.5, dash='dash'),
        marker=dict(size=7)), row=1, col=2)
    fig.add_trace(go.Scatter(x=x, y=data['prop_density'],
        name='Propane Density [g/cm³]', line=dict(color=COLORS['cyan'], width=2),
        marker=dict(size=7)), row=1, col=2)
    fig.add_hline(y=6.0, line=dict(color=COLORS['red'], dash='dot', width=1.2),
                  annotation_text='Colour limit', row=1, col=2,
                  annotation_font=dict(color=COLORS['red'], size=9))

    unit = 'T_bottom  [°C]'
    fig.update_xaxes(title_text=unit)
    fig.update_yaxes(title_text='Yield  [wt%]', row=1, col=1)
    fig.update_yaxes(title_text='Quality / Density', row=1, col=2)
    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(text=f'Temperature — Yield & Quality | {_feed_label(feed_name)} | {solvent.capitalize()}',
                   font=dict(size=14)))
    return _fig_json(fig)


def plot_predilution(data, solvent, feed_name, design_pred=0.226):
    """Pre-dilution effect: yield vs selectivity."""
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=('DAO Yield vs Pre-dilution Fraction',
                        'DAO Contamination & ASTM Colour vs Pre-dilution'))
    x = data['pred_range']

    fig.add_trace(go.Scatter(x=x, y=data['DAO_yield'],
        name='DAO Yield', line=dict(color=COLORS['primary'], width=2.5),
        fill='tozeroy', fillcolor='rgba(26,115,232,0.08)',
        marker=dict(size=7)), row=1, col=1)
    fig.add_vline(x=design_pred, line=dict(color=COLORS['plant'], dash='dot', width=2),
                  annotation_text=f'Design ({design_pred:.3f})',
                  annotation_font=dict(color=COLORS['plant'], size=10),
                  row=1, col=1)
    fig.add_vline(x=design_pred, line=dict(color=COLORS['plant'], dash='dot', width=2),
                  row=1, col=2)

    fig.add_trace(go.Scatter(x=x, y=data['asphal_contam'],
        name='Asphaltene Contam.', line=dict(color=COLORS['red'], width=2.5),
        marker=dict(size=7)), row=1, col=2)
    fig.add_trace(go.Scatter(x=x, y=data['astm_colour'],
        name='ASTM Colour', line=dict(color=COLORS['secondary'], width=2.5, dash='dash'),
        marker=dict(size=7)), row=1, col=2)
    fig.add_hline(y=6.0, line=dict(color=COLORS['red'], dash='dot', width=1.2),
                  annotation_text='Colour limit', row=1, col=2,
                  annotation_font=dict(color=COLORS['red'], size=9))

    fig.add_annotation(
        x=0.5, y=0.02, xref='paper', yref='paper',
        text='Higher pre-dilution → more selective → lower yield, cleaner DAO',
        showarrow=False, font=dict(size=10, color='#8ba4e8'),
        bgcolor='rgba(30,34,48,0.8)', bordercolor='#333', borderwidth=1)

    fig.update_xaxes(title_text='Pre-dilution Fraction [—]')
    fig.update_yaxes(title_text='DAO Yield  [wt%]', row=1, col=1)
    fig.update_yaxes(title_text='wt% / ASTM Scale', row=1, col=2)
    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(text=f'Pre-dilution Effect | {_feed_label(feed_name)} | {solvent.capitalize()}',
                   font=dict(size=14)))
    return _fig_json(fig)


def plot_gradient(data, solvent, feed_name):
    """Temperature gradient effect at constant T_mean."""
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=(f'DAO Yield vs T Gradient  (T_mean={data["T_mean"]:.0f}°C)',
                        'ASTM Colour & Contam. vs T Gradient'))
    x = data['dT_range']

    fig.add_trace(go.Scatter(x=x, y=data['DAO_yield'],
        name='DAO Yield', line=dict(color=COLORS['primary'], width=2.5),
        marker=dict(size=7)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=data['asphal_contam'],
        name='Asphaltene Contam.', line=dict(color=COLORS['red'], width=2.5),
        marker=dict(size=7)), row=1, col=2)
    fig.add_trace(go.Scatter(x=x, y=data['astm_colour'],
        name='ASTM Colour', line=dict(color=COLORS['secondary'], width=2.5, dash='dash'),
        marker=dict(size=7)), row=1, col=2)
    fig.add_hline(y=6.0, line=dict(color=COLORS['red'], dash='dot', width=1.2),
                  annotation_text='Colour limit', row=1, col=2,
                  annotation_font=dict(color=COLORS['red'], size=9))

    fig.add_annotation(
        x=0.5, y=0.02, xref='paper', yref='paper',
        text='Steeper gradient = more internal reflux = better selectivity (LP steam effect)',
        showarrow=False, font=dict(size=10, color='#8ba4e8'),
        bgcolor='rgba(30,34,48,0.8)', bordercolor='#333', borderwidth=1)

    fig.update_xaxes(title_text='Temperature Gradient dT = T_top - T_bottom  [°C]')
    fig.update_yaxes(title_text='DAO Yield  [wt%]', row=1, col=1)
    fig.update_yaxes(title_text='wt% / ASTM Scale', row=1, col=2)
    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(text=f'Temperature Gradient (LP Steam) Effect | {_feed_label(feed_name)}',
                   font=dict(size=14)))
    return _fig_json(fig)


def plot_yield_quality(data, solvent, feed_name, current_SO=None):
    """Yield vs ASTM Colour trade-off scatter — the KEY operator chart."""
    x    = data['DAO_yield']
    y    = data['astm_colour']
    so   = data['SO_range']
    visc = data['viscosity']

    fig = go.Figure()

    # Operating envelope zones
    fig.add_shape(type='rect', x0=14, x1=22, y0=0, y1=4,
                  fillcolor='rgba(52,168,83,0.12)', line=dict(color='rgba(52,168,83,0.5)', width=1),
                  layer='below')
    fig.add_shape(type='rect', x0=22, x1=40, y0=0, y1=6,
                  fillcolor='rgba(244,180,0,0.08)', line=dict(color='rgba(244,180,0,0.4)', width=1),
                  layer='below')
    fig.add_shape(type='rect', x0=0, x1=50, y0=6, y1=8.5,
                  fillcolor='rgba(234,67,53,0.08)', line=dict(color='rgba(234,67,53,0.35)', width=1),
                  layer='below')
    fig.add_annotation(x=18, y=1.5, text='Lube Spec', showarrow=False,
                       font=dict(color='#34a853', size=10))
    fig.add_annotation(x=30, y=4.5, text='FCC Acceptable', showarrow=False,
                       font=dict(color='#f4b400', size=10))
    fig.add_annotation(x=25, y=7.2, text='Reject Zone', showarrow=False,
                       font=dict(color='#ea4335', size=10))

    # Scatter coloured by S/O
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers+text',
        marker=dict(
            size=12, color=so, colorscale='Viridis',
            colorbar=dict(title='S/O ratio', len=0.6),
            line=dict(color='white', width=0.8)),
        text=[f'{s:.0f}' for s in so],
        textposition='top center',
        textfont=dict(size=9, color='#ccc'),
        customdata=np.column_stack([so, visc]),
        hovertemplate='Yield: %{x:.1f}%<br>Colour: %{y:.2f}<br>S/O: %{customdata[0]:.1f}<br>Visc: %{customdata[1]:.1f} cSt<extra></extra>',
        name='Operating points',
    ))

    # Mark current S/O if given
    if current_SO is not None:
        idx = int(np.argmin(np.abs(np.array(so) - current_SO)))
        fig.add_trace(go.Scatter(x=[x[idx]], y=[y[idx]],
            mode='markers', marker=dict(symbol='star', size=18, color='gold',
                                        line=dict(color='white', width=1)),
            name=f'Current (S/O={current_SO:.1f})', showlegend=True))

    fig.add_hline(y=6.0, line=dict(color=COLORS['red'], dash='dot', width=1.5),
                  annotation_text='Colour limit 6.0',
                  annotation_font=dict(color=COLORS['red'], size=10))

    fig.update_xaxes(title_text='DAO Yield  [wt%]', range=[0, max(45, float(x.max()) + 3)])
    fig.update_yaxes(title_text='ASTM Colour', range=[0, 8.5])
    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(
            text=f'Yield vs Quality Trade-off | {_feed_label(feed_name)} | {solvent.capitalize()}',
            font=dict(size=14)))
    return _fig_json(fig)


def plot_operating_map(data, solvent, feed_name, current_SO=None, current_T=None):
    """2D S/O × T_bottom operating map (heatmap + colour contours)."""
    so  = data['SO_range']
    T   = data['T_range']
    z   = data['yield_map']
    zc  = data['colour_map']

    fig = make_subplots(rows=1, cols=2,
        subplot_titles=('DAO Yield  [wt%]  —  Operating Map',
                        'ASTM Colour  —  Operating Map'))

    fig.add_trace(go.Heatmap(z=z, x=so, y=T,
        colorscale='Blues', name='DAO Yield',
        colorbar=dict(title='Yield %', len=0.45, x=0.45),
        hovertemplate='S/O: %{x:.1f}<br>T_bot: %{y:.0f}°C<br>Yield: %{z:.1f}%<extra></extra>'),
        row=1, col=1)
    # Yield contours overlay
    fig.add_trace(go.Contour(z=z, x=so, y=T,
        contours=dict(start=10, end=50, size=5,
                      showlabels=True, labelfont=dict(size=9, color='white')),
        line=dict(width=1, color='rgba(255,255,255,0.5)'),
        showscale=False, name='Yield contours'), row=1, col=1)

    fig.add_trace(go.Heatmap(z=zc, x=so, y=T,
        colorscale='RdYlGn_r', name='ASTM Colour', zmin=0, zmax=8,
        colorbar=dict(title='ASTM', len=0.45, x=1.0),
        hovertemplate='S/O: %{x:.1f}<br>T_bot: %{y:.0f}°C<br>Colour: %{z:.1f}<extra></extra>'),
        row=1, col=2)
    # Colour=6 limit contour
    fig.add_trace(go.Contour(z=zc, x=so, y=T,
        contours=dict(start=6.0, end=6.0, size=0.1,
                      showlabels=True, labelfont=dict(size=10, color='red')),
        line=dict(width=2, color='red'),
        showscale=False, name='Colour=6 limit'), row=1, col=2)

    # Mark current operating point
    if current_SO is not None and current_T is not None:
        for col in [1, 2]:
            fig.add_trace(go.Scatter(x=[current_SO], y=[current_T],
                mode='markers',
                marker=dict(symbol='star', size=15, color='gold',
                            line=dict(color='white', width=1.5)),
                name='Current point', showlegend=(col == 1)), row=1, col=col)

    fig.update_xaxes(title_text='Solvent / Oil Ratio  [kg/kg]')
    fig.update_yaxes(title_text='T_bottom  [°C]', row=1, col=1)
    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(
            text=f'S/O × Temperature Operating Map | {_feed_label(feed_name)} | {solvent.capitalize()}',
            font=dict(size=14)))
    return _fig_json(fig)


def plot_pressure(data, solvent, feed_name):
    """Propane density vs pressure (informational)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['P_kgcm2'], y=data['prop_rho'],
        name='Propane density', line=dict(color=COLORS['cyan'], width=3),
        fill='tozeroy', fillcolor='rgba(0,151,167,0.12)',
        marker=dict(size=8),
        hovertemplate='P: %{x:.1f} kg/cm²g<br>Density: %{y:.4f} g/cm³<extra></extra>',
    ))
    fig.add_annotation(
        x=0.5, y=0.92, xref='paper', yref='paper',
        text=f'Higher pressure → denser propane → stronger solvent power (at T={data["T_bottom"]:.0f}°C)',
        showarrow=False, font=dict(size=11, color='#8ba4e8'),
        bgcolor='rgba(30,34,48,0.85)', bordercolor='#333', borderwidth=1)
    fig.update_xaxes(title_text='Extractor Pressure  [kg/cm²g]')
    fig.update_yaxes(title_text='Propane Density  [g/cm³]')
    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(
            text=f'Pressure Sensitivity — Propane Solvent Power | T_bot={data["T_bottom"]:.0f}°C',
            font=dict(size=14)))
    return _fig_json(fig)


def plot_efficiency(data, solvent, feed_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['E_range'], y=data['DAO_yield'],
        name='DAO Yield', line=dict(color=COLORS['primary'], width=2.5),
        marker=dict(size=7)))
    fig.add_trace(go.Scatter(x=data['E_range'], y=data['asphal_contam'],
        name='Asphal. Contam. in DAO', line=dict(color=COLORS['purple'], width=2.5, dash='dash'),
        fill='tozeroy', fillcolor='rgba(123,31,162,0.12)'))
    fig.add_trace(go.Scatter(x=data['E_range'], y=data['astm_colour'],
        name='ASTM Colour', line=dict(color=COLORS['secondary'], width=2, dash='dot'),
        marker=dict(size=6)))
    fig.add_vline(x=70, line=dict(color=COLORS['secondary'], dash='dot', width=1.5),
                  annotation_text='Default E=70%',
                  annotation_font=dict(color=COLORS['secondary'], size=10))
    fig.update_xaxes(title_text='Murphree Stage Efficiency  [%]')
    fig.update_yaxes(title_text='wt% / ASTM Colour')
    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(text=f'Stage Efficiency Sensitivity (Expert) | {solvent.capitalize()}',
                   font=dict(size=14)))
    return _fig_json(fig)


def plot_stages(data, solvent, feed_name):
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=('DAO Yield & Contamination vs Stages',
                        'Viscosity & ASTM Colour vs Stages'))

    fig.add_trace(go.Bar(x=data['N_range'], y=data['DAO_yield'],
        name='DAO Yield', marker_color=COLORS['primary'],
        marker_line=dict(color='white', width=0.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data['N_range'], y=data['asphal_contam'],
        name='Asphal. Contam.', mode='lines+markers',
        line=dict(color=COLORS['purple'], width=2), marker=dict(size=8)), row=1, col=1)
    # Design point marker
    fig.add_vline(x=3, line=dict(color=COLORS['plant'], dash='dot', width=2),
                  annotation_text='Design (3 stages)', row=1, col=1,
                  annotation_font=dict(color=COLORS['plant'], size=9))

    fig.add_trace(go.Scatter(x=data['N_range'], y=data['viscosity'],
        name='Viscosity [cSt]', line=dict(color=COLORS['purple'], width=2.5),
        marker=dict(size=8)), row=1, col=2)
    fig.add_trace(go.Scatter(x=data['N_range'], y=data['astm_colour'],
        name='ASTM Colour', line=dict(color=COLORS['secondary'], width=2.5, dash='dash'),
        marker=dict(size=8)), row=1, col=2)
    fig.add_hline(y=6.0, line=dict(color=COLORS['red'], dash='dot', width=1.2),
                  annotation_text='Colour limit', row=1, col=2,
                  annotation_font=dict(color=COLORS['red'], size=9))
    fig.add_vline(x=3, line=dict(color=COLORS['plant'], dash='dot', width=2),
                  row=1, col=2)

    fig.update_xaxes(title_text='Number of Stages', dtick=1)
    fig.update_yaxes(title_text='wt%', row=1, col=1)
    fig.update_yaxes(title_text='Quality Metric', row=1, col=2)
    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(text=f'Stage Count + HETP Sensitivity | {solvent.capitalize()}',
                   font=dict(size=14)))
    return _fig_json(fig)


# ─── T_top sweep (Fix 8) ─────────────────────────────────────────────────────

def sweep_temperature_top(feed_name='basra_kuwait_mix', solvent='propane',
                          SO=8.0, N=4, T_bottom_fixed=None,
                          T_top_range=None, K_multiplier=1.0, delta_crit=2.5,
                          predilution_frac=0.0):
    """Sweep T_top while keeping T_bottom fixed."""
    _Tbot, _Ttop = _default_T(solvent)
    if T_bottom_fixed is None:
        T_bottom_fixed = _Tbot
    if T_top_range is None:
        if solvent == 'propane':
            T_top_range = np.linspace(70, 100, 10)
        else:
            T_top_range = np.linspace(130, 170, 10)
    comps = build_residue_distribution(feed_name=feed_name, n_comp=20, solvent_name=solvent)
    yields, viscosities, colours, asph_contams = [], [], [], []
    for T_top in T_top_range:
        r = _run(comps, solvent, SO, T_bottom_fixed, float(T_top), N,
                 K_multiplier=K_multiplier, delta_crit=delta_crit,
                 predilution_frac=predilution_frac)
        visc, col = _quality(r)
        yields.append(float(r['DAO_yield_net']))
        viscosities.append(visc)
        colours.append(col)
        asph_contams.append(float(r['asphal_contam_pct']))
    return {'T_top_range': list(T_top_range), 'T_bottom_fixed': T_bottom_fixed,
            'yields': yields, 'viscosities': viscosities,
            'colours': colours, 'asph_contams': asph_contams}


def plot_temperature_top(data, solvent, feed_name, current_T_top=None, P_bar=40.0):
    if not HAS_PLOTLY:
        return '{}'
    fig = make_subplots(rows=1, cols=2, subplot_titles=['DAO Yield vs T_top', 'Quality vs T_top'])
    Tx = data['T_top_range']
    Tbot = data['T_bottom_fixed']
    p_label = f'at P = {P_bar/0.980665:.0f} kg/cm²g, T_bottom={Tbot:.0f}°C'
    fig.add_trace(go.Scatter(x=Tx, y=data['yields'], name='DAO Yield',
        line=dict(color=COLORS['primary'], width=2.5), marker=dict(size=8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=Tx, y=data['asph_contams'], name='Asph. Contam.',
        line=dict(color=COLORS['red'], width=2, dash='dash'), marker=dict(size=8)), row=1, col=1)
    if current_T_top:
        fig.add_vline(x=current_T_top, line=dict(color=COLORS['plant'], dash='dot', width=2),
                      annotation_text='Current', row=1, col=1,
                      annotation_font=dict(color=COLORS['plant'], size=9))
    fig.add_trace(go.Scatter(x=Tx, y=data['viscosities'], name='Viscosity [cSt]',
        line=dict(color=COLORS['purple'], width=2.5), marker=dict(size=8)), row=1, col=2)
    fig.add_trace(go.Scatter(x=Tx, y=data['colours'], name='ASTM Colour',
        line=dict(color=COLORS['secondary'], width=2.5, dash='dash'), marker=dict(size=8)), row=1, col=2)
    fig.update_xaxes(title_text='Top Bed Temperature [°C]')
    fig.update_yaxes(title_text='wt%', row=1, col=1)
    fig.update_yaxes(title_text='Quality Metric', row=1, col=2)
    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(text=f'Top Temperature Sensitivity | {p_label}', font=dict(size=13)))
    return _fig_json(fig)


# ─── Steam effect pre-computation (Fix 8) ────────────────────────────────────

def sweep_steam_effect(T_feed_mixed=85.0, T_propane_fresh=65.0,
                       steam_range=None, P_bar=40.0,
                       feed_flow_kg_hr=45237.0, solvent_flow_kg_hr=93577.0,
                       bottom_blend=0.35, middle_blend=0.55,
                       steam_effectiveness=0.60):
    """Pre-compute bed temps at different steam flows without running extraction.

    Fast calculation — no LLE needed. Shows operator how steam affects bed temperatures.
    """
    from hydraulics_entrain import estimate_bed_temperatures
    if steam_range is None:
        steam_range = np.linspace(0, 5000, 11)
    results = []
    for sf in steam_range:
        bt = estimate_bed_temperatures(
            T_feed_mixed_C=T_feed_mixed,
            T_propane_fresh_C=T_propane_fresh,
            steam_flow_kg_hr=float(sf),
            feed_flow_kg_hr=feed_flow_kg_hr,
            solvent_flow_kg_hr=solvent_flow_kg_hr,
            P_bar=P_bar,
            bottom_blend=bottom_blend,
            middle_blend=middle_blend,
            steam_effectiveness=steam_effectiveness,
        )
        results.append(bt)
    return {'steam_range': list(steam_range), 'results': results,
            'T_feed_mixed': T_feed_mixed, 'T_propane_fresh': T_propane_fresh}


def plot_steam_effect(data, P_bar=40.0):
    if not HAS_PLOTLY:
        return '{}'
    steam = data['steam_range']
    T_top    = [r['T_top_C']    for r in data['results']]
    T_mid    = [r['T_middle_C'] for r in data['results']]
    T_bot    = [r['T_bottom_C'] for r in data['results']]
    T_sat    = data['results'][0].get('T_sat_propane_C', 87.0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steam, y=T_top, name='Top Bed',
        line=dict(color=COLORS['red'], width=2.5), marker=dict(size=7)))
    fig.add_trace(go.Scatter(x=steam, y=T_mid, name='Middle Bed',
        line=dict(color=COLORS['secondary'], width=2), marker=dict(size=7)))
    fig.add_trace(go.Scatter(x=steam, y=T_bot, name='Bottom Bed',
        line=dict(color=COLORS['primary'], width=2), marker=dict(size=7)))
    fig.add_hline(y=T_sat, line=dict(color=COLORS['red'], dash='dot', width=1.5),
                  annotation_text=f'Propane T_sat {T_sat:.0f}°C',
                  annotation_font=dict(color=COLORS['red'], size=9))
    fig.add_hline(y=T_sat - 3, line=dict(color='orange', dash='dot', width=1),
                  annotation_text='Warning margin',
                  annotation_font=dict(color='orange', size=9))
    p_label = f'P = {P_bar/0.980665:.0f} kg/cm²g'
    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(text=f'Steam Effect on Bed Temperatures | {p_label}', font=dict(size=13)),
        xaxis_title='Steam Flow [kg/hr]',
        yaxis_title='Bed Temperature [°C]')
    return _fig_json(fig)


# ─── Operating margins ────────────────────────────────────────────────────────

def compute_operating_margins(
    baseline_params: dict,
    constraints: dict = None,
    sweep_steps: int = 5,
    refine_steps: int = 5,
) -> dict:
    """
    Compute operating margins: how much each parameter can change
    before violating quality constraints.

    Parameters
    ----------
    baseline_params : dict with keys:
        feed_name, solvent, SO_ratio, T_bottom, T_top, N_stages,
        K_multiplier, delta_crit, predilution_frac
    constraints : dict with optional keys:
        yield_loss_limit (default 1.0 wt%), viscosity_max (default 200 cSt),
        colour_max (default 6.0)
    sweep_steps : number of coarse sweep steps per direction (default 5)
    refine_steps : number of fine sweep steps near boundary (default 5)

    Returns
    -------
    dict : {param_name: {current, max_increase, max_decrease, impact_at_increase, impact_at_decrease}}
    """
    if constraints is None:
        constraints = {
            'yield_loss_limit': 1.0,
            'viscosity_max': 200.0,
            'colour_max': 6.0,
        }

    feed     = baseline_params.get('feed_name', 'basra_kuwait_mix')
    solvent  = baseline_params.get('solvent', 'propane')
    so_base  = float(baseline_params.get('SO_ratio', 8.0))
    T_bot    = float(baseline_params.get('T_bottom', 75.0))
    T_top_b  = float(baseline_params.get('T_top', 85.0))
    N        = max(int(baseline_params.get('N_stages', 4)), 1)
    K_mult   = float(baseline_params.get('K_multiplier', 1.0))
    d_crit   = float(baseline_params.get('delta_crit', 2.5))
    pred     = float(baseline_params.get('predilution_frac', 0.0))

    comps = build_residue_distribution(
        feed_name=feed if feed != 'custom' else 'basra_kuwait_mix',
        n_comp=20, solvent_name=solvent)

    def _run_point(so=so_base, T_b=T_bot, T_t=T_top_b, N=N):
        T_profile = build_T_profile(T_b, T_t, N)
        r = run_extractor(
            components=comps, solvent_name=solvent, solvent_ratio=float(so),
            N_stages=N, T_profile=T_profile,
            kinetics=KineticParams(), efficiency=StageEfficiency(),
            entrainment=EntrainmentParams(), K_multiplier=K_mult,
            delta_crit=d_crit, predilution_frac=pred,
        )
        sara = r.get('SARA_DAO', {})
        visc = float(predict_dao_viscosity(r['MW_DAO_avg'], r['density_DAO'], sara))
        colour = float(predict_astm_colour(r['asphal_contam_pct'], sara))
        return {
            'yield': float(r['DAO_yield_net']),
            'viscosity': visc,
            'colour': colour,
        }

    # Get baseline results
    try:
        baseline = _run_point()
    except Exception as e:
        raise RuntimeError(f'Baseline simulation failed: {e}') from e

    yield_limit = constraints.get('yield_loss_limit', 1.0)
    visc_max = constraints.get('viscosity_max', 200.0)
    colour_max = constraints.get('colour_max', 6.0)

    def _check_constraints(result):
        """Return True if constraints are satisfied."""
        if result['yield'] < baseline['yield'] - yield_limit:
            return False
        if result['viscosity'] > visc_max:
            return False
        if result['colour'] > colour_max:
            return False
        return True

    def _sweep_direction(param, base_val, direction, pct_max=0.20):
        """Sweep param in given direction, return max safe change."""
        steps = np.linspace(0, direction * pct_max * base_val, sweep_steps + 1)[1:]
        last_safe = 0.0
        last_impact = baseline.copy()

        for delta in steps:
            new_val = base_val + delta
            try:
                if param == 'SO_ratio':
                    result = _run_point(so=new_val)
                elif param == 'T_bottom':
                    result = _run_point(T_b=new_val, T_t=new_val + (T_top_b - T_bot))
                else:
                    result = baseline.copy()

                if _check_constraints(result):
                    last_safe = delta
                    last_impact = result
                else:
                    break
            except Exception:
                break

        return last_safe, last_impact

    results = {}

    # Sweep S/O ratio
    try:
        inc_so, inc_impact = _sweep_direction('SO_ratio', so_base, +1)
        dec_so, dec_impact = _sweep_direction('SO_ratio', so_base, -1)
        results['SO_ratio'] = {
            'current': round(so_base, 2),
            'max_increase': round(so_base + inc_so, 2),
            'max_increase_pct': round(inc_so / so_base * 100, 1) if so_base > 0 else 0,
            'max_decrease': round(so_base + dec_so, 2),
            'max_decrease_pct': round(dec_so / so_base * 100, 1) if so_base > 0 else 0,
            'impact_at_increase': {k: round(v - baseline[k], 2) for k, v in inc_impact.items()},
            'impact_at_decrease': {k: round(v - baseline[k], 2) for k, v in dec_impact.items()},
            'baseline': {k: round(v, 2) for k, v in baseline.items()},
        }
    except Exception as e:
        results['SO_ratio'] = {'error': str(e), 'current': so_base}

    # Sweep feed temperature
    try:
        inc_T, inc_T_impact = _sweep_direction('T_bottom', T_bot, +1)
        dec_T, dec_T_impact = _sweep_direction('T_bottom', T_bot, -1)
        results['T_bottom'] = {
            'current': round(T_bot, 1),
            'max_increase': round(T_bot + inc_T, 1),
            'max_increase_pct': round(inc_T / T_bot * 100, 1) if T_bot > 0 else 0,
            'max_decrease': round(T_bot + dec_T, 1),
            'max_decrease_pct': round(dec_T / T_bot * 100, 1) if T_bot > 0 else 0,
            'impact_at_increase': {k: round(v - baseline[k], 2) for k, v in inc_T_impact.items()},
            'impact_at_decrease': {k: round(v - baseline[k], 2) for k, v in dec_T_impact.items()},
            'baseline': {k: round(v, 2) for k, v in baseline.items()},
        }
    except Exception as e:
        results['T_bottom'] = {'error': str(e), 'current': T_bot}

    return results


# ─── Legacy compat wrapper ────────────────────────────────────────────────────

def run_all_and_get_figures(feed_name='basra_kuwait_mix', solvent='propane',
                             SO=8.0, T_C=75.0, N=3, K_multiplier=1.0, delta_crit=2.5):
    """Run all sweeps and return dict of Plotly JSON strings."""
    Tbot, Ttop = _default_T(solvent)
    figs = {}
    for key, sweep_fn, plot_fn, kw in [
        ('so_ratio',    sweep_so_ratio,    plot_so_ratio,    {}),
        ('temperature', sweep_temperature, plot_temperature, {}),
        ('efficiency',  sweep_efficiency,  plot_efficiency,  {}),
        ('stages',      sweep_stages,      plot_stages,      {}),
    ]:
        try:
            data = sweep_fn(feed_name, solvent, T_bottom=Tbot, T_top=Ttop, N=N,
                            K_multiplier=K_multiplier, delta_crit=delta_crit, **kw)
            figs[key] = plot_fn(data, solvent, feed_name)
        except Exception:
            figs[key] = None
    return figs
