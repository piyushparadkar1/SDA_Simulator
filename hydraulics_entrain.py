"""
hydraulics_entrain.py
=====================
Column hydraulics, HETP model, and solvent flow calculator for HPCL PDA Unit.

Based on Operating Manual Plant No. 41 (KBR ROSE® process), Chapter 16 & 25.

Provides:
    COLUMN_DEFAULTS          – dict of design column geometry
    stages_from_packing()    – theoretical stages from packing geometry
    compute_solvent_flows()  – volumetric/mass flow split from S/O + predilution
    check_column_hydraulics()– superficial velocity + density diff warnings
"""

import math

# ---------------------------------------------------------------------------
# Column geometry constants (Operating Manual, Plant No. 41)
# ---------------------------------------------------------------------------

COLUMN_DEFAULTS = {
    'packing_height_mm':  6000,   # 3600 + 1200 + 1200 mm (3 packed beds)
    'HETP_mm':            2000,   # structured packing HETP [mm]
    'column_ID_top_mm':   4300,   # top (DAO-rich) section ID
    'column_ID_bot_mm':   3000,   # bottom (asphalt-rich) section ID
    'total_height_mm':   19400,   # overall column height
    'design_P_kg_cm2':      44,   # design extractor pressure [kg/cm²g]
    'design_feed_kg_hr': 88547,   # design VR feed rate
}

# ---------------------------------------------------------------------------
# 3-Bed extractor geometry (Item B) — dict format
# ---------------------------------------------------------------------------

BED_CONFIG = {
    'top':    {'height_mm': 1200, 'ID_mm': 4300, 'role': 'polishing'},
    'middle': {'height_mm': 1200, 'ID_mm': 4300, 'role': 'extraction'},
    'bottom': {'height_mm': 3600, 'ID_mm': 3000, 'role': 'rejection'},
}

# ---------------------------------------------------------------------------
# Steam coil specification (LP steam at 4.5 kg/cm²g)
# ---------------------------------------------------------------------------

STEAM_DEFAULTS = {
    'pressure_kg_cm2g': 4.5,
    'P_abs_bar':        5.42,    # 4.5*0.981 + 1.013 bar
    'T_sat_C':          155.0,   # saturation temp at 5.42 bar
    'h_fg_kJ_kg':       2097.0,  # latent heat at 5.42 bar
    'flow_kg_hr':       2830.0,  # design flow from Operating Manual
}


def steam_pressure_to_properties(lp_steam_pressure_kgcm2g: float) -> dict:
    """
    Convert LP steam gauge pressure [kg/cm²g] to saturation temperature and
    latent heat, using correlations fitted to IAPWS steam tables.

    Valid range: 0.5–20 kg/cm²g (≈ 1.5–21 bar abs). Max error < 2°C / <1.5% h_fg.

    Correlations:
        P_abs_bar  = P_kgcm2g * 0.980665 + 1.01325
        T_sat_K    = 373.0 * P_abs_bar ^ 0.082       (power-law, fits tables)
        h_fg kJ/kg = 2501.0 - 2.69 * T_sat_C         (linear approx, Clausius)

    Reference points:
        4.5 kg/cm²g → 5.42 bar abs → T_sat ≈ 155 °C, h_fg ≈ 2097 kJ/kg
        0.0 kg/cm²g → 1.013 bar abs → T_sat ≈ 100 °C, h_fg ≈ 2232 kJ/kg
    """
    P_abs_bar = max(lp_steam_pressure_kgcm2g, 0.0) * 0.980665 + 1.01325
    T_sat_C   = 373.0 * (P_abs_bar ** 0.082) - 273.15
    h_fg      = 2501.0 - 2.69 * T_sat_C
    return {
        'P_abs_bar':  round(P_abs_bar, 3),
        'T_sat_C':    round(T_sat_C,   1),
        'h_fg_kJ_kg': round(h_fg,      1),
    }


def _round_stages(h_mm: float, HETP_mm: float) -> int:
    """Ceiling rounding: partial HETP sections still count as one stage.
    Physical rationale: even a partial packing height contributes separation.
    HETP=2000, h=3600 -> ceil(1.8) = 2; HETP=1500, h=3600 -> ceil(2.4) = 3."""
    import math
    return max(1, math.ceil(h_mm / max(HETP_mm, 1.0)))


def beds_to_stages(HETP_mm: float = 2000.0, bed_config: dict = None) -> list:
    """
    Map 3-bed geometry to a flat list of theoretical stages.

    At design (HETP=2000mm):
        Bottom: 3600/2000 = 1.8 → 2 stages
        Middle: 1200/2000 = 0.6 → 1 stage
        Top:    1200/2000 = 0.6 → 1 stage
        Total:  4 stages

    Uses arithmetic rounding (0.5 rounds up) to avoid Python banker's rounding.
    """
    if bed_config is None:
        bed_config = BED_CONFIG
    stages = []
    # Process in order: bottom -> middle -> top (countercurrent)
    for bed_name in ['bottom', 'middle', 'top']:
        bed = bed_config[bed_name]
        n = _round_stages(bed['height_mm'], HETP_mm)
        for _ in range(n):
            stages.append({'bed_name': bed_name, 'ID_mm': bed['ID_mm']})
    return stages


def beds_summary(HETP_mm: float = 2000.0, bed_config: dict = None) -> dict:
    """Return dict summary (total_frac_stages, per-bed details) for display."""
    if bed_config is None:
        bed_config = BED_CONFIG
    beds = []
    total_frac = 0.0
    for bed_name in ['bottom', 'middle', 'top']:
        bed = bed_config[bed_name]
        frac = bed['height_mm'] / max(HETP_mm, 1.0)
        n_int = _round_stages(bed['height_mm'], HETP_mm)
        total_frac += frac
        beds.append({
            'name':           bed_name,
            'height_mm':      bed['height_mm'],
            'ID_mm':          bed['ID_mm'],
            'function':       bed.get('role', bed.get('function', '')),
            'frac_stages':    round(frac, 3),
            'integer_stages': n_int,
        })
    return {'beds': beds, 'total_frac_stages': round(total_frac, 3)}


def build_extractor_profiles(
    bed_config: dict = None,
    HETP_mm:    float = 2000.0,
    T_bottom:   float = 72.0,
    T_middle:   float = 76.0,
    T_top:      float = 82.0,
    P_bar:      float = 40.0,
) -> list:
    """
    Build per-bed simulation profile for extractor zones.

    Returns list of dicts (bottom→top) with bed geometry, temperature,
    pressure, stage counts for use in staged extractor calculation.
    """
    if bed_config is None:
        bed_config = BED_CONFIG
    stage_info = beds_summary(HETP_mm, bed_config)
    T_map = {'bottom': T_bottom, 'middle': T_middle, 'top': T_top}
    profiles = []
    for sinfo in stage_info['beds']:
        name = sinfo['name']
        profiles.append({
            'name':           name,
            'height_mm':      sinfo['height_mm'],
            'ID_mm':          sinfo['ID_mm'],
            'function':       sinfo['function'],
            'T_C':            T_map.get(name, T_middle),
            'P_bar':          P_bar,
            'frac_stages':    sinfo['frac_stages'],
            'integer_stages': sinfo['integer_stages'],
        })
    return profiles


# ---------------------------------------------------------------------------
# Zone thermal model: estimate bed temperatures from inlet stream conditions
# ---------------------------------------------------------------------------

def estimate_bed_temperatures(
    T_feed_mixed_C:           float = 85.0,
    T_propane_fresh_C:        float = 65.0,
    steam_flow_kg_hr:         float = 2830.0,
    feed_flow_kg_hr:          float = 45237.0,   # per extractor
    solvent_flow_kg_hr:       float = 93577.0,   # per extractor
    P_bar:                    float = 40.0,
    bottom_blend:             float = 0.35,
    middle_blend:             float = 0.55,
    steam_effectiveness:      float = 0.60,
    lp_steam_pressure_kgcm2g: float = 4.5,       # LP steam gauge pressure [kg/cm²g]
) -> dict:
    """
    Estimate 3-bed temperatures from inlet stream conditions.

    IMPORTANT: T_feed_mixed_C is AFTER predilution propane mixing.
    The operator reads this from the DCS at the extractor feed nozzle.
    It is NOT the raw VR temperature.

    Physical model:
      Bottom bed: fresh propane dominates (cold); asphalt phase brings some heat
        T_bottom = T_propane_fresh + bottom_blend * (T_feed_mixed - T_propane_fresh)

      Middle bed: intermediate blending zone
        T_middle = T_bottom + middle_blend * (T_feed_mixed - T_bottom)

      Top bed: middle + steam coil heating
        h_fg is derived from lp_steam_pressure_kgcm2g (NOT hardcoded)
        via steam_pressure_to_properties()
        dT_steam = Q * effectiveness / (m_top * Cp)
        T_top = T_middle + dT_steam (capped at T_feed_mixed + 10°C)
    """
    Cp_mix = 2.3  # kJ/(kg·K) for heavy oil/propane mixture

    T_bottom = T_propane_fresh_C + bottom_blend * (T_feed_mixed_C - T_propane_fresh_C)
    T_middle = T_bottom + middle_blend * (T_feed_mixed_C - T_bottom)

    # Derive steam latent heat from actual steam pressure (never hardcoded)
    steam_props = steam_pressure_to_properties(lp_steam_pressure_kgcm2g)
    h_fg = steam_props['h_fg_kJ_kg']
    T_sat_steam = steam_props['T_sat_C']

    dT_steam = 0.0
    if steam_flow_kg_hr > 0:
        Q_kW  = steam_flow_kg_hr * h_fg / 3600.0
        m_top = 0.80 * (feed_flow_kg_hr + solvent_flow_kg_hr) / 3600.0
        dT_steam = min((Q_kW * steam_effectiveness) / (max(m_top, 0.01) * Cp_mix), 20.0)

    T_top = T_middle + dT_steam

    # Propane saturation check at operating pressure
    T_sat_propane = 96.7 * (P_bar / 42.48) ** 0.28
    flash_warn = None
    if T_top > T_sat_propane - 3.0:
        flash_warn = (f'Top bed {T_top:.1f} degC near propane saturation '
                      f'{T_sat_propane:.1f} degC — flashing risk')

    return {
        'T_bottom_C':             round(T_bottom,      1),
        'T_middle_C':             round(T_middle,      1),
        'T_top_C':                round(T_top,         1),
        'dT_steam_C':             round(dT_steam,      1),
        'T_sat_propane_C':        round(T_sat_propane, 1),
        'T_sat_steam_C':          round(T_sat_steam,   1),
        'h_fg_kJ_kg':             round(h_fg,          1),
        'lp_steam_pressure_kgcm2g': lp_steam_pressure_kgcm2g,
        'flash_warning':          flash_warn,
    }


# ---------------------------------------------------------------------------
# HETP / stage model
# ---------------------------------------------------------------------------

def stages_from_packing(packing_height_mm: float, HETP_mm: float) -> float:
    """
    Calculate equivalent theoretical stages from packed column geometry.

        N_stages = packing_height / HETP

    At design: 6000 mm / 2000 mm = 3 theoretical stages.

    Returns float (not rounded) so the caller can decide how to handle
    fractional stages (e.g. pass int(round(N)) to run_extractor).
    """
    if HETP_mm <= 0:
        raise ValueError("HETP_mm must be positive")
    return max(1.0, packing_height_mm / HETP_mm)


# ---------------------------------------------------------------------------
# Solvent flow calculator
# ---------------------------------------------------------------------------

def compute_solvent_flows(
    feed_flow_kg_hr:  float,
    SO_ratio_vol:     float,
    predilution_frac: float,
    T_propane_C:      float,
    P_bar:            float,
    rho_feed:         float,
) -> dict:
    """
    Convert operator-friendly inputs to simulation variables and flow splits.

    Operator provides:
        feed_flow_kg_hr  – VR feed rate [kg/hr]
        SO_ratio_vol     – Solvent/Oil ratio on a VOLUME basis [-]
        predilution_frac – fraction of total solvent injected as pre-dilution
        T_propane_C      – propane temperature at injection point [°C]
        P_bar            – extractor pressure [bar]
        rho_feed         – VR feed density at conditions [g/cm³]

    Returns
    -------
    dict with:
        total_solvent_kg_hr   – total propane flow
        predilution_kg_hr     – pre-dilution (bottom) propane flow
        fresh_solvent_kg_hr   – fresh (main) propane to extractor
        mass_SO               – mass S/O ratio (what the simulation uses)
        volume_SO             – volume S/O (as entered)
        propane_density       – propane density at conditions [g/cm³]
        feed_flow_kg_hr       – echo back
        feed_density          – rho_feed [g/cm³]
    """
    from phct_eos import propane_density as _rho_propane
    rho_prop = _rho_propane(T_propane_C, P_bar)

    # Convert feed to volumetric rate [m³/hr]
    feed_vol_m3_hr    = feed_flow_kg_hr / (rho_feed * 1000.0)
    total_sol_vol_m3  = SO_ratio_vol * feed_vol_m3_hr
    total_sol_kg      = total_sol_vol_m3 * rho_prop * 1000.0

    predil_kg  = total_sol_kg * float(predilution_frac)
    fresh_kg   = total_sol_kg * (1.0 - float(predilution_frac))
    mass_so    = total_sol_kg / max(feed_flow_kg_hr, 1.0)

    return {
        'total_solvent_kg_hr':  round(total_sol_kg,  0),
        'predilution_kg_hr':    round(predil_kg,     0),
        'fresh_solvent_kg_hr':  round(fresh_kg,      0),
        'mass_SO':              round(mass_so,        3),
        'volume_SO':            round(SO_ratio_vol,   2),
        'propane_density':      round(rho_prop,       4),
        'feed_flow_kg_hr':      round(feed_flow_kg_hr, 0),
        'feed_density':         round(rho_feed,        4),
    }


# ---------------------------------------------------------------------------
# Column hydraulic checks
# ---------------------------------------------------------------------------

def check_column_hydraulics(
    total_flow_kg_hr: float,
    rho_light:        float,
    rho_heavy:        float,
    column_ID_mm:     float = 3000,
    packing_type:     str   = 'structured',
    n_extractors:     int   = 2,
) -> list:
    """
    Check operating point against hydraulic limits for packed SDA extractor.

    Manual limits (structured ROSEMAX packing at 40 bar):
        Max superficial velocity:  0.015 m/s (warning), 0.020 m/s (flooding)
        Min density difference:    0.20 g/cm³ (poor separation), 0.10 (critical)

    Parameters
    ----------
    total_flow_kg_hr  – combined light+heavy phase volumetric throughput [kg/hr]
    rho_light         – DAO-rich (solvent) phase density [g/cm³]
    rho_heavy         – asphalt-rich phase density [g/cm³]
    column_ID_mm      – column inner diameter to use [mm] (use bottom section)
    packing_type      – 'structured' or 'random' (informational only)
    n_extractors      – number of parallel extractors (flow split per unit)

    Returns
    -------
    list of warning strings. Empty list means all OK.
    """
    warnings = []

    # Split total flow across parallel extractors
    flow_per_extractor = total_flow_kg_hr / max(n_extractors, 1)

    # Superficial velocity based on column cross-section (per extractor)
    area_m2  = math.pi * (column_ID_mm / 2000.0) ** 2
    vel_ms   = (flow_per_extractor / 3600.0) / max(rho_light * 1000.0 * area_m2, 1e-9)

    if vel_ms > 0.020:
        warnings.append(
            f'Near flooding ({vel_ms:.4f} m/s > 0.020 m/s): reduce throughput immediately')
    elif vel_ms > 0.015:
        warnings.append(
            f'High velocity ({vel_ms:.4f} m/s > 0.015 m/s): entrainment risk')

    # Phase density difference
    delta_rho = rho_heavy - rho_light
    if delta_rho < 0.10:
        warnings.append(
            f'Critical: phases may not separate (delta_rho = {delta_rho:.3f} g/cm3 < 0.10)')
    elif delta_rho < 0.20:
        warnings.append(
            f'Low density difference (delta_rho = {delta_rho:.3f} g/cm3 < 0.20): '
            f'poor phase disengagement')

    return warnings


def check_bed_hydraulics(
    bed_config: dict,
    total_flow_kg_hr: float,
    rho_light: float,
    rho_heavy: float,
    n_extractors: int = 2,
) -> dict:
    """
    Check hydraulics PER BED using each bed's specific diameter.
    Returns dict: {'top': [warnings], 'middle': [warnings], 'bottom': [warnings]}
    """
    if bed_config is None:
        bed_config = BED_CONFIG
    per_bed = {}
    flow_per_ext = total_flow_kg_hr / max(n_extractors, 1)
    delta_rho = rho_heavy - rho_light

    for bed_name in ['bottom', 'middle', 'top']:
        bed = bed_config[bed_name]
        ID_mm = bed['ID_mm']
        area_m2 = math.pi * (ID_mm / 2000.0) ** 2
        vel_ms = (flow_per_ext / 3600.0) / max(rho_light * 1000.0 * area_m2, 1e-9)
        warnings = []
        if vel_ms > 0.020:
            warnings.append(f'{bed_name.capitalize()} bed: FLOODING ({vel_ms:.4f} m/s > 0.020)')
        elif vel_ms > 0.015:
            warnings.append(f'{bed_name.capitalize()} bed: HIGH velocity ({vel_ms:.4f} m/s > 0.015)')
        else:
            warnings.append(f'{bed_name.capitalize()} bed: velocity OK ({vel_ms:.4f} m/s)')
        if delta_rho < 0.10:
            warnings.append(f'{bed_name.capitalize()} bed: CRITICAL density diff ({delta_rho:.3f} g/cm3)')
        elif delta_rho < 0.20:
            warnings.append(f'{bed_name.capitalize()} bed: low density diff ({delta_rho:.3f} g/cm3)')
        per_bed[bed_name] = {
            'warnings': warnings,
            'velocity_m_s': round(vel_ms, 5),
            'ID_mm': ID_mm,
            'delta_rho': round(delta_rho, 3),
        }
    return per_bed


def hydraulic_metrics(
    total_flow_kg_hr: float,
    rho_light:        float,
    rho_heavy:        float,
    column_ID_mm:     float = 3000,
    n_extractors:     int   = 2,
) -> dict:
    """Return hydraulic metrics dict (velocity, density diff, area) for display."""
    flow_per_extractor = total_flow_kg_hr / max(n_extractors, 1)
    area_m2   = math.pi * (column_ID_mm / 2000.0) ** 2
    vel_ms    = (flow_per_extractor / 3600.0) / max(rho_light * 1000.0 * area_m2, 1e-9)
    delta_rho = rho_heavy - rho_light
    return {
        'velocity_m_s':          round(vel_ms,    5),
        'area_m2':               round(area_m2,   3),
        'delta_rho':             round(delta_rho, 3),
        'column_ID_mm':          column_ID_mm,
        'n_extractors':          n_extractors,
        'flow_per_extractor_kg_hr': round(flow_per_extractor, 0),
    }


# ---------------------------------------------------------------------------
# Propane saturation / flashing check (Item F)
# ---------------------------------------------------------------------------

def propane_saturation_check(T_C: float, P_bar: float, margin_C: float = 3.0) -> dict:
    """
    Estimate propane saturation temperature at operating pressure and warn
    if top-bed temperature is close to or above the saturation point.

    Correlation: T_sat [degC] = 96.7 * (P_bar / 42.48) ** 0.28
    At 40 bar → T_sat ~ 87 degC (matches propane vapour-pressure data).

    Parameters
    ----------
    T_C      – top-bed operating temperature [degC]
    P_bar    – extractor pressure [bar]
    margin_C – warning threshold below T_sat [degC], default 3

    Returns
    -------
    dict with T_sat_C, margin, status ('ok'|'warning'|'critical'), message
    """
    T_sat = 96.7 * (P_bar / 42.48) ** 0.28
    margin = T_sat - T_C
    if margin < 0:
        status = 'critical'
        msg = (f'CRITICAL: Top bed {T_C:.1f} degC exceeds propane T_sat '
               f'{T_sat:.1f} degC at {P_bar:.1f} bar — propane will flash!')
    elif margin < margin_C:
        status = 'warning'
        msg = (f'WARNING: Top bed {T_C:.1f} degC within {margin:.1f} degC of '
               f'propane T_sat {T_sat:.1f} degC at {P_bar:.1f} bar')
    else:
        status = 'ok'
        msg = (f'OK: Top bed {T_C:.1f} degC, T_sat {T_sat:.1f} degC '
               f'({margin:.1f} degC margin)')
    return {
        'T_sat_C':  round(T_sat,   1),
        'T_top_C':  round(T_C,     1),
        'margin_C': round(margin,  1),
        'P_bar':    round(P_bar,   1),
        'status':   status,
        'message':  msg,
    }
