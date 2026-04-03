"""
diagnostic_pipeline.py
=======================
Seven-stage diagnostic and calibration orchestration pipeline.

Stages run in order. Any blocking failure halts the pipeline immediately
and returns a structured report with a prescribed fix. The optimizer
(calibration_engine._build_outer_residuals) is only invoked after all
stages pass.

Stage 0 — sensor health             (<1 sec, pure pandas)
Stage 1 — data quality + coverage   (<1 sec, pure pandas)
Stage 2 — LIMS alignment quality    (<1 sec, pure pandas)
Stage 3 — thermal model health      (<5 sec, forward eval only)
Stage 4 — physics readiness         (<10 sec, 20 sampled rows)
Stage 5 — two-speed optimizer       (2 min fast + 3-5 min full)
Stage 6 — residual pattern analysis (<2 sec, post-optimizer)
Stage 7 — physical sanity + profile delta (<5 sec, 6 physics calls)

Public entry points:
  run_diagnostic_pipeline()     — stages 0-4 only (pre-calibration check)
  run_smart_calibration()       — stages 0-7 (full cycle, calls optimizer)
"""

import os
import glob as _glob
import json
import logging
import math
import time
import numpy as np
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Physical bounds for every DCS tag
# ---------------------------------------------------------------------------

SENSOR_BOUNDS = {
    't_bot_a':        (45.0, 110.0),
    't_mid_a':        (50.0, 110.0),
    't_steam_coil_a': (55.0, 110.0),
    't_top_a':        (60.0, 115.0),
    't_bot_b':        (45.0, 110.0),
    't_mid_b':        (50.0, 110.0),
    't_steam_coil_b': (55.0, 110.0),
    't_top_b':        (60.0, 115.0),
    'feed_flow_a':    (5.0,  120.0),
    'feed_flow_b':    (5.0,  120.0),
    'so_ratio_a':     (2.0,  18.0),
    'so_ratio_b':     (2.0,  18.0),
    'dao_flow':       (1.0,  80.0),
    'propane_temp':   (30.0, 80.0),
}

# ---------------------------------------------------------------------------
# Residual pattern definitions
# ---------------------------------------------------------------------------

RESIDUAL_PATTERNS = {
    'so_regime_bias': {
        'description': 'Residuals differ significantly across S/O bins',
        'detect': 'std of per-bin mean residual > 2.0 cSt',
        'meaning': 'K-value S/O dependence exponent (0.50) is wrong for this feed',
        'fix': (
            'Review so_factor exponent in lle_solver.K_value(). '
            'Try 0.40 (softer S/O dependence) or 0.60 (stronger). '
            'This is in lle_solver.py — a code change, not a parameter change.'
        ),
        'severity': 'WARNING',
    },
    'temperature_slope': {
        'description': 'Residuals correlate with T_top',
        'detect': 'Pearson r(visc_residual, t_top) > 0.4 or < -0.4',
        'meaning': 'C_T temperature coefficient in K-value model needs adjustment',
        'fix': (
            'Add C_T as a free parameter in the outer calibration loop. '
            'Current hardcoded value: _C_T = -0.007 in lle_solver.py. '
            'Suggested bounds: [-0.015, -0.002]. Use run_pathb_calibration().'
        ),
        'severity': 'WARNING',
    },
    'train_ab_offset': {
        'description': 'Train A and Train B have systematic viscosity offset',
        'detect': 'abs(mean(resid_A) - mean(resid_B)) > 2.0 cSt',
        'meaning': 'Trains have different effective efficiency or propane quality',
        'fix': (
            'Consider per-train E_murphree calibration. '
            'This requires adding train_id as a feature in simulator_bridge. '
            'Log for Phase 2 implementation.'
        ),
        'severity': 'WARNING',
    },
    'time_drift': {
        'description': 'Residuals show systematic trend over time',
        'detect': 'monthly mean residual changes by > 3.0 cSt over dataset',
        'meaning': (
            'Feed properties or plant condition drifted — model needs more '
            'frequent re-calibration or a shorter calibration window'
        ),
        'fix': (
            'Shorten calibration window: use last 6 months only. '
            'Retry calibration with dcs_train filtered to last 180 days. '
            'Set calibration_window_days=180 parameter.'
        ),
        'severity': 'WARNING',
    },
    'high_yield_bias': {
        'description': 'Model over/under-predicts viscosity at extreme yields',
        'detect': (
            'abs(mean residual for top quartile yield rows) > 3.0 cSt '
            'AND opposite sign from bottom quartile yield rows'
        ),
        'meaning': 'SARA composition model non-linearity not captured',
        'fix': (
            'Phase 2 candidate: add yield-dependent viscosity correction. '
            'For now: report and accept — this is a model limitation.'
        ),
        'severity': 'INFO',
    },
    'poor_yield_r2': {
        'description': 'Yield R² below acceptance threshold',
        'detect': 'yield R² < 0.55',
        'meaning': (
            'Yield residuals structurally high — likely feed density '
            'forward-fill error or S/O meter calibration issue'
        ),
        'fix': (
            'Check feed_density_max_age_hr — if > 336hr (2 weeks), '
            'forward-filled density is causing systematic yield error. '
            'Also verify dao_flow tag 41fic41110.pv calibration.'
        ),
        'severity': 'WARNING',
    },
}

# ---------------------------------------------------------------------------
# Design-point regression tests (Stage 7)
# ---------------------------------------------------------------------------

DESIGN_POINT_TESTS = [
    {
        'name':       'lube_dao_yield',
        'params':     {'K_multiplier': 0.65, 'E_murphree': 0.72,
                       'C_entrain': 0.010, 'delta_crit': 3.0},
        'conditions': {'so_ratio': 8.0,
                       't_profile': [67.0, 72.0, 77.0, 82.0],
                       'predilution_frac': 0.20},
        'check':      'dao_yield_mass_frac_pct in [15, 21]',
        'target':     '~18 wt% DAO yield (lube bright stock mode)',
    },
    {
        'name':       'fcc_dao_yield',
        'params':     {'K_multiplier': 1.05, 'E_murphree': 0.72,
                       'C_entrain': 0.015, 'delta_crit': 2.3},
        'conditions': {'so_ratio': 8.0,
                       't_profile': [62.0, 67.0, 72.0, 77.0],
                       'predilution_frac': 0.15},
        'check':      'dao_yield_mass_frac_pct in [28, 37]',
        'target':     '~32 wt% DAO yield (FCC feed mode)',
    },
    {
        'name':    'yield_increases_with_k_mult',
        'params_sweep': [0.65, 0.85, 1.05],
        'check':   'yields are strictly monotonically increasing',
        'target':  'K_mult monotonically controls yield',
    },
    {
        'name':    'yield_decreases_with_temperature',
        'conditions_sweep': [55.0, 65.0, 75.0],
        'check':   'yields are strictly monotonically decreasing',
        'target':  'Higher T → lower DAO yield (propane selectivity behavior)',
    },
]


# ---------------------------------------------------------------------------
# STAGE 0 — Sensor health
# ---------------------------------------------------------------------------

def _check_sensor_health(dcs: pd.DataFrame) -> dict:
    """
    Check all DCS sensors for stuck readings and out-of-bound values.

    Stuck sensor: rolling std < 0.05 for more than 24 consecutive hours.
    Physical bounds: any value outside SENSOR_BOUNDS[tag] is flagged.

    Auto-remediation (never blocks unless >20% of rows affected):
    - Updates 'train_valid_a' / 'train_valid_b' flags for affected rows.

    Returns structured result with pass/blocking/issues/stuck_sensors.
    """
    result: dict = {
        'pass':             True,
        'blocking':         False,
        'issues':           [],
        'rows_excluded_a':  0,
        'rows_excluded_b':  0,
        'stuck_sensors':    {},
        'fix':              '',
    }

    n_total = len(dcs)
    if n_total == 0:
        result['issues'].append('DCS DataFrame is empty')
        result['pass'] = False
        result['blocking'] = True
        result['fix'] = 'No DCS data loaded. Check extractor_parameters.xlsx path.'
        return result

    # Work on a copy so we can mutate validity flags
    dcs_work = dcs.copy()
    rows_excluded_a_idx = set()
    rows_excluded_b_idx = set()

    for tag, (lo, hi) in SENSOR_BOUNDS.items():
        if tag not in dcs_work.columns:
            continue

        col = dcs_work[tag]

        # ---- Bounds check ----
        oob_mask = col.notna() & ((col < lo) | (col > hi))
        n_oob = int(oob_mask.sum())
        if n_oob > 0:
            pct_oob = n_oob / n_total * 100.0
            result['issues'].append(
                f'{tag}: {n_oob} rows ({pct_oob:.1f}%) outside bounds '
                f'[{lo}, {hi}]'
            )
            # Mark as invalid for the associated train
            oob_idx = set(dcs_work.index[oob_mask].tolist())
            if '_a' in tag:
                rows_excluded_a_idx |= oob_idx
            elif '_b' in tag:
                rows_excluded_b_idx |= oob_idx

        # ---- Stuck sensor detection (only for T and flow tags) ----
        if not any(tag.startswith(p) for p in ('t_', 'feed_flow', 'so_ratio', 'prop')):
            continue

        # Compute rolling std over 6-hour window (assuming hourly data → 6 rows)
        rolling_std = col.rolling(window=6, min_periods=3).std()
        stuck_mask = (rolling_std < 0.05) & col.notna()

        if stuck_mask.any():
            # Count consecutive stuck hours (run-length encoding)
            stuck_arr = stuck_mask.astype(int).values
            max_run = 0
            current_run = 0
            run_start_i = 0
            longest_start_i = 0
            for i, v in enumerate(stuck_arr):
                if v:
                    if current_run == 0:
                        run_start_i = i
                    current_run += 1
                    if current_run > max_run:
                        max_run = current_run
                        longest_start_i = run_start_i
                else:
                    current_run = 0

            if max_run >= 24:  # 24+ consecutive hours stuck
                try:
                    t_start = dcs_work.index[longest_start_i]
                    t_end   = dcs_work.index[
                        min(longest_start_i + max_run - 1, len(dcs_work) - 1)
                    ]
                    result['stuck_sensors'][tag] = {
                        'start': str(t_start)[:16],
                        'end':   str(t_end)[:16],
                        'hours': max_run,
                    }
                except (IndexError, TypeError):
                    result['stuck_sensors'][tag] = {'hours': max_run}

                issue_msg = (
                    f'{tag}: stuck reading for {max_run}h '
                    f'(rolling std < 0.05)'
                )
                result['issues'].append(issue_msg)
                logger.warning(f'[DIAG S0] {issue_msg}')

                # Mark rows in the stuck window as invalid
                stuck_row_idx = set(dcs_work.index[stuck_mask].tolist())
                if '_a' in tag:
                    rows_excluded_a_idx |= stuck_row_idx
                elif '_b' in tag:
                    rows_excluded_b_idx |= stuck_row_idx

    # Apply exclusions to train_valid flags
    if rows_excluded_a_idx and 'train_valid_a' in dcs_work.columns:
        a_idx_list = [i for i in rows_excluded_a_idx if i in dcs_work.index]
        dcs_work.loc[a_idx_list, 'train_valid_a'] = False
        result['rows_excluded_a'] = len(a_idx_list)

    if rows_excluded_b_idx and 'train_valid_b' in dcs_work.columns:
        b_idx_list = [i for i in rows_excluded_b_idx if i in dcs_work.index]
        dcs_work.loc[b_idx_list, 'train_valid_b'] = False
        result['rows_excluded_b'] = len(b_idx_list)

    # ---- Blocking condition: any sensor stuck > 72h AND both trains affected ----
    sensors_stuck_72h = [
        tag for tag, info in result['stuck_sensors'].items()
        if info.get('hours', 0) > 72
    ]
    train_a_sensors_stuck = [t for t in sensors_stuck_72h if '_a' in t]
    train_b_sensors_stuck = [t for t in sensors_stuck_72h if '_b' in t]

    # Both trains affected = blocking
    if train_a_sensors_stuck and train_b_sensors_stuck:
        result['blocking'] = True
        result['pass']     = False
        result['fix'] = (
            f'Both trains have sensors stuck for > 72 hours. '
            f'Train A: {train_a_sensors_stuck}. Train B: {train_b_sensors_stuck}. '
            f'Verify sensor hardware (thermocouple / transmitter). '
            f'Calibration cannot proceed without at least one valid train.'
        )
    # >20% of rows excluded from both trains = blocking
    elif (result['rows_excluded_a'] > 0.20 * n_total and
          result['rows_excluded_b'] > 0.20 * n_total):
        result['blocking'] = True
        result['pass']     = False
        result['fix'] = (
            f'{result["rows_excluded_a"] / n_total * 100:.0f}% of Train A rows and '
            f'{result["rows_excluded_b"] / n_total * 100:.0f}% of Train B rows '
            f'excluded by sensor health checks. '
            f'Investigate stuck sensors before calibration.'
        )
    else:
        result['pass'] = len(sensors_stuck_72h) == 0 and len(result['issues']) == 0

    n_excl_a = result['rows_excluded_a']
    n_excl_b = result['rows_excluded_b']
    logger.info(
        f'[DIAG S0] Sensor health: excluded {n_excl_a} Train A rows, '
        f'{n_excl_b} Train B rows. Blocking={result["blocking"]}'
    )
    return result


# ---------------------------------------------------------------------------
# STAGE 1 — Data quality and operating coverage
# ---------------------------------------------------------------------------

def _check_data_quality(
    dcs: pd.DataFrame,
    visc_anchored: pd.DataFrame,
) -> dict:
    """
    Check row counts, operating coverage, and feed property imputation age.

    Blocking conditions:
      - visc_anchored rows < 200
      - steady_state rows < 500
      - valid rows (both trains) < 30% of total

    Returns structured result with coverage matrix, imputation stats, date gaps.
    """
    result: dict = {
        'pass':                       True,
        'blocking':                   False,
        'visc_anchored_rows':         len(visc_anchored),
        'steady_state_rows':          0,
        'coverage_matrix':            {},
        'empty_regime_count':         0,
        'sparse_regime_count':        0,
        'feed_density_median_age_hr': float('nan'),
        'feed_density_max_age_hr':    float('nan'),
        'feed_ccr_median_age_hr':     float('nan'),
        'feed_ccr_pct_over_168hr':    float('nan'),
        'date_gaps':                  [],
        'issues':                     [],
        'warnings':                   [],
        'fix':                        '',
    }

    n_total = len(dcs)

    # ---- A. Row count checks ----
    if 'steady_state' in dcs.columns:
        ss_mask = dcs['steady_state'].fillna(False)
        if 'train_valid_a' in dcs.columns:
            ss_mask &= dcs['train_valid_a'].fillna(False)
        if 'train_valid_b' in dcs.columns:
            ss_mask &= dcs['train_valid_b'].fillna(False)
        result['steady_state_rows'] = int(ss_mask.sum())
    else:
        result['steady_state_rows'] = n_total

    # Valid rows (both trains)
    if 'train_valid_a' in dcs.columns and 'train_valid_b' in dcs.columns:
        n_valid_both = int(
            (dcs['train_valid_a'].fillna(False) &
             dcs['train_valid_b'].fillna(False)).sum()
        )
    else:
        n_valid_both = n_total

    pct_valid = n_valid_both / max(n_total, 1) * 100.0

    # Blocking checks
    if result['visc_anchored_rows'] < 200:
        result['blocking'] = True
        result['pass'] = False
        est_after = int(result['visc_anchored_rows'] * 1.4)
        result['fix'] = (
            f'visc_anchored rows = {result["visc_anchored_rows"]} (minimum 200 required). '
            f'Widen dao_visc_100 tolerance from 6hr to 9hr in build_calibration_dataset(). '
            f'Or reduce steady_state T variance threshold from 1.5°C to 2.0°C. '
            f'Estimated rows after widening 6→9hr: ~{est_after}.'
        )
        result['issues'].append(
            f'visc_anchored_rows={result["visc_anchored_rows"]} < 200 (BLOCKING)'
        )

    elif result['steady_state_rows'] < 500:
        result['blocking'] = True
        result['pass'] = False
        result['fix'] = (
            f'steady_state rows = {result["steady_state_rows"]} (minimum 500 required). '
            f'Reduce steady_state rolling window from 2hr to 1hr OR '
            f'relax T_std threshold from 1.5°C to 2.5°C.'
        )
        result['issues'].append(
            f'steady_state_rows={result["steady_state_rows"]} < 500 (BLOCKING)'
        )

    elif pct_valid < 30.0:
        result['blocking'] = True
        result['pass'] = False
        result['fix'] = (
            f'Only {pct_valid:.0f}% of DCS rows have both trains valid (<30% threshold). '
            f'Check operating period — may need to widen train_valid bounds '
            f'or select a period when both trains ran consistently.'
        )
        result['issues'].append(
            f'valid_rows={n_valid_both}/{n_total} ({pct_valid:.0f}%) < 30% (BLOCKING)'
        )

    if result['blocking']:
        return result

    # ---- B. Operating coverage matrix (S/O × T_top) ----
    if len(visc_anchored) > 0 and 'so_ratio_a' in visc_anchored.columns:
        so_col = visc_anchored['so_ratio_a'].dropna()
        t_col  = visc_anchored['t_top_a'].dropna() \
            if 't_top_a' in visc_anchored.columns else pd.Series(dtype=float)

        if len(so_col) >= 4 and len(t_col) >= 4:
            try:
                so_bins = pd.qcut(so_col, q=4, labels=['S/O Q1', 'S/O Q2',
                                                         'S/O Q3', 'S/O Q4'],
                                   duplicates='drop')
                t_bins  = pd.qcut(t_col,  q=4, labels=['T Q1', 'T Q2',
                                                         'T Q3', 'T Q4'],
                                   duplicates='drop')
                cov_df = pd.DataFrame({'so_bin': so_bins, 't_bin': t_bins})
                cov_df = cov_df.dropna()
                matrix: dict = {}
                for so_lbl in cov_df['so_bin'].cat.categories:
                    matrix[str(so_lbl)] = {}
                    for t_lbl in cov_df['t_bin'].cat.categories:
                        count = int(
                            ((cov_df['so_bin'] == so_lbl) &
                             (cov_df['t_bin'] == t_lbl)).sum()
                        )
                        matrix[str(so_lbl)][str(t_lbl)] = count

                result['coverage_matrix'] = matrix
                flat_counts = [v for so_d in matrix.values() for v in so_d.values()]
                result['empty_regime_count']  = sum(1 for c in flat_counts if c == 0)
                result['sparse_regime_count'] = sum(1 for c in flat_counts if 0 < c < 5)

                if result['empty_regime_count'] > 0:
                    result['warnings'].append(
                        f'{result["empty_regime_count"]} S/O×T regimes have 0 '
                        f'visc-anchored rows — calibration invalid in those regimes.'
                    )
                if result['sparse_regime_count'] > 0:
                    result['warnings'].append(
                        f'{result["sparse_regime_count"]} S/O×T regimes have <5 rows '
                        f'— model may be unreliable in sparse regimes.'
                    )
            except (ValueError, TypeError) as exc:
                logger.warning(f'[DIAG S1] Coverage matrix failed: {exc}')

    # ---- C. Feed property imputation age ----
    for col_stem, age_col in [('feed_density', 'feed_density_age_hr'),
                               ('feed_ccr',     'feed_ccr_age_hr')]:
        if age_col in visc_anchored.columns:
            age_series = visc_anchored[age_col].dropna()
            if len(age_series) > 0:
                med_age = float(age_series.median())
                max_age = float(age_series.max())
                pct_over = float((age_series > 168).mean() * 100.0)
                if col_stem == 'feed_density':
                    result['feed_density_median_age_hr'] = med_age
                    result['feed_density_max_age_hr']    = max_age
                    if pct_over > 40.0:
                        result['warnings'].append(
                            f'{pct_over:.0f}% of rows have feed_density age > 168hr '
                            f'(max {max_age:.0f}hr). Density uncertainty elevated.'
                        )
                else:
                    result['feed_ccr_median_age_hr']   = med_age
                    result['feed_ccr_pct_over_168hr']  = pct_over
                    if pct_over > 40.0:
                        result['warnings'].append(
                            f'{pct_over:.0f}% of rows have feed_ccr age > 168hr. '
                            f'CCR uncertainty elevated.'
                        )

    # ---- D. Date coverage gaps ----
    if hasattr(dcs.index, 'sort_values') and len(dcs) > 1:
        ts_sorted = dcs.index.sort_values()
        diffs = np.diff(ts_sorted.astype(np.int64)) / 1e9 / 3600  # hours
        gap_indices = np.where(diffs > 14 * 24)[0]  # gaps > 14 days
        for i in gap_indices:
            try:
                gap_days = float(diffs[i]) / 24.0
                result['date_gaps'].append({
                    'start': str(ts_sorted[i])[:10],
                    'end':   str(ts_sorted[i + 1])[:10],
                    'days':  round(gap_days, 1),
                })
            except (IndexError, TypeError):
                pass

        # Freshness check
        try:
            most_recent = ts_sorted[-1].to_pydatetime()
            age_days = (datetime.now() - most_recent).days
            if age_days > 30:
                result['warnings'].append(
                    f'Most recent DCS data is {age_days} days old. '
                    f'Calibration reflects conditions from {str(ts_sorted[-1])[:10]}.'
                )
        except (AttributeError, TypeError):
            pass

    logger.info(
        f'[DIAG S1] visc_anchored={result["visc_anchored_rows"]}, '
        f'steady_state={result["steady_state_rows"]}, '
        f'warnings={len(result["warnings"])}, blocking={result["blocking"]}'
    )
    return result


# ---------------------------------------------------------------------------
# STAGE 2 — LIMS alignment quality
# ---------------------------------------------------------------------------

def _check_lims_alignment(visc_anchored: pd.DataFrame) -> dict:
    """
    Validate LIMS-to-DCS timestamp matching quality.

    Checks: match age, viscosity range sanity, duplicates,
    autocorrelation, and visc-yield correlation.

    Blocking: >30% matches age > 5hr, OR visc-yield r > +0.3.
    """
    result: dict = {
        'pass':                   True,
        'blocking':               False,
        'median_match_age_hr':    float('nan'),
        'pct_age_over_5hr':       float('nan'),
        'visc_range':             {'min': float('nan'), 'max': float('nan'),
                                   'mean': float('nan')},
        'rows_excluded_bad_visc': 0,
        'visc_yield_correlation': float('nan'),
        'autocorrelation_lag1':   float('nan'),
        'duplicates_found':       0,
        'issues':                 [],
        'warnings':               [],
        'fix':                    '',
    }

    if len(visc_anchored) == 0:
        result['issues'].append('visc_anchored is empty — skipping LIMS alignment check')
        return result

    # ---- A. Match age distribution ----
    if 'dao_visc_100_age_hr' in visc_anchored.columns:
        age_series = visc_anchored['dao_visc_100_age_hr'].dropna()
        if len(age_series) > 0:
            result['median_match_age_hr'] = float(age_series.median())
            pct_over_5 = float((age_series > 5.0).mean() * 100.0)
            result['pct_age_over_5hr'] = pct_over_5

            if result['median_match_age_hr'] > 4.0:
                result['warnings'].append(
                    f'Median LIMS match age = {result["median_match_age_hr"]:.1f}hr > 4hr. '
                    f'The 3hr lag assumption may be wrong. Consider increasing lag_hours to 4.'
                )
            if pct_over_5 > 30.0:
                result['blocking'] = True
                result['pass']     = False
                result['issues'].append(
                    f'{pct_over_5:.0f}% of LIMS matches have age > 5hr (threshold 30%) '
                    f'— LIMS timestamps may not align with DCS conditions.'
                )
                result['fix'] = (
                    f'{pct_over_5:.0f}% of LIMS matches have age > 5hr. '
                    f'Increase dao_visc_100 tolerance from 6hr to 9hr in '
                    f'build_calibration_dataset(), or investigate LIMS reporting delay.'
                )
                return result

    # ---- B. Viscosity range sanity ----
    if 'dao_visc_100' in visc_anchored.columns:
        visc_series = visc_anchored['dao_visc_100'].dropna()
        if len(visc_series) > 0:
            result['visc_range'] = {
                'min':  float(visc_series.min()),
                'max':  float(visc_series.max()),
                'mean': float(visc_series.mean()),
            }
            outside_mask = (visc_series < 15.0) | (visc_series > 80.0)
            n_outside = int(outside_mask.sum())
            if n_outside > 0:
                result['rows_excluded_bad_visc'] = n_outside
                if n_outside / len(visc_series) > 0.10:
                    result['warnings'].append(
                        f'{n_outside} visc_anchored rows have dao_visc_100 outside '
                        f'[15, 80] cSt — check LIMS units (should be cSt, not mPa·s).'
                    )
                else:
                    result['warnings'].append(
                        f'{n_outside} rows flagged with dao_visc_100 outside [15, 80] cSt.'
                    )

    # ---- C. Duplicate detection ----
    if 'dao_visc_100' in visc_anchored.columns:
        # Detect repeated identical viscosity values (possible data entry issue)
        visc_counts = visc_anchored['dao_visc_100'].value_counts()
        repeated = int((visc_counts > 2).sum())
        result['duplicates_found'] = repeated
        if repeated > 5:
            result['warnings'].append(
                f'{repeated} DAO viscosity values appear > 2 times in LIMS — '
                f'possible copy-paste or interpolation in LIMS.'
            )

    # ---- D. Autocorrelation check ----
    if 'dao_visc_100' in visc_anchored.columns:
        v = visc_anchored['dao_visc_100'].dropna().values
        if len(v) > 2:
            try:
                ac1 = float(np.corrcoef(v[:-1], v[1:])[0, 1])
                result['autocorrelation_lag1'] = ac1
                if ac1 > 0.98:
                    result['warnings'].append(
                        f'DAO viscosity lag-1 autocorrelation = {ac1:.3f} > 0.98. '
                        f'Data may be interpolated or copy-pasted in LIMS.'
                    )
            except (ValueError, TypeError):
                pass

    # ---- E. Visc vs yield correlation (KEY check) ----
    if ('dao_visc_100' in visc_anchored.columns and
            'dao_yield_vol_pct' in visc_anchored.columns):
        df_paired = visc_anchored[['dao_visc_100', 'dao_yield_vol_pct']].dropna()
        if len(df_paired) >= 10:
            try:
                r = float(np.corrcoef(
                    df_paired['dao_visc_100'].values,
                    df_paired['dao_yield_vol_pct'].values,
                )[0, 1])
                result['visc_yield_correlation'] = r
                if r > 0.3:
                    result['blocking'] = True
                    result['pass']     = False
                    result['issues'].append(
                        f'visc-yield Pearson r = {r:.2f} (positive, expected negative). '
                        f'LIMS-DCS alignment is wrong — calibration will be corrupted.'
                    )
                    result['fix'] = (
                        f'LIMS-DCS alignment is wrong. Actual DAO viscosity should decrease '
                        f'as DAO yield increases. Observed r={r:.2f} (positive). '
                        f'Check dao_lag parameter — try increasing from 3hr to 4hr or 5hr. '
                        f'Also verify DAO flow tag 41fic41110.pv is reading correctly.'
                    )
                    return result
                elif r > 0.0:
                    result['warnings'].append(
                        f'visc-yield correlation r={r:.2f} is positive (expected negative). '
                        f'Signal is weak — watch for misalignment.'
                    )
            except (ValueError, TypeError):
                pass

    logger.info(
        f'[DIAG S2] LIMS alignment: median_age={result["median_match_age_hr"]:.1f}hr, '
        f'visc_yield_r={result["visc_yield_correlation"]:.3f}, '
        f'blocking={result["blocking"]}'
    )
    return result


# ---------------------------------------------------------------------------
# STAGE 3 — Thermal model health
# ---------------------------------------------------------------------------

def _check_thermal_health(
    dcs: pd.DataFrame,
    thermal_profile: dict,
) -> dict:
    """
    Evaluate thermal profile against current DCS dataset (forward eval only).
    Auto-re-calibrates if blocking MAE detected; blocks only if re-cal fails.

    Returns per-train MAE, bias, S/O correlation, time drift, pass/blocking.
    """
    from thermal_calibration import predict_t_profile_calibrated, calibrate_thermal_model, save_thermal_profile

    result: dict = {
        'pass':         True,
        'blocking':     False,
        'recalibrated': False,
        'train_a':      {},
        'train_b':      {},
        'profile_age_days': None,
        'issues':       [],
        'warnings':     [],
        'fix':          '',
    }

    # ---- Profile age ----
    created_at = thermal_profile.get('created_at')
    if created_at:
        try:
            profile_dt = datetime.fromisoformat(created_at)
            result['profile_age_days'] = (datetime.now() - profile_dt).days
            if result['profile_age_days'] > 60:
                result['warnings'].append(
                    f'Thermal profile is {result["profile_age_days"]} days old. '
                    f'Consider re-calibrating.'
                )
        except (ValueError, TypeError):
            pass

    def _eval_train(train: str) -> dict:
        """Evaluate one train's thermal predictions vs DCS measurements."""
        tv = f'train_valid_{train}'
        valid_mask = pd.Series(True, index=dcs.index)
        if tv in dcs.columns:
            valid_mask &= dcs[tv].fillna(False)
        if 'steady_state' in dcs.columns:
            valid_mask &= dcs['steady_state'].fillna(False)

        sub = dcs[valid_mask].copy()
        if len(sub) == 0:
            return {
                'mae_t_bot': float('nan'), 'mae_t_mid': float('nan'),
                'mae_t_steam_coil': float('nan'), 'mae_t_top': float('nan'),
                'bias_t_bot': float('nan'), 'bias_t_top': float('nan'),
                'so_correlation': float('nan'), 'time_drift_detected': False,
                'rows_used': 0,
            }

        t_bot_col  = f't_bot_{train}'
        t_mid_col  = f't_mid_{train}'
        t_sc_col   = f't_steam_coil_{train}'
        t_top_col  = f't_top_{train}'
        tf_col     = f'feed_temp_{train}'
        so_col     = f'so_ratio_{train}'

        required = [t_bot_col, t_mid_col, t_sc_col, t_top_col]
        for c in required:
            if c not in sub.columns:
                return {'rows_used': 0, 'mae_t_bot': float('nan'),
                        'mae_t_mid': float('nan'), 'mae_t_steam_coil': float('nan'),
                        'mae_t_top': float('nan'), 'bias_t_bot': float('nan'),
                        'bias_t_top': float('nan'), 'so_correlation': float('nan'),
                        'time_drift_detected': False}

        pred_bots, pred_mids, pred_scs, pred_tops = [], [], [], []
        meas_bots, meas_mids, meas_scs, meas_tops = [], [], [], []
        so_vals = []

        t_feed_default = 85.0
        t_prop_default = 55.0

        for _, row in sub.iterrows():
            try:
                t_feed = float(row.get(tf_col, t_feed_default) or t_feed_default)
                t_prop = float(row.get('propane_temp', t_prop_default) or t_prop_default)
                so     = float(row.get(so_col, 8.0) or 8.0)
                profile = predict_t_profile_calibrated(
                    thermal_profile, t_feed, t_prop, so, train
                )
                pred_bots.append(profile[0])
                pred_mids.append(profile[1])
                pred_scs.append(profile[2])
                pred_tops.append(profile[3])
                meas_bots.append(float(row[t_bot_col]))
                meas_mids.append(float(row[t_mid_col]))
                meas_scs.append(float(row[t_sc_col]))
                meas_tops.append(float(row[t_top_col]))
                so_vals.append(so)
            except (TypeError, ValueError, KeyError):
                continue

        if not pred_bots:
            return {'rows_used': 0, 'mae_t_bot': float('nan'),
                    'mae_t_mid': float('nan'), 'mae_t_steam_coil': float('nan'),
                    'mae_t_top': float('nan'), 'bias_t_bot': float('nan'),
                    'bias_t_top': float('nan'), 'so_correlation': float('nan'),
                    'time_drift_detected': False}

        pb = np.array(pred_bots); mb = np.array(meas_bots)
        pm = np.array(pred_mids); mm = np.array(meas_mids)
        ps = np.array(pred_scs);  ms = np.array(meas_scs)
        pt = np.array(pred_tops); mt = np.array(meas_tops)

        mae_bot = float(np.mean(np.abs(pb - mb)))
        mae_mid = float(np.mean(np.abs(pm - mm)))
        mae_sc  = float(np.mean(np.abs(ps - ms)))
        mae_top = float(np.mean(np.abs(pt - mt)))
        bias_bot = float(np.mean(pb - mb))
        bias_top = float(np.mean(pt - mt))

        # S/O correlation on T_bot residual
        resid_bot = pb - mb
        so_r = float('nan')
        if len(so_vals) > 3:
            try:
                so_r = float(np.corrcoef(resid_bot, np.array(so_vals))[0, 1])
            except (ValueError, TypeError):
                pass

        # Time drift: does MAE increase in last 30 days?
        time_drift = False
        if len(sub) > 60:
            try:
                cutoff = sub.index[-1] - pd.Timedelta(days=30)
                recent_mask = sub.index >= cutoff
                n_recent = int(recent_mask.sum())
                if n_recent >= 10:
                    recent_resid = np.abs(
                        np.array(pred_bots[-n_recent:]) -
                        np.array(meas_bots[-n_recent:])
                    )
                    full_resid = np.abs(pb - mb)
                    if np.mean(recent_resid) > np.mean(full_resid) * 1.5:
                        time_drift = True
            except (IndexError, TypeError, AttributeError):
                pass

        return {
            'mae_t_bot':          mae_bot,
            'mae_t_mid':          mae_mid,
            'mae_t_steam_coil':   mae_sc,
            'mae_t_top':          mae_top,
            'bias_t_bot':         bias_bot,
            'bias_t_top':         bias_top,
            'so_correlation':     so_r,
            'time_drift_detected': time_drift,
            'rows_used':          len(pred_bots),
        }

    def _needs_recal(train_result: dict) -> bool:
        """Return True if any bed MAE > 3.0°C or phi S/O correlation > 0.6."""
        return (
            any(train_result.get(k, 0.0) > 3.0
                for k in ['mae_t_bot', 'mae_t_mid', 'mae_t_steam_coil', 'mae_t_top'])
            or abs(train_result.get('so_correlation', 0.0)) > 0.6
        )

    result['train_a'] = _eval_train('a')
    result['train_b'] = _eval_train('b')

    needs_recal = _needs_recal(result['train_a']) or _needs_recal(result['train_b'])

    if needs_recal:
        logger.info('[DIAG S3] Thermal MAE too high — auto re-calibrating thermal model.')
        print('[DIAG S3] Thermal profile stale/inaccurate — re-calibrating automatically.',
              flush=True)
        try:
            new_thermal = calibrate_thermal_model(dcs)
            save_thermal_profile(new_thermal)
            # Re-evaluate with new profile
            thermal_profile.update(new_thermal)  # mutate in place for downstream use
            result['train_a'] = _eval_train('a')
            result['train_b'] = _eval_train('b')
            result['recalibrated'] = True
            logger.info('[DIAG S3] Thermal re-calibration complete.')
        except Exception as exc:
            logger.error(f'[DIAG S3] Thermal re-calibration failed: {exc}')
            result['issues'].append(f'Auto thermal re-calibration failed: {exc}')

    # ---- Decision logic ----
    for train_lbl, tr in [('A', result['train_a']), ('B', result['train_b'])]:
        for bed, k in [('bottom', 'mae_t_bot'), ('middle', 'mae_t_mid'),
                       ('steam coil', 'mae_t_steam_coil'), ('top', 'mae_t_top')]:
            mae_val = tr.get(k, float('nan'))
            if math.isnan(mae_val):
                continue
            if mae_val > 3.0:
                result['blocking'] = True
                result['pass'] = False
                result['issues'].append(
                    f'Train {train_lbl} {bed} MAE = {mae_val:.2f}°C > 3.0°C (BLOCKING)'
                )
                result['fix'] = (
                    f'Thermal model accuracy is insufficient (max bed MAE > 3°C after '
                    f'auto re-calibration). '
                    f'Check that steady_state filter is working and DCS temperature '
                    f'sensors are not stuck. Minimum 500 valid DCS rows required for '
                    f'thermal calibration.'
                )
            elif mae_val > 2.0:
                result['warnings'].append(
                    f'Train {train_lbl} {bed} MAE = {mae_val:.2f}°C (2.0-3.0°C range — watch closely)'
                )

        so_r = tr.get('so_correlation', float('nan'))
        if not math.isnan(so_r):
            if abs(so_r) > 0.6:
                result['blocking'] = True
                result['pass'] = False
                result['issues'].append(
                    f'Train {train_lbl} T_bot residual vs S/O r={so_r:.2f} > 0.6 (BLOCKING)'
                )
                result['fix'] = (
                    f'phi parameter is wrong — T_bottom has strong S/O dependence '
                    f'not captured. Check phi bounds in thermal_calibration.py '
                    f'(currently [0.0, 0.30]). Widen upper bound to 0.50.'
                )
            elif abs(so_r) > 0.4:
                result['warnings'].append(
                    f'Train {train_lbl} T_bot residual vs S/O r={so_r:.2f} — phi may be slightly off.'
                )

        if tr.get('time_drift_detected'):
            result['warnings'].append(
                f'Train {train_lbl}: thermal prediction worsens in last 30 days — '
                f'thermal drift detected. Re-calibration recommended.'
            )

    logger.info(
        f'[DIAG S3] Thermal health: A_mae_bot={result["train_a"].get("mae_t_bot", float("nan")):.2f}°C, '
        f'blocking={result["blocking"]}, recalibrated={result["recalibrated"]}'
    )
    return result


# ---------------------------------------------------------------------------
# STAGE 4 — Physics readiness
# ---------------------------------------------------------------------------

def _stratified_subsample(
    df: pd.DataFrame,
    n_total: int,
    so_col: str = 'so_ratio_a',
    t_col:  str = 't_top_a',
) -> pd.DataFrame:
    """
    Sample n_total rows with equal representation across S/O × T bins.
    Every occupied regime is included; empty bins are skipped.
    """
    if len(df) <= n_total:
        return df.copy()

    df_work = df.copy()
    so_col_used = so_col if so_col in df_work.columns else None
    t_col_used  = t_col  if t_col  in df_work.columns else None

    if so_col_used is None or t_col_used is None:
        return df_work.sample(n=min(n_total, len(df_work)), random_state=42)

    try:
        df_work['_so_bin'] = pd.qcut(
            df_work[so_col_used], q=4, labels=False, duplicates='drop'
        )
        df_work['_t_bin'] = pd.qcut(
            df_work[t_col_used], q=4, labels=False, duplicates='drop'
        )
    except ValueError:
        return df_work.sample(n=min(n_total, len(df_work)), random_state=42)

    df_work['_combo_bin'] = (
        df_work['_so_bin'].fillna(-1).astype(int).astype(str) + '_' +
        df_work['_t_bin'].fillna(-1).astype(int).astype(str)
    )
    n_bins_actual = df_work['_combo_bin'].nunique()
    n_per_bin = max(2, n_total // n_bins_actual)

    parts = []
    for _, grp in df_work.groupby('_combo_bin'):
        n = min(len(grp), n_per_bin)
        parts.append(grp.sample(n=n, random_state=42))

    sampled = pd.concat(parts).drop(
        columns=['_so_bin', '_t_bin', '_combo_bin'], errors='ignore'
    )
    return sampled.reset_index(drop=False)  # preserve original index column


def _check_physics_readiness(
    visc_anchored: pd.DataFrame,
    dcs: pd.DataFrame,
    thermal_params: dict,
    default_params: dict,
    feed_cache: dict,
) -> dict:
    """
    Run physics simulation on 20 stratified rows to verify readiness.

    A. Feed component validation (5 rows)
    B. K_multiplier sensitivity sweep (10 rows × 5 K_mult values)
    C. Baseline convergence rate (20 rows)
    D. Jacobian condition number (20 rows)

    Returns structured result with pass/blocking/issues/warnings.
    """
    from simulator_bridge import simulate_parallel_trains, _build_feed_components_from_lab

    result: dict = {
        'pass':                   True,
        'blocking':               False,
        'rows_tested':            0,
        'feed_components_valid':  True,
        'k_mult_responsive':      True,
        'yield_range_from_sweep': [float('nan'), float('nan')],
        'convergence_rate':       float('nan'),
        'condition_number':       float('nan'),
        'correlated_param_pair':  None,
        'issues':                 [],
        'warnings':               [],
        'fix':                    '',
    }

    # Pull 20 stratified rows from visc_anchored (falls back to dcs)
    source_df = visc_anchored if len(visc_anchored) >= 20 else dcs
    if len(source_df) == 0:
        result['blocking'] = True
        result['pass'] = False
        result['issues'].append('No rows available for physics readiness check.')
        result['fix'] = 'Load data first.'
        return result

    test_df = _stratified_subsample(source_df, n_total=20)
    # Restore Timestamp as index if it was moved to a column
    if 'index' in test_df.columns:
        try:
            test_df = test_df.set_index('index')
        except Exception:
            pass

    result['rows_tested'] = len(test_df)

    # ---- A. Feed component check (5 diverse rows) ----
    comp_test_df = test_df.head(5) if len(test_df) >= 5 else test_df
    feed_issues = []
    for _, row in comp_test_df.iterrows():
        try:
            density = float(row.get('feed_density', 1.028) or 1.028)
            ccr     = float(row.get('feed_ccr', 22.8) or 22.8)
            visc135 = float(row.get('feed_visc_135', 230.0) or 230.0)
            comps   = _build_feed_components_from_lab(density, ccr, visc135)
            z_sum   = sum(c.z for c in comps)
            mw_list = [c.MW for c in comps]
            rho_list = [c.rho_liq for c in comps]

            if not (0.999 <= z_sum <= 1.001):
                feed_issues.append(
                    f'z_sum={z_sum:.4f} (expected 1.000±0.001) for density={density:.3f}'
                )
            if any(mw < 50 or mw > 3000 for mw in mw_list):
                bad_mw = [mw for mw in mw_list if mw < 50 or mw > 3000]
                feed_issues.append(f'MW outside [50, 3000]: {bad_mw[:3]}')
            if any(rho < 0.80 or rho > 1.15 for rho in rho_list):
                bad_rho = [rho for rho in rho_list if rho < 0.80 or rho > 1.15]
                feed_issues.append(f'density outside [0.80, 1.15]: {bad_rho[:3]}')
        except Exception as exc:
            feed_issues.append(f'Feed component build failed: {exc}')

    if feed_issues:
        result['feed_components_valid'] = False
        result['blocking'] = True
        result['pass'] = False
        result['issues'].extend(feed_issues)
        result['fix'] = (
            'Feed characterization is degenerate. '
            'Most likely cause: visc135→visc100 conversion producing extreme values. '
            'Check _visc135_to_visc100() output. '
            'Or feed_density values outside expected range [0.95, 1.10] g/cm³.'
        )
        return result

    # ---- B. K_multiplier sensitivity sweep (10 rows) ----
    sweep_df = test_df.head(10) if len(test_df) >= 10 else test_df
    k_sweep_values = [0.5, 0.7, 0.9, 1.1, 1.3]
    sweep_yields_by_k: dict = {k: [] for k in k_sweep_values}

    for _, row in sweep_df.iterrows():
        row_yields = []
        for k_val in k_sweep_values:
            p = dict(default_params)
            p['K_multiplier'] = k_val
            try:
                sim = simulate_parallel_trains(
                    row, p, feed_cache, thermal_params=thermal_params,
                    use_dcs_temperatures=True,
                )
                if sim['converged']:
                    row_yields.append(sim['dao_yield_vol_pct'])
                else:
                    row_yields.append(None)
            except Exception:
                row_yields.append(None)

        for i, k_val in enumerate(k_sweep_values):
            if row_yields[i] is not None:
                sweep_yields_by_k[k_val].append(row_yields[i])

    mean_yields = [
        float(np.mean(sweep_yields_by_k[k])) if sweep_yields_by_k[k] else float('nan')
        for k in k_sweep_values
    ]
    valid_means = [y for y in mean_yields if not math.isnan(y)]

    if len(valid_means) >= 2:
        yield_range = max(valid_means) - min(valid_means)
        result['yield_range_from_sweep'] = [min(valid_means), max(valid_means)]
        if yield_range < 2.0:
            result['k_mult_responsive'] = False
            result['blocking'] = True
            result['pass'] = False
            visc_vals = [
                row.get('feed_visc_135', 230.0)
                for _, row in sweep_df.iterrows()
            ]
            result['issues'].append(
                f'K_mult sweep yield range = {yield_range:.1f} vol% < 2 vol% — '
                f'K_multiplier has no effect on DAO yield.'
            )
            result['fix'] = (
                f'K_multiplier sweep produced yield range of only {yield_range:.1f} vol% '
                f'(expected > 10 vol%). Feed characterization is degenerate. '
                f'Most likely cause: visc135→visc100 conversion producing extreme values. '
                f'Check _visc135_to_visc100() output for feed viscosities: '
                f'{list(set(int(v) for v in visc_vals if v is not None))[:5]}. '
                f'Or: feed_density values are outside expected range [0.95, 1.10] g/cm³.'
            )
            return result

    # ---- C. Baseline convergence rate (20 rows) ----
    n_converged = 0
    rows_list = list(test_df.iterrows())
    for _, row in rows_list:
        try:
            sim = simulate_parallel_trains(
                row, default_params, feed_cache,
                thermal_params=thermal_params,
                use_dcs_temperatures=True,
            )
            if sim['converged']:
                n_converged += 1
        except Exception:
            pass

    convergence_rate = n_converged / max(len(rows_list), 1)
    result['convergence_rate'] = convergence_rate

    if convergence_rate < 0.80:
        # Identify S/O range of the test rows
        so_range = 'unknown'
        if 'so_ratio_a' in test_df.columns:
            so_vals = test_df['so_ratio_a'].dropna()
            if len(so_vals) > 0:
                so_range = f'{so_vals.min():.1f}–{so_vals.max():.1f}'
        result['blocking'] = True
        result['pass'] = False
        result['issues'].append(
            f'Convergence rate = {convergence_rate:.0%} < 80% at default params.'
        )
        result['fix'] = (
            f'Only {convergence_rate:.0%} of test rows converged at default parameters. '
            f'Increase run_extractor max_outer_iter from 60 to 100. '
            f'Or widen outer_tol from 0.3 to 0.8 for the fast-mode pass. '
            f'Affected S/O range: {so_range}. Check if propane density model '
            f'is valid at these conditions.'
        )
        return result
    elif convergence_rate < 0.90:
        result['warnings'].append(
            f'Convergence rate = {convergence_rate:.0%} (80-90% — acceptable but low). '
            f'Consider widening outer_tol.'
        )

    # ---- D. Jacobian condition number (numerical, 20 rows) ----
    param_names = ['K_multiplier', 'E_murphree', 'C_entrain', 'delta_crit']
    x0 = np.array([
        default_params.get('K_multiplier', 1.0),
        default_params.get('E_murphree', 0.70),
        default_params.get('C_entrain', 0.015),
        default_params.get('delta_crit', 2.5),
    ])
    h = np.array([0.05, 0.05, 0.001, 0.1])  # step sizes per parameter

    def _residuals_at(x_arr: np.ndarray) -> np.ndarray:
        p = dict(default_params)
        for name, val in zip(param_names, x_arr):
            p[name] = float(val)
        resids = []
        for _, row in rows_list:
            try:
                sim = simulate_parallel_trains(
                    row, p, feed_cache,
                    thermal_params=thermal_params,
                    use_dcs_temperatures=True,
                )
                if sim['converged']:
                    resids.append(sim['dao_yield_vol_pct'])
                else:
                    resids.append(0.0)
            except Exception:
                resids.append(0.0)
        return np.array(resids)

    try:
        r0 = _residuals_at(x0)
        J_cols = []
        for i, hi in enumerate(h):
            x_fwd = x0.copy(); x_fwd[i] += hi
            r_fwd = _residuals_at(x_fwd)
            J_cols.append((r_fwd - r0) / hi)
        J = np.column_stack(J_cols)
        JtJ = J.T @ J
        cond = float(np.linalg.cond(JtJ))
        result['condition_number'] = cond
        if cond > 1e8:
            # Find most correlated pair
            pair = None
            try:
                JtJ_norm = JtJ / np.sqrt(np.outer(np.diag(JtJ), np.diag(JtJ)) + 1e-30)
                corr_abs = np.abs(JtJ_norm)
                np.fill_diagonal(corr_abs, 0.0)
                i_max, j_max = np.unravel_index(np.argmax(corr_abs), corr_abs.shape)
                pair = f'{param_names[i_max]} vs {param_names[j_max]}'
            except Exception:
                pass
            result['correlated_param_pair'] = pair
            result['warnings'].append(
                f'Jacobian condition number = {cond:.1e} > 1e8 — '
                f'parameters may be correlated ({pair}). '
                f'Calibration can still proceed but parameter identifiability is limited.'
            )
    except Exception as exc:
        logger.debug(f'[DIAG S4] Jacobian check failed (non-blocking): {exc}')

    logger.info(
        f'[DIAG S4] Physics readiness: convergence={convergence_rate:.0%}, '
        f'k_mult_responsive={result["k_mult_responsive"]}, '
        f'blocking={result["blocking"]}'
    )
    return result


# ---------------------------------------------------------------------------
# STAGE 5 — Two-speed optimizer
# ---------------------------------------------------------------------------

def _get_warm_start_params(
    default_params: dict,
    profile_dir: str = 'calibration_profiles',
) -> tuple:
    """
    Load the most recent deployed profile's K_multiplier as optimizer x0.

    Returns (params_dict, source_profile_name) or (default_params, None).
    """
    try:
        pattern = os.path.join(profile_dir, 'plant_calibration_*.json')
        files   = sorted(_glob.glob(pattern))
        if not files:
            # Also check for default name
            default_path = os.path.join(profile_dir, 'plant_calibration_v1.json')
            if os.path.exists(default_path):
                files = [default_path]

        if not files:
            return default_params.copy(), None

        latest = files[-1]
        with open(latest) as f:
            profile = json.load(f)

        cal_params = profile.get('calibration_params', {})
        warm_params = dict(default_params)
        if 'K_multiplier' in cal_params:
            warm_params['K_multiplier'] = float(cal_params['K_multiplier'])
        if 'visc_slope' in cal_params:
            warm_params['visc_slope'] = float(cal_params['visc_slope'])
        if 'visc_bias' in cal_params:
            warm_params['visc_bias']  = float(cal_params['visc_bias'])

        profile_name = os.path.splitext(os.path.basename(latest))[0]
        logger.info(
            f'[DIAG S5] Warm-starting from {profile_name}: '
            f'K_mult={warm_params.get("K_multiplier", "?"):.3f}'
        )
        print(
            f'[DIAG S5] Warm-starting from {profile_name}: '
            f'K_mult={warm_params.get("K_multiplier", 1.0):.3f}',
            flush=True,
        )
        return warm_params, profile_name
    except Exception as exc:
        logger.warning(f'[DIAG S5] Warm-start load failed ({exc}) — using defaults.')
        return default_params.copy(), None


def _run_two_speed_optimizer(
    visc_train: pd.DataFrame,
    dcs_train:  pd.DataFrame,
    warm_start_params: dict,
    thermal_params: dict,
    feed_cache: dict,
    weights: dict,
) -> dict:
    """
    Run K_multiplier optimizer in two sequential passes (fast then full).

    Pass 1: 200 subsampled rows, max_nfev=100 — find the right basin.
    Pass 2: all rows, max_nfev=300 — refine from pass 1 result.

    Includes: cost-decrease check, parameter-at-bound check, 3 random restarts.

    Returns status dict with calibrated_params (or blocking report).
    """
    from scipy.optimize import least_squares
    from calibration_engine import (
        _build_outer_residuals, _subsample_yield_df, _subsample_visc_df,
        _params_dict, _PHASE2_PARAM_NAMES, _PHASE2_BOUNDS_LO, _PHASE2_BOUNDS_HI,
        _FIXED_PHYSICS_PARAMS, _MAX_VISC_ROWS, _MAX_YIELD_ROWS,
        calibrate_visc_correction, _nfev_counter,
    )

    result: dict = {
        'status':            'success',
        'calibrated_params': None,
        'cost_initial':      float('nan'),
        'cost_fast':         float('nan'),
        'cost_full':         float('nan'),
        'nfev_fast':         0,
        'nfev_full':         0,
        'warm_start_source': None,
        'blocking':          False,
        'fix':               '',
    }

    # ---- Phase 1 OLS (visc correction) ----
    visc_train_sub = _subsample_visc_df(visc_train, _MAX_VISC_ROWS)

    ols_params = {
        'K_multiplier': float(warm_start_params.get('K_multiplier', 1.0)),
        'E_murphree':   float(_FIXED_PHYSICS_PARAMS['E_murphree']),
        'C_entrain':    float(_FIXED_PHYSICS_PARAMS['C_entrain']),
        'delta_crit':   float(_FIXED_PHYSICS_PARAMS['delta_crit']),
        'visc_slope':   1.0,
        'visc_bias':    0.0,
    }
    visc_slope, visc_bias, visc_r = calibrate_visc_correction(
        visc_train_sub, thermal_params, ols_params
    )
    fixed_params = {
        'E_murphree': float(_FIXED_PHYSICS_PARAMS['E_murphree']),
        'C_entrain':  float(_FIXED_PHYSICS_PARAMS['C_entrain']),
        'delta_crit': float(_FIXED_PHYSICS_PARAMS['delta_crit']),
        'visc_slope': visc_slope,
        'visc_bias':  visc_bias,
    }

    # ---- Build yield training sets ----
    mask = (
        dcs_train['steady_state'].fillna(False) &
        dcs_train['train_valid_a'].fillna(False) &
        dcs_train['train_valid_b'].fillna(False) &
        dcs_train['dao_yield_vol_pct'].notna()
    )
    yield_full = dcs_train[mask]
    yield_fast = _subsample_yield_df(yield_full, 200)            # fast: 200 rows
    yield_full_sub = _subsample_yield_df(yield_full, _MAX_YIELD_ROWS)  # full: ~500 rows

    k0 = float(warm_start_params.get('K_multiplier', 1.0))

    # ---- Compute initial cost ----
    _nfev_counter[0] = 0
    try:
        r0 = _build_outer_residuals(
            np.array([k0]), yield_fast, feed_cache, thermal_params,
            weights, _PHASE2_PARAM_NAMES, fixed_params
        )
        result['cost_initial'] = float(np.sum(r0 ** 2))
    except Exception as exc:
        logger.warning(f'[DIAG S5] Initial cost computation failed: {exc}')
        result['cost_initial'] = float('nan')

    # ---- PASS 1: Fast mode (200 rows, max_nfev=100) ----
    _nfev_counter[0] = 0
    print('[DIAG S5] Pass 1 (fast): 200 yield rows, max_nfev=100...', flush=True)

    def _run_fast(x_start: float) -> object:
        _nfev_counter[0] = 0
        return least_squares(
            fun=_build_outer_residuals,
            x0=[x_start],
            bounds=(_PHASE2_BOUNDS_LO, _PHASE2_BOUNDS_HI),
            method='trf',
            args=(yield_fast, feed_cache, thermal_params, weights,
                  _PHASE2_PARAM_NAMES, fixed_params),
            max_nfev=100,
            ftol=1e-3,
        )

    fast_result = _run_fast(k0)
    result['nfev_fast'] = fast_result.nfev
    result['cost_fast'] = float(np.sum(fast_result.fun ** 2))

    # ---- Check cost decrease ≥ 5% ----
    cost_initial = result['cost_initial']
    cost_fast    = result['cost_fast']
    pct_decrease = (
        (cost_initial - cost_fast) / max(abs(cost_initial), 1e-10) * 100.0
        if not math.isnan(cost_initial) else 100.0  # skip check if initial failed
    )

    if pct_decrease < 5.0:
        # Try 3 random restarts
        logger.info(
            f'[DIAG S5] Fast mode: cost decrease {pct_decrease:.1f}% < 5% — '
            f'trying 3 random restarts.'
        )
        rng = np.random.RandomState(42)
        k_range = _PHASE2_BOUNDS_HI[0] - _PHASE2_BOUNDS_LO[0]
        best_cost  = cost_fast
        best_result = fast_result

        for restart_i in range(3):
            k_restart = float(
                np.clip(
                    k0 + rng.uniform(-0.20, 0.20) * k_range,
                    _PHASE2_BOUNDS_LO[0] + 0.01,
                    _PHASE2_BOUNDS_HI[0] - 0.01,
                )
            )
            try:
                r_restart = _run_fast(k_restart)
                c_restart  = float(np.sum(r_restart.fun ** 2))
                if c_restart < best_cost:
                    best_cost   = c_restart
                    best_result = r_restart
                    logger.info(
                        f'[DIAG S5] Restart {restart_i + 1}: '
                        f'K={k_restart:.3f} → cost={c_restart:.3f} (new best)'
                    )
            except Exception as exc:
                logger.warning(f'[DIAG S5] Restart {restart_i + 1} failed: {exc}')

        best_pct = (
            (cost_initial - best_cost) / max(abs(cost_initial), 1e-10) * 100.0
            if not math.isnan(cost_initial) else 100.0
        )
        if best_pct < 5.0:
            result['status']   = 'optimizer_stuck_fast_mode'
            result['blocking'] = True
            result['fix'] = (
                f'Cost did not decrease in fast mode (decrease={best_pct:.1f}% < 5%). '
                f'Check: (1) K_mult may need wider bounds — current '
                f'operating yield outside calibrated range. '
                f'(2) Verify residual weights are not zeroing one target. '
                f'(3) Check if the yield training set is too small '
                f'(current: {len(yield_fast)} rows).'
            )
            return result
        fast_result = best_result
        result['cost_fast'] = float(np.sum(fast_result.fun ** 2))

    # ---- Check parameter at bound ----
    k_fast = float(fast_result.x[0])
    if k_fast <= _PHASE2_BOUNDS_LO[0] + 0.02:
        result['status']   = 'parameter_at_bound'
        result['blocking'] = True
        result['fix'] = (
            f'K_multiplier hit its lower bound at {k_fast:.3f}. '
            f'This means the current plant yield is below the physics model range. '
            f'If plant yield < 15%: re-anchor K-value base params (run _find_k_ref). '
            f'If operating in lube mode: widen K_mult lower bound from 0.70 to 0.50.'
        )
        return result
    if k_fast >= _PHASE2_BOUNDS_HI[0] - 0.02:
        result['status']   = 'parameter_at_bound'
        result['blocking'] = True
        result['fix'] = (
            f'K_multiplier hit its upper bound at {k_fast:.3f}. '
            f'This means the current plant yield is above the physics model range. '
            f'Check feed characterization — extremely light feeds can over-predict yield. '
            f'Widen K_mult upper bound from 1.40 to 1.80 if feed is lighter than Basra-Kuwait.'
        )
        return result

    # ---- PASS 2: Full precision (all yield rows, max_nfev=300) ----
    print(
        f'[DIAG S5] Pass 2 (full): {len(yield_full_sub)} yield rows, '
        f'max_nfev=300, warm_start K={k_fast:.4f}...',
        flush=True,
    )
    _nfev_counter[0] = 0
    full_result = least_squares(
        fun=_build_outer_residuals,
        x0=[k_fast],
        bounds=(_PHASE2_BOUNDS_LO, _PHASE2_BOUNDS_HI),
        method='trf',
        args=(yield_full_sub, feed_cache, thermal_params, weights,
              _PHASE2_PARAM_NAMES, fixed_params),
        max_nfev=300,
        ftol=1e-4,
    )
    result['nfev_full'] = full_result.nfev
    result['cost_full'] = float(np.sum(full_result.fun ** 2))

    pct_improve_vs_fast = (
        (result['cost_fast'] - result['cost_full']) /
        max(abs(result['cost_fast']), 1e-10) * 100.0
    )
    if pct_improve_vs_fast < 2.0:
        logger.info(
            f'[DIAG S5] Full-mode pass did not improve on fast-mode result '
            f'({pct_improve_vs_fast:.1f}% < 2%). Using full-mode result anyway.'
        )
        print(
            f'[DIAG S5] Full-mode converged (improvement={pct_improve_vs_fast:.1f}%).',
            flush=True,
        )

    k_final = float(full_result.x[0])
    result['calibrated_params'] = {
        'K_multiplier': k_final,
        'E_murphree':   float(fixed_params['E_murphree']),
        'C_entrain':    float(fixed_params['C_entrain']),
        'delta_crit':   float(fixed_params['delta_crit']),
        'visc_slope':   visc_slope,
        'visc_bias':    visc_bias,
    }

    logger.info(
        f'[DIAG S5] Optimizer complete: K_mult={k_final:.4f}, '
        f'visc_slope={visc_slope:.3f}, visc_bias={visc_bias:.2f}, '
        f'nfev_fast={result["nfev_fast"]}, nfev_full={result["nfev_full"]}'
    )
    return result


# ---------------------------------------------------------------------------
# STAGE 6 — Residual pattern analysis
# ---------------------------------------------------------------------------

def _analyze_residual_patterns(
    visc_test: pd.DataFrame,
    dcs_test:  pd.DataFrame,
    calibrated_params: dict,
    thermal_params: dict,
    feed_cache: dict,
) -> dict:
    """
    Run all residual pattern checks on the held-out TEST set.
    Evaluates each pattern in RESIDUAL_PATTERNS.

    Returns patterns detected, per-bin residuals, R²/MAE metrics, parity data.
    """
    from simulator_bridge import simulate_parallel_trains

    result: dict = {
        'patterns_detected':        [],
        'patterns_checked':         len(RESIDUAL_PATTERNS),
        'no_structural_issues':     True,
        'per_so_bin_visc_residual': {},
        'per_month_visc_residual':  {},
        'train_a_mean_residual':    float('nan'),
        'train_b_mean_residual':    float('nan'),
        'visc_r2':                  float('nan'),
        'yield_r2':                 float('nan'),
        'visc_mae':                 float('nan'),
        'yield_mae':                float('nan'),
        'parity_data': {
            'visc_actual':    [], 'visc_predicted': [],
            'yield_actual':   [], 'yield_predicted': [],
            'so_ratio':       [], 'timestamps':      [],
        },
    }

    # ---- Collect predictions on visc_test ----
    visc_actual, visc_pred = [], []
    visc_resid, visc_so, visc_t_top = [], [], []
    visc_months, visc_train_label = [], []
    visc_yield_vals = []

    for idx, row in visc_test.iterrows():
        try:
            sim = simulate_parallel_trains(
                row, calibrated_params, feed_cache,
                thermal_params=thermal_params, use_dcs_temperatures=True,
            )
            if sim['converged']:
                meas = float(row.get('dao_visc_100', float('nan')))
                pred = float(sim['dao_visc_100_cSt'])
                if math.isnan(meas):
                    continue
                visc_actual.append(meas)
                visc_pred.append(pred)
                visc_resid.append(pred - meas)
                visc_so.append(float(row.get('so_ratio_a', 8.0) or 8.0))
                visc_t_top.append(float(row.get('t_top_a', 75.0) or 75.0))
                visc_train_label.append('a')  # visc_anchored uses combined trains
                visc_yield_vals.append(float(row.get('dao_yield_vol_pct', float('nan'))))
                try:
                    visc_months.append(pd.Timestamp(idx).to_period('M').strftime('%Y-%m'))
                except (TypeError, AttributeError):
                    visc_months.append('unknown')
        except Exception:
            pass

    # ---- Collect predictions on dcs_test (yield) ----
    yield_actual, yield_pred = [], []
    yield_so = []

    ss_mask = dcs_test['steady_state'].fillna(False) \
        if 'steady_state' in dcs_test.columns else pd.Series(True, index=dcs_test.index)
    va_mask = dcs_test['train_valid_a'].fillna(False) \
        if 'train_valid_a' in dcs_test.columns else pd.Series(True, index=dcs_test.index)
    vb_mask = dcs_test['train_valid_b'].fillna(False) \
        if 'train_valid_b' in dcs_test.columns else pd.Series(True, index=dcs_test.index)
    yield_test_rows = dcs_test[ss_mask & va_mask & vb_mask &
                               dcs_test['dao_yield_vol_pct'].notna()]

    for _, row in yield_test_rows.iterrows():
        try:
            sim = simulate_parallel_trains(
                row, calibrated_params, feed_cache,
                thermal_params=thermal_params, use_dcs_temperatures=True,
            )
            if sim['converged']:
                meas = float(row.get('dao_yield_vol_pct', float('nan')))
                if math.isnan(meas):
                    continue
                yield_actual.append(meas)
                yield_pred.append(sim['dao_yield_vol_pct'])
                yield_so.append(float(row.get('so_ratio_a', 8.0) or 8.0))
        except Exception:
            pass

    # ---- Compute R²/MAE ----
    def _r2_mae(actual, predicted):
        if len(actual) < 2:
            return float('nan'), float('nan')
        a = np.array(actual); p = np.array(predicted)
        mae = float(np.mean(np.abs(p - a)))
        ss_res = float(np.sum((a - p) ** 2))
        ss_tot = float(np.sum((a - a.mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        return r2, mae

    result['visc_r2'],  result['visc_mae']  = _r2_mae(visc_actual, visc_pred)
    result['yield_r2'], result['yield_mae'] = _r2_mae(yield_actual, yield_pred)

    result['parity_data'] = {
        'visc_actual':    visc_actual,
        'visc_predicted': visc_pred,
        'yield_actual':   yield_actual,
        'yield_predicted': yield_pred,
        'so_ratio':       visc_so,
        'timestamps':     visc_months,
    }

    if not visc_resid:
        logger.warning('[DIAG S6] No converged visc rows — skipping pattern checks.')
        return result

    visc_resid_arr = np.array(visc_resid)
    visc_so_arr    = np.array(visc_so)
    visc_t_top_arr = np.array(visc_t_top)

    # ---- Per S/O bin residuals ----
    if len(visc_so_arr) >= 4:
        try:
            so_bins = pd.qcut(visc_so_arr, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'],
                              duplicates='drop')
            for lbl in so_bins.cat.categories:
                mask = so_bins == lbl
                if mask.sum() > 0:
                    result['per_so_bin_visc_residual'][str(lbl)] = float(
                        visc_resid_arr[mask].mean()
                    )
        except (ValueError, AttributeError):
            pass

    # ---- Per month residuals ----
    month_groups: dict = {}
    for month, resid in zip(visc_months, visc_resid):
        month_groups.setdefault(month, []).append(resid)
    result['per_month_visc_residual'] = {
        m: float(np.mean(v)) for m, v in sorted(month_groups.items())
    }

    # ---- Pattern detection ----
    detected = []

    # so_regime_bias
    if result['per_so_bin_visc_residual']:
        bin_means = list(result['per_so_bin_visc_residual'].values())
        std_bin_means = float(np.std(bin_means))
        if std_bin_means > 2.0:
            pat = dict(RESIDUAL_PATTERNS['so_regime_bias'])
            pat['detected_value'] = f'std of bin means = {std_bin_means:.2f} cSt (threshold 2.0)'
            detected.append(pat)

    # temperature_slope
    if len(visc_t_top_arr) >= 4:
        try:
            r_t = float(np.corrcoef(visc_resid_arr, visc_t_top_arr)[0, 1])
            if abs(r_t) > 0.4:
                pat = dict(RESIDUAL_PATTERNS['temperature_slope'])
                pat['detected_value'] = f'r(visc_resid, T_top) = {r_t:.3f} (threshold ±0.4)'
                detected.append(pat)
        except (ValueError, TypeError):
            pass

    # train_ab_offset — using combined visc_anchored so approximate with yield split
    # (single train label not easily available in combined visc_anchored; skip)

    # time_drift
    if len(result['per_month_visc_residual']) >= 3:
        monthly_means = list(result['per_month_visc_residual'].values())
        drift = max(monthly_means) - min(monthly_means)
        if drift > 3.0:
            pat = dict(RESIDUAL_PATTERNS['time_drift'])
            pat['detected_value'] = f'monthly mean residual range = {drift:.2f} cSt (threshold 3.0)'
            detected.append(pat)

    # high_yield_bias
    if len(visc_yield_vals) >= 4:
        visc_yield_arr = np.array(visc_yield_vals)
        valid_mask = ~np.isnan(visc_yield_arr)
        if valid_mask.sum() >= 4:
            q75 = float(np.percentile(visc_yield_arr[valid_mask], 75))
            q25 = float(np.percentile(visc_yield_arr[valid_mask], 25))
            high_mask = valid_mask & (visc_yield_arr >= q75)
            low_mask  = valid_mask & (visc_yield_arr <= q25)
            if high_mask.sum() > 0 and low_mask.sum() > 0:
                high_mean = float(visc_resid_arr[high_mask].mean())
                low_mean  = float(visc_resid_arr[low_mask].mean())
                if abs(high_mean) > 3.0 and abs(low_mean) > 3.0 and high_mean * low_mean < 0:
                    pat = dict(RESIDUAL_PATTERNS['high_yield_bias'])
                    pat['detected_value'] = (
                        f'high_yield_mean={high_mean:.2f} cSt, '
                        f'low_yield_mean={low_mean:.2f} cSt (opposite signs)'
                    )
                    detected.append(pat)

    # poor_yield_r2
    if not math.isnan(result['yield_r2']) and result['yield_r2'] < 0.55:
        pat = dict(RESIDUAL_PATTERNS['poor_yield_r2'])
        pat['detected_value'] = f'yield R² = {result["yield_r2"]:.3f} (threshold 0.55)'
        detected.append(pat)

    result['patterns_detected'] = detected
    result['no_structural_issues'] = len(detected) == 0

    logger.info(
        f'[DIAG S6] Residual patterns: {len(detected)} detected, '
        f'visc_r2={result["visc_r2"]:.3f}, yield_r2={result["yield_r2"]:.3f}'
    )
    return result


# ---------------------------------------------------------------------------
# STAGE 7 — Physical sanity and profile delta
# ---------------------------------------------------------------------------

def _check_physical_sanity(
    calibrated_params: dict,
    thermal_params: dict,
    feed_cache: dict,
    profile_dir: str = 'calibration_profiles',
) -> dict:
    """
    Run design-point regression tests and parameter reasonableness checks.
    Compare against previous deployed profile.

    Blocking: any design-point test fails, or K_mult > 25% change vs previous.
    """
    from simulator_bridge import simulate_single_train, _build_feed_components_from_lab
    from simulator_bridge import _DEFAULT_DENSITY, _DEFAULT_CCR, _DEFAULT_VISC135

    result: dict = {
        'pass':                  True,
        'blocking':              False,
        'design_point_results':  [],
        'param_reasonableness':  {},
        'profile_delta':         {},
        'previous_profile_name': None,
        'issues':                [],
        'warnings':              [],
        'fix':                   '',
    }

    # ---- Build default design feed ----
    design_density = _DEFAULT_DENSITY
    design_ccr     = _DEFAULT_CCR
    design_visc135 = _DEFAULT_VISC135
    cache_key = (round(design_density, 3), round(design_ccr, 2), round(design_visc135, 0))
    if cache_key in feed_cache:
        design_comps = feed_cache[cache_key]
    else:
        design_comps = _build_feed_components_from_lab(
            design_density, design_ccr, design_visc135
        )

    # ---- A. Design-point tests ----
    for test in DESIGN_POINT_TESTS:
        test_name = test['name']
        test_pass = False
        actual_val = float('nan')
        expected_range = test.get('check', '')

        try:
            if test_name in ('lube_dao_yield', 'fcc_dao_yield'):
                params_test = dict(calibrated_params)
                params_test.update(test['params'])
                cond = test['conditions']
                sim = simulate_single_train(
                    design_comps,
                    so_ratio=cond['so_ratio'],
                    t_profile=cond['t_profile'],
                    predilution_frac=cond['predilution_frac'],
                    calibration_params=params_test,
                )
                actual_val = sim.get('dao_yield_mass_frac', 0.0) * 100.0
                lo, hi = (15.0, 21.0) if test_name == 'lube_dao_yield' else (28.0, 37.0)
                test_pass = lo <= actual_val <= hi

            elif test_name == 'yield_increases_with_k_mult':
                yields_sweep = []
                for k_val in test['params_sweep']:
                    p = dict(calibrated_params); p['K_multiplier'] = k_val
                    sim = simulate_single_train(
                        design_comps, so_ratio=8.0,
                        t_profile=[65.0, 70.0, 75.0, 80.0],
                        predilution_frac=0.20,
                        calibration_params=p,
                    )
                    yields_sweep.append(sim.get('dao_yield_mass_frac', 0.0) * 100.0)
                test_pass = all(
                    yields_sweep[i] < yields_sweep[i + 1]
                    for i in range(len(yields_sweep) - 1)
                )
                actual_val = yields_sweep[-1] - yields_sweep[0]  # total swing

            elif test_name == 'yield_decreases_with_temperature':
                yields_sweep = []
                for t_top in test['conditions_sweep']:
                    p = dict(calibrated_params)
                    t_profile = [t_top - 20.0, t_top - 15.0, t_top - 10.0, t_top]
                    sim = simulate_single_train(
                        design_comps, so_ratio=8.0,
                        t_profile=t_profile,
                        predilution_frac=0.20,
                        calibration_params=p,
                    )
                    yields_sweep.append(sim.get('dao_yield_mass_frac', 0.0) * 100.0)
                test_pass = all(
                    yields_sweep[i] > yields_sweep[i + 1]
                    for i in range(len(yields_sweep) - 1)
                )
                actual_val = yields_sweep[0] - yields_sweep[-1]  # decrease swing

        except Exception as exc:
            logger.warning(f'[DIAG S7] Design-point test "{test_name}" failed: {exc}')

        result['design_point_results'].append({
            'name':           test_name,
            'pass':           test_pass,
            'actual':         round(float(actual_val), 2) if not math.isnan(actual_val) else None,
            'expected_range': expected_range,
            'target':         test.get('target', ''),
        })

        if not test_pass:
            result['blocking'] = True
            result['pass']     = False
            result['issues'].append(
                f'Design-point test "{test_name}" FAILED: '
                f'actual={actual_val:.2f}, expected: {expected_range}'
            )
            result['fix'] = (
                f'Design-point test "{test_name}" failed. '
                f'Physics model is not producing expected yield at reference conditions. '
                f'Check K_ref anchor value (should produce ~15.9 vol% at S/O=9, T_top=82°C). '
                f'Run _find_k_ref.py to verify K_ref is correct for current lle_solver.py.'
            )

    # ---- B. Parameter reasonableness ----
    bounds = {
        'K_multiplier': (0.40, 2.50, 'K_mult outside [0.40, 2.50]'),
        'E_murphree':   (0.35, 0.95, 'E_murphree outside [0.35, 0.95]'),
        'C_entrain':    (0.002, 0.08, 'C_entrain outside [0.002, 0.08]'),
        'delta_crit':   (1.0, 6.0,   'delta_crit outside [1.0, 6.0]'),
    }
    for param, (lo, hi, msg) in bounds.items():
        val = calibrated_params.get(param)
        if val is None:
            continue
        val = float(val)
        in_range = lo <= val <= hi
        result['param_reasonableness'][param] = {
            'value':    round(val, 4),
            'in_range': in_range,
            'range':    f'[{lo}, {hi}]',
        }
        if not in_range:
            result['blocking'] = True
            result['pass']     = False
            result['issues'].append(f'{param} = {val:.4f}: {msg} (BLOCKING)')
            result['fix'] = (
                f'{param} = {val:.4f} is outside physical reasonableness bounds {bounds[param][:2]}. '
                f'Model is compensating for wrong physics. '
                f'Re-check feed characterization and thermal calibration before re-running.'
            )

    # ---- C. Profile delta vs previous ----
    try:
        pattern    = os.path.join(profile_dir, 'plant_calibration_*.json')
        all_files  = sorted(_glob.glob(pattern))
        if not all_files:
            # Try default
            default_path = os.path.join(profile_dir, 'plant_calibration_v1.json')
            if os.path.exists(default_path):
                all_files = [default_path]

        if all_files:
            prev_path = all_files[-1]
            with open(prev_path) as f:
                prev_profile = json.load(f)
            prev_params = prev_profile.get('calibration_params', {})
            result['previous_profile_name'] = os.path.splitext(
                os.path.basename(prev_path)
            )[0]

            thresholds = {
                'K_multiplier': 15.0,
                'E_murphree':   20.0,
                'C_entrain':    30.0,
                'delta_crit':   20.0,
            }
            for param, flag_pct in thresholds.items():
                old_val = prev_params.get(param)
                new_val = calibrated_params.get(param)
                if old_val is None or new_val is None:
                    continue
                old_val = float(old_val); new_val = float(new_val)
                pct_chg = abs(new_val - old_val) / max(abs(old_val), 1e-10) * 100.0
                flagged = pct_chg > flag_pct
                result['profile_delta'][param] = {
                    'old':        round(old_val, 4),
                    'new':        round(new_val, 4),
                    'pct_change': round(pct_chg, 1),
                    'flagged':    flagged,
                }
                if param == 'K_multiplier' and pct_chg > 25.0:
                    result['blocking'] = True
                    result['pass']     = False
                    result['issues'].append(
                        f'K_multiplier changed {pct_chg:.1f}% vs previous profile '
                        f'({old_val:.3f} → {new_val:.3f}) — likely a data problem (BLOCKING).'
                    )
                    result['fix'] = (
                        f'K_multiplier shifted {pct_chg:.1f}% (>{25}% threshold). '
                        f'This likely indicates a data quality change (new steady-state period, '
                        f'different operating mode, or corrupt LIMS data). '
                        f'Inspect the calibration dataset before deploying this profile.'
                    )
                elif flagged:
                    result['warnings'].append(
                        f'{param} changed {pct_chg:.1f}% vs previous '
                        f'({old_val:.4f} → {new_val:.4f}) — above {flag_pct:.0f}% threshold.'
                    )
    except Exception as exc:
        logger.debug(f'[DIAG S7] Profile delta check skipped: {exc}')

    logger.info(
        f'[DIAG S7] Physical sanity: design_tests_pass='
        f'{all(r["pass"] for r in result["design_point_results"])}, '
        f'blocking={result["blocking"]}'
    )
    return result


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def run_diagnostic_pipeline(
    dcs: pd.DataFrame,
    visc_anchored: pd.DataFrame,
    thermal_profile: dict,
    default_params: dict,
    feed_cache: dict,
) -> dict:
    """
    Run Stages 0-4 only (pre-calibration checks, no optimizer).
    Stops at first blocking failure and returns immediately.
    Called by /api/calibration_dataset_info and by run_smart_calibration().

    Returns:
        {
          'pipeline_pass':  bool,
          'blocking_stage': int | None,
          'stages':         {0: ..., 1: ..., 2: ..., 3: ..., 4: ...},
          'total_time_sec': float,
          'summary':        str,
        }
    """
    t0 = time.time()
    stages: dict = {}
    blocking_stage: int | None = None

    stage_funcs = [
        (0, 'Sensor health',          lambda: _check_sensor_health(dcs)),
        (1, 'Data quality',           lambda: _check_data_quality(dcs, visc_anchored)),
        (2, 'LIMS alignment',         lambda: _check_lims_alignment(visc_anchored)),
        (3, 'Thermal model health',   lambda: _check_thermal_health(dcs, thermal_profile)),
        (4, 'Physics readiness',      lambda: _check_physics_readiness(
                                          visc_anchored, dcs, thermal_profile,
                                          default_params, feed_cache)),
    ]

    for stage_num, stage_name, stage_fn in stage_funcs:
        t_stage = time.time()
        logger.info(f'[DIAG] Running Stage {stage_num}: {stage_name}...')
        print(f'[DIAG] Stage {stage_num}: {stage_name}...', flush=True)
        try:
            stage_result = stage_fn()
        except Exception as exc:
            logger.error(f'[DIAG] Stage {stage_num} raised exception: {exc}')
            stage_result = {
                'pass': False, 'blocking': True,
                'issues': [f'Stage {stage_num} internal error: {exc}'],
                'warnings': [], 'fix': str(exc),
            }
        elapsed_s = time.time() - t_stage
        stage_result['elapsed_sec'] = round(elapsed_s, 2)
        stages[stage_num] = stage_result

        status = 'BLOCKED' if stage_result.get('blocking') else (
            'PASS' if stage_result.get('pass') else 'WARN'
        )
        logger.info(f'[DIAG] Stage {stage_num} {status} in {elapsed_s:.1f}s')
        print(f'[DIAG] Stage {stage_num} → {status} ({elapsed_s:.1f}s)', flush=True)

        if stage_result.get('blocking'):
            blocking_stage = stage_num
            break

    pipeline_pass = blocking_stage is None

    if pipeline_pass:
        summary = (
            f'All {len(stages)} pre-calibration stages passed. '
            f'Ready for full calibration.'
        )
    else:
        fix_text = stages[blocking_stage].get('fix', 'See stage report.')
        summary = (
            f'Stage {blocking_stage} BLOCKED. '
            f'Fix: {fix_text[:120]}{"..." if len(fix_text) > 120 else ""}'
        )

    return {
        'pipeline_pass':  pipeline_pass,
        'blocking_stage': blocking_stage,
        'stages':         stages,
        'total_time_sec': round(time.time() - t0, 2),
        'summary':        summary,
    }


def run_smart_calibration(
    dcs_filepath:     str,
    lims_filepath:    str,
    weights:          dict | None = None,
    profile_name:     str = 'plant_calibration_v1',
    max_retry_cycles: int = 2,
    use_pinn:         bool = False,
) -> dict:
    """
    Full seven-stage calibration cycle with two-speed optimizer.

    Steps:
    1.  Load data
    2.  Run Stages 0-4 (run_diagnostic_pipeline)
    3.  Chronological split
    4.  Pre-calibration metrics
    5.  Warm-start from last profile
    6.  Stage 5: two-speed optimizer
    7.  Stage 6: residual pattern analysis
    8.  Stage 7: physical sanity
    9.  Quality gate (visc R² > 0.70 AND yield R² > 0.65)
    9b. (use_pinn=True) Phase 3-4: regime detection + PINN training
    10. Save profile if gate passes

    Parameters
    ----------
    use_pinn : bool
        If True, run PINN phases 3-4 after Stage 7. Adds ~5-6 min.
        PINN is discarded if quality gate margin is not met (OLS kept).

    Returns structured result with status, metrics, patterns, fix_instructions.
    """
    from plant_data_loader import build_calibration_dataset
    from thermal_calibration import calibrate_thermal_model, save_thermal_profile, load_thermal_profile
    from calibration_engine import (
        compute_metrics, _make_train_test_split, _prepopulate_feed_cache,
        _DEFAULT_PARAMS,
    )

    t0 = time.time()
    if weights is None:
        weights = {'yield': 0.4, 'visc': 1.0}

    result: dict = {
        'status':             'blocked_pre_calibration',
        'blocking_stage':     None,
        'calibrated_params':  None,
        'metrics_before':     None,
        'metrics_after':      None,
        'thermal_metrics':    {},
        'dataset_info':       {},
        'diagnostic_report':  {},
        'residual_patterns':  [],
        'physical_sanity':    {},
        'profile_name':       None,
        'fix_instructions':   [],
        'total_time_sec':     0.0,
        'optimizer_nfev':     0,
    }

    # ------------------------------------------------------------------ 1. Load data
    print('[SMART_CAL] Loading calibration dataset...', flush=True)
    dataset = build_calibration_dataset(dcs_filepath, lims_filepath)
    dcs_hourly    = dataset['dcs_hourly']
    visc_anchored = dataset['visc_anchored']
    dataset_info  = dataset['dataset_info']
    result['dataset_info'] = dataset_info

    # ------------------------------------------------------------------ Thermal profile
    thermal_path = os.path.join('calibration_profiles', 'thermal_profile.json')
    try:
        thermal_profile = load_thermal_profile(thermal_path)
        print('[SMART_CAL] Loaded existing thermal profile.', flush=True)
    except FileNotFoundError:
        print('[SMART_CAL] No thermal profile found — calibrating now.', flush=True)
        thermal_profile = calibrate_thermal_model(dcs_hourly)
        os.makedirs('calibration_profiles', exist_ok=True)
        save_thermal_profile(thermal_profile, thermal_path)

    result['thermal_metrics'] = {
        tr: {k: thermal_profile.get(tr, {}).get(k)
             for k in ['mae_t_bottom', 'mae_t_middle', 'mae_t_steam_coil', 'mae_t_top']}
        for tr in ['train_a', 'train_b']
    }

    # ------------------------------------------------------------------ Feed cache
    feed_cache: dict = {}
    _prepopulate_feed_cache(
        pd.concat([visc_anchored,
                   dcs_hourly[dcs_hourly.get('steady_state',
                              pd.Series(True, index=dcs_hourly.index)).fillna(False)].head(3000)])
        if len(visc_anchored) > 0 else dcs_hourly.head(1000),
        feed_cache,
    )

    # ------------------------------------------------------------------ 2. Pre-cal diagnostic (Stages 0-4)
    print('[SMART_CAL] Running pre-calibration diagnostic (Stages 0-4)...', flush=True)
    default_params = dict(_DEFAULT_PARAMS)
    diag = run_diagnostic_pipeline(
        dcs_hourly, visc_anchored, thermal_profile, default_params, feed_cache
    )
    result['diagnostic_report'] = diag

    if not diag['pipeline_pass']:
        result['blocking_stage'] = diag['blocking_stage']
        blocking_fix = diag['stages'].get(
            diag['blocking_stage'], {}
        ).get('fix', 'See diagnostic report.')
        result['fix_instructions'] = [blocking_fix]
        result['total_time_sec'] = round(time.time() - t0, 2)
        print(
            f'[SMART_CAL] BLOCKED at Stage {diag["blocking_stage"]}. '
            f'Fix: {blocking_fix[:100]}',
            flush=True,
        )
        return result

    # ------------------------------------------------------------------ 3. Chronological split
    visc_train, visc_test = _make_train_test_split(visc_anchored, 0.20)
    dcs_train,  dcs_test  = _make_train_test_split(dcs_hourly,    0.20)
    print(
        f'[SMART_CAL] Split: visc train={len(visc_train)}/test={len(visc_test)}, '
        f'dcs train={len(dcs_train)}/test={len(dcs_test)}',
        flush=True,
    )

    # ------------------------------------------------------------------ 4. Pre-cal metrics
    metrics_before = compute_metrics(
        visc_test, dcs_test, default_params, feed_cache, thermal_profile, label='before'
    )
    result['metrics_before'] = metrics_before

    # ------------------------------------------------------------------ 5. Warm-start
    warm_params, warm_source = _get_warm_start_params(default_params)
    result['diagnostic_report']['warm_start_source'] = warm_source

    # ------------------------------------------------------------------ 6. Stage 5: Two-speed optimizer
    print('[SMART_CAL] Stage 5: Two-speed optimizer...', flush=True)
    t_opt = time.time()
    opt_result = _run_two_speed_optimizer(
        visc_train, dcs_train, warm_params, thermal_profile, feed_cache, weights
    )
    result['optimizer_nfev'] = opt_result.get('nfev_fast', 0) + opt_result.get('nfev_full', 0)
    result['diagnostic_report']['stage5_optimizer'] = opt_result

    if opt_result.get('blocking'):
        result['status']          = 'optimizer_failed'
        result['fix_instructions'] = [opt_result.get('fix', '')]
        result['total_time_sec']   = round(time.time() - t0, 2)
        return result

    calibrated_params = opt_result['calibrated_params']
    result['calibrated_params'] = calibrated_params
    print(
        f'[SMART_CAL] Optimizer done in {time.time() - t_opt:.0f}s: '
        f'K_mult={calibrated_params["K_multiplier"]:.4f}',
        flush=True,
    )

    # ------------------------------------------------------------------ Post-cal metrics
    metrics_after = compute_metrics(
        visc_test, dcs_test, calibrated_params, feed_cache, thermal_profile, label='after'
    )
    result['metrics_after'] = metrics_after

    # ------------------------------------------------------------------ 7. Stage 6: Residual patterns
    print('[SMART_CAL] Stage 6: Residual pattern analysis...', flush=True)
    patterns_result = _analyze_residual_patterns(
        visc_test, dcs_test, calibrated_params, thermal_profile, feed_cache
    )
    result['residual_patterns'] = patterns_result.get('patterns_detected', [])
    result['diagnostic_report']['stage6_patterns'] = patterns_result

    # ------------------------------------------------------------------ 8. Stage 7: Physical sanity
    print('[SMART_CAL] Stage 7: Physical sanity checks...', flush=True)
    sanity_result = _check_physical_sanity(
        calibrated_params, thermal_profile, feed_cache
    )
    result['physical_sanity'] = sanity_result

    if sanity_result.get('blocking'):
        result['status']          = 'physical_sanity_failed'
        result['fix_instructions'] = sanity_result.get('issues', [])
        result['total_time_sec']   = round(time.time() - t0, 2)
        return result

    # ------------------------------------------------------------------ 9. Quality gate
    visc_r2  = metrics_after.get('visc', {}).get('r2', float('nan'))
    yield_r2 = metrics_after.get('yield', {}).get('r2', float('nan'))

    gate_visc  = (not math.isnan(visc_r2))  and visc_r2  > 0.70
    gate_yield = (not math.isnan(yield_r2)) and yield_r2 > 0.65
    design_pass = all(r['pass'] for r in sanity_result.get('design_point_results', []))

    if not (gate_visc and gate_yield and design_pass):
        result['status'] = 'quality_gate_failed'
        result['fix_instructions'] = []
        if not gate_visc:
            result['fix_instructions'].append(
                f'visc R² = {visc_r2:.3f} < 0.70 threshold. '
                f'See residual patterns for structural causes. '
                f'Most likely: narrow dao_visc_100 operating range relative to LIMS noise.'
            )
        if not gate_yield:
            result['fix_instructions'].append(
                f'yield R² = {yield_r2:.3f} < 0.65 threshold. '
                f'Check feed_density imputation quality and DAO flow meter calibration.'
            )
        for pat in result['residual_patterns']:
            result['fix_instructions'].append(pat.get('fix', ''))
        result['total_time_sec'] = round(time.time() - t0, 2)
        print(
            f'[SMART_CAL] Quality gate FAILED: visc_r2={visc_r2:.3f}, '
            f'yield_r2={yield_r2:.3f}, design_pass={design_pass}',
            flush=True,
        )
        return result

    # ---------------------------------------------------------------- 9b. PINN phases
    pinn_result = None
    correction_mode = 'ols'
    if use_pinn:
        print('[SMART_CAL] Step 9b: Running PINN phases...', flush=True)
        try:
            from pinn_calibration_engine import run_pinn_phases
            pinn_result = run_pinn_phases(
                dcs_hourly=dcs_hourly,
                visc_anchored=visc_anchored,
                dcs_train=dcs_train,
                visc_train=visc_train,
                dcs_test=dcs_test,
                visc_test=visc_test,
                calibrated_params=calibrated_params,
                thermal_params=thermal_profile,
                feed_cache=feed_cache,
                ols_metrics_after=metrics_after,
                profile_name=profile_name,
            )
            correction_mode = pinn_result.get('correction_mode', 'ols')
            if correction_mode == 'pinn':
                pinn_m = pinn_result.get('pinn_metrics', {})
                if pinn_m.get('visc') and pinn_m.get('yield'):
                    metrics_after = pinn_m
            result['pinn_result'] = pinn_result
            result['correction_mode'] = correction_mode
            print(
                f"[SMART_CAL] PINN: status={pinn_result.get('status')} "
                f"correction={correction_mode} "
                f"clusters={pinn_result.get('n_clusters', 0)}",
                flush=True,
            )
        except Exception as exc:
            logger.warning('[SMART_CAL] PINN phases failed: %s', exc)
            print(f'[SMART_CAL] PINN phases failed ({exc}) — OLS kept', flush=True)
            result['correction_mode'] = 'ols'
    else:
        result['correction_mode'] = 'ols'

    # ------------------------------------------------------------------ 10. Save profile
    import json, math as _math
    from datetime import datetime as _dt

    def _clean(d):
        if isinstance(d, dict):
            return {k: _clean(v) for k, v in d.items()}
        if isinstance(d, list):
            return [_clean(v) for v in d]
        if isinstance(d, float) and (_math.isnan(d) or _math.isinf(d)):
            return None
        if hasattr(d, 'item'):           # numpy scalar
            return float(d)
        return d

    os.makedirs('calibration_profiles', exist_ok=True)
    profile_path = os.path.join('calibration_profiles', f'{profile_name}.json')

    # Split info
    split_date = str(visc_test.index.min().date()) if len(visc_test) > 0 else 'N/A'

    profile_json = {
        'profile_name':      profile_name,
        'created_at':        _dt.now().isoformat(timespec='seconds'),
        'profile_type':      'smart_calibration',
        'calibration_params': calibrated_params,
        'thermal_params': {
            tr: {k: thermal_profile.get(tr, {}).get(k)
                 for k in ['alpha', 'beta', 'gamma', 'phi']}
            for tr in ['train_a', 'train_b']
        },
        'metrics_before': {
            'visc':  {k: metrics_before['visc'].get(k)
                      for k in ['mae', 'rmse', 'r2', 'bias']},
            'yield': {k: metrics_before['yield'].get(k)
                      for k in ['mae', 'rmse', 'r2', 'bias']},
        },
        'metrics_after': {
            'visc':  {k: metrics_after['visc'].get(k)
                      for k in ['mae', 'rmse', 'r2', 'bias']},
            'yield': {k: metrics_after['yield'].get(k)
                      for k in ['mae', 'rmse', 'r2', 'bias']},
        },
        'thermal_metrics': result['thermal_metrics'],
        'dataset_info':    dataset_info,
        'split_info': {
            'train_rows_visc': len(visc_train),
            'test_rows_visc':  len(visc_test),
            'train_rows_dcs':  len(dcs_train),
            'test_rows_dcs':   len(dcs_test),
            'split_date':      split_date,
        },
        'weights_used':    weights,
        'optimizer_nfev':  result['optimizer_nfev'],
        'quality_gate': {
            'visc_r2_threshold':  0.70,
            'yield_r2_threshold': 0.65,
            'visc_r2_actual':     visc_r2,
            'yield_r2_actual':    yield_r2,
            'design_tests_pass':  design_pass,
        },
        'residual_patterns': [
            {'name': p.get('name', p.get('description', '')),
             'severity': p.get('severity', 'INFO'),
             'fix': p.get('fix', '')}
            for p in result['residual_patterns']
        ],
        'correction_mode': correction_mode,
        'pinn': {
            'status':              pinn_result.get('status') if pinn_result else None,
            'n_clusters':          pinn_result.get('n_clusters', 0) if pinn_result else 0,
            'pinn_checkpoint_dir': pinn_result.get('pinn_checkpoint_dir') if pinn_result else None,
            'quality_gate':        pinn_result.get('quality_gate') if pinn_result else None,
            'training_result':     pinn_result.get('training_result') if pinn_result else None,
            'regime_summary':      pinn_result.get('regime_summary') if pinn_result else None,
        } if pinn_result else None,
    }

    with open(profile_path, 'w') as f:
        json.dump(_clean(profile_json), f, indent=2)

    result['status']        = 'deployed'
    result['profile_name']  = profile_name
    result['total_time_sec'] = round(time.time() - t0, 2)
    result['fix_instructions'] = [
        p.get('fix', '') for p in result['residual_patterns']
        if p.get('severity') in ('WARNING',)
    ]

    print(
        f'[SMART_CAL] DEPLOYED: {profile_path} | '
        f'visc_r2={visc_r2:.3f}, yield_r2={yield_r2:.3f} | '
        f'{result["total_time_sec"]:.0f}s total',
        flush=True,
    )
    logger.info(
        f'[SMART_CAL] Profile deployed: {profile_path} | '
        f'visc_r2={visc_r2:.3f}, yield_r2={yield_r2:.3f}'
    )
    return result
