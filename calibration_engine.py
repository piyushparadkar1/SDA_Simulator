"""
calibration_engine.py
=====================
3-Phase calibration engine for HPCL PDA Unit (DAO-only mode).

Phase 0: Thermal model calibration (thermal_calibration.py) — inner loop.
Phase 1: Viscosity bias calibration via OLS — closed-form, no optimizer.
         Fits visc_slope and visc_bias in one forward pass through visc_anchored
         training rows. Decouples viscosity from the phase-split optimizer.
Phase 2: K_multiplier optimizer — single 1-D TRF minimisation on DAO yield.
         All other extraction physics params are fixed at design values.
Phase 2b removed (STEP13): replaced by Path B C_T calibration (run_pathb_calibration).

IMPORTANT: Chronological (not random) train/test split is always used.
           Time-series plant data must not be randomly shuffled — doing so
           would create data leakage from future to past and give
           artificially optimistic metrics.

Performance notes:
  - Phase 1 (visc OLS): ONE forward pass — ~30s for 200 rows. No nfev loop.
  - Phase 2: single Parallel() call per nfev (yield only). Was two calls
    (visc + yield) — eliminates duplicate worker-pool startup overhead.
  - _MAX_VISC_ROWS: caps visc OLS forward pass at 200 rows (stratified).
  - _MAX_YIELD_ROWS: caps Phase 2 yield block at 500 rows per nfev.
  - _MAX_METRIC_YIELD_ROWS: caps compute_metrics() yield rows at 1,000.
  - joblib loky backend: true multiprocessing (GIL bypassed).
    With 12 cores, loky gives ~10-12x wallclock speedup per nfev.
  - _process_local_cache: module-level per-worker dict. PseudoComponent
    objects live in the worker's address space — never pickled across
    process boundaries. Workers self-populate on first call.
  - Feed cache pre-populated in main process for warm start.
"""

import os
import json
import logging
import math
import time
import numpy as np
import pandas as pd
from datetime import datetime
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)

# Default extraction physics params — engineering / design values.
# visc_slope and visc_bias are solved analytically in Phase 1 OLS and are
# listed here as placeholders only — they are always overwritten before use.
# alpha_density is FIXED at 3.0 inside simulator_bridge.py (STEP13).
_DEFAULT_PARAMS = {
    'K_multiplier':  1.0,
    'E_murphree':    0.70,
    'C_entrain':     0.015,
    'delta_crit':    2.5,
    'visc_slope':    1.0,
    'visc_bias':     0.0,
}

# Phase 2 — optimise K_multiplier only.
# Lower bound 0.85: avoids the K-value phase-collapse dead zone that occurs when
# T_bot_a >= 68°C (verified empirically 2026-04-01 with real plant data).
# Upper bound 2.00: allows model to reach high-yield months (Oct 2025: 23 vol%)
# without hitting ceiling.  K_mult=1.0 → ~20% at mean conditions; K_mult=1.8 → ~32%.
_PHASE2_PARAM_NAMES = ['K_multiplier']
_PHASE2_BOUNDS_LO   = [0.85]
_PHASE2_BOUNDS_HI   = [2.00]

# Phase 2b (K_mult + alpha_density) removed — replaced by Path B C_T approach.
# See _PATHB_* constants below and run_pathb_calibration().

# Path B: C_T optimizer — direct temperature term in lnK.
# c_t_sat and c_t_aro are free; c_t_res and c_t_asp fixed at 0.0.
# Physical range: [-0.06, -0.005] per °C.
# Tie: c_t_aro = 0.7 * c_t_sat (one degree of freedom, reduces to 2-param problem).
_PATHB_PARAM_NAMES  = ['K_multiplier', 'c_t_sat']
_PATHB_BOUNDS_LO    = [0.85, -0.060]
_PATHB_BOUNDS_HI    = [2.00, -0.005]
_PATHB_CT_ARO_RATIO = 0.7   # c_t_aro = ratio * c_t_sat

# Fixed extraction physics params — not calibrated by optimizer.
# visc_slope / visc_bias are overwritten after Phase 1 OLS.
# alpha_density is hardcoded to 3.0 in simulator_bridge.py (STEP13).
_FIXED_PHYSICS_PARAMS = {
    'E_murphree':    0.70,
    'C_entrain':     0.015,
    'delta_crit':    2.5,
    'visc_slope':    1.0,   # placeholder — overwritten by Phase 1 OLS
    'visc_bias':     0.0,   # placeholder — overwritten by Phase 1 OLS
}

# Maximum visc rows per optimizer evaluation.
# Plant dataset: ~8,783 visc-anchored rows → cap at 200 = 44× row speedup.
# Stratified by DAO viscosity decile to retain full operating range.
_MAX_VISC_ROWS = 200

# Maximum yield rows per optimizer evaluation.
# Reduces from ~10,000 rows to 500 (20× speedup) while retaining S/O coverage.
_MAX_YIELD_ROWS = 500

# Maximum yield rows passed to compute_metrics(). Enough for stable statistics
# without waiting through all test rows at metrics time.
_MAX_METRIC_YIELD_ROWS = 1000

# Number of parallel workers. -1 = os.cpu_count() (all physical+logical cores).
# joblib loky backend: true multiprocessing, GIL bypassed.
_N_JOBS = -1


def _n_workers() -> int:
    """Return effective number of parallel workers (processes, not threads)."""
    n = os.cpu_count() or 4
    return n if _N_JOBS == -1 else max(1, _N_JOBS)


# ---------------------------------------------------------------------------
# Process-local feed cache
# ---------------------------------------------------------------------------

# Each loky worker process gets its own empty dict on fork.
# Workers populate it on first access and reuse it on all subsequent calls
# within the same persistent worker process — no cross-process pickle needed.
_process_local_cache: dict = {}


# ---------------------------------------------------------------------------
# Parallel row-evaluation helpers (module-level for thread safety)
# ---------------------------------------------------------------------------

def _eval_one_visc(row_tuple, calibration_params, feed_cache, thermal_params, w_visc):
    """
    Evaluate one visc-anchored row. Returns log-scale residual.

    row_tuple: (index, pd.Series) from iterrows()
    """
    from simulator_bridge import simulate_parallel_trains
    idx, row = row_tuple
    try:
        sim = simulate_parallel_trains(
            row, calibration_params, feed_cache,
            thermal_params=thermal_params,
            use_dcs_temperatures=True,
        )
        if sim['converged']:
            meas = float(row.get('dao_visc_100', 0) or 0)
            pred = float(sim['dao_visc_100_cSt'])
            if meas > 0 and pred > 0:
                return w_visc * (math.log(pred) - math.log(meas))
    except Exception:
        pass
    return 10.0


def _eval_one_yield(row_tuple, calibration_params, feed_cache, thermal_params,
                    w_yield, sigma_yield):
    """
    Evaluate one yield row. Returns normalised residual.

    row_tuple: (index, pd.Series) from iterrows()
    """
    from simulator_bridge import simulate_parallel_trains
    idx, row = row_tuple
    try:
        sim = simulate_parallel_trains(
            row, calibration_params, feed_cache,
            thermal_params=thermal_params,
            use_dcs_temperatures=True,
        )
        if sim['converged']:
            meas = float(row.get('dao_yield_vol_pct', 0) or 0)
            return w_yield * (sim['dao_yield_vol_pct'] - meas) / sigma_yield
    except Exception:
        pass
    return 10.0


def _eval_one_for_metrics(row_tuple, params, feed_cache, thermal_params):
    """
    Evaluate one row for compute_metrics. Returns (converged, visc_pred, yield_pred).

    visc_pred is None if no visc measurement; yield_pred is None if no yield.
    """
    from simulator_bridge import simulate_parallel_trains
    idx, row = row_tuple
    try:
        sim = simulate_parallel_trains(
            row, params, feed_cache,
            thermal_params=thermal_params,
            use_dcs_temperatures=True,
        )
        if sim['converged']:
            visc_meas = row.get('dao_visc_100')
            try:
                visc_meas_f = float(visc_meas)
                has_visc = not math.isnan(visc_meas_f)
            except (TypeError, ValueError):
                has_visc = False

            yield_meas = row.get('dao_yield_vol_pct')
            try:
                yield_meas_f = float(yield_meas)
                has_yield = not math.isnan(yield_meas_f)
            except (TypeError, ValueError):
                has_yield = False

            so_avg = (
                float(row.get('so_ratio_a') or 8.0) +
                float(row.get('so_ratio_b') or 8.0)
            ) / 2.0

            return {
                'converged':    True,
                'visc_pred':    float(sim['dao_visc_100_cSt']) if has_visc else None,
                'visc_meas':    visc_meas_f if has_visc else None,
                'yield_pred':   float(sim['dao_yield_vol_pct']) if has_yield else None,
                'yield_meas':   yield_meas_f if has_yield else None,
                'timestamp':    str(idx)[:10],
                'so_ratio':     so_avg,
            }
    except Exception:
        pass
    return {'converged': False}


# ---------------------------------------------------------------------------
# Process-safe eval functions (use _process_local_cache, not passed feed_cache)
# ---------------------------------------------------------------------------
# These are called by joblib loky workers. Each worker process has its own
# _process_local_cache dict that starts empty and fills up on first access.
# Because the dict is module-level in the worker's address space, it persists
# across all calls dispatched to the same persistent loky worker — so the
# cache warms up quickly and subsequent nfev calls are fast.
# ---------------------------------------------------------------------------

def _eval_one_visc_proc(row_tuple, calibration_params, thermal_params, w_visc):
    """
    Process-safe viscosity residual evaluator for joblib Parallel.

    Uses module-level _process_local_cache instead of a passed feed_cache dict
    so that PseudoComponent objects never need to be pickled across processes.

    row_tuple : (index, pd.Series) from DataFrame.iterrows()
    Returns   : float residual (log-scale, or 10.0 on failure)
    """
    from simulator_bridge import simulate_parallel_trains
    idx, row = row_tuple
    try:
        sim = simulate_parallel_trains(
            row, calibration_params, _process_local_cache,
            thermal_params=thermal_params,
            use_dcs_temperatures=True,
        )
        if sim['converged']:
            meas = float(row.get('dao_visc_100', 0) or 0)
            pred = float(sim['dao_visc_100_cSt'])
            if meas > 0 and pred > 0:
                return w_visc * (math.log(pred) - math.log(meas))
    except Exception:
        pass
    return 10.0


def _eval_one_yield_proc(row_tuple, calibration_params, thermal_params,
                         w_yield, sigma_yield):
    """
    Process-safe yield residual evaluator for joblib Parallel.

    Uses module-level _process_local_cache instead of a passed feed_cache dict.

    row_tuple : (index, pd.Series) from DataFrame.iterrows()
    Returns   : float residual (normalised, or 10.0 on failure)
    """
    from simulator_bridge import simulate_parallel_trains
    idx, row = row_tuple
    try:
        sim = simulate_parallel_trains(
            row, calibration_params, _process_local_cache,
            thermal_params=thermal_params,
            use_dcs_temperatures=True,
        )
        if sim['converged']:
            meas = float(row.get('dao_yield_vol_pct', 0) or 0)
            return w_yield * (sim['dao_yield_vol_pct'] - meas) / sigma_yield
    except Exception:
        pass
    return 10.0


def _eval_one_for_metrics_proc(row_tuple, params, thermal_params):
    """
    Process-safe metrics evaluator for joblib Parallel.

    Uses module-level _process_local_cache instead of a passed feed_cache dict.

    row_tuple : (index, pd.Series) from DataFrame.iterrows()
    Returns   : dict with converged, visc_pred, visc_meas, yield_pred, yield_meas
    """
    from simulator_bridge import simulate_parallel_trains
    idx, row = row_tuple
    try:
        sim = simulate_parallel_trains(
            row, params, _process_local_cache,
            thermal_params=thermal_params,
            use_dcs_temperatures=True,
        )
        if sim['converged']:
            visc_meas = row.get('dao_visc_100')
            try:
                visc_meas_f = float(visc_meas)
                has_visc = not math.isnan(visc_meas_f)
            except (TypeError, ValueError):
                has_visc = False

            yield_meas = row.get('dao_yield_vol_pct')
            try:
                yield_meas_f = float(yield_meas)
                has_yield = not math.isnan(yield_meas_f)
            except (TypeError, ValueError):
                has_yield = False

            so_avg = (
                float(row.get('so_ratio_a') or 8.0) +
                float(row.get('so_ratio_b') or 8.0)
            ) / 2.0

            return {
                'converged':  True,
                'visc_pred':  float(sim['dao_visc_100_cSt']) if has_visc else None,
                'visc_meas':  visc_meas_f if has_visc else None,
                'yield_pred': float(sim['dao_yield_vol_pct']) if has_yield else None,
                'yield_meas': yield_meas_f if has_yield else None,
                'timestamp':  str(idx)[:10],
                'so_ratio':   so_avg,
            }
    except Exception:
        pass
    return {'converged': False}


def _get_raw_visc_pred_proc(row_tuple, physics_params_no_visc_corr, thermal_params):
    """
    Process-safe: return (predicted_visc_raw, measured_visc) for Phase 1 OLS.

    Runs simulate_parallel_trains with visc_slope=1.0, visc_bias=0.0 so that
    the returned dao_visc_100_cSt is the RAW Walther model prediction before
    any linear correction. The OLS in calibrate_visc_correction() then fits
    slope/bias against the measured LIMS values.

    physics_params_no_visc_corr : dict of K_mult, E, C, delta, alpha,
                                  visc_slope=1.0, visc_bias=0.0 already set.

    Returns (pred_cSt, meas_cSt) on convergence with valid visc, else None.
    """
    from simulator_bridge import simulate_parallel_trains
    idx, row = row_tuple
    try:
        sim = simulate_parallel_trains(
            row, physics_params_no_visc_corr, _process_local_cache,
            thermal_params=thermal_params,
            use_dcs_temperatures=True,
        )
        if sim['converged']:
            meas = float(row.get('dao_visc_100', 0) or 0)
            pred = float(sim['dao_visc_100_cSt'])
            if meas > 0 and pred > 0:
                return (pred, meas)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Path B process-safe helpers (pass c_t_params through to simulate_parallel_trains)
# ---------------------------------------------------------------------------

def _eval_one_yield_pathb_proc(row_tuple, calibration_params, thermal_params,
                                w_yield, sigma_yield, c_t_params):
    """
    Process-safe yield residual evaluator for Path B (C_T pass).

    Identical to _eval_one_yield_proc but forwards c_t_params so that
    K_value() applies the direct temperature sensitivity correction.

    row_tuple   : (index, pd.Series) from DataFrame.iterrows()
    c_t_params  : dict with 'saturates', 'aromatics', 'resins', 'asphaltenes' keys
    Returns     : float residual (normalised, or 10.0 on failure)
    """
    from simulator_bridge import simulate_parallel_trains
    idx, row = row_tuple
    try:
        sim = simulate_parallel_trains(
            row, calibration_params, _process_local_cache,
            thermal_params=thermal_params,
            use_dcs_temperatures=True,
            c_t_params=c_t_params,
        )
        if sim['converged']:
            meas = float(row.get('dao_yield_vol_pct', 0) or 0)
            return w_yield * (sim['dao_yield_vol_pct'] - meas) / sigma_yield
    except Exception:
        pass
    return 10.0


def _eval_one_for_metrics_pathb_proc(row_tuple, params, thermal_params, c_t_params):
    """
    Process-safe metrics evaluator for Path B (C_T pass).

    Identical to _eval_one_for_metrics_proc but forwards c_t_params.

    row_tuple  : (index, pd.Series) from DataFrame.iterrows()
    c_t_params : dict with SARA keys
    Returns    : dict with converged, visc_pred, visc_meas, yield_pred, yield_meas
    """
    from simulator_bridge import simulate_parallel_trains
    idx, row = row_tuple
    try:
        sim = simulate_parallel_trains(
            row, params, _process_local_cache,
            thermal_params=thermal_params,
            use_dcs_temperatures=True,
            c_t_params=c_t_params,
        )
        if sim['converged']:
            visc_meas = row.get('dao_visc_100')
            try:
                visc_meas_f = float(visc_meas)
                has_visc = not math.isnan(visc_meas_f)
            except (TypeError, ValueError):
                has_visc = False

            yield_meas = row.get('dao_yield_vol_pct')
            try:
                yield_meas_f = float(yield_meas)
                has_yield = not math.isnan(yield_meas_f)
            except (TypeError, ValueError):
                has_yield = False

            so_avg = (
                float(row.get('so_ratio_a') or 8.0) +
                float(row.get('so_ratio_b') or 8.0)
            ) / 2.0

            return {
                'converged':  True,
                'visc_pred':  float(sim['dao_visc_100_cSt']) if has_visc else None,
                'visc_meas':  visc_meas_f if has_visc else None,
                'yield_pred': float(sim['dao_yield_vol_pct']) if has_yield else None,
                'yield_meas': yield_meas_f if has_yield else None,
                'timestamp':  str(idx)[:10],
                'so_ratio':   so_avg,
            }
    except Exception:
        pass
    return {'converged': False}


# nfev counter for Path B (separate from Path A counter)
_nfev_pathb_counter = [0]
_nfev_pathb_t0      = [0.0]


def _build_pathb_residuals(
    params: np.ndarray,
    yield_train_df: pd.DataFrame,
    feed_cache: dict,
    thermal_params: dict,
    weights: dict,
    fixed_params: dict,
) -> np.ndarray:
    """
    Build yield residuals for Path B (K_multiplier + c_t_sat) optimizer.

    params[0] = K_multiplier
    params[1] = c_t_sat  (c_t_aro = _PATHB_CT_ARO_RATIO * c_t_sat; res/asp = 0.0)
    fixed_params : dict with E_murphree, C_entrain, delta_crit, visc_slope, visc_bias
                   (same as Path A Phase 2 fixed_params; c_t_sat is NOT in here)

    Builds c_t_params dict from c_t_sat, then evaluates yield residuals with
    K_value()'s C_T correction active.  Single Parallel() call per nfev.
    """
    _nfev_pathb_counter[0] += 1
    call_n = _nfev_pathb_counter[0]
    if call_n == 1:
        _nfev_pathb_t0[0] = time.time()

    K_mult  = float(params[0])
    c_t_sat = float(params[1])

    c_t_params = {
        'saturates':   c_t_sat,
        'aromatics':   _PATHB_CT_ARO_RATIO * c_t_sat,
        'resins':      0.0,
        'asphaltenes': 0.0,
    }

    # Build full calibration params: fixed_params (visc_slope/bias + E/C/delta)
    # overridden by optimizer K_multiplier.
    calibration_params_b = _params_dict(
        np.array([K_mult]), ['K_multiplier'], fixed_params
    )

    w_yield = float(weights.get('yield', 0.4))
    n_workers = _n_workers()

    yield_tuples = list(yield_train_df.iterrows())
    sigma_yield = float(yield_train_df['dao_yield_vol_pct'].std()) \
        if len(yield_train_df) > 1 else 1.0
    if sigma_yield < 0.1:
        sigma_yield = 1.0

    yield_residuals = Parallel(n_jobs=n_workers, backend='loky')(
        delayed(_eval_one_yield_pathb_proc)(
            t, calibration_params_b, thermal_params, w_yield, sigma_yield, c_t_params
        )
        for t in yield_tuples
    )

    n_yield_conv = sum(1 for r in yield_residuals if r != 10.0)
    elapsed = time.time() - _nfev_pathb_t0[0]
    cost = float(np.sum(np.square(yield_residuals)))

    msg = (
        f"PathB nfev={call_n:3d} | K={K_mult:.4f} c_t_sat={c_t_sat:.4f} | "
        f"cost={cost:.3f} | yield_conv={n_yield_conv}/{len(yield_residuals)} | "
        f"elapsed={elapsed:.0f}s"
    )
    logger.info(msg)
    print(f"[CALIB] {msg}", flush=True)

    return np.array(yield_residuals)


# ---------------------------------------------------------------------------
# Feed cache pre-population
# ---------------------------------------------------------------------------

def _prepopulate_feed_cache(df: pd.DataFrame, cache: dict) -> None:
    """
    Pre-build PseudoComponent lists for every unique feed composition in df.

    Keys the cache by (round(density,3), round(ccr,2), round(visc135,0))
    exactly as simulate_parallel_trains does — so parallel threads find every
    key already present and never write concurrently.
    """
    from simulator_bridge import _build_feed_components_from_lab
    from simulator_bridge import _DEFAULT_DENSITY, _DEFAULT_CCR, _DEFAULT_VISC135

    def _safe(val, default):
        try:
            v = float(val)
            return default if math.isnan(v) else v
        except (TypeError, ValueError):
            return default

    unique_keys = set()
    for _, row in df.iterrows():
        density = _safe(row.get('feed_density'),  _DEFAULT_DENSITY)
        ccr     = _safe(row.get('feed_ccr'),      _DEFAULT_CCR)
        visc135 = _safe(row.get('feed_visc_135'), _DEFAULT_VISC135)
        unique_keys.add((round(density, 3), round(ccr, 2), round(visc135, 0)))

    new_keys = [k for k in unique_keys if k not in cache]
    logger.info(f"  Pre-populating feed cache: {len(new_keys)} new / "
                f"{len(unique_keys)} total unique compositions")

    for key in new_keys:
        cache[key] = _build_feed_components_from_lab(key[0], key[1], key[2])

    logger.info(f"  Feed cache ready: {len(cache)} entries")


# ---------------------------------------------------------------------------
# Yield-block stratified subsampling
# ---------------------------------------------------------------------------

def _subsample_yield_df(df: pd.DataFrame, max_rows: int, random_seed: int = 42) -> pd.DataFrame:
    """
    Return at most max_rows from df, stratified by DAO yield deciles.

    Stratification by YIELD (not S/O) ensures the optimizer sees the true
    empirical distribution of the calibration target variable.

    Previous S/O stratification gave equal weight to rare high-S/O operating
    points (S/O 14-20) where the model systematically over-predicts yield.
    This pulled K_multiplier well below 1.0 on those rare rows, then collapsed
    yield predictions on the test set which has a normal S/O distribution.
    Stratifying by yield preserves the shape of what the optimizer is actually
    minimising, preventing this overfitting to rare operating conditions.
    """
    if len(df) <= max_rows:
        return df

    yield_col = 'dao_yield_vol_pct'
    if yield_col not in df.columns or df[yield_col].isna().all():
        sampled = df.sample(n=max_rows, random_state=random_seed)
        logger.info(f"  Yield subsampled (random, no yield col): {max_rows} / {len(df)} rows")
        return sampled.sort_index()

    # 10-bin stratified sample by DAO yield decile
    n_bins = 10
    df = df.copy()
    try:
        df['_yield_bin'] = pd.qcut(df[yield_col], q=n_bins,
                                   labels=False, duplicates='drop')
    except ValueError:
        df['_yield_bin'] = pd.cut(df[yield_col],
                                   bins=min(n_bins, df[yield_col].nunique()),
                                   labels=False)
    df['_yield_bin'] = df['_yield_bin'].fillna(0).astype(int)
    n_unique_bins = df['_yield_bin'].nunique()
    per_bin = max(1, max_rows // n_unique_bins)

    parts = []
    for _, grp in df.groupby('_yield_bin'):
        n = min(len(grp), per_bin)
        parts.append(grp.sample(n=n, random_state=random_seed))

    sampled = pd.concat(parts).drop(columns=['_yield_bin'])
    # top up if rounding left us short
    if len(sampled) < max_rows:
        orig_no_bin = df.drop(columns=['_yield_bin'])
        remaining = orig_no_bin.drop(index=sampled.index, errors='ignore')
        extra = remaining.sample(
            n=min(max_rows - len(sampled), len(remaining)),
            random_state=random_seed,
        )
        sampled = pd.concat([sampled, extra])

    sampled = sampled.sort_index()
    logger.info(
        f"  Yield subsampled (stratified by yield): {len(sampled)} / {len(df)} rows "
        f"({len(sampled)/len(df)*100:.1f}% coverage)"
    )
    return sampled


# ---------------------------------------------------------------------------
# Viscosity-block stratified subsampling
# ---------------------------------------------------------------------------

def _subsample_visc_df(df: pd.DataFrame, max_rows: int, random_seed: int = 42) -> pd.DataFrame:
    """
    Return at most max_rows from a visc-anchored DataFrame, stratified by
    DAO viscosity deciles.

    Stratification ensures the subsample covers the full viscosity range
    rather than over-representing the most common operating viscosity.

    df       : visc_anchored DataFrame (all rows have dao_visc_100 measured)
    max_rows : cap (default _MAX_VISC_ROWS = 200)
    Returns  : subsampled DataFrame sorted chronologically
    """
    if len(df) <= max_rows:
        return df

    visc_col = 'dao_visc_100'
    if visc_col not in df.columns or df[visc_col].isna().all():
        sampled = df.sample(n=max_rows, random_state=random_seed)
        logger.info(f"  Visc subsampled (random, no visc col): {max_rows} / {len(df)} rows")
        return sampled.sort_index()

    # 10-bin stratified sample by viscosity
    n_bins = 10
    df = df.copy()
    try:
        df['_visc_bin'] = pd.qcut(df[visc_col], q=n_bins,
                                   labels=False, duplicates='drop')
    except ValueError:
        # Fallback if too few unique values to bin
        df['_visc_bin'] = pd.cut(df[visc_col], bins=min(n_bins, df[visc_col].nunique()),
                                  labels=False)
    df['_visc_bin'] = df['_visc_bin'].fillna(0).astype(int)
    n_unique_bins = df['_visc_bin'].nunique()
    per_bin = max(1, max_rows // n_unique_bins)

    parts = []
    for _, grp in df.groupby('_visc_bin'):
        n = min(len(grp), per_bin)
        parts.append(grp.sample(n=n, random_state=random_seed))

    sampled = pd.concat(parts).drop(columns=['_visc_bin'])
    # Top up if rounding left us short of max_rows
    if len(sampled) < max_rows:
        orig_no_bin = df.drop(columns=['_visc_bin'])
        remaining = orig_no_bin.drop(index=sampled.index, errors='ignore')
        extra = remaining.sample(
            n=min(max_rows - len(sampled), len(remaining)),
            random_state=random_seed,
        )
        sampled = pd.concat([sampled, extra])

    sampled = sampled.sort_index()
    logger.info(
        f"  Visc subsampled (stratified by viscosity): {len(sampled)} / {len(df)} rows "
        f"({len(sampled) / len(df) * 100:.1f}% coverage)"
    )
    return sampled


# ---------------------------------------------------------------------------
# Phase 1: Viscosity OLS calibration
# ---------------------------------------------------------------------------

def calibrate_visc_correction(
    visc_train_df: pd.DataFrame,
    thermal_params: dict,
    initial_physics_params: dict,
) -> tuple:
    """
    Phase 1: Fit visc_slope and visc_bias via OLS against visc_anchored rows.

    Runs ONE forward pass through visc_train_df using initial_physics_params
    (with visc_slope=1.0, visc_bias=0.0 forced) to obtain raw Walther DAO
    viscosity predictions.  Then fits:

        measured_visc = visc_slope * predicted_visc + visc_bias   (OLS)

    This closed-form solution decouples the viscosity correction from the
    phase-split optimizer.  K_multiplier (Phase 2) then only needs to match
    yield — not viscosity.

    Parameters
    ----------
    visc_train_df         : visc-anchored training rows (dao_visc_100 present)
    thermal_params        : thermal calibration result dict from Phase 0
    initial_physics_params: dict with K_mult, E, C, delta, alpha (visc_slope
                            and visc_bias are overridden to 1.0 / 0.0 internally)

    Returns
    -------
    (visc_slope, visc_bias) : floats clamped to [0.3, 5.0] and [-30.0, +30.0]
    """
    n_workers = _n_workers()

    # Force visc_slope=1.0, visc_bias=0.0 so simulate_parallel_trains returns
    # the raw Walther prediction without any correction applied.
    params_for_ols = dict(initial_physics_params)
    params_for_ols['visc_slope'] = 1.0
    params_for_ols['visc_bias']  = 0.0

    _m = (f"Phase 1 visc OLS: {len(visc_train_df)} rows | {n_workers} workers | "
          f"K_mult={params_for_ols.get('K_multiplier', 1.0):.2f}")
    logger.info(_m); print(f"[CALIB] {_m}", flush=True)
    t0 = time.time()

    row_tuples = list(visc_train_df.iterrows())
    raw_results = Parallel(n_jobs=n_workers, backend='loky')(
        delayed(_get_raw_visc_pred_proc)(t, params_for_ols, thermal_params)
        for t in row_tuples
    )

    # Collect converged (pred, meas) pairs
    pairs = [r for r in raw_results if r is not None]

    if len(pairs) < 10:
        logger.warning(
            f"Phase 1 OLS: only {len(pairs)} converged rows — "
            f"defaulting to slope=2.2, bias=0.0 (plant ~31 / model ~14 cSt)"
        )
        print(f"[CALIB] Phase 1 OLS: too few converged rows ({len(pairs)}) — "
              f"defaulting slope=2.2, bias=0.0", flush=True)
        return 2.2, 0.0, 0.0

    x = np.array([p[0] for p in pairs])   # raw Walther predictions
    y = np.array([p[1] for p in pairs])   # LIMS measurements

    # Pearson r — must be checked BEFORE committing to OLS slope.
    # If the model's visc predictions have near-zero variance (DAO composition
    # doesn't change enough across the operating envelope), polyfit returns
    # a numerically unstable slope near 0 that gets clamped to 0.3.
    # A slope of 0.3 is WORSE than no correction: it collapses all predictions
    # toward the bias constant, destroying any residual model signal.
    r_pearson = float(np.corrcoef(x, y)[0, 1]) if len(x) >= 4 else 0.0
    if np.isnan(r_pearson):
        r_pearson = 0.0

    slope_raw, bias_raw = np.polyfit(x, y, 1)

    _m_diag = (
        f"  Phase 1 OLS diagnostics: r_pearson={r_pearson:.3f} | "
        f"mean_pred={np.mean(x):.2f} cSt | mean_meas={np.mean(y):.2f} cSt | "
        f"var_pred={np.var(x):.3f} | var_meas={np.var(y):.3f} | "
        f"slope_raw={slope_raw:.3f} | bias_raw={bias_raw:.2f}"
    )
    logger.info(_m_diag); print(f"[CALIB] {_m_diag}", flush=True)

    if abs(r_pearson) < 0.15:
        # Insufficient linear correlation — model visc predictions do not track
        # LIMS variation. Use bias-only: slope=1.0 preserves model shape and
        # only shifts the mean. This is always better than slope=0.3 which
        # ignores the model prediction almost entirely.
        bias_only = float(np.clip(np.mean(y) - np.mean(x), -30.0, 30.0))
        _m_fb = (
            f"Phase 1 OLS: r={r_pearson:.3f} < 0.15 → bias-only correction: "
            f"slope=1.0, bias={bias_only:.2f} cSt  "
            f"[mean shift only; OLS slope unreliable at this r]"
        )
        logger.info(_m_fb); print(f"[CALIB] {_m_fb}", flush=True)
        return 1.0, bias_only, r_pearson

    # OLS slope is meaningful — clamp to physical bounds
    slope = float(np.clip(slope_raw, 0.3, 5.0))
    bias  = float(np.clip(bias_raw, -30.0, 30.0))

    # Log OLS quality (R²)
    y_hat = slope * x + bias
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2_ols = 1.0 - ss_res / max(ss_tot, 1e-12)

    _m = (f"Phase 1 OLS done: slope={slope:.3f} bias={bias:.2f} cSt "
          f"r={r_pearson:.3f} R²={r2_ols:.3f} on {len(pairs)} rows [{time.time()-t0:.1f}s]")
    logger.info(_m); print(f"[CALIB] {_m}", flush=True)

    return slope, bias, r_pearson


# ---------------------------------------------------------------------------
# Core split / params helpers
# ---------------------------------------------------------------------------

def _make_train_test_split(
    df: pd.DataFrame,
    test_fraction: float = 0.20,
) -> tuple:
    """
    Chronological split — NOT random.

    Sort by timestamp (ascending). Use first (1-test_fraction) as train,
    last test_fraction as test. Return (train_df, test_df).

    NEVER use random_state shuffle for time-series plant data.
    Random splits would allow future data to inform past predictions,
    creating data leakage and overestimated generalisation metrics.
    """
    df_sorted = df.sort_index()
    n = len(df_sorted)
    n_train = int(n * (1.0 - test_fraction))
    train_df = df_sorted.iloc[:n_train].copy()
    test_df  = df_sorted.iloc[n_train:].copy()
    return train_df, test_df


def _params_dict(
    params_array: np.ndarray,
    param_names: list,
    fixed_params: dict | None = None,
) -> dict:
    """
    Build a full calibration_params dict from optimizer values + fixed params.

    params_array : array of optimised values (length = len(param_names))
    param_names  : ordered list of names matching params_array entries
    fixed_params : dict of params not in the optimizer (merged first;
                   optimizer values take precedence on key conflicts)

    Returns a flat dict safe to pass to simulate_parallel_trains().
    """
    result = {} if fixed_params is None else dict(fixed_params)
    for k, v in zip(param_names, params_array):
        result[k] = float(v)
    return result


# ---------------------------------------------------------------------------
# Outer loop residuals
# ---------------------------------------------------------------------------

# Module-level call counter shared across calls to _build_outer_residuals
_nfev_counter = [0]
_nfev_t0      = [0.0]


def _build_outer_residuals(
    params: np.ndarray,
    yield_train_df: pd.DataFrame,
    feed_cache: dict,
    thermal_params: dict,
    weights: dict,
    param_names: list,
    fixed_params: dict,
) -> np.ndarray:
    """
    Build yield residuals for scipy.optimize.least_squares Phase 2.

    Phase 2 only optimises yield — viscosity is pre-solved by Phase 1 OLS.
    Single Parallel() call per nfev: eliminates the dual-block worker overhead
    that added ~1s per nfev (×100 nfev = ~100s wasted).

    params       : array of Phase 2 values — [K_multiplier] only (Path A)
                   or [K_multiplier, c_t_sat] for Path B (see run_pathb_calibration)
    yield_train_df: pre-subsampled steady-state DCS training rows
    feed_cache   : main-process feed component cache (not used by loky workers,
                   which use _process_local_cache — kept for metrics compat)
    thermal_params: thermal calibration result dict
    weights      : {'yield': float, 'visc': float} — only 'yield' used here
    param_names  : ordered names matching params (e.g. ['K_multiplier'])
    fixed_params : dict of all fixed params merged with optimiser values
    """
    _nfev_counter[0] += 1
    call_n = _nfev_counter[0]
    if call_n == 1:
        _nfev_t0[0] = time.time()

    # Merge optimizer values over fixed_params to build full calibration dict
    calibration_params = _params_dict(params, param_names, fixed_params)
    w_yield = float(weights.get('yield', 0.4))
    n_workers = _n_workers()

    # --- Single yield block (loky multiprocessing) ---
    yield_tuples = list(yield_train_df.iterrows())
    sigma_yield = float(yield_train_df['dao_yield_vol_pct'].std()) \
        if len(yield_train_df) > 1 else 1.0
    if sigma_yield < 0.1:
        sigma_yield = 1.0

    yield_residuals = Parallel(n_jobs=n_workers, backend='loky')(
        delayed(_eval_one_yield_proc)(t, calibration_params, thermal_params,
                                     w_yield, sigma_yield)
        for t in yield_tuples
    )

    n_yield_conv = sum(1 for r in yield_residuals if r != 10.0)
    elapsed = time.time() - _nfev_t0[0]
    cost = float(np.sum(np.square(yield_residuals)))

    K = float(calibration_params.get('K_multiplier', params[0]))
    msg = (
        f"nfev={call_n:3d} | K={K:.4f} | "
        f"cost={cost:.3f} | yield_conv={n_yield_conv}/{len(yield_residuals)} | "
        f"elapsed={elapsed:.0f}s"
    )
    logger.info(msg)
    print(f"[CALIB] {msg}", flush=True)

    return np.array(yield_residuals)


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(
    df: pd.DataFrame,
    dcs_df: pd.DataFrame,
    params: dict,
    feed_cache: dict,
    thermal_params: dict,
    label: str = '',
    c_t_params: dict = None,
) -> dict:
    """
    Compute prediction metrics on a DataFrame (either train or test split).

    Runs simulate_parallel_trains for every valid row in parallel.
    Collects: dao_visc_100 predictions vs actuals (visc_anchored rows)
              dao_yield_vol_pct predictions vs actuals (dcs steady-state rows)
    """
    n_workers = _n_workers()

    # --- Viscosity rows ---
    if 'dao_visc_100' in df.columns:
        visc_rows = df[df['dao_visc_100'].notna()]
    else:
        visc_rows = df.iloc[0:0]   # empty

    # --- Yield rows from dcs_df ---
    if 'steady_state' in dcs_df.columns:
        y_mask = dcs_df['steady_state'].fillna(False)
    else:
        y_mask = pd.Series(True, index=dcs_df.index)
    if 'train_valid_a' in dcs_df.columns:
        y_mask &= dcs_df['train_valid_a'].fillna(False)
    if 'train_valid_b' in dcs_df.columns:
        y_mask &= dcs_df['train_valid_b'].fillna(False)
    if 'dao_yield_vol_pct' in dcs_df.columns:
        y_mask &= dcs_df['dao_yield_vol_pct'].notna()
    yield_rows_full = dcs_df[y_mask]

    # Cap yield rows — full test set can be 4,000+ rows; 1,000 gives stable
    # statistics while keeping metrics time under 2 minutes.
    if len(yield_rows_full) > _MAX_METRIC_YIELD_ROWS:
        yield_rows = yield_rows_full.head(_MAX_METRIC_YIELD_ROWS)
        logger.info(
            f"  compute_metrics [{label}]: yield rows capped at "
            f"{_MAX_METRIC_YIELD_ROWS} / {len(yield_rows_full)} (chronological head)"
        )
    else:
        yield_rows = yield_rows_full

    # Combine all rows to evaluate (visc first, yield second — order matters
    # for result collection below)
    all_rows = list(visc_rows.iterrows()) + list(yield_rows.iterrows())
    n_total = len(all_rows)
    n_visc_block = len(visc_rows)

    _cm = (f"compute_metrics [{label}]: "
           f"{len(visc_rows)} visc + {len(yield_rows)} yield = {n_total} rows | "
           f"{n_workers} workers")
    logger.info(f"  {_cm}")
    print(f"[CALIB]   {_cm}", flush=True)
    t_start = time.time()

    # joblib Parallel preserves result order — results[i] corresponds to all_rows[i]
    # Use Path B evaluator when c_t_params is provided; Path A evaluator otherwise.
    if c_t_params is not None:
        results_ordered = Parallel(n_jobs=n_workers, backend='loky', verbose=0)(
            delayed(_eval_one_for_metrics_pathb_proc)(t, params, thermal_params, c_t_params)
            for t in all_rows
        )
    else:
        results_ordered = Parallel(n_jobs=n_workers, backend='loky', verbose=0)(
            delayed(_eval_one_for_metrics_proc)(t, params, thermal_params)
            for t in all_rows
        )

    elapsed_total = time.time() - t_start
    _cm = (f"compute_metrics [{label}] done: "
           f"{n_total} rows in {elapsed_total:.1f}s ({n_total/elapsed_total:.0f} rows/s)")
    logger.info(f"  {_cm}")
    print(f"[CALIB]   {_cm}", flush=True)

    # --- Collect results ---
    visc_actual, visc_pred = [], []
    yield_actual, yield_pred = [], []
    timestamps, so_ratios = [], []
    n_converged = 0

    # First n_visc_block results are visc; rest are yield
    for i, r in enumerate(results_ordered):
        if r is None or not r.get('converged'):
            continue
        n_converged += 1
        if i < n_visc_block:
            if r['visc_pred'] is not None:
                visc_actual.append(r['visc_meas'])
                visc_pred.append(r['visc_pred'])
                timestamps.append(r['timestamp'])
                so_ratios.append(r['so_ratio'])
        else:
            if r['yield_pred'] is not None:
                yield_actual.append(r['yield_meas'])
                yield_pred.append(r['yield_pred'])

    def _stats(actual, predicted, within_thresholds=None):
        if not actual:
            return {'n': 0, 'mae': float('nan'), 'rmse': float('nan'),
                    'r2': float('nan'), 'bias': float('nan')}
        a = np.array(actual)
        p = np.array(predicted)
        mae  = float(np.mean(np.abs(p - a)))
        rmse = float(np.sqrt(np.mean((p - a) ** 2)))
        bias = float(np.mean(p - a))
        ss_res = np.sum((a - p) ** 2)
        ss_tot = np.sum((a - a.mean()) ** 2)
        r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))
        out = {'n': len(a), 'mae': mae, 'rmse': rmse, 'r2': r2, 'bias': bias}
        if within_thresholds:
            for key, thresh in within_thresholds.items():
                out[key] = float(np.mean(np.abs(p - a) <= thresh) * 100.0)
        return out

    visc_stats  = _stats(visc_actual, visc_pred,
                         {'pct_within_3cst': 3.0, 'pct_within_5cst': 5.0})
    yield_stats = _stats(yield_actual, yield_pred,
                         {'pct_within_2pct': 2.0})

    return {
        'label': label,
        'visc':  visc_stats,
        'yield': yield_stats,
        'convergence_rate': float(n_converged / max(n_total, 1)),
        'parity_data': {
            'visc_actual':     visc_actual,
            'visc_predicted':  visc_pred,
            'yield_actual':    yield_actual,
            'yield_predicted': yield_pred,
            'timestamps':      timestamps,
            'so_ratio':        so_ratios,
        },
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_full_calibration(
    dcs_filepath: str,
    lims_filepath: str,
    weights: dict | None = None,
    initial_params: dict | None = None,
    profile_name: str = 'plant_calibration_v1',
    enable_pinn: bool = False,
) -> dict:
    """
    Orchestrate the 3-phase calibration workflow (DAO-only mode).

    Phase 0 : Thermal model calibration per train (inner loop).
    Phase 1 : Viscosity OLS — closed-form, one forward pass, no optimizer loop.
              Fits visc_slope and visc_bias analytically.
    Phase 2 : K_multiplier TRF optimizer — 1 parameter, yield target only.
              max_nfev=50. Expected: < 20 nfev, < 60 s.
    Phase 3 : (enable_pinn=True) RegimeDetector — PCA+GMM on DCS telemetry.
    Phase 4 : (enable_pinn=True) PINN training — discrepancy networks on residuals.
    Path B  : Use run_pathb_calibration() for C_T temperature-sensitivity pass.

    Arguments
    ---------
    dcs_filepath   : path to extractor_parameters.xlsx
    lims_filepath  : path to lims.xlsx
    weights        : {'yield': 0.4, 'visc': 1.0}  (yield weight used in Phase 2)
    initial_params : dict with any of K_multiplier, E_murphree, C_entrain,
                     delta_crit.  visc_slope / visc_bias are ignored here —
                     solved analytically in Phase 1.  alpha_density is hardcoded.
    profile_name   : filename stem for saving the result JSON
    enable_pinn    : if True, run Phases 3-4 (regime detection + PINN training)
                     after Phase 2 completes. Adds ~5-6 min. Falls back to OLS
                     if PINN quality gate fails.
    """
    from scipy.optimize import least_squares
    from plant_data_loader import build_calibration_dataset
    from thermal_calibration import calibrate_thermal_model, save_thermal_profile

    if weights is None:
        weights = {'yield': 0.4, 'visc': 1.0}
    if initial_params is None:
        initial_params = _DEFAULT_PARAMS.copy()

    # Reset nfev counter at the start of every new calibration run
    _nfev_counter[0] = 0

    # -----------------------------------------------------------------------
    # Step 1: Load data
    # -----------------------------------------------------------------------
    print("[CALIB] " + "=" * 58, flush=True)
    print("[CALIB] Step 1: Loading calibration dataset (mode=dao_only)...", flush=True)
    logger.info("Step 1: Loading calibration dataset (mode=dao_only)...")
    t_step = time.time()
    dataset = build_calibration_dataset(dcs_filepath, lims_filepath)
    dcs_hourly    = dataset['dcs_hourly']
    visc_anchored = dataset['visc_anchored']
    dataset_info  = dataset['dataset_info']
    _m = (f"  DCS rows: {dataset_info['dcs_rows_total']}, "
          f"visc_anchored: {dataset_info['visc_anchored_rows']} "
          f"[{time.time()-t_step:.1f}s]")
    logger.info(_m); print(f"[CALIB] {_m}", flush=True)

    # -----------------------------------------------------------------------
    # Step 2: Phase 0 — Thermal calibration
    # -----------------------------------------------------------------------
    print("[CALIB] Step 2: Phase 0 — Thermal model calibration...", flush=True)
    logger.info("Step 2: Phase 0 — Calibrating thermal model...")
    t_step = time.time()
    thermal_result = calibrate_thermal_model(dcs_hourly)
    save_thermal_profile(thermal_result)
    for train in ['a', 'b']:
        t = thermal_result.get(f'train_{train}', {})
        _m = (f"  Train {train.upper()}: alpha={t.get('alpha', float('nan')):.3f}, "
              f"beta={t.get('beta', float('nan')):.3f}, "
              f"MAE_bot={t.get('mae_t_bottom', float('nan')):.2f}C, "
              f"MAE_top={t.get('mae_t_top', float('nan')):.2f}C")
        logger.info(_m); print(f"[CALIB] {_m}", flush=True)
    _m = f"  Thermal done [{time.time()-t_step:.1f}s]"
    logger.info(_m); print(f"[CALIB] {_m}", flush=True)

    # -----------------------------------------------------------------------
    # Step 3: Chronological split + subsampling
    # -----------------------------------------------------------------------
    print("[CALIB] Step 3: Train/test split (80/20 chronological)...", flush=True)
    logger.info("Step 3: Chronological train/test split (80/20)...")
    visc_train, visc_test = _make_train_test_split(visc_anchored, 0.20)
    dcs_train,  dcs_test  = _make_train_test_split(dcs_hourly, 0.20)
    split_date = str(visc_test.index.min().date()) if len(visc_test) > 0 else 'N/A'
    _m = (f"  visc: train={len(visc_train)}, test={len(visc_test)}, split={split_date}  "
          f"dcs: train={len(dcs_train)}, test={len(dcs_test)}")
    logger.info(_m); print(f"[CALIB] {_m}", flush=True)

    print("[CALIB] Step 3b: Subsampling training rows...", flush=True)
    logger.info("Step 3b: Pre-filtering + subsampling visc and yield training rows...")

    visc_train_sub = _subsample_visc_df(visc_train, _MAX_VISC_ROWS)
    _m = (f"  visc_train: {len(visc_train)} -> {len(visc_train_sub)} rows "
          f"(cap={_MAX_VISC_ROWS})" if len(visc_train) > 0 else "  visc_train: 0 rows")
    logger.info(_m); print(f"[CALIB] {_m}", flush=True)

    mask = (
        dcs_train['steady_state'].fillna(False) &
        dcs_train['train_valid_a'].fillna(False) &
        dcs_train['train_valid_b'].fillna(False) &
        dcs_train['dao_yield_vol_pct'].notna()
    )
    yield_full = dcs_train[mask]
    _m = f"  Full yield_df after steady-state filter: {len(yield_full)} rows"
    logger.info(_m); print(f"[CALIB] {_m}", flush=True)
    yield_df_sub = _subsample_yield_df(yield_full, _MAX_YIELD_ROWS)
    _m = f"  yield_train subsampled: {len(yield_df_sub)} rows"
    logger.info(_m); print(f"[CALIB] {_m}", flush=True)

    # --- Pre-populate feed cache ---
    print("[CALIB] Step 3c: Pre-populating feed cache...", flush=True)
    logger.info("Step 3c: Pre-populating feed component cache...")
    t_step = time.time()
    feed_cache: dict = {}
    all_rows_for_cache = pd.concat([
        visc_train_sub, visc_test,
        yield_df_sub,
        dcs_test[dcs_test['steady_state'].fillna(False)].head(2000),
    ])
    _prepopulate_feed_cache(all_rows_for_cache, feed_cache)
    _m = f"  Feed cache ready: {len(feed_cache)} compositions [{time.time()-t_step:.1f}s]"
    logger.info(_m); print(f"[CALIB] {_m}", flush=True)

    # -----------------------------------------------------------------------
    # Step 4: Phase 1 — Viscosity OLS (closed-form, single forward pass)
    # -----------------------------------------------------------------------
    print("[CALIB] Step 4: Phase 1 — Viscosity OLS (closed-form)...", flush=True)
    logger.info("Step 4: Phase 1 — Viscosity bias calibration (OLS)...")
    t_step = time.time()

    # Build params for OLS forward pass: use initial K_mult/E/C/delta
    # but force visc_slope=1.0, visc_bias=0.0 to get raw Walther predictions.
    # alpha_density is hardcoded in simulator_bridge.py — not included here.
    ols_physics_params = {
        'K_multiplier':  float(initial_params.get('K_multiplier',
                               _DEFAULT_PARAMS['K_multiplier'])),
        'E_murphree':    float(initial_params.get('E_murphree',
                               _DEFAULT_PARAMS['E_murphree'])),
        'C_entrain':     float(initial_params.get('C_entrain',
                               _DEFAULT_PARAMS['C_entrain'])),
        'delta_crit':    float(initial_params.get('delta_crit',
                               _DEFAULT_PARAMS['delta_crit'])),
        'visc_slope':    1.0,
        'visc_bias':     0.0,
    }
    visc_slope_ols, visc_bias_ols, visc_ols_r = calibrate_visc_correction(
        visc_train_sub, thermal_result, ols_physics_params
    )
    _m = (f"  Phase 1 result: visc_slope={visc_slope_ols:.3f}, "
          f"visc_bias={visc_bias_ols:.2f} cSt [{time.time()-t_step:.1f}s]")
    logger.info(_m); print(f"[CALIB] {_m}", flush=True)

    # Build the fixed_params dict for Phase 2.
    # All extraction physics params are fixed; only K_mult is optimised.
    # alpha_density is hardcoded at 3.0 in simulator_bridge.py — not in this dict.
    fixed_params = {
        'E_murphree':    float(_FIXED_PHYSICS_PARAMS['E_murphree']),
        'C_entrain':     float(_FIXED_PHYSICS_PARAMS['C_entrain']),
        'delta_crit':    float(_FIXED_PHYSICS_PARAMS['delta_crit']),
        'visc_slope':    visc_slope_ols,
        'visc_bias':     visc_bias_ols,
    }

    k_initial = float(initial_params.get('K_multiplier',
                                         _DEFAULT_PARAMS['K_multiplier']))

    # Build pre-calibration full params (for metrics_before)
    initial_full_params = dict(fixed_params)
    initial_full_params['K_multiplier'] = k_initial

    # -----------------------------------------------------------------------
    # Step 5: Pre-calibration metrics (uses Phase 1 visc correction already)
    # -----------------------------------------------------------------------
    print("[CALIB] Step 5: Pre-calibration metrics (test set)...", flush=True)
    logger.info("Step 5: Computing pre-calibration metrics (test set)...")
    t_step = time.time()
    metrics_before = compute_metrics(
        visc_test, dcs_test, initial_full_params, feed_cache, thermal_result,
        label='before'
    )
    _m = (f"  Before: visc MAE={metrics_before['visc'].get('mae', float('nan')):.1f} cSt  "
          f"yield MAE={metrics_before['yield'].get('mae', float('nan')):.1f} vol%  "
          f"visc R2={metrics_before['visc'].get('r2', float('nan')):.3f}  "
          f"[{time.time()-t_step:.1f}s]")
    logger.info(_m); print(f"[CALIB] {_m}", flush=True)

    # -----------------------------------------------------------------------
    # Step 6: Phase 2 — K_multiplier optimizer (yield only, 1 parameter)
    #
    # SKIPPED when enable_pinn=True (Option C mode):
    #   The K_mult optimizer converges to ~0.85 which sits at the edge of the
    #   Rachford-Rice dead zone (yield collapses to <2% for T_bot>=68C rows).
    #   Post-calibration OLS metrics (visc MAE=11.5, yield MAE=10.1) are WORSE
    #   than pre-calibration (visc MAE=3.2, yield MAE=3.8) at K_mult=1.0.
    #   In PINN mode we lock K_mult=1.0 — the PINN learns the residual
    #   correction directly from a physically valid baseline.
    # -----------------------------------------------------------------------
    if enable_pinn:
        k_calibrated      = 1.0
        optimizer_success = True   # Phase 2 skipped, not failed
        optimizer_message = 'Phase 2 skipped — K_mult locked at 1.0 (PINN mode)'
        _m = ("Step 6: Phase 2 SKIPPED (enable_pinn=True — K_mult locked at 1.0 "
              "to avoid dead-zone; PINN learns yield correction directly)")
        logger.info(_m); print(f"[CALIB] {_m}", flush=True)
        # Phase 1.5 would just re-run OLS at K_mult=1.0 = identical to Phase 1.
        # Skip it; use Phase 1 OLS params directly.
        visc_slope_phase1 = visc_slope_ols
        visc_bias_phase1  = visc_bias_ols
        _m = (f"Step 6b: Phase 1.5 SKIPPED (K_mult=1.0, Phase 1 OLS is already "
              f"optimal: slope={visc_slope_ols:.3f} bias={visc_bias_ols:.2f})")
        logger.info(_m); print(f"[CALIB] {_m}", flush=True)
    else:
        n_w = _n_workers()
        est_per_nfev = len(yield_df_sub) * 0.013 / max(n_w, 1)
        _m = (f"Step 6: Phase 2 optimizer START | mode=dao_only | "
              f"{len(yield_df_sub)} yield rows | {n_w} workers | "
              f"params={_PHASE2_PARAM_NAMES} | max_nfev=50 | "
              f"est ~{est_per_nfev:.1f}s/nfev -> ~{est_per_nfev*50/60:.1f} min")
        logger.info(_m); print(f"[CALIB] {_m}", flush=True)

        _nfev_counter[0] = 0
        t_step = time.time()
        opt_result = least_squares(
            fun=_build_outer_residuals,
            x0=[k_initial],
            bounds=(_PHASE2_BOUNDS_LO, _PHASE2_BOUNDS_HI),
            method='trf',
            args=(yield_df_sub, feed_cache, thermal_result, weights,
                  _PHASE2_PARAM_NAMES, fixed_params),
            verbose=0,
            max_nfev=50,
        )

        k_calibrated      = float(opt_result.x[0])
        optimizer_success = bool(opt_result.success)
        optimizer_message = str(opt_result.message)
        _m = (f"  Phase 2 DONE: K_mult={k_calibrated:.4f}, "
              f"success={optimizer_success}, nfev={opt_result.nfev}, "
              f"msg={optimizer_message} [{time.time()-t_step:.1f}s]")
        logger.info(_m); print(f"[CALIB] {_m}", flush=True)

        # -----------------------------------------------------------------------
        # Step 6b: Phase 1.5 — re-run OLS visc correction at calibrated K_mult
        # -----------------------------------------------------------------------
        _m = (f"Step 6b: Phase 1.5 — re-run visc OLS at K_mult={k_calibrated:.4f}...")
        logger.info(_m); print(f"[CALIB] {_m}", flush=True)
        t_step = time.time()
        ols_params_at_k2 = {
            'K_multiplier': k_calibrated,
            'E_murphree':   float(fixed_params['E_murphree']),
            'C_entrain':    float(fixed_params['C_entrain']),
            'delta_crit':   float(fixed_params['delta_crit']),
            'visc_slope':   1.0,
            'visc_bias':    0.0,
        }
        visc_slope_ols2, visc_bias_ols2, visc_ols_r2 = calibrate_visc_correction(
            visc_train_sub, thermal_result, ols_params_at_k2
        )
        _m = (f"  Phase 1.5 done: slope={visc_slope_ols2:.3f} bias={visc_bias_ols2:.2f} "
              f"r={visc_ols_r2:.3f}  (Phase 1 was slope={visc_slope_ols:.3f} "
              f"bias={visc_bias_ols:.2f} r={visc_ols_r:.3f}) [{time.time()-t_step:.1f}s]")
        logger.info(_m); print(f"[CALIB] {_m}", flush=True)
        # Save Phase 1 OLS params for PINN (in case PINN is re-enabled later).
        visc_slope_phase1 = visc_slope_ols
        visc_bias_phase1  = visc_bias_ols
        # Use Phase 1.5 values going forward for OLS output
        visc_slope_ols = visc_slope_ols2
        visc_bias_ols  = visc_bias_ols2

    # --- Final calibrated params dict ---
    calibrated_params = {
        'K_multiplier':  k_calibrated,
        'E_murphree':    float(fixed_params['E_murphree']),
        'C_entrain':     float(fixed_params['C_entrain']),
        'delta_crit':    float(fixed_params['delta_crit']),
        'visc_slope':    visc_slope_ols,
        'visc_bias':     visc_bias_ols,
    }
    _m = f"  Final calibrated params: {calibrated_params}"
    logger.info(_m); print(f"[CALIB] {_m}", flush=True)

    # -----------------------------------------------------------------------
    # Step 7: Post-calibration metrics (test set)
    # -----------------------------------------------------------------------
    print("[CALIB] Step 7: Post-calibration metrics (test set)...", flush=True)
    logger.info("Step 7: Computing post-calibration metrics (test set)...")
    t_step = time.time()
    metrics_after = compute_metrics(
        visc_test, dcs_test, calibrated_params, feed_cache, thermal_result,
        label='after'
    )
    _m = (f"  After: visc MAE={metrics_after['visc'].get('mae', float('nan')):.1f} cSt  "
          f"yield MAE={metrics_after['yield'].get('mae', float('nan')):.1f} vol%  "
          f"visc R2={metrics_after['visc'].get('r2', float('nan')):.3f}  "
          f"[{time.time()-t_step:.1f}s]")
    logger.info(_m); print(f"[CALIB] {_m}", flush=True)

    # -----------------------------------------------------------------------
    # Steps 3-4 (optional): PINN regime detection + discrepancy training
    # -----------------------------------------------------------------------
    pinn_result = None
    correction_mode = 'ols'
    if enable_pinn:
        print("[CALIB] Steps 3-4: Running PINN phases (regime detection + "
              "discrepancy training)...", flush=True)
        logger.info("Steps 3-4: PINN phases (enable_pinn=True)")
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
                thermal_params=thermal_result,
                feed_cache=feed_cache,
                ols_metrics_after=metrics_after,
                profile_name=profile_name,
                phase1_visc_slope=visc_slope_phase1,
                phase1_visc_bias=visc_bias_phase1,
            )
            correction_mode = pinn_result.get('correction_mode', 'ols')
            if correction_mode == 'pinn':
                # Use PINN test metrics as the authoritative "after" metrics
                pinn_m = pinn_result.get('pinn_metrics', {})
                if pinn_m.get('visc') and pinn_m.get('yield'):
                    metrics_after = pinn_m
            _m = (f"  PINN status={pinn_result.get('status')} "
                  f"correction_mode={correction_mode} "
                  f"n_clusters={pinn_result.get('n_clusters', 0)} "
                  f"[{pinn_result.get('elapsed_sec', 0):.1f}s]")
            logger.info(_m)
            print(f"[CALIB] {_m}", flush=True)
        except Exception as exc:
            logger.warning("PINN phases failed — continuing with OLS: %s", exc)
            print(f"[CALIB] PINN phases failed ({exc}) — OLS corrections kept",
                  flush=True)

    # -----------------------------------------------------------------------
    # Step 8: Save profile JSON
    # -----------------------------------------------------------------------
    print(f"[CALIB] Step 8: Saving profile '{profile_name}'...", flush=True)
    logger.info(f"Step 8: Saving calibration profile '{profile_name}'...")
    os.makedirs('calibration_profiles', exist_ok=True)
    profile_path = os.path.join('calibration_profiles', f'{profile_name}.json')

    def _clean(d):
        """Recursively convert nan/inf/numpy scalars to JSON-safe Python types."""
        if isinstance(d, dict):
            return {k: _clean(v) for k, v in d.items()}
        if isinstance(d, list):
            return [_clean(v) for v in d]
        if isinstance(d, float) and (math.isnan(d) or math.isinf(d)):
            return None
        if isinstance(d, (np.floating, np.integer)):
            return float(d)
        return d

    profile = {
        'profile_name':      profile_name,
        'created_at':        datetime.now().isoformat(timespec='seconds'),
        'profile_type':      'plant_workbook',
        'calibration_mode':  'dao_only',
        'k_params_anchored': True,
        'k_ref_used':        None,   # K_ref=0.644844 from _find_k_ref.py (STEP13)
        'alpha_density_fixed': 3.0,
        'calibration_params': calibrated_params,
        'thermal_params': {
            'train_a': {k: thermal_result['train_a'].get(k)
                        for k in ['alpha', 'beta', 'gamma', 'phi']},
            'train_b': {k: thermal_result['train_b'].get(k)
                        for k in ['alpha', 'beta', 'gamma', 'phi']},
        },
        'phase1_visc_ols': {
            'visc_slope':  visc_slope_ols,
            'visc_bias':   visc_bias_ols,
            'r_pearson':   round(float(visc_ols_r), 4),
            'mode':        'bias_only' if abs(visc_ols_r) < 0.15 else 'ols',
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
        'thermal_metrics': {
            'train_a': {k: thermal_result['train_a'].get(k)
                        for k in ['mae_t_bottom', 'mae_t_middle',
                                  'mae_t_steam_coil', 'mae_t_top']},
            'train_b': {k: thermal_result['train_b'].get(k)
                        for k in ['mae_t_bottom', 'mae_t_middle',
                                  'mae_t_steam_coil', 'mae_t_top']},
        },
        'dataset_info':  dataset_info,
        'split_info': {
            'train_rows_visc':  len(visc_train),
            'visc_rows_used':   len(visc_train_sub),
            'test_rows_visc':   len(visc_test),
            'train_rows_dcs':   len(dcs_train),
            'test_rows_dcs':    len(dcs_test),
            'yield_rows_used':  len(yield_df_sub),
            'yield_rows_total': len(yield_full),
            'split_date':       split_date,
        },
        'weights_used':      weights,
        'max_visc_rows':     _MAX_VISC_ROWS,
        'max_yield_rows':    _MAX_YIELD_ROWS,
        'n_workers_loky':    _n_workers(),
        'correction_mode':   correction_mode,
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
        json.dump(_clean(profile), f, indent=2)
    logger.info(f"  Profile saved to {profile_path}")
    print(f"[CALIB] DONE — profile saved to {profile_path}", flush=True)
    print("[CALIB] " + "=" * 58, flush=True)

    return {
        'calibrated_params':  calibrated_params,
        'thermal_params':     thermal_result,
        'metrics_before':     metrics_before,
        'metrics_after':      metrics_after,
        'dataset_info':       dataset_info,
        'profile_name':       profile_name,
        'calibration_mode':   'dao_only',
        'correction_mode':    correction_mode,
        'optimizer_success':  optimizer_success,
        'optimizer_message':  optimizer_message,
        'pinn_result':        pinn_result,
        'split_info': {
            'train_rows_visc':  len(visc_train),
            'visc_rows_used':   len(visc_train_sub),
            'test_rows_visc':   len(visc_test),
            'train_rows_dcs':   len(dcs_train),
            'test_rows_dcs':    len(dcs_test),
            'yield_rows_used':  len(yield_df_sub),
            'yield_rows_total': len(yield_full),
            'split_date':       split_date,
        },
    }


# ---------------------------------------------------------------------------
# Path B calibration — K_multiplier + C_T direct temperature term
# ---------------------------------------------------------------------------

def run_pathb_calibration(
    dcs_filepath: str,
    lims_filepath: str,
    weights: dict | None = None,
    initial_params: dict | None = None,
    profile_name: str = 'plant_calibration_pathb_v1',
) -> dict:
    """
    Path B calibration: K_multiplier + c_t_sat (direct temperature sensitivity).

    Runs the same Phase 0 (thermal) and Phase 1 (visc OLS) as run_full_calibration(),
    then uses a 2-parameter TRF optimizer on [K_multiplier, c_t_sat] targeting yield.

    c_t_aro = _PATHB_CT_ARO_RATIO * c_t_sat (tied, reduces to 1 free C_T param).
    c_t_res = c_t_asp = 0.0 (T-insensitive in propane SDA range).

    If c_t_sat converges to < -0.01, the temperature sensitivity is physically
    meaningful and Path B is preferred.  If it sits at a bound (-0.005 or -0.060),
    Path B adds no value — use Path A (run_full_calibration) instead.

    Saves profile with 'path': 'B' and 'c_t_params' keys.
    """
    from scipy.optimize import least_squares
    from plant_data_loader import build_calibration_dataset
    from thermal_calibration import calibrate_thermal_model, save_thermal_profile

    if weights is None:
        weights = {'yield': 0.4, 'visc': 1.0}
    if initial_params is None:
        initial_params = _DEFAULT_PARAMS.copy()

    _nfev_pathb_counter[0] = 0

    # -----------------------------------------------------------------------
    # Steps 1–4: identical to run_full_calibration (data load, thermal, split,
    # visc OLS).  Code is duplicated here to keep this function self-contained.
    # -----------------------------------------------------------------------
    print("[CALIB-B] " + "=" * 55, flush=True)
    print("[CALIB-B] Step 1: Loading calibration dataset...", flush=True)
    t_step = time.time()
    dataset = build_calibration_dataset(dcs_filepath, lims_filepath)
    dcs_hourly    = dataset['dcs_hourly']
    visc_anchored = dataset['visc_anchored']
    dataset_info  = dataset['dataset_info']
    _m = (f"  DCS rows: {dataset_info['dcs_rows_total']}, "
          f"visc_anchored: {dataset_info['visc_anchored_rows']} "
          f"[{time.time()-t_step:.1f}s]")
    print(f"[CALIB-B] {_m}", flush=True)

    print("[CALIB-B] Step 2: Phase 0 — Thermal calibration...", flush=True)
    t_step = time.time()
    thermal_result = calibrate_thermal_model(dcs_hourly)
    save_thermal_profile(thermal_result)
    _m = f"  Thermal done [{time.time()-t_step:.1f}s]"
    print(f"[CALIB-B] {_m}", flush=True)

    print("[CALIB-B] Step 3: Train/test split + subsampling...", flush=True)
    visc_train, visc_test = _make_train_test_split(visc_anchored, 0.20)
    dcs_train,  dcs_test  = _make_train_test_split(dcs_hourly, 0.20)
    split_date = str(visc_test.index.min().date()) if len(visc_test) > 0 else 'N/A'
    visc_train_sub = _subsample_visc_df(visc_train, _MAX_VISC_ROWS)
    mask = (
        dcs_train['steady_state'].fillna(False) &
        dcs_train['train_valid_a'].fillna(False) &
        dcs_train['train_valid_b'].fillna(False) &
        dcs_train['dao_yield_vol_pct'].notna()
    )
    yield_full   = dcs_train[mask]
    yield_df_sub = _subsample_yield_df(yield_full, _MAX_YIELD_ROWS)

    feed_cache: dict = {}
    all_rows_for_cache = pd.concat([
        visc_train_sub, visc_test,
        yield_df_sub,
        dcs_test[dcs_test['steady_state'].fillna(False)].head(2000),
    ])
    _prepopulate_feed_cache(all_rows_for_cache, feed_cache)

    print("[CALIB-B] Step 4: Phase 1 — Viscosity OLS...", flush=True)
    t_step = time.time()
    ols_physics_params = {
        'K_multiplier':  float(initial_params.get('K_multiplier',
                               _DEFAULT_PARAMS['K_multiplier'])),
        'E_murphree':    float(initial_params.get('E_murphree',
                               _DEFAULT_PARAMS['E_murphree'])),
        'C_entrain':     float(initial_params.get('C_entrain',
                               _DEFAULT_PARAMS['C_entrain'])),
        'delta_crit':    float(initial_params.get('delta_crit',
                               _DEFAULT_PARAMS['delta_crit'])),
        'visc_slope':    1.0,
        'visc_bias':     0.0,
    }
    visc_slope_ols, visc_bias_ols, visc_ols_r = calibrate_visc_correction(
        visc_train_sub, thermal_result, ols_physics_params
    )
    _m = (f"  Phase 1 OLS: slope={visc_slope_ols:.3f} bias={visc_bias_ols:.2f} cSt "
          f"[{time.time()-t_step:.1f}s]")
    print(f"[CALIB-B] {_m}", flush=True)

    # fixed_params shared by all Path B nfev calls
    fixed_params_b = {
        'E_murphree':  float(_FIXED_PHYSICS_PARAMS['E_murphree']),
        'C_entrain':   float(_FIXED_PHYSICS_PARAMS['C_entrain']),
        'delta_crit':  float(_FIXED_PHYSICS_PARAMS['delta_crit']),
        'visc_slope':  visc_slope_ols,
        'visc_bias':   visc_bias_ols,
    }

    k_initial = float(initial_params.get('K_multiplier',
                                         _DEFAULT_PARAMS['K_multiplier']))

    # Pre-calibration metrics (Path A comparison baseline)
    initial_full_params = dict(fixed_params_b)
    initial_full_params['K_multiplier'] = k_initial
    # c_t_sat initial guess: start at -0.02 (moderate, physically plausible)
    c_t_sat_initial = -0.020

    print("[CALIB-B] Step 5: Pre-calibration metrics (test set)...", flush=True)
    t_step = time.time()
    ct_initial = {
        'saturates':   c_t_sat_initial,
        'aromatics':   _PATHB_CT_ARO_RATIO * c_t_sat_initial,
        'resins':      0.0,
        'asphaltenes': 0.0,
    }
    metrics_before = compute_metrics(
        visc_test, dcs_test, initial_full_params, feed_cache, thermal_result,
        label='before_pathb', c_t_params=ct_initial,
    )
    _m = (f"  Before: visc MAE={metrics_before['visc'].get('mae', float('nan')):.1f} cSt  "
          f"yield MAE={metrics_before['yield'].get('mae', float('nan')):.1f} vol%  "
          f"[{time.time()-t_step:.1f}s]")
    print(f"[CALIB-B] {_m}", flush=True)

    # -----------------------------------------------------------------------
    # Path B Phase 2: K_multiplier + c_t_sat (2-param TRF)
    # -----------------------------------------------------------------------
    n_w = _n_workers()
    _m = (f"Step 6: Path B optimizer START | params={_PATHB_PARAM_NAMES} | "
          f"{len(yield_df_sub)} yield rows | {n_w} workers | max_nfev=60")
    print(f"[CALIB-B] {_m}", flush=True)

    _nfev_pathb_counter[0] = 0
    t_step = time.time()
    opt_result_b = least_squares(
        fun=_build_pathb_residuals,
        x0=[k_initial, c_t_sat_initial],
        bounds=(_PATHB_BOUNDS_LO, _PATHB_BOUNDS_HI),
        method='trf',
        args=(yield_df_sub, feed_cache, thermal_result, weights, fixed_params_b),
        verbose=0,
        max_nfev=60,
    )

    k_calibrated_b   = float(opt_result_b.x[0])
    c_t_sat_cal      = float(opt_result_b.x[1])
    optimizer_success = bool(opt_result_b.success)
    optimizer_message = str(opt_result_b.message)

    c_t_params_cal = {
        'saturates':   c_t_sat_cal,
        'aromatics':   _PATHB_CT_ARO_RATIO * c_t_sat_cal,
        'resins':      0.0,
        'asphaltenes': 0.0,
    }

    _m = (f"  Path B DONE: K_mult={k_calibrated_b:.4f} c_t_sat={c_t_sat_cal:.4f} "
          f"c_t_aro={c_t_params_cal['aromatics']:.4f} "
          f"success={optimizer_success} nfev={opt_result_b.nfev} "
          f"[{time.time()-t_step:.1f}s]")
    print(f"[CALIB-B] {_m}", flush=True)

    calibrated_params_b = {
        'K_multiplier': k_calibrated_b,
        'E_murphree':   float(fixed_params_b['E_murphree']),
        'C_entrain':    float(fixed_params_b['C_entrain']),
        'delta_crit':   float(fixed_params_b['delta_crit']),
        'visc_slope':   visc_slope_ols,
        'visc_bias':    visc_bias_ols,
    }

    # -----------------------------------------------------------------------
    # Post-calibration metrics with Path B c_t_params applied
    # -----------------------------------------------------------------------
    print("[CALIB-B] Step 7: Post-calibration metrics (test set)...", flush=True)
    t_step = time.time()
    metrics_after = compute_metrics(
        visc_test, dcs_test, calibrated_params_b, feed_cache, thermal_result,
        label='after_pathb', c_t_params=c_t_params_cal,
    )
    _m = (f"  After: visc MAE={metrics_after['visc'].get('mae', float('nan')):.1f} cSt  "
          f"yield MAE={metrics_after['yield'].get('mae', float('nan')):.1f} vol%  "
          f"visc R2={metrics_after['visc'].get('r2', float('nan')):.3f}  "
          f"[{time.time()-t_step:.1f}s]")
    print(f"[CALIB-B] {_m}", flush=True)

    # -----------------------------------------------------------------------
    # Save Path B profile JSON
    # -----------------------------------------------------------------------
    print(f"[CALIB-B] Step 8: Saving profile '{profile_name}'...", flush=True)
    os.makedirs('calibration_profiles', exist_ok=True)
    profile_path = os.path.join('calibration_profiles', f'{profile_name}.json')

    def _clean_b(d):
        if isinstance(d, dict):
            return {k: _clean_b(v) for k, v in d.items()}
        if isinstance(d, list):
            return [_clean_b(v) for v in d]
        if isinstance(d, float) and (math.isnan(d) or math.isinf(d)):
            return None
        if isinstance(d, (np.floating, np.integer)):
            return float(d)
        return d

    profile_b = {
        'profile_name':       profile_name,
        'created_at':         datetime.now().isoformat(timespec='seconds'),
        'profile_type':       'plant_workbook',
        'calibration_mode':   'dao_only',
        'path':               'B',
        'k_params_anchored':  True,
        'k_ref_used':         None,
        'alpha_density_fixed': 3.0,
        'c_t_params':         c_t_params_cal,
        'c_t_aro_ratio':      _PATHB_CT_ARO_RATIO,
        'calibration_params': calibrated_params_b,
        'thermal_params': {
            'train_a': {k: thermal_result['train_a'].get(k)
                        for k in ['alpha', 'beta', 'gamma', 'phi']},
            'train_b': {k: thermal_result['train_b'].get(k)
                        for k in ['alpha', 'beta', 'gamma', 'phi']},
        },
        'phase1_visc_ols': {
            'visc_slope':  visc_slope_ols,
            'visc_bias':   visc_bias_ols,
            'r_pearson':   round(float(visc_ols_r), 4),
            'mode':        'bias_only' if abs(visc_ols_r) < 0.15 else 'ols',
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
        'dataset_info':   dataset_info,
        'split_info': {
            'train_rows_visc':  len(visc_train),
            'visc_rows_used':   len(visc_train_sub),
            'test_rows_visc':   len(visc_test),
            'train_rows_dcs':   len(dcs_train),
            'test_rows_dcs':    len(dcs_test),
            'yield_rows_used':  len(yield_df_sub),
            'yield_rows_total': len(yield_full),
            'split_date':       split_date,
        },
        'weights_used':      weights,
        'max_visc_rows':     _MAX_VISC_ROWS,
        'max_yield_rows':    _MAX_YIELD_ROWS,
        'n_workers_loky':    _n_workers(),
    }

    with open(profile_path, 'w') as f:
        json.dump(_clean_b(profile_b), f, indent=2)
    print(f"[CALIB-B] DONE — profile saved to {profile_path}", flush=True)
    print("[CALIB-B] " + "=" * 55, flush=True)

    return {
        'calibrated_params':  calibrated_params_b,
        'c_t_params':         c_t_params_cal,
        'thermal_params':     thermal_result,
        'metrics_before':     metrics_before,
        'metrics_after':      metrics_after,
        'dataset_info':       dataset_info,
        'profile_name':       profile_name,
        'calibration_mode':   'dao_only',
        'path':               'B',
        'optimizer_success':  optimizer_success,
        'optimizer_message':  optimizer_message,
        'split_info': {
            'train_rows_visc':  len(visc_train),
            'visc_rows_used':   len(visc_train_sub),
            'test_rows_visc':   len(visc_test),
            'train_rows_dcs':   len(dcs_train),
            'test_rows_dcs':    len(dcs_test),
            'yield_rows_used':  len(yield_df_sub),
            'yield_rows_total': len(yield_full),
            'split_date':       split_date,
        },
    }
