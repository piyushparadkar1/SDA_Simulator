"""
thermal_calibration.py
======================
Calibrates the column temperature profile model for HPCL PDA Unit.

The current estimate_bed_temperatures() in run_simulation.py uses
hardcoded blend factors. This module calibrates those factors against
measured DCS bed temperatures, producing a thermal profile that
accurately predicts T_bottom, T_middle, T_steam_coil, T_top from
only T_feed and T_propane — which are the inputs available at
simulation time.

Model (4 parameters per train):
    T_bottom     = T_propane + alpha * (T_feed - T_propane)
                              * (1 - phi * ln(SO_ratio / 8))
    T_middle     = T_bottom  + beta  * (T_feed - T_bottom)
    T_steam_coil = T_middle  + gamma * (T_feed - T_middle)
    T_top        = T_steam_coil  (no 5th parameter; T_top residuals
                                  carry steam-heating uncertainty, weighted 0.5)

Output: save_thermal_profile() writes calibration_profiles/thermal_profile.json
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

# Initial guess and bounds for [alpha, beta, gamma, phi]
_X0 = np.array([0.35, 0.55, 0.20, 0.05])
_LB = np.array([0.05, 0.05, 0.00, 0.00])
_UB = np.array([0.80, 0.95, 0.60, 0.30])


def _predict_t_profile(
    params: np.ndarray,
    t_feed: float,
    t_propane: float,
    so_ratio: float,
) -> np.ndarray:
    """
    Predict [T_bottom, T_middle, T_steam_coil, T_top] from thermal params.
    Returns array of 4 temperatures in °C.

    params: [alpha, beta, gamma, phi]
    """
    alpha, beta, gamma, phi = params
    # Clip SO ratio to avoid log(0) or negative log arguments
    so_safe = np.clip(so_ratio / 8.0, 0.1, 20.0)
    so_correction = 1.0 - phi * np.log(so_safe)

    T_bot = t_propane + alpha * (t_feed - t_propane) * so_correction
    T_mid = T_bot  + beta  * (t_feed - T_bot)
    T_sc  = T_mid  + gamma * (t_feed - T_mid)
    T_top = T_sc   # no additional parameter for steam heating

    return np.array([T_bot, T_mid, T_sc, T_top])


def _thermal_residuals(
    params: np.ndarray,
    dcs: pd.DataFrame,
    train: str,
) -> np.ndarray:
    """
    Build residuals vector: predicted_T_bed - measured_T_bed for all rows.

    params: [alpha, beta, gamma, phi]
    train: 'a' or 'b'

    For each valid DCS row (train_valid_{train}=True, steady_state=True):
      predicted = _predict_t_profile(params, T_feed, T_propane, SO_ratio)
      residuals = [T_bot_pred - T_bot_meas,
                   T_mid_pred - T_mid_meas,
                   T_steam_coil_pred - T_steam_coil_meas,
                   T_top_pred - T_top_meas * 0.5]

    T_top residuals are weighted by 0.5 — the top bed is steam-heated
    and harder to model; over-weighting it distorts bottom/middle.

    Returns flat 1-D numpy array of residuals.
    """
    alpha, beta, gamma, phi = params
    so_safe = np.clip(
        dcs[f'so_ratio_{train}'].values.astype(float) / 8.0,
        0.1, 20.0
    )
    so_corr = 1.0 - phi * np.log(so_safe)

    t_feed = dcs[f'feed_temp_{train}'].values.astype(float)
    t_prop = dcs['propane_temp'].values.astype(float)

    T_bot = t_prop + alpha * (t_feed - t_prop) * so_corr
    T_mid = T_bot  + beta  * (t_feed - T_bot)
    T_sc  = T_mid  + gamma * (t_feed - T_mid)
    T_top = T_sc

    r_bot = T_bot - dcs[f't_bot_{train}'].values.astype(float)
    r_mid = T_mid - dcs[f't_mid_{train}'].values.astype(float)
    r_sc  = T_sc  - dcs[f't_steam_coil_{train}'].values.astype(float)
    r_top = 0.5 * (T_top - dcs[f't_top_{train}'].values.astype(float))

    return np.concatenate([r_bot, r_mid, r_sc, r_top])


def calibrate_thermal_model(
    dcs: pd.DataFrame,
) -> dict:
    """
    Calibrate thermal parameters for both trains independently.

    For each train ('a', 'b'):
      1. Filter to rows where train_valid_{train}=True AND steady_state=True.
      2. Run scipy.optimize.least_squares (TRF method).
      3. Compute per-bed MAE in °C on the calibration set.

    Returns:
        {
          'train_a': {
            'alpha': float, 'beta': float, 'gamma': float, 'phi': float,
            'mae_t_bottom':     float,
            'mae_t_middle':     float,
            'mae_t_steam_coil': float,
            'mae_t_top':        float,
            'rows_used':        int,
            'converged':        bool,
          },
          'train_b': { ... same structure ... },
        }
    """
    from scipy.optimize import least_squares

    result = {}

    for train in ['a', 'b']:
        logger.info(f"  Calibrating thermal model for train {train.upper()}...")

        # Required columns
        required = [
            f'train_valid_{train}', 'steady_state',
            f'feed_temp_{train}', 'propane_temp', f'so_ratio_{train}',
            f't_bot_{train}', f't_mid_{train}',
            f't_steam_coil_{train}', f't_top_{train}',
        ]
        missing = [c for c in required if c not in dcs.columns]
        if missing:
            logger.warning(
                f"  Train {train.upper()}: missing columns {missing}. "
                f"Using default parameters."
            )
            result[f'train_{train}'] = {
                'alpha': float(_X0[0]), 'beta': float(_X0[1]),
                'gamma': float(_X0[2]), 'phi':  float(_X0[3]),
                'mae_t_bottom': np.nan, 'mae_t_middle': np.nan,
                'mae_t_steam_coil': np.nan, 'mae_t_top': np.nan,
                'rows_used': 0, 'converged': False,
            }
            continue

        mask = (
            dcs[f'train_valid_{train}'].fillna(False) &
            dcs['steady_state'].fillna(False)
        )
        # Drop rows with NaN in temperature columns
        temp_cols = [f'feed_temp_{train}', 'propane_temp', f'so_ratio_{train}',
                     f't_bot_{train}', f't_mid_{train}',
                     f't_steam_coil_{train}', f't_top_{train}']
        mask &= dcs[temp_cols].notna().all(axis=1)

        dcs_sub = dcs[mask].copy()
        n_rows = len(dcs_sub)
        logger.info(f"  Train {train.upper()}: {n_rows} rows for calibration")

        if n_rows < 10:
            logger.warning(
                f"  Train {train.upper()}: only {n_rows} rows, using defaults"
            )
            result[f'train_{train}'] = {
                'alpha': float(_X0[0]), 'beta': float(_X0[1]),
                'gamma': float(_X0[2]), 'phi':  float(_X0[3]),
                'mae_t_bottom': np.nan, 'mae_t_middle': np.nan,
                'mae_t_steam_coil': np.nan, 'mae_t_top': np.nan,
                'rows_used': n_rows, 'converged': False,
            }
            continue

        opt = least_squares(
            fun=_thermal_residuals,
            x0=_X0.copy(),
            bounds=(_LB, _UB),
            method='trf',
            args=(dcs_sub, train),
            verbose=0,
            max_nfev=1000,
        )

        params_opt = opt.x
        converged = bool(opt.success or opt.cost < 1e6)

        # Compute per-bed MAE
        pred = _predict_t_profile(
            params_opt,
            dcs_sub[f'feed_temp_{train}'].values,
            dcs_sub['propane_temp'].values,
            dcs_sub[f'so_ratio_{train}'].values,
        )
        # pred is 4×N array when called with arrays
        # Re-vectorise
        so_safe = np.clip(
            dcs_sub[f'so_ratio_{train}'].values / 8.0, 0.1, 20.0
        )
        so_corr = 1.0 - params_opt[3] * np.log(so_safe)
        t_feed = dcs_sub[f'feed_temp_{train}'].values
        t_prop = dcs_sub['propane_temp'].values

        T_bot = t_prop + params_opt[0] * (t_feed - t_prop) * so_corr
        T_mid = T_bot  + params_opt[1] * (t_feed - T_bot)
        T_sc  = T_mid  + params_opt[2] * (t_feed - T_mid)
        T_top = T_sc

        mae_bot = float(np.mean(np.abs(T_bot - dcs_sub[f't_bot_{train}'].values)))
        mae_mid = float(np.mean(np.abs(T_mid - dcs_sub[f't_mid_{train}'].values)))
        mae_sc  = float(np.mean(np.abs(T_sc  - dcs_sub[f't_steam_coil_{train}'].values)))
        mae_top = float(np.mean(np.abs(T_top - dcs_sub[f't_top_{train}'].values)))

        logger.info(
            f"  Train {train.upper()}: alpha={params_opt[0]:.3f}, "
            f"beta={params_opt[1]:.3f}, gamma={params_opt[2]:.3f}, "
            f"phi={params_opt[3]:.3f}, converged={converged}"
        )
        logger.info(
            f"  MAE: bot={mae_bot:.2f}°C  mid={mae_mid:.2f}°C  "
            f"sc={mae_sc:.2f}°C  top={mae_top:.2f}°C"
        )

        result[f'train_{train}'] = {
            'alpha': float(params_opt[0]),
            'beta':  float(params_opt[1]),
            'gamma': float(params_opt[2]),
            'phi':   float(params_opt[3]),
            'mae_t_bottom':     mae_bot,
            'mae_t_middle':     mae_mid,
            'mae_t_steam_coil': mae_sc,
            'mae_t_top':        mae_top,
            'rows_used':        n_rows,
            'converged':        converged,
        }

    return result


def predict_t_profile_calibrated(
    thermal_params: dict,
    t_feed: float,
    t_propane: float,
    so_ratio: float,
    train: str,
) -> list:
    """
    Predict T-profile using calibrated parameters.
    Returns list [T_bottom, T_middle, T_steam_coil, T_top] in °C.
    Used by simulator_bridge.py to produce T_profile for run_extractor().

    train: 'a' or 'b'
    """
    tp = thermal_params.get(f'train_{train}', thermal_params)
    params = np.array([tp['alpha'], tp['beta'], tp['gamma'], tp['phi']])
    pred = _predict_t_profile(params, t_feed, t_propane, so_ratio)
    return [float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3])]


def save_thermal_profile(
    thermal_result: dict,
    filepath: str = 'calibration_profiles/thermal_profile.json',
) -> None:
    """
    Save thermal calibration result to JSON.
    Includes: params per train, MAE per bed, rows_used, calibration timestamp.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    payload = {
        'calibrated_at': datetime.now().isoformat(timespec='seconds'),
        'model': 'T_bottom = T_propane + alpha*(T_feed-T_propane)*(1-phi*ln(SO/8))',
        'train_a': thermal_result.get('train_a', {}),
        'train_b': thermal_result.get('train_b', {}),
    }
    with open(filepath, 'w') as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Thermal profile saved to {filepath}")


def load_thermal_profile(
    filepath: str = 'calibration_profiles/thermal_profile.json',
) -> dict:
    """
    Load thermal profile. Returns dict matching save_thermal_profile output.
    Raises FileNotFoundError with clear message if file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Thermal profile not found at '{filepath}'. "
            f"Run calibration first via /api/calibrate_workbooks."
        )
    with open(filepath) as f:
        return json.load(f)
