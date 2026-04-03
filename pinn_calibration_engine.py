"""
PINN Calibration Engine — Orchestration of Phase 3 (Regime Detection)
and Phase 4 (PINN Training) on top of the existing 3-phase calibration.

This module is the integration point between the classic calibration
pipeline (calibration_engine.py) and the new PINN-based discrepancy
correction system (pinn_network.py + pinn_trainer.py + regime_detector.py).

Flow
----
Phase 0-2 : Unchanged (thermal → OLS visc → K_mult optimizer)
Phase 3   : RegimeDetector.fit(dcs_hourly) → cluster_id per row
Phase 4   : Forward pass cache → PINNTrainer.train() on residuals
Phase 5   : Combined metrics + quality gate (PINN vs OLS)

The PINN does NOT replace the physics engine — it learns a multiplicative
correction on top of existing physics predictions:

    corrected_visc  = physics_visc  * (1 + δ_visc(features, cluster))
    corrected_yield = physics_yield * (1 + δ_yield(features, cluster))

ISO 23247 Entity: digital_twin_core
"""
_ISO23247_ENTITY = 'digital_twin_core'

import json
import logging
import os
import time
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_PINN_CHECKPOINT_DIR = 'calibration_profiles'
_QUALITY_GATE_MARGIN = 0.10   # PINN must be ≤10% worse than OLS to deploy
_MAX_FORWARD_ROWS = 2000      # cap cached forward-pass rows


def run_pinn_phases(
    dcs_hourly: pd.DataFrame,
    visc_anchored: pd.DataFrame,
    dcs_train: pd.DataFrame,
    visc_train: pd.DataFrame,
    dcs_test: pd.DataFrame,
    visc_test: pd.DataFrame,
    calibrated_params: dict,
    thermal_params: dict,
    feed_cache: dict,
    ols_metrics_after: dict,
    profile_name: str = 'plant_calibration_v1',
    lambda_config: Optional[Dict] = None,
    training_config: Optional[Dict] = None,
    phase1_visc_slope: Optional[float] = None,
    phase1_visc_bias: Optional[float] = None,
) -> dict:
    """
    Run Phase 3 (regime detection) and Phase 4 (PINN training) on top
    of already-completed Phases 0-2.

    Parameters
    ----------
    dcs_hourly : pd.DataFrame
        Full DCS dataset (used for regime detection).
    visc_anchored : pd.DataFrame
        Full LIMS-anchored dataset.
    dcs_train, visc_train : pd.DataFrame
        Chronological training splits.
    dcs_test, visc_test : pd.DataFrame
        Chronological test splits.
    calibrated_params : dict
        Result of Phase 2 (K_multiplier, visc_slope, visc_bias, etc.).
    thermal_params : dict
        Calibrated thermal model from Phase 0.
    feed_cache : dict
        Pre-populated feed component cache from calibration_engine.
    ols_metrics_after : dict
        OLS baseline metrics on test set (to compare PINN against).
    profile_name : str
        Profile name for checkpoint file naming.
    lambda_config : dict, optional
        Physics penalty weights for PINNTrainer.
    training_config : dict, optional
        Training hyperparameters for PINNTrainer.
    phase1_visc_slope : float, optional
        Visc correction slope calibrated at K_mult=1.0 (Phase 1 OLS, before
        Phase 1.5 re-calibration at the converged K_mult). When provided,
        overrides calibrated_params['visc_slope'] in the PINN physics cache
        so the baseline is consistent with K_mult=1.0 (which PINN forces).
    phase1_visc_bias : float, optional
        Visc correction bias at K_mult=1.0 (same rationale as above).

    Returns
    -------
    dict with keys:
        status : 'deployed' | 'discarded_quality_gate' | 'failed' | 'unavailable'
        correction_mode : 'pinn' | 'ols'
        pinn_metrics : dict (test-set metrics with PINN corrections)
        ols_metrics : dict (passed-in ols_metrics_after for comparison)
        pinn_checkpoint_dir : str or None
        regime_summary : dict
        training_result : dict
        n_clusters : int
        elapsed_sec : float
    """
    t0 = time.time()

    # Check PyTorch availability
    from pinn_network import is_torch_available, PINNCorrector, \
        extract_features_batch, N_CONTINUOUS, MAX_CLUSTERS
    from regime_detector import RegimeDetector, REGIME_FEATURES

    if not is_torch_available():
        logger.warning("PyTorch unavailable — PINN phases skipped. "
                       "Install: pip install torch")
        return _pinn_unavailable_result(ols_metrics_after, time.time() - t0)

    # Build physics params for PINN forward passes.
    # When called from Option-C mode (enable_pinn=True with Phase 2 skipped),
    # calibrated_params already has K_multiplier=1.0 and Phase 1 OLS visc params.
    # We still defensively override K_mult to 1.0 here in case this function is
    # ever called from a non-Option-C path with a calibrated K_mult≠1.0.
    _ols_kmult = calibrated_params.get('K_multiplier', 1.0)
    pinn_physics_params = dict(calibrated_params)
    pinn_physics_params['K_multiplier'] = 1.0
    if abs(_ols_kmult - 1.0) > 0.001:
        logger.warning("PINN physics override: K_mult %.4f → 1.0 (dead-zone avoidance; "
                       "Phase 2 should have been skipped in PINN mode)", _ols_kmult)
        print(f"[PINN] WARNING: K_mult {_ols_kmult:.4f} → 1.0 override "
              f"(Phase 2 should be skipped when enable_pinn=True)", flush=True)
    else:
        logger.info("PINN physics params: K_mult=1.0 (already correct from Phase 2 skip)")
        print(f"[PINN] K_mult=1.0 confirmed (Phase 2 was skipped)", flush=True)

    # Override visc params to Phase 1 OLS values if provided.
    # In Option-C mode calibrated_params already has Phase 1 values; this is
    # a no-op. In non-Option-C mode it corrects Phase 1.5 contamination.
    _cur_slope = calibrated_params.get('visc_slope', 1.0)
    _cur_bias  = calibrated_params.get('visc_bias',  0.0)
    if phase1_visc_slope is not None:
        pinn_physics_params['visc_slope'] = float(phase1_visc_slope)
        pinn_physics_params['visc_bias']  = float(phase1_visc_bias) if phase1_visc_bias is not None else _cur_bias
        if abs(float(phase1_visc_slope) - _cur_slope) > 0.001:
            logger.info("PINN visc override: slope %.3f→%.3f  bias %.2f→%.2f  (Phase 1 OLS)",
                        _cur_slope, pinn_physics_params['visc_slope'],
                        _cur_bias,  pinn_physics_params['visc_bias'])
            print(f"[PINN] Visc override: slope {_cur_slope:.3f}→{pinn_physics_params['visc_slope']:.3f}"
                  f"  bias {_cur_bias:.2f}→{pinn_physics_params['visc_bias']:.2f}  (Phase 1 OLS)",
                  flush=True)
        else:
            print(f"[PINN] Visc params confirmed: slope={_cur_slope:.3f}  bias={_cur_bias:.2f}  "
                  f"(Phase 1 OLS, already correct)", flush=True)

    # -----------------------------------------------------------------------
    # Phase 3: Regime Detection
    # -----------------------------------------------------------------------
    print("[PINN] Phase 3: Fitting operating regime detector...", flush=True)
    logger.info("Phase 3: Regime detection on %d DCS rows", len(dcs_hourly))

    regime_result = _fit_regime_detector(dcs_hourly, profile_name)
    detector = regime_result['detector']
    n_clusters = detector.n_clusters
    logger.info("Phase 3 complete: %d clusters in %.1fs",
                n_clusters, regime_result['elapsed_sec'])

    # Predict cluster IDs for training + test rows
    cluster_ids_train = detector.predict(dcs_train, smooth=False) \
        if all(c in dcs_train.columns for c in REGIME_FEATURES) \
        else np.zeros(len(dcs_train), dtype=np.int32)

    cluster_ids_visc_train = detector.predict(visc_train, smooth=False) \
        if all(c in visc_train.columns for c in REGIME_FEATURES) \
        else np.zeros(len(visc_train), dtype=np.int32)

    cluster_ids_test = detector.predict(dcs_test, smooth=False) \
        if all(c in dcs_test.columns for c in REGIME_FEATURES) \
        else np.zeros(len(dcs_test), dtype=np.int32)

    cluster_ids_visc_test = detector.predict(visc_test, smooth=False) \
        if all(c in visc_test.columns for c in REGIME_FEATURES) \
        else np.zeros(len(visc_test), dtype=np.int32)

    # -----------------------------------------------------------------------
    # Phase 4: Physics Forward Pass Cache
    # -----------------------------------------------------------------------
    print("[PINN] Phase 4a: Caching physics engine predictions "
          "(~2-3 min)...", flush=True)
    logger.info("Phase 4a: Forward pass on %d visc_train + %d dcs_train rows",
                len(visc_train), len(dcs_train))

    cache_result = _build_physics_cache(
        visc_train=visc_train,
        dcs_train=dcs_train,
        calibrated_params=pinn_physics_params,
        thermal_params=thermal_params,
        feed_cache=feed_cache,
    )

    # -----------------------------------------------------------------------
    # Phase 4b: PINN Training
    # -----------------------------------------------------------------------
    print("[PINN] Phase 4b: Training discrepancy networks...", flush=True)
    logger.info("Phase 4b: PINN training with %d visc targets, "
                "%d yield targets",
                int(cache_result['visc_mask'].sum()),
                int(cache_result['yield_mask'].sum()))

    try:
        train_result = _train_pinn(
            cache_result=cache_result,
            visc_train=visc_train,
            dcs_train=dcs_train,
            cluster_ids_visc=cluster_ids_visc_train,
            cluster_ids_dcs=cluster_ids_train,
            n_clusters=n_clusters,
            lambda_config=lambda_config,
            training_config=training_config,
        )
        corrector = train_result['corrector']
    except Exception as exc:
        logger.error("PINN training failed: %s", exc, exc_info=True)
        return _pinn_failed_result(ols_metrics_after, str(exc), time.time() - t0)

    # -----------------------------------------------------------------------
    # Phase 5: Test-Set Evaluation + Quality Gate
    # -----------------------------------------------------------------------
    print("[PINN] Phase 5: Evaluating PINN on test set...", flush=True)
    pinn_metrics = _evaluate_pinn_on_test(
        corrector=corrector,
        detector=detector,
        visc_test=visc_test,
        dcs_test=dcs_test,
        cluster_ids_visc_test=cluster_ids_visc_test,
        cluster_ids_dcs_test=cluster_ids_test,
        calibrated_params=pinn_physics_params,
        thermal_params=thermal_params,
        feed_cache=feed_cache,
        n_clusters=n_clusters,
    )

    # Quality gate: keep PINN only if it doesn't degrade beyond tolerance
    passed_gate, gate_detail = _quality_gate(pinn_metrics, ols_metrics_after)

    if passed_gate:
        corrector._trained = True
        checkpoint_dir = os.path.join(_PINN_CHECKPOINT_DIR, profile_name)
        corrector.save(checkpoint_dir)
        detector.save(checkpoint_dir)
        status = 'deployed'
        correction_mode = 'pinn'
        logger.info("PINN deployed: visc_r2=%.3f (OLS=%.3f), "
                    "yield_r2=%.3f (OLS=%.3f)",
                    pinn_metrics.get('visc', {}).get('r2', 0),
                    ols_metrics_after.get('visc', {}).get('r2', 0),
                    pinn_metrics.get('yield', {}).get('r2', 0),
                    ols_metrics_after.get('yield', {}).get('r2', 0))
        print(f"[PINN] DEPLOYED — corrections saved to {checkpoint_dir}",
              flush=True)
    else:
        status = 'discarded_quality_gate'
        correction_mode = 'ols'
        checkpoint_dir = None
        logger.warning("PINN discarded by quality gate: %s", gate_detail)
        print(f"[PINN] DISCARDED by quality gate — OLS corrections kept. "
              f"Reason: {gate_detail}", flush=True)

    elapsed = time.time() - t0
    logger.info("PINN phases complete in %.1fs (status=%s)", elapsed, status)

    return {
        'status': status,
        'correction_mode': correction_mode,
        'pinn_metrics': pinn_metrics,
        'ols_metrics': ols_metrics_after,
        'quality_gate': {'passed': passed_gate, 'detail': gate_detail},
        'pinn_checkpoint_dir': checkpoint_dir,
        'regime_summary': detector.summary(),
        'training_result': {
            'epochs_trained': train_result['training_history'].get('epochs_trained', 0),
            'final_loss': train_result['training_history'].get('final_loss', 0),
            'val_loss': train_result['training_history'].get('val_loss', 0),
            'best_epoch': train_result['training_history'].get('best_epoch', 0),
            'elapsed_sec': train_result['training_history'].get('elapsed_sec', 0),
        },
        'n_clusters': n_clusters,
        'elapsed_sec': elapsed,
    }


def load_pinn_corrector(profile_name: str):
    """
    Load a saved PINN corrector from a calibration profile directory.

    Parameters
    ----------
    profile_name : str

    Returns
    -------
    (PINNCorrector, RegimeDetector) or (None, None) if not found.
    """
    from pinn_network import PINNCorrector, is_torch_available
    from regime_detector import RegimeDetector

    if not is_torch_available():
        return None, None

    checkpoint_dir = os.path.join(_PINN_CHECKPOINT_DIR, profile_name)
    try:
        corrector = PINNCorrector.load(checkpoint_dir)
        detector = RegimeDetector.load(checkpoint_dir)
        logger.info("PINN corrector loaded from %s", checkpoint_dir)
        return corrector, detector
    except FileNotFoundError:
        logger.info("No PINN checkpoint found at %s — OLS mode", checkpoint_dir)
        return None, None
    except Exception as exc:
        logger.warning("Failed to load PINN checkpoint: %s", exc)
        return None, None


def apply_pinn_correction(sim_result: dict, features: np.ndarray,
                           corrector) -> dict:
    """
    Apply PINN multiplicative corrections to a single simulation result.

    Parameters
    ----------
    sim_result : dict
        Output of simulate_parallel_trains() with 'dao_visc_100_cSt'
        and 'dao_yield_vol_pct'.
    features : np.ndarray, shape (input_dim,)
        Feature vector from extract_features_from_row().
    corrector : PINNCorrector

    Returns
    -------
    dict — sim_result with corrected visc and yield, plus
    'pinn_delta_visc' and 'pinn_delta_yield' diagnostics.
    """
    if corrector is None or not corrector.is_trained:
        return sim_result

    try:
        d_visc = corrector.predict_visc_correction(features)
        d_yield = corrector.predict_yield_correction(features)

        result = dict(sim_result)
        raw_visc = result.get('dao_visc_100_cSt', 0.0)
        raw_yield = result.get('dao_yield_vol_pct', 0.0)

        result['dao_visc_100_cSt'] = max(raw_visc * (1.0 + d_visc), 0.1)
        result['dao_yield_vol_pct'] = float(np.clip(
            raw_yield * (1.0 + d_yield), 0.0, 100.0
        ))
        result['pinn_delta_visc'] = round(d_visc, 4)
        result['pinn_delta_yield'] = round(d_yield, 4)
        result['correction_mode'] = 'pinn'
    except Exception as exc:
        logger.warning("PINN correction failed, using uncorrected result: %s", exc)
        sim_result['correction_mode'] = 'ols_fallback'

    return sim_result


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _fit_regime_detector(dcs_hourly: pd.DataFrame,
                         profile_name: str) -> dict:
    """Fit RegimeDetector on DCS data."""
    from regime_detector import RegimeDetector
    t0 = time.time()
    detector = RegimeDetector()
    detector.fit(dcs_hourly)
    return {'detector': detector, 'elapsed_sec': time.time() - t0}


def _build_physics_cache(visc_train: pd.DataFrame,
                         dcs_train: pd.DataFrame,
                         calibrated_params: dict,
                         thermal_params: dict,
                         feed_cache: dict) -> dict:
    """
    Run physics forward pass on training rows and cache predictions.
    Uses joblib for parallelism (same pattern as calibration_engine).
    """
    from simulator_bridge import simulate_parallel_trains

    def _run_row_visc(row_tuple):
        idx, row = row_tuple
        try:
            result = simulate_parallel_trains(
                row, calibrated_params, feed_cache,
                thermal_params=thermal_params, use_dcs_temperatures=True,
            )
            if result['converged']:
                return {
                    'physics_visc': result['dao_visc_100_cSt'],
                    'measured_visc': float(row.get('dao_visc_100', np.nan)),
                    'converged': True,
                }
        except Exception:
            pass
        return {'physics_visc': np.nan, 'measured_visc': np.nan, 'converged': False}

    def _run_row_yield(row_tuple):
        idx, row = row_tuple
        try:
            result = simulate_parallel_trains(
                row, calibrated_params, feed_cache,
                thermal_params=thermal_params, use_dcs_temperatures=True,
            )
            if result['converged']:
                return {
                    'physics_yield': result['dao_yield_vol_pct'],
                    'measured_yield': float(row.get('dao_yield_vol_pct', np.nan)),
                    'converged': True,
                }
        except Exception:
            pass
        return {'physics_yield': np.nan, 'measured_yield': np.nan, 'converged': False}

    # Run visc rows
    visc_rows = list(visc_train.iterrows())[:_MAX_FORWARD_ROWS]
    visc_results = Parallel(n_jobs=-1, backend='loky')(
        delayed(_run_row_visc)(row) for row in visc_rows
    )

    # Run yield rows (use steady_state from dcs_train)
    steady = dcs_train[dcs_train.get('steady_state', pd.Series(True, index=dcs_train.index)).fillna(True)] \
        if 'steady_state' in dcs_train.columns else dcs_train
    yield_rows = list(steady.iterrows())[:_MAX_FORWARD_ROWS]
    yield_results = Parallel(n_jobs=-1, backend='loky')(
        delayed(_run_row_yield)(row) for row in yield_rows
    )

    # Unpack visc arrays
    physics_visc = np.array([r['physics_visc'] for r in visc_results], dtype=np.float32)
    measured_visc = np.array([r['measured_visc'] for r in visc_results], dtype=np.float32)
    visc_mask = np.array([
        r['converged'] and not np.isnan(r['physics_visc']) and not np.isnan(r['measured_visc'])
        for r in visc_results
    ], dtype=bool)

    # Unpack yield arrays
    physics_yield = np.array([r['physics_yield'] for r in yield_results], dtype=np.float32)
    measured_yield = np.array([r['measured_yield'] for r in yield_results], dtype=np.float32)
    yield_mask = np.array([
        r['converged'] and not np.isnan(r['physics_yield']) and not np.isnan(r['measured_yield'])
        for r in yield_results
    ], dtype=bool)

    logger.info("Physics cache: %d visc rows (%d valid), %d yield rows (%d valid)",
                len(visc_results), int(visc_mask.sum()),
                len(yield_results), int(yield_mask.sum()))

    return {
        'visc_df': visc_train.iloc[:len(visc_results)].reset_index(drop=True),
        'yield_df': steady.iloc[:len(yield_results)].reset_index(drop=True),
        'physics_visc': physics_visc,
        'measured_visc': measured_visc,
        'visc_mask': visc_mask,
        'physics_yield': physics_yield,
        'measured_yield': measured_yield,
        'yield_mask': yield_mask,
    }


def _train_pinn(cache_result: dict, visc_train: pd.DataFrame,
                dcs_train: pd.DataFrame,
                cluster_ids_visc: np.ndarray,
                cluster_ids_dcs: np.ndarray,
                n_clusters: int,
                lambda_config: Optional[Dict],
                training_config: Optional[Dict]) -> dict:
    """Extract features and run the PINN training loop."""
    from pinn_network import PINNCorrector, extract_features_batch
    from pinn_trainer import PINNTrainer

    visc_df = cache_result['visc_df']
    yield_df = cache_result['yield_df']

    # Extract features (visc rows and yield rows separately, then stack)
    n_visc = len(visc_df)
    n_yield = len(yield_df)
    n_total = n_visc + n_yield

    feats_visc = extract_features_batch(visc_df, n_clusters, cluster_ids_visc[:n_visc])
    feats_yield = extract_features_batch(yield_df, n_clusters, cluster_ids_dcs[:n_yield])

    # Stack into unified feature / target arrays for the trainer
    features = np.vstack([feats_visc, feats_yield]).astype(np.float32)

    physics_visc_full = np.concatenate([
        cache_result['physics_visc'],
        np.full(n_yield, np.nan, dtype=np.float32)
    ])
    measured_visc_full = np.concatenate([
        cache_result['measured_visc'],
        np.full(n_yield, np.nan, dtype=np.float32)
    ])
    visc_mask_full = np.concatenate([
        cache_result['visc_mask'],
        np.zeros(n_yield, dtype=bool)
    ])

    physics_yield_full = np.concatenate([
        np.full(n_visc, np.nan, dtype=np.float32),
        cache_result['physics_yield']
    ])
    measured_yield_full = np.concatenate([
        np.full(n_visc, np.nan, dtype=np.float32),
        cache_result['measured_yield']
    ])
    yield_mask_full = np.concatenate([
        np.zeros(n_visc, dtype=bool),
        cache_result['yield_mask']
    ])

    # S/O ratios for monotonicity pairs
    so_visc = visc_df.get('so_ratio_a', pd.Series(8.0, index=visc_df.index)).fillna(8.0).values
    so_yield = yield_df.get('so_ratio_a', pd.Series(8.0, index=yield_df.index)).fillna(8.0).values
    so_ratios = np.concatenate([so_visc, so_yield]).astype(np.float32)

    # Build and train corrector
    corrector = PINNCorrector(n_clusters=n_clusters)
    trainer = PINNTrainer(corrector, lambdas=lambda_config,
                          training_config=training_config)

    history = trainer.train(
        features=features,
        physics_visc=physics_visc_full,
        measured_visc=measured_visc_full,
        visc_mask=visc_mask_full,
        physics_yield=physics_yield_full,
        measured_yield=measured_yield_full,
        yield_mask=yield_mask_full,
        so_ratios=so_ratios,
    )

    return {'corrector': corrector, 'training_history': history}


def _evaluate_pinn_on_test(corrector, detector, visc_test: pd.DataFrame,
                            dcs_test: pd.DataFrame,
                            cluster_ids_visc_test: np.ndarray,
                            cluster_ids_dcs_test: np.ndarray,
                            calibrated_params: dict, thermal_params: dict,
                            feed_cache: dict, n_clusters: int) -> dict:
    """Evaluate PINN-corrected predictions on the test set.

    Uses joblib outer parallelism (same pattern as compute_metrics) to avoid
    the 12× throughput penalty of a serial outer loop with 2-worker inner
    parallelism (simulate_parallel_trains).
    """
    import multiprocessing
    from pinn_network import extract_features_from_row, PINNCorrector
    from simulator_bridge import simulate_parallel_trains

    n_jobs = min(multiprocessing.cpu_count(), 12)

    # --- helper: one visc row ---
    def _eval_visc_row(i, row, cid):
        try:
            result = simulate_parallel_trains(
                row, calibrated_params, feed_cache,
                thermal_params=thermal_params, use_dcs_temperatures=True,
            )
            if not result['converged']:
                return None
            feats = extract_features_from_row(row, n_clusters, int(cid))
            corrected = apply_pinn_correction(result, feats, corrector)
            meas_v = float(row.get('dao_visc_100', float('nan')))
            if not (meas_v != meas_v):   # isnan check without numpy
                return (corrected['dao_visc_100_cSt'], meas_v)
        except Exception:
            pass
        return None

    # --- helper: one yield row ---
    def _eval_yield_row(i, row, cid):
        try:
            result = simulate_parallel_trains(
                row, calibrated_params, feed_cache,
                thermal_params=thermal_params, use_dcs_temperatures=True,
            )
            if not result['converged']:
                return None
            feats = extract_features_from_row(row, n_clusters, int(cid))
            corrected = apply_pinn_correction(result, feats, corrector)
            meas_y = float(row.get('dao_yield_vol_pct', float('nan')))
            if not (meas_y != meas_y):
                return (corrected['dao_yield_vol_pct'], meas_y)
        except Exception:
            pass
        return None

    visc_rows = list(visc_test.iterrows())
    dcs_rows  = list(dcs_test.iterrows())

    visc_results = Parallel(n_jobs=n_jobs, prefer='processes')(
        delayed(_eval_visc_row)(
            i, row,
            int(cluster_ids_visc_test[i]) if i < len(cluster_ids_visc_test) else -1
        )
        for i, (_, row) in enumerate(visc_rows)
    )
    yield_results = Parallel(n_jobs=n_jobs, prefer='processes')(
        delayed(_eval_yield_row)(
            i, row,
            int(cluster_ids_dcs_test[i]) if i < len(cluster_ids_dcs_test) else -1
        )
        for i, (_, row) in enumerate(dcs_rows)
    )

    visc_preds = [r[0] for r in visc_results if r is not None]
    visc_meas  = [r[1] for r in visc_results if r is not None]
    yield_preds = [r[0] for r in yield_results if r is not None]
    yield_meas  = [r[1] for r in yield_results if r is not None]

    return {
        'visc': _metrics_dict(np.array(visc_preds), np.array(visc_meas), 'visc'),
        'yield': _metrics_dict(np.array(yield_preds), np.array(yield_meas), 'yield'),
    }


def _metrics_dict(pred: np.ndarray, meas: np.ndarray, kind: str) -> dict:
    """Compute MAE, RMSE, R² for pred vs meas arrays."""
    if len(pred) < 2:
        return {'mae': float('nan'), 'rmse': float('nan'),
                'r2': float('nan'), 'n': len(pred)}

    err = pred - meas
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((meas - meas.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-9)

    return {'mae': round(mae, 3), 'rmse': round(rmse, 3),
            'r2': round(r2, 4), 'n': len(pred)}


def _quality_gate(pinn_metrics: dict, ols_metrics: dict) -> Tuple[bool, str]:
    """
    Determine if PINN metrics are good enough to deploy.

    Two criteria — PINN passes if it satisfies either:
      (A) Absolute targets: yield MAE ≤ 5 vol% AND visc MAE ≤ 8 cSt
          (industry-acceptable thresholds for process simulation)
      (B) Relative improvement: PINN is not more than _QUALITY_GATE_MARGIN
          (10%) worse than OLS on BOTH visc MAE and yield MAE

    Criterion B exists as a fallback so PINN is deployed when OLS is very
    poor (R² < 0) and any improvement (even partial) is useful.
    """
    details = []

    pinn_visc_mae  = pinn_metrics.get('visc', {}).get('mae', float('nan'))
    pinn_yield_mae = pinn_metrics.get('yield', {}).get('mae', float('nan'))
    ols_visc_mae   = ols_metrics.get('visc', {}).get('mae', float('nan'))
    ols_yield_mae  = ols_metrics.get('yield', {}).get('mae', float('nan'))

    # Check criterion A (absolute targets)
    if not np.isnan(pinn_visc_mae) and not np.isnan(pinn_yield_mae):
        if pinn_yield_mae <= 5.0 and pinn_visc_mae <= 8.0:
            return True, (f'absolute targets met: yield_MAE={pinn_yield_mae:.2f}≤5 '
                          f'visc_MAE={pinn_visc_mae:.2f}≤8')

    # Check criterion B (relative: PINN not worse than OLS by >10%)
    for key, pinn_mae, ols_mae in [
        ('visc', pinn_visc_mae, ols_visc_mae),
        ('yield', pinn_yield_mae, ols_yield_mae),
    ]:
        if np.isnan(pinn_mae) or np.isnan(ols_mae):
            details.append(f"{key}: insufficient data "
                           f"(n={pinn_metrics.get(key, {}).get('n', 0)})")
            continue
        threshold = ols_mae * (1.0 + _QUALITY_GATE_MARGIN)
        if pinn_mae > threshold:
            details.append(
                f"{key} MAE degraded: PINN={pinn_mae:.3f} > OLS×1.1={threshold:.3f}"
            )

    passed = len(details) == 0
    return passed, '; '.join(details) if details else 'relative improvement confirmed'


def _pinn_unavailable_result(ols_metrics: dict, elapsed: float) -> dict:
    return {
        'status': 'unavailable',
        'correction_mode': 'ols',
        'pinn_metrics': {},
        'ols_metrics': ols_metrics,
        'quality_gate': {'passed': False, 'detail': 'PyTorch not installed'},
        'pinn_checkpoint_dir': None,
        'regime_summary': {},
        'training_result': {},
        'n_clusters': 0,
        'elapsed_sec': elapsed,
    }


def _pinn_failed_result(ols_metrics: dict, error: str, elapsed: float) -> dict:
    return {
        'status': 'failed',
        'correction_mode': 'ols',
        'pinn_metrics': {},
        'ols_metrics': ols_metrics,
        'quality_gate': {'passed': False, 'detail': f'Training error: {error}'},
        'pinn_checkpoint_dir': None,
        'regime_summary': {},
        'training_result': {},
        'n_clusters': 0,
        'elapsed_sec': elapsed,
    }
