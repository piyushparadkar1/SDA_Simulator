"""
Unsupervised Operating Mode Detection for PDA Simulator.

Uses PCA for dimensionality reduction + Gaussian Mixture Model (GMM)
for soft clustering of DCS telemetry into distinct operating regimes
(e.g., seasonal shifts, crude slate changes, lube vs FCC mode).

Design rationale
----------------
- PCA over SDAE: 9 DCS features with strong linear correlations.
  PCA captures 95%+ variance in 4-5 components. SDAE adds tuning
  complexity (depth, corruption, learning rate) for marginal gain.
- GMM over FCM: GMM provides soft memberships (posterior probabilities)
  plus proper model selection via BIC. FCM requires manual cluster count.
- No LSTM: Regime detection is static clustering, not sequence modeling.
  Temporal smoothing uses rolling-window majority vote if needed.

ISO 23247 Entity: device_communication
"""
_ISO23247_ENTITY = 'device_communication'

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# DCS features used for regime detection — must exist in dcs_hourly DataFrame
REGIME_FEATURES = [
    'so_ratio_a',
    'so_ratio_b',
    't_top_a',
    't_top_b',
    't_bot_a',
    't_bot_b',
    'feed_flow_a',
    'feed_flow_b',
    'propane_temp',
]

# GMM model selection range
MIN_CLUSTERS = 2
MAX_CLUSTERS = 5

# PCA variance retention target
PCA_VARIANCE_TARGET = 0.95

# Minimum cluster size as fraction of total data
MIN_CLUSTER_FRACTION = 0.05

# Temporal smoothing window (hours)
SMOOTHING_WINDOW_HOURS = 3


class RegimeDetector:
    """
    Detects operating regimes from high-frequency DCS telemetry.

    Pipeline: StandardScaler → PCA (95% variance) → GMM (BIC-selected clusters).

    Parameters
    ----------
    max_clusters : int
        Maximum number of clusters to evaluate (2 to max_clusters inclusive).
    pca_variance : float
        Fraction of variance to retain in PCA (default 0.95).
    smoothing_hours : int
        Rolling window for temporal smoothing of cluster labels (0 = disabled).
    """

    def __init__(self, max_clusters: int = MAX_CLUSTERS,
                 pca_variance: float = PCA_VARIANCE_TARGET,
                 smoothing_hours: int = SMOOTHING_WINDOW_HOURS):
        self.max_clusters = max(max_clusters, MIN_CLUSTERS)
        self.pca_variance = pca_variance
        self.smoothing_hours = smoothing_hours

        self._scaler: Optional[StandardScaler] = None
        self._pca_components: Optional[np.ndarray] = None  # manual PCA via SVD
        self._pca_mean: Optional[np.ndarray] = None
        self._n_pca_components: int = 0
        self._explained_variance_ratio: Optional[np.ndarray] = None
        self._gmm: Optional[GaussianMixture] = None
        self._n_clusters: int = 0
        self._bic_scores: Dict[int, float] = {}
        self._cluster_centers: Optional[np.ndarray] = None
        self._cluster_sizes: Optional[np.ndarray] = None
        self._fitted = False

    @property
    def n_clusters(self) -> int:
        """Number of clusters found by BIC model selection."""
        return self._n_clusters

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, dcs_df: pd.DataFrame) -> 'RegimeDetector':
        """
        Fit the regime detection pipeline on DCS hourly data.

        Parameters
        ----------
        dcs_df : pd.DataFrame
            The dcs_hourly DataFrame from plant_data_loader. Must contain
            columns listed in REGIME_FEATURES.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If required columns are missing or data has fewer than 100 valid rows.
        """
        # Validate columns
        missing = [c for c in REGIME_FEATURES if c not in dcs_df.columns]
        if missing:
            raise ValueError(
                f"RegimeDetector.fit: missing DCS columns: {missing}. "
                f"Expected: {REGIME_FEATURES}"
            )

        # Extract feature matrix and drop NaN rows
        X_raw = dcs_df[REGIME_FEATURES].values.astype(np.float64)
        valid_mask = ~np.isnan(X_raw).any(axis=1)
        X_valid = X_raw[valid_mask]

        if len(X_valid) < 100:
            raise ValueError(
                f"RegimeDetector.fit: only {len(X_valid)} valid rows "
                f"(need ≥100). Check DCS data quality."
            )

        logger.info("RegimeDetector.fit: %d valid rows from %d total "
                     "(%.1f%% coverage)",
                     len(X_valid), len(dcs_df),
                     100.0 * len(X_valid) / max(len(dcs_df), 1))

        # Step 1: StandardScaler
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_valid)

        # Step 2: PCA via SVD (no sklearn.decomposition dependency)
        self._pca_mean = X_scaled.mean(axis=0)
        X_centered = X_scaled - self._pca_mean
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Explained variance ratio
        explained_var = (S ** 2) / (len(X_centered) - 1)
        total_var = explained_var.sum()
        self._explained_variance_ratio = explained_var / total_var

        # Select number of components for target variance
        cumvar = np.cumsum(self._explained_variance_ratio)
        self._n_pca_components = int(np.searchsorted(cumvar, self.pca_variance) + 1)
        self._n_pca_components = min(self._n_pca_components, len(REGIME_FEATURES))
        self._pca_components = Vt[:self._n_pca_components]  # (n_comp, n_features)

        X_pca = X_centered @ self._pca_components.T  # (n_samples, n_comp)

        logger.info("PCA: %d components retain %.1f%% variance",
                     self._n_pca_components,
                     100.0 * cumvar[self._n_pca_components - 1])

        # Step 3: GMM model selection by BIC
        best_bic = np.inf
        best_k = MIN_CLUSTERS
        best_gmm = None

        for k in range(MIN_CLUSTERS, self.max_clusters + 1):
            gmm = GaussianMixture(
                n_components=k,
                covariance_type='full',
                n_init=3,
                max_iter=200,
                random_state=42,
            )
            gmm.fit(X_pca)
            bic = gmm.bic(X_pca)
            self._bic_scores[k] = float(bic)
            logger.info("GMM k=%d: BIC=%.1f", k, bic)

            if bic < best_bic:
                best_bic = bic
                best_k = k
                best_gmm = gmm

        self._gmm = best_gmm
        self._n_clusters = best_k

        # Compute cluster statistics
        labels = self._gmm.predict(X_pca)
        self._cluster_sizes = np.bincount(labels, minlength=best_k)
        self._cluster_centers = np.array([
            X_pca[labels == k].mean(axis=0) for k in range(best_k)
        ])

        # Warn if any cluster is too small
        min_size = self._cluster_sizes.min()
        min_frac = min_size / len(X_valid)
        if min_frac < MIN_CLUSTER_FRACTION:
            logger.warning(
                "Cluster %d has only %d rows (%.1f%% of data) — "
                "may be unreliable. Consider reducing max_clusters.",
                int(np.argmin(self._cluster_sizes)), min_size, 100.0 * min_frac
            )

        self._fitted = True
        logger.info("RegimeDetector fitted: %d clusters selected (BIC=%.1f). "
                     "Cluster sizes: %s",
                     best_k, best_bic,
                     ', '.join(f'{s}' for s in self._cluster_sizes))
        return self

    def predict(self, dcs_df: pd.DataFrame,
                smooth: bool = True) -> np.ndarray:
        """
        Predict operating regime cluster labels for each row.

        Parameters
        ----------
        dcs_df : pd.DataFrame
            DCS data rows (must contain REGIME_FEATURES columns).
        smooth : bool
            If True, apply temporal smoothing (rolling majority vote).

        Returns
        -------
        np.ndarray, shape (len(dcs_df),), dtype int
            Cluster labels (0-indexed). -1 for rows with NaN features.
        """
        self._check_fitted()

        X_raw = dcs_df[REGIME_FEATURES].values.astype(np.float64)
        valid_mask = ~np.isnan(X_raw).any(axis=1)

        labels = np.full(len(dcs_df), -1, dtype=np.int32)

        if valid_mask.sum() == 0:
            logger.warning("RegimeDetector.predict: no valid rows")
            return labels

        X_valid = X_raw[valid_mask]
        X_pca = self._transform_to_pca(X_valid)
        labels_valid = self._gmm.predict(X_pca)
        labels[valid_mask] = labels_valid

        if smooth and self.smoothing_hours > 0:
            labels = self._temporal_smooth(labels, dcs_df)

        return labels

    def predict_proba(self, dcs_df: pd.DataFrame) -> np.ndarray:
        """
        Predict soft cluster membership probabilities.

        Parameters
        ----------
        dcs_df : pd.DataFrame

        Returns
        -------
        np.ndarray, shape (len(dcs_df), n_clusters)
            Posterior probabilities per row. Rows with NaN features get
            uniform distribution (1/n_clusters).
        """
        self._check_fitted()

        X_raw = dcs_df[REGIME_FEATURES].values.astype(np.float64)
        valid_mask = ~np.isnan(X_raw).any(axis=1)

        proba = np.full((len(dcs_df), self._n_clusters),
                        1.0 / self._n_clusters, dtype=np.float64)

        if valid_mask.sum() > 0:
            X_valid = X_raw[valid_mask]
            X_pca = self._transform_to_pca(X_valid)
            proba[valid_mask] = self._gmm.predict_proba(X_pca)

        return proba

    def _transform_to_pca(self, X_raw: np.ndarray) -> np.ndarray:
        """Scale and project raw features to PCA space."""
        X_scaled = self._scaler.transform(X_raw)
        X_centered = X_scaled - self._pca_mean
        return X_centered @ self._pca_components.T

    def _temporal_smooth(self, labels: np.ndarray,
                         dcs_df: pd.DataFrame) -> np.ndarray:
        """
        Apply rolling majority vote for temporal smoothing.

        Uses the DataFrame index (assumed datetime) to determine the window.
        Falls back to fixed-count window if index is not datetime.
        """
        if len(labels) < 3:
            return labels

        smoothed = labels.copy()

        # Try datetime-based window
        if hasattr(dcs_df.index, 'freq') or pd.api.types.is_datetime64_any_dtype(dcs_df.index):
            window = f'{self.smoothing_hours}h'
            series = pd.Series(labels, index=dcs_df.index)
            for i in range(len(labels)):
                if labels[i] == -1:
                    continue
                # Get window around this point
                t = dcs_df.index[i]
                mask = (dcs_df.index >= t - pd.Timedelta(window)) & \
                       (dcs_df.index <= t + pd.Timedelta(window))
                window_labels = labels[mask]
                window_labels = window_labels[window_labels >= 0]
                if len(window_labels) > 0:
                    smoothed[i] = int(np.bincount(window_labels).argmax())
        else:
            # Fixed-count window fallback
            half_w = max(self.smoothing_hours, 1)
            for i in range(len(labels)):
                if labels[i] == -1:
                    continue
                lo = max(0, i - half_w)
                hi = min(len(labels), i + half_w + 1)
                window_labels = labels[lo:hi]
                window_labels = window_labels[window_labels >= 0]
                if len(window_labels) > 0:
                    smoothed[i] = int(np.bincount(window_labels).argmax())

        return smoothed

    def _check_fitted(self) -> None:
        """Raise if detector has not been fitted."""
        if not self._fitted:
            raise RuntimeError(
                "RegimeDetector has not been fitted. Call .fit(dcs_df) first."
            )

    def save(self, directory: str) -> str:
        """
        Save fitted detector to directory.

        Parameters
        ----------
        directory : str
            Path to save directory. Created if it doesn't exist.

        Returns
        -------
        str
            Path to the saved state file.
        """
        self._check_fitted()
        os.makedirs(directory, exist_ok=True)
        state_path = os.path.join(directory, 'regime_detector.npz')
        meta_path = os.path.join(directory, 'regime_detector_meta.json')

        # Save numpy arrays
        np.savez(state_path,
                 scaler_mean=self._scaler.mean_,
                 scaler_scale=self._scaler.scale_,
                 pca_mean=self._pca_mean,
                 pca_components=self._pca_components,
                 explained_variance_ratio=self._explained_variance_ratio,
                 gmm_weights=self._gmm.weights_,
                 gmm_means=self._gmm.means_,
                 gmm_covariances=self._gmm.covariances_,
                 gmm_precisions_cholesky=self._gmm.precisions_cholesky_,
                 cluster_centers=self._cluster_centers,
                 cluster_sizes=self._cluster_sizes)

        # Save metadata
        meta = {
            'n_clusters': self._n_clusters,
            'n_pca_components': self._n_pca_components,
            'max_clusters': self.max_clusters,
            'pca_variance': self.pca_variance,
            'smoothing_hours': self.smoothing_hours,
            'bic_scores': self._bic_scores,
            'features': REGIME_FEATURES,
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        logger.info("RegimeDetector saved to %s", state_path)
        return state_path

    @classmethod
    def load(cls, directory: str) -> 'RegimeDetector':
        """
        Load a fitted detector from directory.

        Parameters
        ----------
        directory : str

        Returns
        -------
        RegimeDetector

        Raises
        ------
        FileNotFoundError
            If saved files are missing.
        """
        state_path = os.path.join(directory, 'regime_detector.npz')
        meta_path = os.path.join(directory, 'regime_detector_meta.json')

        if not os.path.exists(state_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"RegimeDetector state not found in {directory}. "
                f"Expected regime_detector.npz and regime_detector_meta.json."
            )

        with open(meta_path, 'r') as f:
            meta = json.load(f)

        data = np.load(state_path)

        detector = cls(
            max_clusters=meta['max_clusters'],
            pca_variance=meta['pca_variance'],
            smoothing_hours=meta['smoothing_hours'],
        )

        # Restore scaler
        detector._scaler = StandardScaler()
        detector._scaler.mean_ = data['scaler_mean']
        detector._scaler.scale_ = data['scaler_scale']
        detector._scaler.var_ = data['scaler_scale'] ** 2
        detector._scaler.n_features_in_ = len(REGIME_FEATURES)

        # Restore PCA
        detector._pca_mean = data['pca_mean']
        detector._pca_components = data['pca_components']
        detector._n_pca_components = meta['n_pca_components']
        detector._explained_variance_ratio = data['explained_variance_ratio']

        # Restore GMM
        n_clusters = meta['n_clusters']
        detector._gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type='full',
        )
        detector._gmm.weights_ = data['gmm_weights']
        detector._gmm.means_ = data['gmm_means']
        detector._gmm.covariances_ = data['gmm_covariances']
        detector._gmm.precisions_cholesky_ = data['gmm_precisions_cholesky']
        detector._gmm.converged_ = True
        detector._gmm.n_iter_ = 0

        detector._n_clusters = n_clusters
        detector._bic_scores = meta.get('bic_scores', {})
        detector._cluster_centers = data['cluster_centers']
        detector._cluster_sizes = data['cluster_sizes']
        detector._fitted = True

        logger.info("RegimeDetector loaded from %s: %d clusters",
                     state_path, n_clusters)
        return detector

    def summary(self) -> Dict:
        """
        Return a summary dict for diagnostics and logging.

        Returns
        -------
        dict with cluster count, sizes, BIC scores, PCA variance.
        """
        self._check_fitted()
        return {
            'n_clusters': self._n_clusters,
            'n_pca_components': self._n_pca_components,
            'pca_variance_retained': float(
                sum(self._explained_variance_ratio[:self._n_pca_components])
            ),
            'bic_scores': {str(k): v for k, v in self._bic_scores.items()},
            'cluster_sizes': [int(s) for s in self._cluster_sizes],
            'cluster_fractions': [
                round(float(s) / max(int(self._cluster_sizes.sum()), 1), 3)
                for s in self._cluster_sizes
            ],
            'features_used': REGIME_FEATURES,
            'smoothing_hours': self.smoothing_hours,
        }
