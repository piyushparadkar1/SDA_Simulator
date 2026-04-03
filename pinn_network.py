"""
PINN Discrepancy Networks for PDA Simulator Hybrid Digital Twin.

Two lightweight MLPs learn the multiplicative correction factor between
the physics engine predictions and real plant measurements:

    corrected_visc  = physics_visc  * (1 + delta_visc)
    corrected_yield = physics_yield * (1 + delta_yield)

Network sizing is deliberately tiny to prevent overfitting on sparse
LIMS data (~20-30 unique viscosity targets per 2-year window).

ISO 23247 Entity: digital_twin_core
"""
_ISO23247_ENTITY = 'digital_twin_core'

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy torch import — falls back gracefully if PyTorch is not installed
# ---------------------------------------------------------------------------
_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not installed — PINN correction disabled. "
                   "Install with: pip install torch")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Feature names expected by the PINN (6 continuous + up to MAX_CLUSTERS one-hot)
CONTINUOUS_FEATURES = [
    'so_ratio_avg',    # average S/O ratio across trains
    't_top_avg',       # average top temperature [°C]
    't_bot_avg',       # average bottom temperature [°C]
    'feed_density',    # feed density [g/cm³]
    'feed_ccr',        # feed CCR [wt%]
    'delta_t',         # t_top - t_bot [°C]
]
N_CONTINUOUS = len(CONTINUOUS_FEATURES)
MAX_CLUSTERS = 5  # maximum number of operating regime clusters

# Default network hyperparameters
VISC_HIDDEN_DIM = 16   # enlarged from 8 — more capacity for regime-dependent visc
YIELD_HIDDEN_DIM = 32  # enlarged from 16 — needed for wide yield range (10–23 vol%)
VISC_CLAMP = 0.8   # delta_visc clamped to [-0.8, +0.8] → ±80% correction
YIELD_CLAMP = 0.5  # delta_yield clamped to [-0.5, +0.5] → ±50% correction


def is_torch_available() -> bool:
    """Check whether PyTorch is importable."""
    return _TORCH_AVAILABLE


# ═══════════════════════════════════════════════════════════════════════════
# Feature extraction (numpy-only, no torch dependency)
# ═══════════════════════════════════════════════════════════════════════════

def extract_features_from_row(row, n_clusters: int = 0,
                              cluster_id: int = -1) -> np.ndarray:
    """
    Extract the PINN input feature vector from a single plant data row.

    Parameters
    ----------
    row : pd.Series or dict
        A single row from the calibration dataset (visc_anchored or dcs_hourly).
    n_clusters : int
        Total number of operating regime clusters (for one-hot encoding).
        If 0, no cluster features are appended.
    cluster_id : int
        The cluster assignment for this row (0-indexed). Ignored if n_clusters=0.

    Returns
    -------
    np.ndarray, shape (N_CONTINUOUS + n_clusters,)
        Feature vector with continuous features followed by one-hot cluster encoding.
    """
    # Continuous features — use .get() for dict compatibility, fallback to defaults
    _get = row.get if hasattr(row, 'get') else lambda k, d=None: getattr(row, k, d)

    # S/O ratio: average of trains A and B
    so_a = _safe_float(_get('so_ratio_a', None), 8.0)
    so_b = _safe_float(_get('so_ratio_b', None), so_a)
    so_avg = (so_a + so_b) / 2.0

    # Temperatures: average of trains A and B
    t_top_a = _safe_float(_get('t_top_a', None), 82.0)
    t_top_b = _safe_float(_get('t_top_b', None), t_top_a)
    t_top_avg = (t_top_a + t_top_b) / 2.0

    t_bot_a = _safe_float(_get('t_bot_a', None), 67.0)
    t_bot_b = _safe_float(_get('t_bot_b', None), t_bot_a)
    t_bot_avg = (t_bot_a + t_bot_b) / 2.0

    feed_density = _safe_float(_get('feed_density', None), 1.028)
    feed_ccr = _safe_float(_get('feed_ccr', None), 22.8)
    delta_t = t_top_avg - t_bot_avg

    continuous = np.array([so_avg, t_top_avg, t_bot_avg,
                           feed_density, feed_ccr, delta_t],
                          dtype=np.float32)

    # One-hot cluster encoding
    if n_clusters > 0 and 0 <= cluster_id < n_clusters:
        onehot = np.zeros(n_clusters, dtype=np.float32)
        onehot[cluster_id] = 1.0
        return np.concatenate([continuous, onehot])
    elif n_clusters > 0:
        # Unknown cluster — use uniform distribution as soft prior
        onehot = np.full(n_clusters, 1.0 / n_clusters, dtype=np.float32)
        return np.concatenate([continuous, onehot])
    else:
        return continuous


def extract_features_batch(df, n_clusters: int = 0,
                           cluster_ids: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Extract feature matrix for a DataFrame of plant data rows.

    Parameters
    ----------
    df : pd.DataFrame
        Multiple rows from the calibration dataset.
    n_clusters : int
        Number of operating regime clusters.
    cluster_ids : np.ndarray or None
        Cluster assignments aligned with df index. Shape (len(df),).

    Returns
    -------
    np.ndarray, shape (len(df), N_CONTINUOUS + n_clusters)
    """
    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        cid = int(cluster_ids[i]) if cluster_ids is not None else -1
        rows.append(extract_features_from_row(row, n_clusters, cid))
    return np.vstack(rows)


def _safe_float(val, default: float) -> float:
    """Convert value to float, returning default on None/NaN/error."""
    if val is None:
        return default
    try:
        f = float(val)
        if np.isnan(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


# ═══════════════════════════════════════════════════════════════════════════
# Neural Network Definitions (require PyTorch)
# ═══════════════════════════════════════════════════════════════════════════

if _TORCH_AVAILABLE:

    class DiscrepancyNet(nn.Module):
        """
        Small MLP that predicts a multiplicative correction factor delta.

        The output is clamped to [-clamp, +clamp] via hard tanh to prevent
        extreme corrections that would violate physical plausibility.

        Parameters
        ----------
        input_dim : int
            Number of input features (continuous + one-hot cluster).
        hidden_dim : int
            Number of neurons in the single hidden layer.
        clamp : float
            Maximum absolute value of the output delta.
        """

        def __init__(self, input_dim: int, hidden_dim: int, clamp: float):
            super().__init__()
            self.clamp = clamp
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
                nn.Hardtanh(min_val=-clamp, max_val=clamp),
            )
            # Xavier initialization for stable gradients with Tanh
            for layer in self.net:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass.

            Parameters
            ----------
            x : torch.Tensor, shape (batch, input_dim)

            Returns
            -------
            torch.Tensor, shape (batch, 1) — delta values in [-clamp, +clamp]
            """
            return self.net(x)

    class FeatureNormalizer(nn.Module):
        """
        Learned mean/std normalization stored with the model checkpoint.
        Fitted once on training data, then frozen during inference.
        """

        def __init__(self, n_features: int):
            super().__init__()
            self.register_buffer('mean', torch.zeros(n_features))
            self.register_buffer('std', torch.ones(n_features))
            self._fitted = False

        def fit(self, features: np.ndarray) -> 'FeatureNormalizer':
            """
            Compute mean/std from training data.

            Parameters
            ----------
            features : np.ndarray, shape (n_samples, n_features)

            Returns
            -------
            self
            """
            self.mean = torch.tensor(np.nanmean(features, axis=0),
                                     dtype=torch.float32)
            std = np.nanstd(features, axis=0)
            # Floor std at 1e-6 to avoid division by zero
            std = np.where(std < 1e-6, 1.0, std)
            self.std = torch.tensor(std, dtype=torch.float32)
            self._fitted = True
            return self

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Normalize input features using stored mean/std."""
            return (x - self.mean) / self.std

        @property
        def is_fitted(self) -> bool:
            return self._fitted

    class PINNCorrector:
        """
        Wraps viscosity and yield discrepancy networks with feature normalization.

        This is the main interface for PINN-based corrections. It handles:
        - Feature normalization (learned from training data)
        - Viscosity correction prediction
        - Yield correction prediction
        - Checkpoint save/load

        Parameters
        ----------
        n_clusters : int
            Number of operating regime clusters (0 = no clustering).
        visc_hidden : int
            Hidden layer size for viscosity correction net.
        yield_hidden : int
            Hidden layer size for yield correction net.
        visc_clamp : float
            Max absolute correction for viscosity.
        yield_clamp : float
            Max absolute correction for yield.
        """

        def __init__(self, n_clusters: int = 0,
                     visc_hidden: int = VISC_HIDDEN_DIM,
                     yield_hidden: int = YIELD_HIDDEN_DIM,
                     visc_clamp: float = VISC_CLAMP,
                     yield_clamp: float = YIELD_CLAMP):
            self.n_clusters = n_clusters
            input_dim = N_CONTINUOUS + n_clusters

            self.normalizer = FeatureNormalizer(input_dim)
            self.visc_net = DiscrepancyNet(input_dim, visc_hidden, visc_clamp)
            self.yield_net = DiscrepancyNet(input_dim, yield_hidden, yield_clamp)

            self._trained = False
            self._metadata = {
                'n_clusters': n_clusters,
                'visc_hidden': visc_hidden,
                'yield_hidden': yield_hidden,
                'visc_clamp': visc_clamp,
                'yield_clamp': yield_clamp,
                'input_dim': input_dim,
            }

            logger.info("PINNCorrector initialized: input_dim=%d, "
                        "visc_params=%d, yield_params=%d",
                        input_dim, self.param_count_visc, self.param_count_yield)

        @property
        def param_count_visc(self) -> int:
            """Total learnable parameters in viscosity network."""
            return sum(p.numel() for p in self.visc_net.parameters())

        @property
        def param_count_yield(self) -> int:
            """Total learnable parameters in yield network."""
            return sum(p.numel() for p in self.yield_net.parameters())

        @property
        def is_trained(self) -> bool:
            return self._trained

        def fit_normalizer(self, features: np.ndarray) -> None:
            """
            Fit the feature normalizer on training data.

            Parameters
            ----------
            features : np.ndarray, shape (n_samples, input_dim)
            """
            self.normalizer.fit(features)

        def predict_visc_correction(self, features: np.ndarray) -> float:
            """
            Predict viscosity correction delta for a single row.

            Parameters
            ----------
            features : np.ndarray, shape (input_dim,)
                Feature vector from extract_features_from_row().

            Returns
            -------
            float
                delta_visc in [-visc_clamp, +visc_clamp].
            """
            self.visc_net.eval()
            with torch.no_grad():
                x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                x = self.normalizer(x)
                delta = self.visc_net(x).item()
            return delta

        def predict_yield_correction(self, features: np.ndarray) -> float:
            """
            Predict yield correction delta for a single row.

            Parameters
            ----------
            features : np.ndarray, shape (input_dim,)
                Feature vector from extract_features_from_row().

            Returns
            -------
            float
                delta_yield in [-yield_clamp, +yield_clamp].
            """
            self.yield_net.eval()
            with torch.no_grad():
                x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                x = self.normalizer(x)
                delta = self.yield_net(x).item()
            return delta

        def predict_batch(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """
            Predict corrections for a batch of rows.

            Parameters
            ----------
            features : np.ndarray, shape (n_rows, input_dim)

            Returns
            -------
            delta_visc : np.ndarray, shape (n_rows,)
            delta_yield : np.ndarray, shape (n_rows,)
            """
            self.visc_net.eval()
            self.yield_net.eval()
            with torch.no_grad():
                x = torch.tensor(features, dtype=torch.float32)
                x_norm = self.normalizer(x)
                d_visc = self.visc_net(x_norm).squeeze(-1).numpy()
                d_yield = self.yield_net(x_norm).squeeze(-1).numpy()
            return d_visc, d_yield

        def save(self, directory: str) -> str:
            """
            Save PINN checkpoint (networks + normalizer + metadata) to directory.

            Parameters
            ----------
            directory : str
                Path to save directory. Created if it doesn't exist.

            Returns
            -------
            str
                Path to the saved checkpoint file.
            """
            os.makedirs(directory, exist_ok=True)
            checkpoint_path = os.path.join(directory, 'pinn_checkpoint.pt')
            meta_path = os.path.join(directory, 'pinn_metadata.json')

            checkpoint = {
                'visc_net_state': self.visc_net.state_dict(),
                'yield_net_state': self.yield_net.state_dict(),
                'normalizer_state': self.normalizer.state_dict(),
                'trained': self._trained,
            }
            torch.save(checkpoint, checkpoint_path)

            with open(meta_path, 'w') as f:
                json.dump(self._metadata, f, indent=2)

            logger.info("PINN checkpoint saved to %s", checkpoint_path)
            return checkpoint_path

        @classmethod
        def load(cls, directory: str) -> 'PINNCorrector':
            """
            Load PINN checkpoint from directory.

            Parameters
            ----------
            directory : str
                Path to directory containing pinn_checkpoint.pt and pinn_metadata.json.

            Returns
            -------
            PINNCorrector
                Loaded and ready-to-use corrector.

            Raises
            ------
            FileNotFoundError
                If checkpoint files are missing.
            """
            meta_path = os.path.join(directory, 'pinn_metadata.json')
            checkpoint_path = os.path.join(directory, 'pinn_checkpoint.pt')

            if not os.path.exists(meta_path) or not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    f"PINN checkpoint not found in {directory}. "
                    f"Expected pinn_checkpoint.pt and pinn_metadata.json."
                )

            with open(meta_path, 'r') as f:
                metadata = json.load(f)

            corrector = cls(
                n_clusters=metadata['n_clusters'],
                visc_hidden=metadata['visc_hidden'],
                yield_hidden=metadata['yield_hidden'],
                visc_clamp=metadata['visc_clamp'],
                yield_clamp=metadata['yield_clamp'],
            )

            checkpoint = torch.load(checkpoint_path, weights_only=True)
            corrector.visc_net.load_state_dict(checkpoint['visc_net_state'])
            corrector.yield_net.load_state_dict(checkpoint['yield_net_state'])
            corrector.normalizer.load_state_dict(checkpoint['normalizer_state'])
            corrector._trained = checkpoint.get('trained', True)

            logger.info("PINN checkpoint loaded from %s (trained=%s)",
                        checkpoint_path, corrector._trained)
            return corrector

        def summary(self) -> Dict:
            """
            Return a summary dict for diagnostics and logging.

            Returns
            -------
            dict with network sizes, parameter counts, training status.
            """
            return {
                'n_clusters': self.n_clusters,
                'input_dim': self._metadata['input_dim'],
                'visc_net': {
                    'hidden_dim': self._metadata['visc_hidden'],
                    'clamp': self._metadata['visc_clamp'],
                    'param_count': self.param_count_visc,
                },
                'yield_net': {
                    'hidden_dim': self._metadata['yield_hidden'],
                    'clamp': self._metadata['yield_clamp'],
                    'param_count': self.param_count_yield,
                },
                'total_params': self.param_count_visc + self.param_count_yield,
                'trained': self._trained,
                'normalizer_fitted': self.normalizer.is_fitted,
            }
