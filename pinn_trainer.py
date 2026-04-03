"""
Physics-Informed Training Loop for PDA Discrepancy Networks.

Implements a composite loss function that balances:
    L_total = L_data_visc + L_data_yield
              + lambda_mb  * L_mass_balance
              + lambda_mono * L_monotonicity
              + lambda_L2  * L_weight_decay

The physics engine is called ONCE to cache predictions; the PINN
then trains on residuals without calling the physics engine during
gradient computation.

ISO 23247 Entity: digital_twin_core
"""
_ISO23247_ENTITY = 'digital_twin_core'

import logging
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Lazy torch import
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------
DEFAULT_LAMBDAS = {
    'lambda_mb': 1.0,       # was 10.0/5.0 — relaxed; tight lambda caused Hardtanh
                             # saturation at init (all ±clamp) → frozen gradients
    'lambda_mono': 0.1,     # was 1.0 — soft monotonicity nudge, not hard constraint
    'lambda_L2': 0.001,     # was 0.01 — lighter regularization for sparse LIMS data
}

DEFAULT_TRAINING = {
    'max_epochs': 500,       # was 200 — PINN still improving at epoch 50 in tests
    'patience': 80,          # was 20 — generous patience for slow convergence
    'lr': 1e-3,              # initial learning rate
    'lr_factor': 0.5,        # ReduceLROnPlateau factor
    'lr_patience': 10,       # ReduceLROnPlateau patience
    'min_lr': 1e-5,          # minimum learning rate
    'val_fraction': 0.20,    # chronological validation split from training set
    'mono_n_pairs': 50,      # number of S/O-sorted pairs for monotonicity
}


if _TORCH_AVAILABLE:

    class PINNTrainer:
        """
        Trains PINNCorrector networks with physics-informed composite loss.

        Parameters
        ----------
        corrector : PINNCorrector
            The corrector instance containing visc_net and yield_net.
        lambdas : dict, optional
            Penalty weights {lambda_mb, lambda_mono, lambda_L2}.
        training_config : dict, optional
            Training hyperparameters (max_epochs, patience, lr, etc.).
        """

        def __init__(self, corrector, lambdas: Optional[Dict] = None,
                     training_config: Optional[Dict] = None):
            from pinn_network import PINNCorrector
            if not isinstance(corrector, PINNCorrector):
                raise TypeError("corrector must be a PINNCorrector instance")

            self.corrector = corrector
            self.lambdas = {**DEFAULT_LAMBDAS, **(lambdas or {})}
            self.config = {**DEFAULT_TRAINING, **(training_config or {})}
            self._history: List[Dict] = []

        def train(self, features: np.ndarray,
                  physics_visc: np.ndarray,
                  measured_visc: np.ndarray,
                  visc_mask: np.ndarray,
                  physics_yield: np.ndarray,
                  measured_yield: np.ndarray,
                  yield_mask: np.ndarray,
                  so_ratios: np.ndarray) -> Dict:
            """
            Train the PINN corrector on cached physics predictions.

            Parameters
            ----------
            features : np.ndarray, shape (N, input_dim)
                Normalized feature vectors for all training rows.
            physics_visc : np.ndarray, shape (N,)
                Physics engine viscosity predictions (cSt).
            measured_visc : np.ndarray, shape (N,)
                Measured viscosity (cSt). NaN where not available.
            visc_mask : np.ndarray, shape (N,), bool
                True where measured_visc is valid (not NaN, converged).
            physics_yield : np.ndarray, shape (N,)
                Physics engine yield predictions (vol%).
            measured_yield : np.ndarray, shape (N,)
                Measured yield (vol%). NaN where not available.
            yield_mask : np.ndarray, shape (N,), bool
                True where measured_yield is valid.
            so_ratios : np.ndarray, shape (N,)
                S/O ratios for monotonicity penalty construction.

            Returns
            -------
            dict with keys:
                'epochs_trained': int
                'final_loss': float
                'loss_components': dict
                'val_loss': float
                'best_epoch': int
                'history': list of per-epoch dicts
                'elapsed_sec': float
            """
            t0 = time.time()

            # Seed for reproducible initialization — Xavier init result varies with
            # random seed; some seeds saturate Hardtanh immediately (all outputs at
            # ±clamp), which freezes gradients on the first epoch and prevents training.
            torch.manual_seed(42)

            # Chronological validation split (last val_fraction of training data)
            n = len(features)
            n_val = max(int(n * self.config['val_fraction']), 1)
            n_train = n - n_val

            # Convert to tensors
            X_train = torch.tensor(features[:n_train], dtype=torch.float32)
            X_val = torch.tensor(features[n_train:], dtype=torch.float32)

            pv_train = torch.tensor(physics_visc[:n_train], dtype=torch.float32)
            mv_train = torch.tensor(measured_visc[:n_train], dtype=torch.float32)
            vm_train = torch.tensor(visc_mask[:n_train], dtype=torch.bool)

            py_train = torch.tensor(physics_yield[:n_train], dtype=torch.float32)
            my_train = torch.tensor(measured_yield[:n_train], dtype=torch.float32)
            ym_train = torch.tensor(yield_mask[:n_train], dtype=torch.bool)

            so_train = torch.tensor(so_ratios[:n_train], dtype=torch.float32)

            pv_val = torch.tensor(physics_visc[n_train:], dtype=torch.float32)
            mv_val = torch.tensor(measured_visc[n_train:], dtype=torch.float32)
            vm_val = torch.tensor(visc_mask[n_train:], dtype=torch.bool)

            py_val = torch.tensor(physics_yield[n_train:], dtype=torch.float32)
            my_val = torch.tensor(measured_yield[n_train:], dtype=torch.float32)
            ym_val = torch.tensor(yield_mask[n_train:], dtype=torch.bool)

            # Fit normalizer on training features
            self.corrector.fit_normalizer(features[:n_train])
            normalizer = self.corrector.normalizer

            # Yield std for normalization (floored at 1.0)
            if ym_train.sum() > 0:
                sigma_yield = max(float(my_train[ym_train].std()), 1.0)
            else:
                sigma_yield = 1.0

            # Generate monotonicity pairs from training data
            mono_pairs = self._generate_monotonicity_pairs(
                so_train, n_pairs=self.config['mono_n_pairs']
            )

            # Optimizer setup — single optimizer for both networks
            all_params = list(self.corrector.visc_net.parameters()) + \
                         list(self.corrector.yield_net.parameters())
            optimizer = optim.Adam(all_params, lr=self.config['lr'],
                                  weight_decay=0.0)  # L2 handled manually
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min',
                factor=self.config['lr_factor'],
                patience=self.config['lr_patience'],
                min_lr=self.config['min_lr'],
            )

            # Early stopping state
            best_val_loss = float('inf')
            best_epoch = 0
            best_visc_state = None
            best_yield_state = None
            epochs_no_improve = 0

            logger.info("PINN training: %d train / %d val rows, "
                        "%d visc targets, %d yield targets, "
                        "sigma_yield=%.2f",
                        n_train, n_val,
                        int(vm_train.sum()), int(ym_train.sum()),
                        sigma_yield)

            self._history = []

            for epoch in range(self.config['max_epochs']):
                # --- Training step ---
                self.corrector.visc_net.train()
                self.corrector.yield_net.train()
                optimizer.zero_grad()

                X_norm = normalizer(X_train)
                delta_v = self.corrector.visc_net(X_norm).squeeze(-1)
                delta_y = self.corrector.yield_net(X_norm).squeeze(-1)

                loss_components = self._compute_loss(
                    delta_v, delta_y,
                    pv_train, mv_train, vm_train,
                    py_train, my_train, ym_train, sigma_yield,
                    mono_pairs, X_norm,
                )
                total_loss = loss_components['total']
                total_loss.backward()
                optimizer.step()

                # --- Validation step ---
                self.corrector.visc_net.eval()
                self.corrector.yield_net.eval()
                with torch.no_grad():
                    X_val_norm = normalizer(X_val)
                    dv_val = self.corrector.visc_net(X_val_norm).squeeze(-1)
                    dy_val = self.corrector.yield_net(X_val_norm).squeeze(-1)
                    val_components = self._compute_loss(
                        dv_val, dy_val,
                        pv_val, mv_val, vm_val,
                        py_val, my_val, ym_val, sigma_yield,
                        mono_pairs=None, X_norm=None,  # skip physics penalties on val
                    )
                    val_loss = val_components['data_visc'] + val_components['data_yield']

                scheduler.step(val_loss)

                # Record history
                epoch_record = {
                    'epoch': epoch,
                    'train_loss': float(total_loss),
                    'val_loss': float(val_loss),
                    'lr': optimizer.param_groups[0]['lr'],
                }
                for k, v in loss_components.items():
                    if k != 'total':
                        epoch_record[f'train_{k}'] = float(v)
                self._history.append(epoch_record)

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = float(val_loss)
                    best_epoch = epoch
                    best_visc_state = {k: v.clone()
                                       for k, v in self.corrector.visc_net.state_dict().items()}
                    best_yield_state = {k: v.clone()
                                        for k, v in self.corrector.yield_net.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= self.config['patience']:
                    logger.info("Early stopping at epoch %d (best=%d, "
                                "val_loss=%.6f)", epoch, best_epoch, best_val_loss)
                    break

                # Log every 25 epochs
                if epoch % 25 == 0 or epoch == self.config['max_epochs'] - 1:
                    logger.info(
                        "Epoch %3d: train=%.5f  val=%.5f  "
                        "visc=%.5f  yield=%.5f  mb=%.5f  mono=%.5f  "
                        "lr=%.1e",
                        epoch, float(total_loss), float(val_loss),
                        float(loss_components['data_visc']),
                        float(loss_components['data_yield']),
                        float(loss_components['mass_balance']),
                        float(loss_components['monotonicity']),
                        optimizer.param_groups[0]['lr'],
                    )

            # Restore best model
            if best_visc_state is not None:
                self.corrector.visc_net.load_state_dict(best_visc_state)
                self.corrector.yield_net.load_state_dict(best_yield_state)

            self.corrector._trained = True
            elapsed = time.time() - t0

            # Final loss components at best epoch
            final_components = {}
            if best_epoch < len(self._history):
                rec = self._history[best_epoch]
                for k, v in rec.items():
                    if k.startswith('train_'):
                        final_components[k.replace('train_', '')] = v

            result = {
                'epochs_trained': epoch + 1,
                'final_loss': float(self._history[best_epoch]['train_loss'])
                              if best_epoch < len(self._history) else 0.0,
                'val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'loss_components': final_components,
                'elapsed_sec': elapsed,
                'n_train': n_train,
                'n_val': n_val,
                'n_visc_targets': int(vm_train.sum()),
                'n_yield_targets': int(ym_train.sum()),
                'sigma_yield': sigma_yield,
            }

            logger.info("PINN training complete: %d epochs in %.1fs, "
                        "best_epoch=%d, val_loss=%.6f",
                        result['epochs_trained'], elapsed,
                        best_epoch, best_val_loss)

            return result

        def _compute_loss(self, delta_v: 'torch.Tensor', delta_y: 'torch.Tensor',
                          physics_visc: 'torch.Tensor', measured_visc: 'torch.Tensor',
                          visc_mask: 'torch.Tensor',
                          physics_yield: 'torch.Tensor', measured_yield: 'torch.Tensor',
                          yield_mask: 'torch.Tensor', sigma_yield: float,
                          mono_pairs: Optional[List[Tuple[int, int]]],
                          X_norm: Optional['torch.Tensor']) -> Dict[str, 'torch.Tensor']:
            """
            Compute the composite physics-informed loss.

            Parameters
            ----------
            delta_v, delta_y : Tensor (N,)
                Predicted correction factors.
            physics_visc, measured_visc : Tensor (N,)
                Physics and measured viscosity values.
            visc_mask, yield_mask : Tensor (N,) bool
                Validity masks.
            sigma_yield : float
                Yield standard deviation for normalization.
            mono_pairs : list of (i, j) tuples or None
                S/O-sorted pairs for monotonicity penalty.
            X_norm : Tensor or None
                Normalized features (needed for L2 only; passed for interface consistency).

            Returns
            -------
            dict of loss components including 'total'.
            """
            device = delta_v.device
            zero = torch.tensor(0.0, device=device)

            # --- L_data_visc: MSE on log-scale ---
            if visc_mask.sum() > 0:
                corrected_visc = physics_visc[visc_mask] * (1.0 + delta_v[visc_mask])
                corrected_visc = torch.clamp(corrected_visc, min=0.1)
                target_visc = measured_visc[visc_mask]
                target_visc = torch.clamp(target_visc, min=0.1)
                l_data_visc = torch.mean(
                    (torch.log(corrected_visc) - torch.log(target_visc)) ** 2
                )
            else:
                l_data_visc = zero

            # --- L_data_yield: MSE normalized by sigma ---
            if yield_mask.sum() > 0:
                corrected_yield = physics_yield[yield_mask] * (1.0 + delta_y[yield_mask])
                target_yield = measured_yield[yield_mask]
                l_data_yield = torch.mean(
                    ((corrected_yield - target_yield) / sigma_yield) ** 2
                )
            else:
                l_data_yield = zero

            # --- L_mass_balance: penalize corrections that break physics ---
            # Mask to rows with valid (non-NaN) physics_yield before computing any
            # yield-related penalty.  Visc-only rows have NaN physics_yield; without
            # masking, relu(NaN) = NaN propagates through mean() → total loss = NaN.
            valid_y = ~torch.isnan(physics_yield)

            if valid_y.sum() > 0:
                # Penalty 1: corrected yield must stay in [0, 100]
                corrected_y_valid = physics_yield[valid_y] * (1.0 + delta_y[valid_y])
                l_mb_bounds = torch.mean(
                    torch.relu(corrected_y_valid - 100.0) ** 2 +
                    torch.relu(-corrected_y_valid) ** 2
                )
                # Penalty 2: soft-constrain yield |delta| to clamp range
                l_mb_yield_delta = torch.mean(
                    torch.relu(torch.abs(delta_y[valid_y]) - 0.5) ** 2
                )
            else:
                device = delta_y.device
                l_mb_bounds = torch.tensor(0.0, device=device)
                l_mb_yield_delta = torch.tensor(0.0, device=device)

            # Visc delta penalty applies to all rows (visc rows always have valid physics)
            valid_v = ~torch.isnan(physics_visc)
            if valid_v.sum() > 0:
                l_mb_visc_delta = torch.mean(
                    torch.relu(torch.abs(delta_v[valid_v]) - 0.8) ** 2
                )
            else:
                l_mb_visc_delta = torch.tensor(0.0, device=delta_v.device)

            l_mass_balance = l_mb_bounds + l_mb_yield_delta + l_mb_visc_delta

            # --- L_monotonicity: higher S/O → higher corrected yield ---
            # Only apply to pairs where BOTH rows have valid (non-NaN) physics_yield.
            # Visc-only rows have NaN physics_yield; including them produces NaN loss.
            if mono_pairs is not None and len(mono_pairs) > 0:
                valid_y_np = (~torch.isnan(physics_yield)).cpu().numpy().astype(bool)
                clean_pairs = [(lo, hi) for lo, hi in mono_pairs
                               if valid_y_np[lo] and valid_y_np[hi]]
                if len(clean_pairs) > 0:
                    idx_lo = torch.tensor([p[0] for p in clean_pairs],
                                          dtype=torch.long, device=device)
                    idx_hi = torch.tensor([p[1] for p in clean_pairs],
                                          dtype=torch.long, device=device)
                    cy_lo = physics_yield[idx_lo] * (1.0 + delta_y[idx_lo])
                    cy_hi = physics_yield[idx_hi] * (1.0 + delta_y[idx_hi])
                    l_mono = torch.mean(torch.relu(cy_lo - cy_hi) ** 2)
                else:
                    l_mono = zero
            else:
                l_mono = zero

            # --- L_weight_decay: L2 on all network weights ---
            l_L2 = zero
            for net in [self.corrector.visc_net, self.corrector.yield_net]:
                for param in net.parameters():
                    l_L2 = l_L2 + torch.sum(param ** 2)

            # --- Total ---
            total = (l_data_visc + l_data_yield
                     + self.lambdas['lambda_mb'] * l_mass_balance
                     + self.lambdas['lambda_mono'] * l_mono
                     + self.lambdas['lambda_L2'] * l_L2)

            return {
                'total': total,
                'data_visc': l_data_visc,
                'data_yield': l_data_yield,
                'mass_balance': l_mass_balance,
                'monotonicity': l_mono,
                'weight_decay': l_L2,
            }

        def _generate_monotonicity_pairs(self, so_ratios: 'torch.Tensor',
                                         n_pairs: int = 50) -> List[Tuple[int, int]]:
            """
            Generate pairs of indices where so_ratios[i] < so_ratios[j].

            Selects pairs that span a meaningful S/O gap to enforce the
            physical constraint: higher S/O → higher DAO yield.

            Parameters
            ----------
            so_ratios : Tensor, shape (N,)
            n_pairs : int
                Number of pairs to generate.

            Returns
            -------
            list of (lo_idx, hi_idx) tuples
            """
            so_np = so_ratios.numpy()
            sorted_idx = np.argsort(so_np)
            n = len(sorted_idx)

            if n < 4:
                return []

            pairs = []
            # Stratified sampling: pick pairs from different quartiles
            q1 = n // 4
            q3 = 3 * n // 4
            rng = np.random.RandomState(42)

            for _ in range(n_pairs):
                # Pick one from low S/O region, one from high S/O region
                i_lo = sorted_idx[rng.randint(0, q1)]
                i_hi = sorted_idx[rng.randint(q3, n)]
                # Only include if S/O gap is meaningful (> 1.0 difference)
                if so_np[i_hi] - so_np[i_lo] > 1.0:
                    pairs.append((int(i_lo), int(i_hi)))

            logger.info("Generated %d monotonicity pairs (requested %d) "
                        "with S/O gap > 1.0", len(pairs), n_pairs)
            return pairs

        @property
        def history(self) -> List[Dict]:
            """Training history: list of per-epoch loss records."""
            return self._history
