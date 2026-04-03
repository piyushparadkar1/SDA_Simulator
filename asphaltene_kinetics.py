"""
asphaltene_kinetics.py
======================
Asphaltene precipitation kinetics model for SDA simulation.

The LLE solver predicts the *equilibrium* precipitation amount A_eq.
Actual precipitation at stage conditions follows first-order kinetics:

    dA/dt = k_precip * (A_eq - A)

Discretised over stage residence time τ:

    A_stage = A_prev + k_precip * τ * (A_eq - A_prev)
            = A_eq + (A_prev - A_eq) * exp(-k_precip * τ)

The second form (exact solution to the ODE) is used here, avoiding
numerical instability for large k_precip * τ.

Units for all masses: kg  (or any consistent mass unit)
"""

import numpy as np
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Kinetic parameters
# ---------------------------------------------------------------------------

@dataclass
class KineticParams:
    """Container for asphaltene precipitation kinetic parameters."""
    k_precip: float = 0.5    # precipitation rate constant  [1/s]
    tau:      float = 10.0   # stage residence time          [s]

    @property
    def conversion(self) -> float:
        """Fractional approach to equilibrium = 1 - exp(-k*τ)."""
        return 1.0 - np.exp(-self.k_precip * self.tau)


DEFAULT_KINETICS = KineticParams(k_precip=0.5, tau=10.0)


# ---------------------------------------------------------------------------
# Per-stage precipitation update
# ---------------------------------------------------------------------------

def apply_precipitation_kinetics(
    A_prev:     np.ndarray,    # [kg] precipitable component masses before this stage
    A_eq:       np.ndarray,    # [kg] equilibrium precipitate from LLE
    params:     KineticParams = DEFAULT_KINETICS,
) -> np.ndarray:
    """
    Apply first-order precipitation kinetics to each precipitable component.

    Uses the exact ODE solution:
        A_stage = A_eq + (A_prev - A_eq) * exp(-k_precip * τ)

    Parameters
    ----------
    A_prev : precipitable component masses at inlet of this stage  [kg, n_precip]
    A_eq   : equilibrium precipitate predicted by LLE              [kg, n_precip]
    params : KineticParams with k_precip and tau

    Returns
    -------
    A_stage : np.ndarray  [kg, n_precip] – actual precipitate at stage exit
    """
    A_prev = np.asarray(A_prev, dtype=float)
    A_eq   = np.asarray(A_eq,   dtype=float)

    if A_prev.shape != A_eq.shape:
        raise ValueError(
            f"Shape mismatch: A_prev {A_prev.shape} vs A_eq {A_eq.shape}"
        )

    decay   = np.exp(-params.k_precip * params.tau)
    A_stage = A_eq + (A_prev - A_eq) * decay

    # Physical constraint: precipitation cannot be negative
    A_stage = np.maximum(A_stage, 0.0)
    return A_stage


def precipitation_efficiency(params: KineticParams) -> float:
    """
    Return fractional completion of precipitation  (0 = none, 1 = equilibrium).

    η = 1 - exp(-k_precip * τ)

    At default k=0.5, τ=10: η = 1 - e^(-5) ≈ 0.9933  (99% of equilibrium)
    """
    return 1.0 - np.exp(-params.k_precip * params.tau)


# ---------------------------------------------------------------------------
# Sensitivity: how kinetics affect actual yield at different k and τ
# ---------------------------------------------------------------------------

def kinetics_sensitivity(
    A_eq_total:   float,
    A_prev_total: float = 0.0,
    k_range:      np.ndarray = None,
    tau_range:    np.ndarray = None,
) -> dict:
    """
    Compute actual precipitate fraction at a range of k and τ values.

    Parameters
    ----------
    A_eq_total   : equilibrium precipitate (normalised, e.g. 1.0)
    A_prev_total : initial precipitate (usually 0 at fresh feed)
    k_range      : array of k_precip values to test  [1/s]
    tau_range    : array of tau values to test        [s]

    Returns
    -------
    dict with 'k_range', 'tau_range', 'A_vs_k', 'A_vs_tau'
    """
    if k_range is None:
        k_range = np.logspace(-2, 1, 50)    # 0.01 to 10 s⁻¹
    if tau_range is None:
        tau_range = np.linspace(1, 60, 50)  # 1 to 60 s

    tau_fixed = DEFAULT_KINETICS.tau
    k_fixed   = DEFAULT_KINETICS.k_precip

    A_vs_k   = A_eq_total + (A_prev_total - A_eq_total) * np.exp(-k_range * tau_fixed)
    A_vs_tau = A_eq_total + (A_prev_total - A_eq_total) * np.exp(-k_fixed  * tau_range)

    return {
        'k_range':   k_range,
        'tau_range': tau_range,
        'A_vs_k':    np.maximum(A_vs_k,   0.0),
        'A_vs_tau':  np.maximum(A_vs_tau, 0.0),
    }


if __name__ == '__main__':
    params = KineticParams(k_precip=0.5, tau=10.0)
    eff = precipitation_efficiency(params)
    print(f"Default kinetics: k={params.k_precip} s⁻¹, τ={params.tau} s")
    print(f"  Approach to equilibrium: {eff*100:.2f}%")

    A_eq   = np.array([0.8, 0.6, 0.3])   # equilibrium precipitate [kg]
    A_prev = np.array([0.0, 0.0, 0.0])   # fresh feed
    A_out  = apply_precipitation_kinetics(A_prev, A_eq, params)
    print(f"\nEquilibrium precipitate:  {A_eq}")
    print(f"Actual precipitate:       {A_out.round(4)}")
