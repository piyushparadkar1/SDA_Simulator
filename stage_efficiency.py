"""
stage_efficiency.py
===================
Murphree stage efficiency model for the SDA extraction column.

The Murphree efficiency E_M corrects the ideal (equilibrium) separation
achieved by LLE to account for imperfect mixing, axial dispersion, and
mass-transfer limitations within each extractor stage.

Murphree efficiency definition (for the DAO-rich Phase I):
    E_M = (w_I_actual - w_I_in) / (w_I_eq - w_I_in)

Rearranged to give the actual exit composition:
    w_I_actual = w_I_in + E_M * (w_I_eq - w_I_in)

where:
    w_I_in  : composition of Phase I entering the stage  (from stage below)
    w_I_eq  : equilibrium composition from the LLE solver
    E_M     : Murphree efficiency  (0 = no separation, 1 = full equilibrium)
"""

import numpy as np
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Efficiency parameter container
# ---------------------------------------------------------------------------

@dataclass
class StageEfficiency:
    """Murphree stage efficiency configuration."""
    E_murphree: float = 0.70   # dimensionless, 0–1

    def validate(self) -> None:
        if not (0.0 < self.E_murphree <= 1.0):
            raise ValueError(
                f"Murphree efficiency must be in (0, 1], got {self.E_murphree}"
            )


DEFAULT_EFFICIENCY = StageEfficiency(E_murphree=0.70)


# ---------------------------------------------------------------------------
# Apply efficiency correction
# ---------------------------------------------------------------------------

def apply_stage_efficiency(
    mass_I_eq:   np.ndarray,   # [kg] Phase I mass per component at equilibrium
    mass_II_eq:  np.ndarray,   # [kg] Phase II mass per component at equilibrium
    mass_I_in:   np.ndarray,   # [kg] Phase I mass per component entering from below
    efficiency:  StageEfficiency = DEFAULT_EFFICIENCY,
) -> tuple:
    """
    Apply Murphree efficiency to correct equilibrium stage results.

    The actual Phase I exit is interpolated between the inlet Phase I
    (no separation) and the equilibrium Phase I (perfect separation):

        mass_I_actual  = mass_I_in  + E_M * (mass_I_eq  - mass_I_in)
        mass_II_actual = total_feed - mass_I_actual

    Parameters
    ----------
    mass_I_eq  : equilibrium Phase I masses  [kg, n_comp]
    mass_II_eq : equilibrium Phase II masses [kg, n_comp]
    mass_I_in  : Phase I masses entering this stage from the stage below
    efficiency : StageEfficiency object

    Returns
    -------
    (mass_I_actual, mass_II_actual) : tuple of np.ndarray
    """
    efficiency.validate()
    E = efficiency.E_murphree

    total_feed   = mass_I_eq + mass_II_eq       # conserved total at this stage

    mass_I_actual  = mass_I_in + E * (mass_I_eq - mass_I_in)
    mass_I_actual  = np.maximum(mass_I_actual, 0.0)
    mass_I_actual  = np.minimum(mass_I_actual, total_feed)

    mass_II_actual = total_feed - mass_I_actual
    mass_II_actual = np.maximum(mass_II_actual, 0.0)

    return mass_I_actual, mass_II_actual


def effective_stages(N_actual: int, E_murphree: float) -> float:
    """
    Compute number of theoretical (equilibrium) stages equivalent to
    N actual stages each with Murphree efficiency E_M.

    For a linear equilibrium curve (Kremser approximation):
        N_theoretical = N_actual * E_M

    (Simplified; rigorous calculation requires operating/equilibrium line slopes.)
    """
    return N_actual * E_murphree


def efficiency_sensitivity(
    N_stages:    int   = 3,
    base_yield:  float = 45.0,      # DAO yield at E=1.0  [wt%]
    E_range:     np.ndarray = None,
) -> dict:
    """
    Approximate DAO yield as a function of Murphree efficiency.

    Uses the simple linear approximation:
        DAO_yield(E) ≈ base_yield * E  (bounded at 100%)

    Returns dict for plotting.
    """
    if E_range is None:
        E_range = np.linspace(0.3, 1.0, 50)

    yield_vs_E = np.minimum(base_yield * E_range / 0.70, 100.0)

    return {
        'E_range':    E_range,
        'yield_vs_E': yield_vs_E,
        'N_stages':   N_stages,
    }


if __name__ == '__main__':
    eff = StageEfficiency(E_murphree=0.70)
    eff.validate()

    # Example: equilibrium splits 60/40, stage inlet is 40/60
    mass_I_eq  = np.array([0.60, 0.30, 0.05])
    mass_II_eq = np.array([0.40, 0.70, 0.95])
    mass_I_in  = np.array([0.40, 0.30, 0.05])   # entering from below

    m_I, m_II = apply_stage_efficiency(mass_I_eq, mass_II_eq, mass_I_in, eff)

    print(f"Murphree efficiency = {eff.E_murphree}")
    print(f"Equilibrium Phase I mass : {mass_I_eq}")
    print(f"Actual Phase I mass      : {m_I.round(4)}")
    print(f"Actual Phase II mass     : {m_II.round(4)}")
