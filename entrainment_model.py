"""
entrainment_model.py
====================
Asphalt entrainment into DAO model for HPCL PDA Unit.

REVISED MODEL: Asphalt (including asphaltenes) entrained into DAO phase.
This is the critical quality concern — asphaltenes reaching DAO product
damage viscosity index and colour specifications in downstream lube processing.

Physical mechanism:
    In the extractor, asphalt-rich droplets can be mechanically entrained
    upward with the rising solvent/DAO phase. Asphaltenes are particularly
    problematic because they:
        - Degrade DAO colour (dark coloration)
        - Increase DAO viscosity beyond spec
        - Contaminate downstream dewaxing / hydrofinishing units

Empirical model:
    asphalt_in_DAO_fraction = C_entrain / S_O_ratio^n_exp

    Default: C_entrain = 0.015,  n_exp = 1.2  (slightly super-linear: higher
    S/O turbulence increases entrainment less than proportionally)

The entrained asphalt is removed from Phase II and added to Phase I (DAO),
penalising DAO quality (raises asphaltene content of DAO product).
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class EntrainmentParams:
    C_entrain: float = 0.015    # entrainment constant
    n_exp:     float = 1.20     # S/O exponent  (>1 → entrainment falls faster with S/O)

    def validate(self) -> None:
        if self.C_entrain < 0:
            raise ValueError(f"C_entrain must be >= 0, got {self.C_entrain}")
        if self.n_exp <= 0:
            raise ValueError(f"n_exp must be > 0, got {self.n_exp}")


DEFAULT_ENTRAINMENT = EntrainmentParams(C_entrain=0.015, n_exp=1.20)


def apply_entrainment(
    mass_I:        np.ndarray,
    mass_II:       np.ndarray,
    solvent_ratio: float,
    params:        EntrainmentParams = DEFAULT_ENTRAINMENT,
    precip_mask:   np.ndarray = None,
) -> tuple:
    """
    Apply asphalt-into-DAO entrainment correction.

    A fraction of the Phase II (asphalt) mass is transferred to Phase I (DAO),
    representing upward mechanical entrainment of asphalt droplets.

    Precipitable / asphaltene components are preferentially entrained
    (heavier, stickier droplets more easily mechanically trapped).

    Parameters
    ----------
    mass_I        : Phase I (DAO) component masses    [kg, n_comp]
    mass_II       : Phase II (asphalt) component masses [kg, n_comp]
    solvent_ratio : effective S/O at this stage
    params        : EntrainmentParams
    precip_mask   : bool array — if provided, precipitable comps entrain 2x

    Returns
    -------
    (mass_I_net, mass_II_net, entrained_mass)
    """
    params.validate()

    # Base entrainment fraction
    frac = params.C_entrain / max(solvent_ratio, 0.1) ** params.n_exp
    frac = float(np.clip(frac, 0.0, 0.30))   # cap at 30%

    # Precipitable components entrain at 1.5x (asphaltene aggregates)
    entrain_factor = np.ones(len(mass_II))
    if precip_mask is not None:
        entrain_factor[precip_mask] = 1.5

    entrained   = mass_II * frac * entrain_factor
    entrained   = np.minimum(entrained, mass_II)   # cannot entrain more than exists

    mass_I_net  = mass_I  + entrained
    mass_II_net = mass_II - entrained

    return (np.maximum(mass_I_net, 0.0),
            np.maximum(mass_II_net, 0.0),
            entrained)


def asphalt_entrainment_in_dao(
    DAO_yield_gross:       float,
    asphalt_yield_gross:   float,
    solvent_ratio:         float,
    params:                EntrainmentParams = DEFAULT_ENTRAINMENT,
) -> tuple:
    """
    Scalar version: compute entrained asphalt contamination in DAO.

    Returns
    -------
    (DAO_yield_net, asphalt_yield_net, asphalt_in_DAO_wt)
        asphalt_in_DAO_wt : absolute asphalt contamination [wt% of total residue]
    """
    frac = params.C_entrain / max(solvent_ratio, 0.1) ** params.n_exp
    frac = float(np.clip(frac, 0.0, 0.30))

    asphalt_entrained  = asphalt_yield_gross * frac
    DAO_yield_net      = DAO_yield_gross    + asphalt_entrained  # DAO gets contaminated
    asphalt_yield_net  = asphalt_yield_gross - asphalt_entrained

    return float(DAO_yield_net), float(asphalt_yield_net), float(asphalt_entrained)


def entrainment_sensitivity(
    DAO_yield_gross:     float = 35.0,
    asphalt_yield_gross: float = 65.0,
    SO_range:            np.ndarray = None,
    params:              EntrainmentParams = DEFAULT_ENTRAINMENT,
) -> dict:
    if SO_range is None:
        SO_range = np.linspace(3, 16, 50)

    frac_arr   = params.C_entrain / SO_range ** params.n_exp
    contam_arr = asphalt_yield_gross * frac_arr

    return {
        'SO_range':            SO_range,
        'contamination_wt':    contam_arr,
        'DAO_gross':           DAO_yield_gross,
        'asphalt_gross':       asphalt_yield_gross,
        'C_entrain':           params.C_entrain,
    }


if __name__ == '__main__':
    params = DEFAULT_ENTRAINMENT
    for so in [4, 6, 8, 10, 12]:
        _, _, contam = asphalt_entrainment_in_dao(35.0, 65.0, so, params)
        print(f"S/O={so:3d}: asphalt contamination in DAO = {contam:.2f} wt%")
