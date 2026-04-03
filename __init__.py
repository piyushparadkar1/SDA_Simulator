"""
pda_simulator
=============
Propane Deasphalting (PDA) unit simulator.

Modules
-------
residue_distribution   : continuous mixture representation of vacuum residue
phct_eos               : PHCT-like equation of state (density, fugacity, μ)
lle_solver             : liquid-liquid equilibrium solver (Rachford-Rice)
asphaltene_kinetics    : first-order precipitation kinetics
stage_efficiency       : Murphree stage efficiency correction
entrainment_model      : DAO entrainment loss model
hunter_nash_extractor  : Hunter-Nash countercurrent extractor
sensitivity_analysis   : parameter sensitivity sweeps and plotting
run_simulation         : main entry point
"""

from .residue_distribution  import build_residue_distribution, distribution_summary
from .lle_solver             import solve_lle, K_value
from .asphaltene_kinetics    import KineticParams, apply_precipitation_kinetics
from .stage_efficiency       import StageEfficiency, apply_stage_efficiency
from .entrainment_model      import EntrainmentParams, apply_entrainment
from .hunter_nash_extractor  import run_extractor
