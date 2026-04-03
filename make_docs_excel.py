"""
Generate PDA_Calibration_Documentation.xlsx — project summary workbook.
Run once: python make_docs_excel.py
"""
import pandas as pd, os

out = 'PDA_Calibration_Documentation.xlsx'
writer = pd.ExcelWriter(out, engine='openpyxl')

# ── Sheet 1: Calibration Runs ─────────────────────────────────────────────────
runs = [
    dict(Run='Pre-calib baseline', K_mult=1.000, visc_slope=0.300, visc_bias=25.77,
         enable_pinn=False, Phase2_skipped=True,
         Visc_MAE_cSt=3.19, Visc_R2=-0.521, Yield_MAE_vol=3.81, Yield_R2=-0.330,
         best_epoch='N/A', val_loss='N/A',
         Notes='K=1.0 + Phase 1 OLS. Best directional result. Visc MAE meets target.'),
    dict(Run='calib_ols_v1', K_mult=0.8005, visc_slope=0.300, visc_bias=25.77,
         enable_pinn=False, Phase2_skipped=False,
         Visc_MAE_cSt=11.5, Visc_R2=-13.30, Yield_MAE_vol=10.1, Yield_R2=-2.10,
         best_epoch='N/A', val_loss='N/A',
         Notes='Phase 2 K-opt converged to 0.80 (dead zone). Worse than pre-calib on ALL metrics.'),
    dict(Run='calib_pinn_v1', K_mult=0.856, visc_slope=1.000, visc_bias=11.67,
         enable_pinn=True, Phase2_skipped=False,
         Visc_MAE_cSt=8.17, Visc_R2=-7.59, Yield_MAE_vol=4.38, Yield_R2=-0.113,
         best_epoch=0, val_loss='NaN (bug)',
         Notes='NaN loss bug: mass-balance penalty computed on NaN physics rows. best_epoch=0 = random init.'),
    dict(Run='calib_pinn_v2', K_mult=0.856, visc_slope=1.000, visc_bias=11.67,
         enable_pinn=True, Phase2_skipped=False,
         Visc_MAE_cSt=8.17, Visc_R2=-7.59, Yield_MAE_vol=4.38, Yield_R2=-0.113,
         best_epoch=0, val_loss='NaN (bug)',
         Notes='Same as v1. NaN bugs not yet fixed.'),
    dict(Run='calib_pinn_v3', K_mult=0.856, visc_slope=1.000, visc_bias=11.67,
         enable_pinn=True, Phase2_skipped=False,
         Visc_MAE_cSt=8.17, Visc_R2=-7.59, Yield_MAE_vol=4.38, Yield_R2=-0.113,
         best_epoch=499, val_loss=4.705,
         Notes='NaN fixed, training converges (best_epoch=499). Metrics unchanged: Phase 1.5 visc inconsistent with PINN K=1.0 override.'),
    dict(Run='calib_pinn_v4', K_mult=0.856, visc_slope=0.300, visc_bias=25.77,
         enable_pinn=True, Phase2_skipped=False,
         Visc_MAE_cSt=3.146, Visc_R2=-0.493, Yield_MAE_vol=4.378, Yield_R2=-0.113,
         best_epoch=46, val_loss='NaN',
         Notes='Phase 1 visc params passed to PINN. Visc MAE recovers to 3.15 cSt. Yield MAE unchanged. PINN trained but corrections do not improve R2.'),
    dict(Run='calib_pinn_v5 (DEPLOYED)', K_mult=1.000, visc_slope=0.300, visc_bias=25.77,
         enable_pinn=True, Phase2_skipped=True,
         Visc_MAE_cSt=3.146, Visc_R2=-0.493, Yield_MAE_vol=4.378, Yield_R2=-0.113,
         best_epoch=36, val_loss='NaN',
         Notes='Option C: Phase 2 skipped, K=1.0 locked. Identical to v4 (v4 already used K=1.0 PINN override). Confirms: PINN cannot achieve R2>0 on current data.'),
]
df_runs = pd.DataFrame(runs)
df_runs.to_excel(writer, sheet_name='Calibration_Runs', index=False)

# ── Sheet 2: Dataset Summary ──────────────────────────────────────────────────
ds = [
    ('Plant', 'HPCL Mumbai Refinery, Plant 41 — Propane Deasphalting Unit (PDA/SDA)'),
    ('Trains', 'Two parallel extractors: Train A, Train B'),
    ('DCS total rows', '14,363 hourly rows'),
    ('After steady-state filter', '~10,624 rows (T std < 1.5 deg C over 2-hr rolling window)'),
    ('LIMS visc measurements (visc_anchored)', '13,159 rows (dao_visc_100, 24-hr tolerance match)'),
    ('Date range', 'May 2024 to March 2026 (~22 months)'),
    ('Train/Test split method', 'Chronological 80/20 — NEVER random shuffle on time-series'),
    ('Split date', '2025-11-21'),
    ('Train visc rows used', '200 (stratified subsample from 10,527)'),
    ('Train yield rows used', '500 (stratified subsample from 10,624 DCS)'),
    ('Test visc rows', '2,632'),
    ('Test yield rows (DCS)', '2,873'),
    ('Feed density LIMS samples', '~14 per year (forward-filled)'),
    ('Feed CCR LIMS samples', '~10 per year (forward-filled)'),
    ('DAO visc_100 LIMS samples', '~14 per year (NOT forward-filled — calibration target)'),
    ('Feed cache entries', '271 unique compositions (density, CCR, visc135 keyed)'),
]
pd.DataFrame(ds, columns=['Item','Value']).to_excel(writer, sheet_name='Dataset_Summary', index=False)

# ── Sheet 3: Targets vs Achieved ─────────────────────────────────────────────
tgt = [
    ('Viscosity MAE', '<=5 cSt', '3.146 cSt (v4/v5)', 'MET', 'calib_pinn_v4/v5'),
    ('Viscosity R2', '>=0.40', '-0.493', 'NOT MET', 'Structurally unachievable — see RCA'),
    ('Yield MAE', '<=3 vol%', '3.81 vol% (pre-calib)', 'NOT MET', 'K=1.0 + Phase 1 OLS baseline'),
    ('Yield R2', '>=0.50', '-0.113', 'NOT MET', 'Structurally unachievable — see RCA'),
    ('Simulation directional accuracy', 'Yield 15-25%, Visc 25-40 cSt at typical DCS', 'Yield 23.5%, Visc 35 cSt (K=1.0)', 'MET', 'calib_pinn_v5'),
    ('Profile load: dead-zone prevention', 'No yield collapse at T_bot>=68C', 'K_mult>=0.87 enforced in all deployed profiles', 'MET', 'All profiles updated 2026-04-02'),
]
pd.DataFrame(tgt, columns=['Metric','Target','Best_Achieved','Status','Notes']).to_excel(writer, sheet_name='Targets_vs_Achieved', index=False)

# ── Sheet 4: Root Cause Analysis ─────────────────────────────────────────────
rca = [
    ('K_mult dead zone',
     'Optimizer converges to K~0.80-0.86. At T_bot>=68C with K<0.87, Rachford-Rice flash collapses (yield<2%). Physics model is singular near this K range for high-temperature plant operation.',
     'Post-calib visc MAE=11.5 cSt, yield MAE=10.1 vol% — WORSE than pre-calib. UI shows 2% yield.',
     'enable_pinn=True now locks K=1.0 (Phase 2 skipped). All default profiles updated to K>=0.87.',
     'Re-anchor K-value correlation to actual Plant 41 crude fractions. Requires SARA analysis of feed.'),
    ('Negative R2 — structural ceiling',
     'Physics model yield direction anti-correlated with test set (Nov25-Mar26). When plant yield rises, model predicts fall. Root cause: K-values use generic propane/crude correlation; plant feed quality swings (API, SARA) not captured.',
     'R2 remains negative for ALL calibration runs (v1-v5). No PINN or parametric correction achieves R2>0.',
     'None — confirmed structural after exhausting PINN options.',
     'Monthly LIMS sampling of feed API gravity + SARA fractions as K-value model inputs.'),
    ('PINN NaN training loss',
     'Mass-balance and monotonicity penalties computed over ALL rows. Visc-only rows have physics_yield=NaN; yield-only rows have physics_visc=NaN. relu(NaN)=NaN; mean([...NaN...])=NaN. Loss=NaN from epoch 0.',
     'PINN training stalled at epoch 0 (v1, v2). Random-init model used as best checkpoint.',
     'Masked penalties to valid_y and valid_v rows. Added torch.manual_seed(42). Relaxed lambda_mb 10->1, lambda_mono 1->0.1.',
     'Fixed. Training now converges (best_epoch=36-499). val_loss=NaN in v4/v5 is monitoring artifact (val set NaN, not model NaN).'),
    ('Phase 1.5 visc / PINN inconsistency',
     'Phase 1.5 re-runs OLS visc at K=0.856 (slope=1.0, bias=11.67). PINN forces K=1.0 in physics cache. At K=1.0, slope=1.0/bias=11.67 gives physics_visc~31 cSt but PINN learns large corrections that do not generalise.',
     'PINN visc MAE degraded from 3.2 to 8.17 cSt in v1-v3.',
     'Phase 1 visc params (slope=0.30, bias=25.77 at K=1.0) now passed to PINN separately. In v5, Phase 2 skipped entirely so no Phase 1.5 inconsistency.',
     'Fixed in v4/v5.'),
    ('Sparse LIMS feed quality',
     '~14 feed density measurements and ~10 CCR measurements per year, both forward-filled with constants. K-values depend on feed fractions but those fractions do not vary in the model.',
     'Yield direction cannot be predicted correctly. Model has no signal for crude quality swings.',
     'None possible without additional data.',
     'Regular LIMS sampling: monthly API gravity and SARA fractions of deasphalter feed.'),
]
pd.DataFrame(rca, columns=['Issue','Description','Impact','Fix_Applied','Permanent_Fix']).to_excel(writer, sheet_name='Root_Cause_Analysis', index=False)

# ── Sheet 5: Profile Status ───────────────────────────────────────────────────
profs = [
    ('calib_pinn_v5', 1.000, 0.300, 25.77, 0.70, 0.015, 2.5, 'DEPLOYED', 'Best plant calibration. Thermal + Phase 1 OLS + PINN.'),
    ('plant_calibration_v1', 1.000, 0.300, 25.77, 0.70, 0.015, 2.5, 'Updated to v5 params', 'Was K=0.8005 (dead zone). Fixed 2026-04-02.'),
    ('sda_default', 1.000, None, None, 0.70, 0.015, 2.5, 'OK', 'Generic default. No visc correction. Raw physics visc shown.'),
    ('sda_fcc_dao', 1.037, None, None, 0.72, 0.015, 2.3, 'OK', 'FCC-mode DAO. Higher K, lower T.'),
    ('sda_lube_dao', 0.950, None, None, 0.70, 0.015, 2.8, 'Fixed (was 0.581)', 'K=0.581 valid in v1.8. Dead zone in v2.0 after Step-13 K re-anchor. Fixed to 0.95.'),
    ('step1_test', 0.930, None, None, 0.71, 0.039, 3.18, 'Fixed (was 0.853)', 'v1.7 synthetic calibration. K=0.8533 hits dead zone at T_bot>=68C. Fixed to 0.93.'),
    ('v17_test', 0.937, None, None, 0.56, 0.007, 0.50, 'OK', 'Best v1.7 result. yield_MAE=2.35% on 16 synthetic Basra-Kuwait blend points.'),
]
pd.DataFrame(profs, columns=['Profile','K_mult','visc_slope','visc_bias','E_murphree','C_entrain','delta_crit','Status','Notes']).to_excel(writer, sheet_name='Profile_Status', index=False)

# ── Sheet 6: PINN Architecture ────────────────────────────────────────────────
pinn = [
    ('Architecture', 'Multiplicative correction: corrected = physics * (1 + delta). delta in [-clamp, +clamp].'),
    ('Visc network', '3-layer MLP: [N_cont + N_clusters] -> 16 -> 16 -> 1. Hardtanh clamp +-0.8.'),
    ('Yield network', '3-layer MLP: [N_cont + N_clusters] -> 32 -> 32 -> 1. Hardtanh clamp +-0.5.'),
    ('Visc params', '89 trainable parameters'),
    ('Yield params', '177 trainable parameters'),
    ('Input features', '7 continuous (T_bot, T_top, so_ratio, propane_purity, pressure, feed_density, feed_visc_100) + N_clusters one-hot'),
    ('Regime detection', 'RegimeDetector: StandardScaler -> PCA -> GMM (BIC model selection). n_clusters=5 for Plant 41.'),
    ('Training loss', 'MSE_data + lambda_mb * L_mass_balance + lambda_mono * L_monotonicity + lambda_L2 * L2_reg'),
    ('lambda_mb', '1.0 (was 10.0 — relaxed to prevent Hardtanh saturation at init)'),
    ('lambda_mono', '0.1 (was 1.0)'),
    ('lambda_L2', '0.001 (was 0.01)'),
    ('max_epochs', '500 with patience=80'),
    ('Random seed', 'torch.manual_seed(42) for reproducibility'),
    ('Quality gate', 'PINN deployed only if visc_MAE<=8.0 AND yield_MAE<=5.0 (absolute) OR <=10% worse than OLS (relative)'),
    ('Checkpoint', 'calibration_profiles/calib_pinn_v5/pinn_checkpoint.pt + regime_detector.npz'),
]
pd.DataFrame(pinn, columns=['Item','Value']).to_excel(writer, sheet_name='PINN_Architecture', index=False)

writer.close()
print(f'Written: {os.path.abspath(out)}')
