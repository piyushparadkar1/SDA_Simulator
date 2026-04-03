"""
Direct calibration test — calib_pinn_v4
Bypasses Flask; calls run_full_calibration directly with enable_pinn=True.
"""
import sys, os
sys.stdout = __import__('io').TextIOWrapper(
    sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True
)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

DCS_PATH  = r'C:\Users\piyush\AppData\Local\Temp\tmpft33al6f.xlsx'
LIMS_PATH = r'C:\Users\piyush\AppData\Local\Temp\tmpo8b5e4fi.xlsx'
PROFILE   = 'calib_pinn_v4'

from calibration_engine import run_full_calibration
import json

print(f"\n{'='*60}")
print(f"  calib_pinn_v4 — enable_pinn=True, Phase 1 visc fix")
print(f"{'='*60}\n")

result = run_full_calibration(
    dcs_filepath=DCS_PATH,
    lims_filepath=LIMS_PATH,
    profile_name=PROFILE,
    enable_pinn=True,
)

print("\n" + "="*60)
print("  RESULT SUMMARY")
print("="*60)
# Pull out key metrics
m = result.get('metrics_after', {})
vm = m.get('visc', {})
ym = m.get('yield', {})
print(f"  Visc  MAE={vm.get('mae', float('nan')):.3f} cSt   R2={vm.get('r2', float('nan')):.4f}")
print(f"  Yield MAE={ym.get('mae', float('nan')):.3f} vol%  R2={ym.get('r2', float('nan')):.4f}")
print(f"  correction_mode: {result.get('correction_mode', 'ols')}")

pinn_r = result.get('pinn_result', {})
if pinn_r:
    pm = pinn_r.get('pinn_metrics', {})
    pvm = pm.get('visc', {}) if pm else {}
    pym = pm.get('yield', {}) if pm else {}
    print(f"\n  PINN test metrics:")
    print(f"    Visc  MAE={pvm.get('mae', float('nan')):.3f} cSt   R2={pvm.get('r2', float('nan')):.4f}")
    print(f"    Yield MAE={pym.get('mae', float('nan')):.3f} vol%  R2={pym.get('r2', float('nan')):.4f}")
    tr = pinn_r.get('training_result', {})
    print(f"    best_epoch={tr.get('best_epoch', -1)}  val_loss={tr.get('best_val_loss', float('nan')):.4f}")
    print(f"    status={pinn_r.get('status','?')}  mode={pinn_r.get('correction_mode','?')}")

print("\n" + "="*60 + "\n")
