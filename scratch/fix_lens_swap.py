import json
import sys
import os
import tempfile
from scipy.optimize import minimize

sys.path.append('c:/Users/Juan/Documents/GitHub/2D_Optics')
from optics_app.engine.tracer import OpticalSystem, Ray
import numpy as np

with open('c:/Users/Juan/Desktop/spectrometer_raman_50mm_100mm.json') as f:
    config = json.load(f)

surfs = config['surfaces']

# ============================================================
# 1. Fix Collection lens (S3-5): 30mm → 50mm (AC254-050-C)
#    True Thorlabs AC254-050-C prescription:
#    R1=22.9  t=9.0  N-LAK22
#    R2=-25.9 t=1.8  N-SF6
#    R3=194.5 (BFL = 41.2 mm)
# ============================================================
surfs[3]['name']      = 'Collect_AC254-050-C_S1'
surfs[3]['radius']    = 22.9
surfs[3]['thickness'] = 9.0
surfs[3]['glass']     = 'N-LAK22'

surfs[4]['name']      = 'Collect_AC254-050-C_S2'
surfs[4]['radius']    = -25.9
surfs[4]['thickness'] = 1.8
surfs[4]['glass']     = 'N-SF6'

surfs[5]['name']      = 'Collect_AC254-050-C_S3'
surfs[5]['radius']    = 194.5
surfs[5]['glass']     = 'AIR'
surfs[5]['thickness'] = 41.2   # BFL of 50mm lens → slit distance

# Slit-to-collimator distance: symmetric, same as current
# The FFL of the reversed 30mm collimator is ~19.31 mm (from original), keep it
surfs[6]['thickness'] = 19.31  # Slit → Collimator front

# ============================================================
# 2. Fix Focusing lens (S15-17): 50mm → 30mm (AC254-030-C)
#    True Thorlabs AC254-030-C prescription:
#    R1=19.4  t=7.0  N-BAF10
#    R2=-14.2 t=2.0  F2
#    R3=-64.3 (BFL ≈ 23.5 mm)
# ============================================================
surfs[15]['name']      = 'Focusing_AC254-030-C_S1'
surfs[15]['radius']    = 19.4
surfs[15]['thickness'] = 7.0
surfs[15]['glass']     = 'N-BAF10'

surfs[16]['name']      = 'Focusing_AC254-030-C_S2'
surfs[16]['radius']    = -14.2
surfs[16]['thickness'] = 2.0
surfs[16]['glass']     = 'F2'

surfs[17]['name']      = 'Focusing_AC254-030-C_S3'
surfs[17]['radius']    = -64.3
surfs[17]['glass']     = 'AIR'
surfs[17]['thickness'] = 23.54  # BFL of 30mm lens → DMD

# ============================================================
# 3. Re-optimize back-end distances: DMD-to-relay, relay-to-cond, cond-to-det
# ============================================================
print("Testing initial trace with fixed lenses...")
with tempfile.NamedTemporaryFile('w', delete=False, suffix='.json') as tf:
    json.dump(config, tf)
    tmp = tf.name

sys_t = OpticalSystem.from_json(tmp)
ys = []
for wl in [1.137, 1.227, 1.316]:
    tr = sys_t.trace_ray(Ray(pos=(0,0,0), dir=(0,0,1), wavelength=wl))
    y = tr.history[-1]['y_global']
    print(f'  WL {wl} -> detector Y = {y:.4f} mm  (vignetted={tr.vignetted})')
    ys.append(y)
print(f'  Spread = {max(ys)-min(ys):.4f} mm')
os.remove(tmp)

# Optimize [dmd_thickness, relay_to_cond, cond_to_det] for smallest spread
def evaluate(x):
    config['surfaces'][18]['thickness'] = x[0]   # DMD → CB
    config['surfaces'][22]['thickness'] = x[1]   # Relay S3 → Condenser
    config['surfaces'][24]['thickness'] = x[2]   # Condenser S2 → detector
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.json') as tf:
        json.dump(config, tf); tmp = tf.name
    sys_temp = OpticalSystem.from_json(tmp)
    ys = []
    for wl in [1.137, 1.227, 1.316]:
        tr = sys_temp.trace_ray(Ray(pos=(0,0,0), dir=(0,0,1), wavelength=wl))
        if len(tr.history) < 26:
            os.remove(tmp); return 1e6
        ys.append(tr.history[-1]['y_global'])
    spread = max(ys) - min(ys)
    tr_m = sys_temp.trace_ray(Ray(pos=(0, 0.005, 0), dir=(0,0,1), wavelength=1.225))
    if len(tr_m.history) < 26:
        os.remove(tmp); return 1e6
    defocus = abs(tr_m.history[-1]['y_global'] - ys[1])
    os.remove(tmp)
    return defocus * 10000 + max(0, spread - 0.79) * 500

print("\nOptimizing back-end distances...")
res = minimize(evaluate, [55, 26, 7.6], method='Nelder-Mead')
print(f"  Best [dmd_t, relay_cond, cond_det] = {res.x}")
print(f"  Penalty = {res.fun:.6f}")

config['surfaces'][18]['thickness'] = res.x[0]
config['surfaces'][22]['thickness'] = res.x[1]
config['surfaces'][24]['thickness'] = res.x[2]

with tempfile.NamedTemporaryFile('w', delete=False, suffix='.json') as tf:
    json.dump(config, tf); tmp = tf.name
sys_t = OpticalSystem.from_json(tmp)
ys = []
print("\nFinal trace verification:")
for wl in [1.137, 1.227, 1.316]:
    tr = sys_t.trace_ray(Ray(pos=(0,0,0), dir=(0,0,1), wavelength=wl))
    y = tr.history[-1]['y_global']
    print(f'  WL {wl} -> detector Y = {y:.4f} mm  (vignetted={tr.vignetted})')
    ys.append(y)
print(f'  Detector Spread = {max(ys)-min(ys):.4f} mm  (target < 0.8 mm)')

# Check DMD spread
dmd_ys = []
for wl in [1.137, 1.227, 1.316]:
    tr = sys_t.trace_ray(Ray(pos=(0,0,0), dir=(0,0,1), wavelength=wl))
    dmd_ys.append(tr.history[18]['y_local'])
print(f'  DMD Spread     = {max(dmd_ys)-min(dmd_ys):.4f} mm  (target: fit on 8.66mm)')

os.remove(tmp)

# Save result
out = 'c:/Users/Juan/Desktop/spectrometer_raman_50mm_100mm.json'
with open(out, 'w') as f:
    json.dump(config, f, indent=4)
print(f"\nSaved to {out}")
