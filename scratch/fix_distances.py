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
# Print current prescriptions
# ============================================================
print("=== Current lens prescriptions ===")
for idx in [3,4,5, 15,16,17]:
    s = surfs[idx]
    print(f"  {idx}: {s['name']:30} R={s['radius']:>8.2f}  t={s['thickness']:>6.2f}  glass={s['glass']}")

# ============================================================
# Helper: trace 3 wavelengths and return (dmd_spread, det_spread)
# ============================================================
def get_spread(cfg):
    import copy
    tmp_cfg = copy.deepcopy(cfg)
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.json') as tf:
        json.dump(tmp_cfg, tf); tmp = tf.name
    sys_t = OpticalSystem.from_json(tmp)
    dmd_ys, det_ys = [], []
    for wl in [1.137, 1.227, 1.316]:
        tr = sys_t.trace_ray(Ray(pos=(0,0,0), dir=(0,0,1), wavelength=wl))
        if len(tr.history) < 26:
            os.remove(tmp); return None, None
        dmd_ys.append(tr.history[18]['y_local'])
        det_ys.append(tr.history[-1]['y_global'])
    os.remove(tmp)
    return max(dmd_ys)-min(dmd_ys), max(det_ys)-min(det_ys)

dmd_sp, det_sp = get_spread(config)
print(f"\n=== Initial spreads ===")
print(f"  DMD spread     = {dmd_sp:.3f} mm")
print(f"  Detector spread= {det_sp:.3f} mm")

# ============================================================
# The 30mm focusing lens has BFL ~23.5mm. But the grating-to-lens
# distance (CB_After_Grating, surf 14) controls how much the beam
# diverges before the lens → affects magnification on DMD.
# Scan it to find where the DMD gets best focus (smallest spot).
# ============================================================
print("\n=== Scanning grating-to-lens distance ===")
best_cb = None
best_spread = 1e9
for cb_t in np.arange(15, 60, 1):
    surfs[14]['thickness'] = float(cb_t)
    d, _ = get_spread(config)
    if d is not None:
        print(f"  CB={cb_t:.0f}mm -> DMD spread={d:.3f} mm")
        if d is not None and abs(d - 6.5) < abs(best_spread - 6.5):
            best_cb = float(cb_t)
            best_spread = d

print(f"\n  Best CB for ~6.5mm DMD spread: {best_cb:.1f} mm (spread={best_spread:.3f} mm)")
surfs[14]['thickness'] = best_cb

# ============================================================
# Optimize [bfl→DMD, dmd_t, relay_cond, cond_det] for <0.8mm detector
# ============================================================
print("\n=== Optimizing back-end distances ===")

def evaluate(x):
    import copy
    cfg = copy.deepcopy(config)
    cfg['surfaces'][17]['thickness'] = float(x[0])
    cfg['surfaces'][18]['thickness'] = float(x[1])
    cfg['surfaces'][22]['thickness'] = float(x[2])
    cfg['surfaces'][24]['thickness'] = float(x[3])
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.json') as tf:
        json.dump(cfg, tf); tmp = tf.name
    sys_t = OpticalSystem.from_json(tmp)
    ys = []
    for wl in [1.137, 1.227, 1.316]:
        tr = sys_t.trace_ray(Ray(pos=(0,0,0), dir=(0,0,1), wavelength=wl))
        if len(tr.history) < 26:
            os.remove(tmp); return 1e6
        ys.append(tr.history[-1]['y_global'])
    spread = max(ys) - min(ys)
    center = ys[1]
    tr_m = sys_t.trace_ray(Ray(pos=(0, 0.001, 0), dir=(0,0,1), wavelength=1.227))
    defocus = abs(tr_m.history[-1]['y_global'] - center) if len(tr_m.history) >= 26 else 10.0
    os.remove(tmp)
    return defocus * 5000 + max(0, spread - 0.79) * 200

res = minimize(evaluate, [23.54, 55.0, 26.0, 7.6], method='Nelder-Mead',
               options={'xatol':0.01, 'fatol':1e-5, 'maxiter':5000})
print(f"  Result: bfl={res.x[0]:.2f}  dmd_t={res.x[1]:.2f}  relay_cond={res.x[2]:.2f}  cond_det={res.x[3]:.2f}")
print(f"  Penalty = {res.fun:.6f}")

surfs[17]['thickness'] = float(res.x[0])
surfs[18]['thickness'] = float(res.x[1])
surfs[22]['thickness'] = float(res.x[2])
surfs[24]['thickness'] = float(res.x[3])

dmd_sp, det_sp = get_spread(config)
print(f"\n=== Final spreads ===")
print(f"  DMD spread     = {dmd_sp:.3f} mm  (target ~6-8 mm on 8.66mm DMD)")
print(f"  Detector spread= {det_sp:.3f} mm  (target < 0.8 mm)")

print("\n=== Final surface distances ===")
for i, s in enumerate(surfs):
    print(f"  {i:02d}: {s.get('name',''):35} thickness={s.get('thickness',0):.2f} mm")

out_path = 'c:/Users/Juan/Desktop/spectrometer_raman_50mm_100mm.json'
with open(out_path, 'w') as f:
    json.dump(config, f, indent=4)
print(f"\nSaved -> {out_path}")
