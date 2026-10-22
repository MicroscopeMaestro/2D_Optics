import json
import sys
import os
import tempfile
import numpy as np

sys.path.append('c:/Users/Juan/Documents/GitHub/2D_Optics')
from optics_app.engine.tracer import OpticalSystem, Ray

# -------------------------------------------------------
# 1. Trace the MANUAL edition to see its DMD and detector spread
# -------------------------------------------------------
print("=== MANUAL EDITION trace ===")
sys_m = OpticalSystem.from_json('c:/Users/Juan/Desktop/spectrometer_raman_50mm_100mm_manual.json')
dmd_ys_m, det_ys_m = [], []
for wl in [1.137, 1.227, 1.316]:
    tr = sys_m.trace_ray(Ray(pos=(0,0,0), dir=(0,0,1), wavelength=wl))
    print(f"  WL {wl}: vignetted={tr.vignetted}  len={len(tr.history)}", end='')
    if len(tr.history) >= 19:
        y_dmd = tr.history[18]['y_local']
        print(f"  DMD_y={y_dmd:.3f}", end='')
        dmd_ys_m.append(y_dmd)
    if len(tr.history) >= 26:
        y_det = tr.history[-1]['y_global']
        print(f"  Det_y={y_det:.3f}", end='')
        det_ys_m.append(y_det)
    print()

if len(dmd_ys_m) >= 2:
    print(f"  Manual DMD spread = {max(dmd_ys_m)-min(dmd_ys_m):.3f} mm")
if len(det_ys_m) >= 2:
    print(f"  Manual Det spread = {max(det_ys_m)-min(det_ys_m):.3f} mm")

# -------------------------------------------------------
# 2. Trace the CURRENT edition
# -------------------------------------------------------
print("\n=== CURRENT EDITION trace ===")
sys_c = OpticalSystem.from_json('c:/Users/Juan/Desktop/spectrometer_raman_50mm_100mm.json')
dmd_ys_c, det_ys_c = [], []
for wl in [1.137, 1.227, 1.316]:
    tr = sys_c.trace_ray(Ray(pos=(0,0,0), dir=(0,0,1), wavelength=wl))
    print(f"  WL {wl}: vignetted={tr.vignetted}  len={len(tr.history)}", end='')
    if len(tr.history) >= 19:
        y_dmd = tr.history[18]['y_local']
        print(f"  DMD_y={y_dmd:.3f}", end='')
        dmd_ys_c.append(y_dmd)
    if len(tr.history) >= 26:
        y_det = tr.history[-1]['y_global']
        print(f"  Det_y={y_det:.3f}", end='')
        det_ys_c.append(y_det)
    print()

if len(dmd_ys_c) >= 2:
    print(f"  Current DMD spread = {max(dmd_ys_c)-min(dmd_ys_c):.3f} mm")
if len(det_ys_c) >= 2:
    print(f"  Current Det spread = {max(det_ys_c)-min(det_ys_c):.3f} mm")

# -------------------------------------------------------
# 3. Check angles at grating output for each wavelength
# -------------------------------------------------------
print("\n=== Angles after grating (current) ===")
for wl in [1.137, 1.227, 1.316]:
    tr = sys_c.trace_ray(Ray(pos=(0,0,0), dir=(0,0,1), wavelength=wl))
    if len(tr.history) > 13:
        dy = tr.history[13]['dir_y']
        dz = tr.history[13]['dir_z']
        ang = np.degrees(np.arctan2(dy, abs(dz)))
        print(f"  WL {wl}: grating output angle = {ang:.3f} deg  (dir_y={dy:.4f})")
