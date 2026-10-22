import json
import copy
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
import numpy as np

with open("spectrometer_raman_50mm_100mm_manual.json", "r") as f:
    data = json.load(f)

# Fresh full stack
stack = SurfaceStack.from_dict(data)
opt = OpticalSystem(stack)

print("="*60)
print("STAGE 3: Grating (S15) to DMD (S21) - Spectral Focus")
print("="*60)
for wl in [1.087, 1.225, 1.350]:
    spots = []
    for ang in [-3.0, -1.5, 0.0, 1.5, 3.0]:
        ang_rad = np.radians(ang)
        d = np.array([0.0, np.sin(ang_rad), np.cos(ang_rad)])
        r = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=d, wavelength=wl)
        res = opt.trace_ray(r)
        s21 = next((s for s in res.history if s['surface_index'] == 21), None)
        if s21 and not res.vignetted:
            spots.append(s21['y_local'])
    if spots:
        spread_um = (max(spots) - min(spots)) * 1000.0
        mean_y = np.mean(spots)
        print(f"wl={wl:.3f} um: mean y_DMD = {mean_y:+7.3f} mm, spot spread = {spread_um:6.1f} um across ray fan {spots}")

print("\n" + "="*60)
print("STAGE 4: DMD (S21) to Detector (S25) - Condenser Focus")
print("="*60)
for wl in [1.087, 1.225, 1.350]:
    spots = []
    for ang in [-3.0, -1.5, 0.0, 1.5, 3.0]:
        ang_rad = np.radians(ang)
        d = np.array([0.0, np.sin(ang_rad), np.cos(ang_rad)])
        r = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=d, wavelength=wl)
        res = opt.trace_ray(r)
        s25 = next((s for s in res.history if s['surface_index'] == 25), None)
        if s25 and not res.vignetted:
            spots.append(s25['y_local'])
    if spots:
        spread_um = (max(spots) - min(spots)) * 1000.0
        mean_y = np.mean(spots)
        print(f"wl={wl:.3f} um: mean y_Det = {mean_y:+7.3f} mm, spot spread = {spread_um:6.1f} um on Detector")
