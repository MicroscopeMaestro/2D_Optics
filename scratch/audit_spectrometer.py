import json
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.engine.materials import get_refractive_index
import numpy as np

with open("spectrometer_raman_50mm_100mm_manual.json", "r") as f:
    data = json.load(f)

stack = SurfaceStack.from_dict(data)
opt = OpticalSystem(stack)

print("="*60)
print("STAGE 1: Sample (S0) to Slit (S8)")
print("="*60)
# Sub-stack from S0 to S8
sub1 = SurfaceStack()
for i in range(9):
    sub1.add_surface(stack.surfaces[i])
opt1 = OpticalSystem(sub1)

# Trace rays from point source at Sample (S0) at angles up to NA 0.22 (theta ~ 12 deg)
# Wavelength = 1.225 um (Raman Stokes signal from 1064nm pump)
wl = 1.225
for angle_deg in [-10.0, -5.0, 0.0, 5.0, 10.0]:
    ang_rad = np.radians(angle_deg)
    d = np.array([0.0, np.sin(ang_rad), np.cos(ang_rad)])
    r = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=d, wavelength=wl)
    res = opt1.trace_ray(r)
    slit_step = next((s for s in res.history if s['surface_index'] == 8), None)
    if slit_step:
        print(f"angle={angle_deg:+5.1f} deg: pos on slit y={slit_step['y_local']:+8.4f} mm, dir={slit_step['dir']}")

print("\n" + "="*60)
print("STAGE 2: Slit (S8) to Grating (S15) - Collimator Check")
print("="*60)
sub2 = SurfaceStack()
for i in range(8, 16):
    sub2.add_surface(stack.surfaces[i])
opt2 = OpticalSystem(sub2)

for y_slit in [-0.025, 0.0, 0.025]: # 50 um slit
    for angle_deg in [-6.0, 0.0, 6.0]:
        ang_rad = np.radians(angle_deg)
        d = np.array([0.0, np.sin(ang_rad), np.cos(ang_rad)])
        r = Ray(pos=np.array([0.0, y_slit, 0.0]), dir=d, wavelength=wl)
        res = opt2.trace_ray(r)
        # Check direction after collimator (surface 12, index 4 in sub2)
        s12 = next((s for s in res.history if s['surface_index'] == 4), None)
        s15 = next((s for s in res.history if s['surface_index'] == 7), None)
        if s12:
            print(f"y_slit={y_slit*1e3:+5.0f} um, ang={angle_deg:+4.1f} deg -> after collimator dir_y={s12['dir'][1]:+8.5f}, beam y={s12['y_local']:+7.3f} mm")

print("\n" + "="*60)
print("STAGE 3: Grating (S15) to DMD (S21) - Spectral Focus")
print("="*60)
# Check focal spot at DMD for 1.087, 1.225, 1.350 um
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
        print(f"wl={wl:.3f} um: mean y_DMD = {mean_y:+7.3f} mm, spot spread = {spread_um:6.1f} um across ray fan")
    else:
        print(f"wl={wl:.3f} um: vignetted or no hits on DMD")

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
    else:
        print(f"wl={wl:.3f} um: vignetted or no hits on Detector")
