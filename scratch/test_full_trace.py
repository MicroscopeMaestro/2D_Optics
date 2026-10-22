import sys, os
sys.path.insert(0, os.path.abspath('.'))

from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
import numpy as np

res = SurfaceStack.load_from_json(r'C:\Users\Juan\Desktop\spectrometer_raman_50mm_100mm_manual_optimized.json')
stack = res['stack']
wavelengths = res.get('wavelengths', [1.087, 1.225, 1.35])

system = OpticalSystem(stack)
traced_rays = []

d0 = stack.surfaces[0].thickness if len(stack.surfaces) > 0 else 0.0
r1 = stack.surfaces[1].semi_diameter if len(stack.surfaces) > 1 else 10.0

for wl in wavelengths:
    for py in np.linspace(-0.8, 0.8, 7):
        start_pos = np.array([0.0, 0.0, 0.0])
        target_y = py * r1
        target_z = d0
        length = np.sqrt(target_y**2 + target_z**2)
        start_dir = np.array([0.0, target_y / length, target_z / length])
        r = Ray(pos=start_pos, dir=start_dir, wavelength=wl)
        traced = system.trace_ray(r)
        traced_rays.append(traced)

print(f"Traced {len(traced_rays)} rays.")
for i, tr in enumerate(traced_rays):
    hist = getattr(tr, 'history', [])
    valid = getattr(tr, 'valid', True)
    err = getattr(tr, 'error_message', '')
    steps = len(hist)
    last_z = hist[-1].get('z_global', 0.0) if hist else 0.0
    wl = tr.wavelength
    print(f"Ray {i:02d} (wl={wl:.3f}): valid={valid} steps={steps} last_z={last_z:.2f} err='{err}'")
