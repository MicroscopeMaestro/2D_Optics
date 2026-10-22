import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scratch.analyze_exact_paths import build_system
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import RAMAN_WAVELENGTHS_UM, generate_fiber_phase_space_rays

stack = build_system(relay_name="AC254-030-C", reverse_relay=False, t_dmd_relay=15.0, t_relay_cond=5.0, t_cond_det=5.2)
opt = OpticalSystem(stack)

rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
traced = opt.trace_rays(rays)

print(f"{'Wavelength (um)':<18} {'Rays':<6} {'Centroid (X,Y) mm':<22} {'Max Radius (mm)':<18} {'Max Diam (mm)':<16}")
print("-" * 80)
for wl in RAMAN_WAVELENGTHS_UM:
    wl_rays = [r for r in traced if abs(r.wavelength - wl) < 1e-4]
    det_hits = [h for r in wl_rays for h in r.history if h["surface_index"] == 14]
    if det_hits:
        xs = [h["x"] for h in det_hits]
        ys = [h["y"] for h in det_hits]
        rs = [np.sqrt(x**2 + y**2) for x, y in zip(xs, ys)]
        cx = np.mean(xs)
        cy = np.mean(ys)
        print(f"{wl:<18.5f} {len(det_hits):<6} ({cx:+.3f}, {cy:+.3f}){'':<7} {max(rs):<18.4f} {2*max(rs):<16.4f}")
    else:
        print(f"{wl:<18.5f} {'0':<6} NO HITS")
