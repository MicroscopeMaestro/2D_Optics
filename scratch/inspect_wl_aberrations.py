import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import RAMAN_WAVELENGTHS_UM, generate_fiber_phase_space_rays

stack = SurfaceStack.load_from_json('hybrid_raman_1064_full_range.json')['stack']
stack.surfaces[14].semi_diameter = 25.0

rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
opt = OpticalSystem(stack)
traced = opt.trace_rays(rays)

print(f"Detailed analysis for hybrid_raman_1064_full_range.json:")
for wl in RAMAN_WAVELENGTHS_UM:
    wl_rays = [r for r in traced if abs(r.wavelength - wl) < 1e-4]
    det_hits = [h for r in wl_rays for h in r.history if h['surface_index'] == 14]
    if det_hits:
        rs = [np.sqrt(h['x']**2 + h['y']**2) for h in det_hits]
        xs = [h['x'] for h in det_hits]
        ys = [h['y'] for h in det_hits]
        print(f"WL {wl:.4f} um: hits={len(det_hits)} | x in [{min(xs):7.3f}, {max(xs):7.3f}] | y in [{min(ys):7.3f}, {max(ys):7.3f}] | max_r = {max(rs):7.3f} mm")
