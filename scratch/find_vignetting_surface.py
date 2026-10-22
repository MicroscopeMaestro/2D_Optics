import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import RAMAN_WAVELENGTHS_UM, generate_fiber_phase_space_rays
from scratch.evaluate_final_design import build_spectrometer

stack = build_spectrometer(d_dmd_relay=50.0, d_relay_cond=10.0, d_cond_det=8.0, cond_model="ACL25416U-C")
# Set detector semi-diameter large
stack.surfaces[-1].semi_diameter = 25.0

rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
opt = OpticalSystem(stack)
traced = opt.trace_rays(rays)

print(f"Total rays: {len(traced)}")
vig_by_surf = {}
for r in traced:
    if r.vignetted:
        last = r.history[-1]['surface_index'] if r.history else -1
        vig_by_surf[last] = vig_by_surf.get(last, 0) + 1

print("Vignetting by surface index:")
for s_idx, count in sorted(vig_by_surf.items()):
    s_name = stack.surfaces[s_idx].name if 0 <= s_idx < len(stack.surfaces) else "BEFORE S0"
    print(f"  Surface {s_idx:2d} ({s_name}): {count} rays vignetted")

det_hits = [h for r in traced for h in r.history if h['surface_index'] == len(stack.surfaces)-1]
print(f"Detector hits: {len(det_hits)} / {len(rays)}")
if det_hits:
    rs = [np.sqrt(h['x']**2 + h['y']**2) for h in det_hits]
    print(f"Max spot radius: {max(rs):.4f} mm (Diameter: {2*max(rs):.4f} mm)")
    print(f"RMS spot radius: {np.sqrt(np.mean(np.array(rs)**2)):.4f} mm")
