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

stack = build_spectrometer(d_dmd_relay=10.0, d_relay_cond=5.0, d_cond_det=5.0, cond_model="ACL12708U-C")
# Open all apertures wide to see where rays go
for s in stack.surfaces:
    s.semi_diameter = 50.0

rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
opt = OpticalSystem(stack)
traced = opt.trace_rays(rays)

print(f"Wide open apertures: total={len(traced)}, vignetted={sum(1 for r in traced if r.vignetted)}")

# Now reset authentic Thorlabs clear apertures:
# S0: 25.0
# S1-3: 12.7 (1 inch)
# S4: 12.7 (1 inch grating)
# S5-7: 12.7 (1 inch focus)
# S8: 12.7 (DMD)
# S9-11: 25.4 (2 inch relay AC508-075-C) or 12.7
# S12-13: 12.7 (1/2 inch / 1 inch condenser)
# S14: 0.50 (Detector 1mm)

stack.surfaces[0].semi_diameter = 25.0
for i in [1, 2, 3, 4, 5, 6, 7, 8]:
    stack.surfaces[i].semi_diameter = 12.7
for i in [9, 10, 11]:
    stack.surfaces[i].semi_diameter = 25.4 # or 12.7
stack.surfaces[12].semi_diameter = 6.35 # 1/2 inch ACL12708U-C
stack.surfaces[13].semi_diameter = 6.35
stack.surfaces[14].semi_diameter = 0.50

traced_thor = opt.trace_rays(rays)
print(f"\nAuthentic Thorlabs apertures:")
for s_idx in range(len(stack.surfaces)):
    hits = [h for r in traced_thor for h in r.history if h['surface_index'] == s_idx]
    xs = [h['x'] for h in hits]
    ys = [h['y'] for h in hits]
    rs = [np.sqrt(h['x']**2 + h['y']**2) for h in hits]
    sd = stack.surfaces[s_idx].semi_diameter
    print(f"S{s_idx:2d} ({stack.surfaces[s_idx].name:24s}): hits={len(hits):3d}/315 | x=[{min(xs):7.3f}, {max(xs):7.3f}] | y=[{min(ys):7.3f}, {max(ys):7.3f}] | max_r={max(rs):6.3f} (CA={sd:5.2f})")
