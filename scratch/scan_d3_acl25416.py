import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import RAMAN_WAVELENGTHS_UM, generate_fiber_phase_space_rays
from scratch.test_acl25416u_full import build_system_acl25416

rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)

print("Scanning d3 for ACL25416U-C:")
for d3 in np.linspace(2.0, 15.0, 27):
    stack = build_system_acl25416(d_dmd_relay=50.0, d_relay_cond=10.0, d_cond_det=d3)
    stack.surfaces[-1].semi_diameter = 25.0
    opt = OpticalSystem(stack)
    traced = opt.trace_rays(rays)
    
    det_hits = [h for r in traced for h in r.history if h['surface_index'] == 14]
    if len(det_hits) == len(rays):
        rs = [np.sqrt(h['x']**2 + h['y']**2) for h in det_hits]
        max_r = max(rs)
        rms_r = np.sqrt(np.mean(np.array(rs)**2))
        print(f"d3 = {d3:5.2f} mm -> max_r = {max_r:.4f} mm, rms_r = {rms_r:.4f} mm")
