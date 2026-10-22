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
from scratch.test_all_architectures import make_full_system

rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
print(f"Total test rays: {len(rays)}")

# Test Architecture A: Relay AC254-050-C + Condenser AC254-030-C (Exclusively 1-inch Thorlabs doublets!)
print("\n--- Architecture A: Relay AC254-050-C + Condenser AC254-030-C ---")
for d1 in [15.0, 20.0, 25.0]:
    for d2 in [5.0, 10.0]:
        for d3 in [18.0, 20.0, 22.0, 24.0]:
            stack = make_full_system(relay_f=50, cond_type="30mm_doublet", d_dmd_relay=d1, d_relay_cond=d2, d_cond_det=d3)
            opt = OpticalSystem(stack)
            traced = opt.trace_rays(rays)
            
            vig_optics = [r for r in traced if r.vignetted and (not r.history or r.history[-1]['surface_index'] < len(stack.surfaces)-1)]
            det_hits = [h for r in traced for h in r.history if h['surface_index'] == len(stack.surfaces)-1]
            
            if len(vig_optics) == 0 and len(det_hits) == len(rays):
                rs = [np.sqrt(h['x']**2 + h['y']**2) for h in det_hits]
                max_r = max(rs)
                print(f"A: d1={d1:4.1f}, d2={d2:4.1f}, d3={d3:4.1f} -> vig=0, max_r={max_r:.4f}mm (diam={2*max_r:.4f}mm)")

# Test Architecture B: Relay AC254-050-C + Condenser ACL12708U-C (f=8mm Asphere)
print("\n--- Architecture B: Relay AC254-050-C + Condenser ACL12708U-C ---")
for d1 in [10.0, 15.0, 20.0]:
    for d2 in [3.0, 5.0, 8.0]:
        for d3 in [4.5, 4.8, 5.0, 5.2]:
            stack = make_full_system(relay_f=50, cond_type="8mm_asph", d_dmd_relay=d1, d_relay_cond=d2, d_cond_det=d3)
            opt = OpticalSystem(stack)
            traced = opt.trace_rays(rays)
            
            vig_optics = [r for r in traced if r.vignetted and (not r.history or r.history[-1]['surface_index'] < len(stack.surfaces)-1)]
            det_hits = [h for r in traced for h in r.history if h['surface_index'] == len(stack.surfaces)-1]
            
            if len(vig_optics) == 0 and len(det_hits) == len(rays):
                rs = [np.sqrt(h['x']**2 + h['y']**2) for h in det_hits]
                max_r = max(rs)
                print(f"B: d1={d1:4.1f}, d2={d2:4.1f}, d3={d3:4.1f} -> vig=0, max_r={max_r:.4f}mm (diam={2*max_r:.4f}mm)")
