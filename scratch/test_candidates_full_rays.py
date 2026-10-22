import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import RAMAN_WAVELENGTHS_UM, generate_fiber_phase_space_rays
from scratch.test_collector_designs import make_stack

rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
print(f"Total test rays: {len(rays)}", flush=True)

# Candidate 1: 75mm + 8mm_asph (d1=10.0, d2=5.0, d3=4.85 to 5.2)
# Candidate 2: 75mm + 16mm_asph (d1=50.0, d2=10.0, d3=7.0 to 9.0)

print("\nEvaluating Candidate 1: 75mm relay + 8mm asphere across d1 in [8, 15], d2 in [3, 8], d3 in [4.5, 5.2]:", flush=True)
for d1 in [8.0, 10.0, 12.0, 15.0]:
    for d2 in [3.0, 5.0, 8.0]:
        for d3 in [4.6, 4.8, 5.0, 5.2]:
            stack = make_stack("75mm", "8mm_asph", d1, d2, d3)
            opt = OpticalSystem(stack)
            traced = opt.trace_rays(rays)
            
            vig_optics = [r for r in traced if r.vignetted and (not r.history or r.history[-1]['surface_index'] < len(stack.surfaces)-1)]
            det_hits = [h for r in traced for h in r.history if h['surface_index'] == len(stack.surfaces)-1]
            
            if len(vig_optics) == 0 and len(det_hits) == len(rays):
                rs = [np.sqrt(h['x']**2 + h['y']**2) for h in det_hits]
                max_r = max(rs)
                print(f"Cand 1: d1={d1:4.1f}, d2={d2:4.1f}, d3={d3:4.2f} -> vignetting=0.0%, max_r={max_r:.4f} mm (diam={2*max_r:.4f} mm)", flush=True)

print("\nEvaluating Candidate 2: 75mm relay + 16mm asphere across d1 in [40, 60], d2 in [5, 15], d3 in [7.0, 9.0]:", flush=True)
for d1 in [40.0, 45.0, 50.0, 55.0, 60.0]:
    for d2 in [5.0, 8.0, 10.0, 12.0]:
        for d3 in [7.5, 8.0, 8.5]:
            stack = make_stack("75mm", "16mm_asph", d1, d2, d3)
            opt = OpticalSystem(stack)
            traced = opt.trace_rays(rays)
            
            vig_optics = [r for r in traced if r.vignetted and (not r.history or r.history[-1]['surface_index'] < len(stack.surfaces)-1)]
            det_hits = [h for r in traced for h in r.history if h['surface_index'] == len(stack.surfaces)-1]
            
            if len(vig_optics) == 0 and len(det_hits) == len(rays):
                rs = [np.sqrt(h['x']**2 + h['y']**2) for h in det_hits]
                max_r = max(rs)
                print(f"Cand 2: d1={d1:4.1f}, d2={d2:4.1f}, d3={d3:4.2f} -> vignetting=0.0%, max_r={max_r:.4f} mm (diam={2*max_r:.4f} mm)", flush=True)
