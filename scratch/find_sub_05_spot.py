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
stack.surfaces[14].semi_diameter = 10.0  # Open aperture to measure spot accurately

rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
print(f"Total test rays: {len(rays)}", flush=True)

best = None
min_max_r = 1e9

for d1 in [16.0, 18.0, 20.0, 22.0, 24.0]:
    for d2 in [3.0, 5.0, 8.0, 10.0]:
        for d3 in [4.4, 4.6, 4.8, 5.0, 5.2]:
            stack.surfaces[8].thickness = float(d1)
            stack.surfaces[11].thickness = float(d2)
            stack.surfaces[13].thickness = float(d3)
            stack._update_accumulated_positions()
            
            opt = OpticalSystem(stack)
            traced = opt.trace_rays(rays)
            
            vig_optics = [r for r in traced if r.vignetted and (not r.history or r.history[-1]['surface_index'] < 14)]
            if len(vig_optics) > 0:
                continue
                
            det_hits = [h for r in traced for h in r.history if h['surface_index'] == 14]
            if len(det_hits) == len(rays):
                rs = [np.sqrt(h['x']**2 + h['y']**2) for h in det_hits]
                max_r = max(rs)
                if max_r < min_max_r:
                    min_max_r = max_r
                    best = (d1, d2, d3, max_r)
                    print(f"New best: d1={d1:5.2f}, d2={d2:5.2f}, d3={d3:5.2f} -> max_r={max_r:.4f} mm (diam={2*max_r:.4f} mm)", flush=True)

if best:
    print(f"\nFinal Best: d1={best[0]:.3f}, d2={best[1]:.3f}, d3={best[2]:.3f}, max_r={best[3]:.4f} mm (diam={2*best[3]:.4f} mm)", flush=True)
else:
    print("No valid configuration found")
