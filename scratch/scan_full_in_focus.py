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
stack.surfaces[-1].semi_diameter = 25.0

rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
print(f"Total rays: {len(rays)}")

print("\nScanning d3 across 2.0 to 3.0 mm for d1 in [10, 15, 20, 25]:")
for d1 in [10.0, 15.0, 20.0, 25.0]:
    for d2 in [3.0, 5.0, 8.0]:
        for d3 in [2.2, 2.3, 2.4, 2.5, 2.6]:
            stack.surfaces[8].thickness = float(d1)
            stack.surfaces[11].thickness = float(d2)
            stack.surfaces[13].thickness = float(d3)
            stack._update_accumulated_positions()
            
            opt = OpticalSystem(stack)
            traced = opt.trace_rays(rays)
            
            vig_optics = [r for r in traced if r.vignetted and (not r.history or r.history[-1]['surface_index'] < 14)]
            det_hits = [h for r in traced for h in r.history if h['surface_index'] == 14 and not r.vignetted]
            
            if len(vig_optics) == 0 and len(det_hits) == len(rays):
                rs = [np.sqrt(h['x']**2 + h['y']**2) for h in det_hits]
                max_r = max(rs)
                rms_r = np.sqrt(np.mean(np.array(rs)**2))
                print(f"d1={d1:4.1f}, d2={d2:4.1f}, d3={d3:4.2f} -> max_r={max_r:.4f}mm (diam={2*max_r:.4f}mm), rms_r={rms_r:.4f}mm")
