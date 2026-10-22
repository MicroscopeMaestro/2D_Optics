import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import generate_fiber_phase_space_rays, RAMAN_WAVELENGTHS_UM

stack = SurfaceStack.load_from_json('hybrid_raman_1064_full_range.json')['stack']

# Temporarily enlarge detector semi-diameter to 10mm so we can measure exact spot radius without clipping
stack.surfaces[14].semi_diameter = 10.0

print("Testing d_dmd_relay (S8 thickness) around 60 to 80 mm:")
for d1 in np.linspace(60.0, 80.0, 21):
    for d2 in [2.0, 5.0, 10.0, 15.0]:
        for d3 in np.linspace(4.0, 6.0, 11):
            stack.surfaces[8].thickness = float(d1)
            stack.surfaces[11].thickness = float(d2)
            stack.surfaces[13].thickness = float(d3)
            stack._update_accumulated_positions()
            
            rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
            opt = OpticalSystem(stack)
            traced = opt.trace_rays(rays)
            
            # Check optical vignetting before detector
            vig_optics = [r for r in traced if r.vignetted and (not r.history or r.history[-1]['surface_index'] < 14)]
            if len(vig_optics) > 0:
                continue
                
            det_hits = [h for r in traced for h in r.history if h['surface_index'] == 14]
            if len(det_hits) == len(rays):
                rs = [np.sqrt(h['x']**2 + h['y']**2) for h in det_hits]
                max_r = max(rs)
                if max_r < 0.60:
                    print(f"d1={d1:5.1f}, d2={d2:4.1f}, d3={d3:5.2f} -> max_r={max_r:.4f} mm (diam={2*max_r:.4f} mm)")
