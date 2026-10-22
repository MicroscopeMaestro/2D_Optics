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
stack.surfaces[14].semi_diameter = 10.0

chief_rays = [Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl) for wl in RAMAN_WAVELENGTHS_UM]

print("Evaluating specific candidate positions...", flush=True)

for d1 in [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 75.0, 80.0]:
    for d2 in [5.0, 10.0, 20.0]:
        for d3 in [4.5, 4.85, 5.0, 5.2, 5.5]:
            stack.surfaces[8].thickness = float(d1)
            stack.surfaces[11].thickness = float(d2)
            stack.surfaces[13].thickness = float(d3)
            stack._update_accumulated_positions()
            
            opt = OpticalSystem(stack)
            traced = opt.trace_rays(chief_rays)
            
            det_hits = [h for r in traced for h in r.history if h['surface_index'] == 14]
            if len(det_hits) == len(chief_rays):
                rs = [np.sqrt(h['x']**2 + h['y']**2) for h in det_hits]
                max_r = max(rs)
                print(f"d1={d1:5.1f}, d2={d2:4.1f}, d3={d3:5.2f} -> chief max_r = {max_r:.4f} mm", flush=True)
