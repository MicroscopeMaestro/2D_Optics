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

# 4f / Afocal telecentric relay setup:
# DMD to Relay: d1 ~ 70 - 75 mm
# Relay to Condenser: d2 ~ 70 - 80 mm
# Condenser to Detector: d3 ~ 4.5 - 5.2 mm (BFL of ACL12708U-C is ~4.85 mm)

print("Testing 4f telecentric pupil relay:")
for d1 in [65.0, 70.0, 72.0, 75.0, 78.0]:
    for d2 in [40.0, 50.0, 60.0, 70.0, 75.0]:
        for d3 in [4.5, 4.8, 5.0, 5.2]:
            stack.surfaces[8].thickness = float(d1)
            stack.surfaces[11].thickness = float(d2)
            stack.surfaces[13].thickness = float(d3)
            stack._update_accumulated_positions()
            
            # Trace chief rays first
            chief_rays = [Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl) for wl in RAMAN_WAVELENGTHS_UM]
            opt = OpticalSystem(stack)
            traced = opt.trace_rays(chief_rays)
            
            # Check hits on condenser front (S12) and detector (S14)
            s12_hits = [h for r in traced for h in r.history if h['surface_index'] == 12]
            s14_hits = [h for r in traced for h in r.history if h['surface_index'] == 14]
            
            if len(s12_hits) == 7 and len(s14_hits) == 7:
                s12_r = [np.sqrt(h['x']**2 + h['y']**2) for h in s12_hits]
                s14_r = [np.sqrt(h['x']**2 + h['y']**2) for h in s14_hits]
                print(f"d1={d1:4.1f}, d2={d2:4.1f}, d3={d3:4.2f} -> S12 max_r={max(s12_r):5.2f} mm | DET chief max_r={max(s14_r):6.3f} mm", flush=True)
