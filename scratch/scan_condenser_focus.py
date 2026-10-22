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

rays_center = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=[1.38919])

print("Focus scan on d3 (condenser to detector):")
for d3 in np.linspace(3.0, 5.5, 51):
    stack.surfaces[13].thickness = float(d3)
    stack._update_accumulated_positions()
    
    opt = OpticalSystem(stack)
    traced = opt.trace_rays(rays_center)
    
    det_hits = [h for r in traced for h in r.history if h['surface_index'] == 14]
    rs = [np.sqrt(h['x']**2 + h['y']**2) for h in det_hits]
    max_r = max(rs)
    rms_r = np.sqrt(np.mean(np.array(rs)**2))
    print(f"d3 = {d3:5.2f} mm -> max_r = {max_r:.4f} mm, rms_r = {rms_r:.4f} mm")
