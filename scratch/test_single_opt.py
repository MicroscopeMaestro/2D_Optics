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

stack = SurfaceStack.load_from_json('hybrid_raman_1064_full_range.json')['stack']
stack.surfaces[8].thickness = 20.0
stack.surfaces[11].thickness = 5.0
stack.surfaces[13].thickness = 2.40
stack.surfaces[14].semi_diameter = 25.0
stack._update_accumulated_positions()

rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
opt = OpticalSystem(stack)
traced = opt.trace_rays(rays)

print(f"Results for d1=20.0, d2=5.0, d3=2.40 mm:")
for wl in RAMAN_WAVELENGTHS_UM:
    wl_rays = [r for r in traced if abs(r.wavelength - wl) < 1e-4]
    det_hits = [h for r in wl_rays for h in r.history if h['surface_index'] == 14]
    rs = [np.sqrt(h['x']**2 + h['y']**2) for h in det_hits]
    xs = [h['x'] for h in det_hits]
    ys = [h['y'] for h in det_hits]
    print(f"  WL {wl:.4f} um: x in [{min(xs):6.3f}, {max(xs):6.3f}], y in [{min(ys):6.3f}, {max(ys):6.3f}], max_r = {max(rs):6.3f} mm")

all_det_hits = [h for r in traced for h in r.history if h['surface_index'] == 14]
all_rs = [np.sqrt(h['x']**2 + h['y']**2) for h in all_det_hits]
print(f"\nALL 315 RAYS: max_r = {max(all_rs):.4f} mm (Diameter: {2*max(all_rs):.4f} mm)")
