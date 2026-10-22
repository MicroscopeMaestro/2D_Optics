import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from verify_layout import generate_fiber_phase_space_rays, RAMAN_WAVELENGTHS_UM

stack = SurfaceStack.load_from_json("hybrid_raman_1064.json")
if isinstance(stack, dict): stack = stack["stack"]

# Set exact 0.5-inch semi-diameter on ACL12708U-C surfaces
stack.surfaces[12].semi_diameter = 6.35
stack.surfaces[13].semi_diameter = 6.35
stack._update_accumulated_positions()

opt = OpticalSystem(stack)
rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
traced = opt.trace_rays(rays)
det_idx = len(stack.surfaces) - 1

det_hits = [(h['x'], h['y'], r.wavelength) for r in traced for h in r.history if h['surface_index'] == det_idx and not (r.vignetted and r.history[-1]['surface_index'] < det_idx)]
vig_count = sum(1 for r in traced if r.vignetted and (not r.history or r.history[-1]['surface_index'] < det_idx))

print(f"Total rays: {len(rays)}, Det hits: {len(det_hits)}, Vignetted in optics: {vig_count}")
if det_hits:
    rads = [np.sqrt(x**2 + y**2) for x, y, wl in det_hits]
    print(f"Max detector spot radius: {max(rads):.4f} mm (dia = {max(rads)*2:.4f} mm)")
    print(f"RMS detector spot radius: {np.sqrt(np.mean(np.array(rads)**2)):.4f} mm")
