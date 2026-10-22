import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from verify_layout import generate_fiber_phase_space_rays

stack = SurfaceStack.load_from_json("hybrid_raman_1064.json")
if isinstance(stack, dict): stack = stack["stack"]

primary_wls = [1.11130, 1.20000, 1.30000, 1.36200, 1.50000, 1.60000, 1.61330]
rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=primary_wls)

opt = OpticalSystem(stack)
traced = opt.trace_rays(rays)
det_idx = len(stack.surfaces) - 1

det_hits = [(h['x'], h['y'], r.wavelength) for r in traced for h in r.history if h['surface_index'] == det_idx and not (r.vignetted and r.history[-1]['surface_index'] < det_idx)]
vig_count = sum(1 for r in traced if r.vignetted and (not r.history or r.history[-1]['surface_index'] < det_idx))

print(f"Total rays: {len(rays)}, Det hits: {len(det_hits)}, Vignetted in optics: {vig_count}")
if det_hits:
    rads = [np.sqrt(x**2 + y**2) for x, y, wl in det_hits]
    print(f"Max detector spot radius: {max(rads):.4f} mm (dia = {max(rads)*2:.4f} mm)")
    print(f"RMS detector spot radius: {np.sqrt(np.mean(np.array(rads)**2)):.4f} mm")
    print(f"Centroid: ({np.mean([x for x, y, wl in det_hits]):+.4f}, {np.mean([y for x, y, wl in det_hits]):+.4f}) mm")
