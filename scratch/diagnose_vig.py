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
rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
opt = OpticalSystem(stack)
traced = opt.trace_rays(rays)

print("Loaded hybrid_raman_1064_full_range.json:")
print("Total rays:", len(traced))
print("Vignetted:", sum(1 for r in traced if r.vignetted))

# Let's inspect where rays are clipped
for r in traced:
    if r.vignetted:
        last_surf = r.history[-1]['surface_index'] if r.history else -1
        # print first 5 vignetted
        print(f"Vignetted ray wl={r.wavelength:.4f} stopped at surface {last_surf} ({stack.surfaces[last_surf].name})")
        break
