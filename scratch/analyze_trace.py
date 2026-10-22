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
opt = OpticalSystem(stack)

rays = [Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl) for wl in RAMAN_WAVELENGTHS_UM]
traced = opt.trace_rays(rays)

print("Surface global positions & rotations:")
for i, s in enumerate(stack.surfaces):
    print(f"S{i:2d} ({s.name:24s}): origin={s.global_origin.round(2)}, tilt_y={s.tilt_y}")

print("\nChief rays at DMD (S8) vs Detector (S14):")
for r in traced:
    h8 = next(h for h in r.history if h['surface_index'] == 8)
    h14 = next(h for h in r.history if h['surface_index'] == 14)
    print(f"WL {r.wavelength:.4f} um: DMD x={h8['x']:7.3f}, y={h8['y']:7.3f} | DET x={h14['x']:7.3f}, y={h14['y']:7.3f}, r={np.sqrt(h14['x']**2 + h14['y']**2):7.3f}")
