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

rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=[1.66708])
opt = OpticalSystem(stack)
traced = opt.trace_rays(rays)

print(f"Surface-by-surface trace for 1.66708 um (45 rays):")
for s_idx in range(8, 15):
    s = stack.surfaces[s_idx]
    hits = [h for r in traced for h in r.history if h['surface_index'] == s_idx]
    xs = [h['x'] for h in hits]
    ys = [h['y'] for h in hits]
    rs = [np.sqrt(h['x']**2 + h['y']**2) for h in hits]
    dirs_z = [h.get('dir_z', 0.0) for h in hits]
    print(f"S{s_idx:2d} ({s.name:24s}): x=[{min(xs):7.3f}, {max(xs):7.3f}], y=[{min(ys):7.3f}, {max(ys):7.3f}], max_r={max(rs):6.3f} (SD={s.semi_diameter:5.2f})")
