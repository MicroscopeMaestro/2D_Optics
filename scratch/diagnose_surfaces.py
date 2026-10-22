import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import RAMAN_WAVELENGTHS_UM, generate_fiber_phase_space_rays

stack = SurfaceStack.load_from_json('hybrid_raman_1064.json')['stack']
opt = OpticalSystem(stack)

rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
traced = opt.trace_rays(rays)

print("Surface-by-surface ray statistics for hybrid_raman_1064.json:")
for s_idx, surf in enumerate(stack.surfaces):
    hits = [h for r in traced for h in r.history if h['surface_index'] == s_idx]
    if hits:
        xs = [h['x'] for h in hits]
        ys = [h['y'] for h in hits]
        rs = [np.sqrt(x**2 + y**2) for x, y in zip(xs, ys)]
        dirs_x = [h['dir_x'] for h in hits]
        dirs_y = [h['dir_y'] for h in hits]
        dirs_z = [h['dir_z'] for h in hits]
        print(f"[{s_idx:02d}] {surf.name:<25} Hits: {len(hits):<4} X:[{min(xs):+6.2f}, {max(xs):+6.2f}] Y:[{min(ys):+6.2f}, {max(ys):+6.2f}] MaxR: {max(rs):5.2f} mm  DirX:[{min(dirs_x):+6.3f}, {max(dirs_x):+6.3f}]")
    else:
        print(f"[{s_idx:02d}] {surf.name:<25} Hits: 0")

