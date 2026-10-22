import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scratch.analyze_exact_paths import build_system
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import RAMAN_WAVELENGTHS_UM, generate_fiber_phase_space_rays

stack = build_system(relay_name="AC254-030-C", reverse_relay=False, t_dmd_relay=10.0, t_relay_cond=3.0, t_cond_det=4.2)
opt = OpticalSystem(stack)

rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
traced = opt.trace_rays(rays)

print("Surface-by-surface ray statistics for AC254-030-C configuration:")
for s_idx in range(8, 15):
    surf = stack.surfaces[s_idx]
    hits = [h for r in traced for h in r.history if h['surface_index'] == s_idx]
    if hits:
        xs = [h['x'] for h in hits]
        ys = [h['y'] for h in hits]
        rs = [np.sqrt(x**2 + y**2) for x, y in zip(xs, ys)]
        dirs_x = [h['dir_x'] for h in hits]
        dirs_y = [h['dir_y'] for h in hits]
        dirs_z = [h['dir_z'] for h in hits]
        print(f"[{s_idx:02d}] {surf.name:<25} Hits: {len(hits):<4} X:[{min(xs):+6.2f}, {max(xs):+6.2f}] Y:[{min(ys):+6.2f}, {max(ys):+6.2f}] MaxR: {max(rs):5.2f} mm  DirX:[{min(dirs_x):+6.3f}, {max(dirs_x):+6.3f}]")

