import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scratch.solve_relay import create_spectrometer
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import RAMAN_WAVELENGTHS_UM, generate_fiber_phase_space_rays

stack = create_spectrometer(relay_name="AC254-075-C", t_dmd_relay=20.0, t_relay_cond=5.0, t_cond_det=5.2)
opt = OpticalSystem(stack)

rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
traced = opt.trace_rays(rays)

print(f"Total rays: {len(rays)}")
for i, s in enumerate(stack.surfaces):
    hits = [h for r in traced for h in r.history if h['surface_index'] == i]
    vigs = [r for r in traced if r.vignetted and r.history and r.history[-1]['surface_index'] == i]
    if hits:
        xs = [h['x'] for h in hits]
        ys = [h['y'] for h in hits]
        rs = [np.sqrt(x**2 + y**2) for x, y in zip(xs, ys)]
        print(f"[{i:02d}] {s.name:<28} SemiD: {s.semi_diameter:5.2f} mm | Hits: {len(hits):3d} | X: [{min(xs):+6.2f}, {max(xs):+6.2f}] Y: [{min(ys):+6.2f}, {max(ys):+6.2f}] MaxR: {max(rs):5.2f} mm | Lost here: {len(vigs)}")
    else:
        print(f"[{i:02d}] {s.name:<28} SemiD: {s.semi_diameter:5.2f} mm | Hits: 0")
