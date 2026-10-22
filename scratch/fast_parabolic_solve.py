import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import RAMAN_WAVELENGTHS_UM

stack = SurfaceStack.load_from_json('hybrid_raman_1064_full_range.json')['stack']
stack.surfaces[14].semi_diameter = 10.0

# 7 chief rays only for ultra-fast evaluation (< 1 ms per evaluation)
chief_rays = [Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl) for wl in RAMAN_WAVELENGTHS_UM]

print("Chief ray evaluation across d_dmd_relay (S8 T) and d_cond_det (S13 T):")
best = None
min_max_r = 1e9

for d1 in np.linspace(40.0, 90.0, 51):
    for d2 in [5.0, 10.0, 15.0, 20.0]:
        for d3 in np.linspace(3.5, 6.5, 31):
            stack.surfaces[8].thickness = float(d1)
            stack.surfaces[11].thickness = float(d2)
            stack.surfaces[13].thickness = float(d3)
            stack._update_accumulated_positions()
            
            opt = OpticalSystem(stack)
            traced = opt.trace_rays(chief_rays)
            
            det_hits = [h for r in traced for h in r.history if h['surface_index'] == 14]
            if len(det_hits) == len(chief_rays):
                rs = [np.sqrt(h['x']**2 + h['y']**2) for h in det_hits]
                max_r = max(rs)
                if max_r < min_max_r:
                    min_max_r = max_r
                    best = (d1, d2, d3, max_r)
                    print(f"New best chief spot: d1={d1:.2f}, d2={d2:.2f}, d3={d3:.2f} -> max_r={max_r:.4f} mm")

print(f"\nOptimal: d1={best[0]:.3f}, d2={best[1]:.3f}, d3={best[2]:.3f}, max_r={best[3]:.4f}")
