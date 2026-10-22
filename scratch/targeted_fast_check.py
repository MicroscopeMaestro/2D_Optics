import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scratch.analyze_exact_paths import build_system, CATALOG
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import RAMAN_WAVELENGTHS_UM, generate_fiber_phase_space_rays

rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
print(f"Targeted check with {len(rays)} rays:")

for name in ["AC254-030-C", "AC254-040-C", "AC254-050-C"]:
    for d1 in [3.0, 5.0, 7.0, 10.0]:
        for d2 in [1.0, 2.0, 3.0]:
            for d3 in [4.2, 4.6, 5.0, 5.2]:
                stack = build_system(relay_name=name, reverse_relay=False, t_dmd_relay=d1, t_relay_cond=d2, t_cond_det=d3)
                opt = OpticalSystem(stack)
                traced = opt.trace_rays(rays)
                vig = len([r for r in traced if r.vignetted and (not r.history or r.history[-1]["surface_index"] < 14)])
                hits = [h for r in traced for h in r.history if h["surface_index"] == 14]
                if vig == 0 and len(hits) == len(rays):
                    rs = [np.sqrt(h["x"]**2 + h["y"]**2) for h in hits]
                    max_r = max(rs)
                    cx = np.mean([h["x"] for h in hits])
                    cy = np.mean([h["y"] for h in hits])
                    if max_r < 3.0:
                        print(f"Relay: {name} | d1={d1:4.1f}, d2={d2:4.1f}, d3={d3:4.2f} -> MaxR: {max_r:.4f} mm (Diam: {2*max_r:.4f} mm), Centroid: ({cx:+.3f}, {cy:+.3f})")
