import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scratch.analyze_exact_paths import build_system
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import RAMAN_WAVELENGTHS_UM, generate_fiber_phase_space_rays

rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)

print("Testing AC254-075-C and AC254-100-C with proper focal conjugates:")
for name in ["AC254-075-C", "AC254-100-C"]:
    for rev in [False, True]:
        for d1 in np.linspace(65.0, 80.0, 16) if "75" in name else np.linspace(90.0, 105.0, 16):
            for d2 in [1.0, 3.0, 5.0, 10.0]:
                for d3 in np.linspace(4.5, 6.0, 16):
                    stack = build_system(relay_name=name, reverse_relay=rev, t_dmd_relay=d1, t_relay_cond=d2, t_cond_det=d3)
                    opt = OpticalSystem(stack)
                    traced = opt.trace_rays(rays)
                    vig_optics = [r for r in traced if r.vignetted and (not r.history or r.history[-1]["surface_index"] < 14)]
                    if len(vig_optics) > 0:
                        continue
                    det_hits = [h for r in traced for h in r.history if h["surface_index"] == 14]
                    if len(det_hits) == len(rays):
                        rs = [np.sqrt(h["x"]**2 + h["y"]**2) for h in det_hits]
                        max_r = max(rs)
                        if max_r < 1.0:
                            cx = np.mean([h["x"] for h in det_hits])
                            print(f"Candidate: {name} (rev={rev}), d1={d1:5.1f}, d2={d2:4.1f}, d3={d3:4.2f} -> MaxR: {max_r:.4f} mm (Diam: {2*max_r:.4f} mm), cx: {cx:+.3f}")
