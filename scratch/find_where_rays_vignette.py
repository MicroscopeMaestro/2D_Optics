import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from scratch.superfast_optimizer import build_spectrometer, full_rays
from optics_app.engine.tracer import OpticalSystem

stack = build_spectrometer(
    f_coll=47.5, c_g=20.0, g_f=20.0, f_dmd=44.0,
    dmd_r=20.0, relay_name='AC254-075-C', orient='forward', r_c=5.0, c_d=5.2, pri_wl=1.3620
)
opt = OpticalSystem(stack)
traced = opt.trace_rays(full_rays)

print(f"Total Surfaces: {len(stack.surfaces)}")
for i, s in enumerate(stack.surfaces):
    print(f"  Surf {i:2d}: {s.name:<30} | semi_dia = {s.semi_diameter:.2f} mm")

vig_counts = {i: 0 for i in range(len(stack.surfaces))}
for r in traced:
    if r.vignetted:
        last_s = r.history[-1]['surface_index'] if r.history else 0
        vig_counts[last_s] += 1

print("\nVignetting breakdown across all 315 rays:")
for s_idx, count in vig_counts.items():
    if count > 0:
        print(f"  Surf {s_idx:2d} ({stack.surfaces[s_idx].name}): {count} rays vignetted / blocked")
        # Print max local height at that surface
        heights = []
        for r in traced:
            for h in r.history:
                if h['surface_index'] == s_idx:
                    heights.append(np.sqrt(h['x']**2 + h['y']**2))
        if heights:
            print(f"    Max ray height at Surf {s_idx}: {max(heights):.4f} mm (semi-diameter is {stack.surfaces[s_idx].semi_diameter:.4f} mm)")
