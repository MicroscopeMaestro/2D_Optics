import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from scratch.test_short_doublets_7wl import build_system, rays

for relay_name in ['AC254-030-C', 'AC254-040-C', 'AC254-050-C']:
    for dmd_r in [5.0, 8.0, 10.0]:
        stack = build_system(47.5, 38.5, dmd_r, relay_name, 'reversed', 2.0, 5.2, 1.38919)
        opt = OpticalSystem(stack)
        traced = opt.trace_rays(rays)
        
        print(f"\n--- {relay_name} (reversed) | dmd_r={dmd_r} ---")
        for s_idx, s in enumerate(stack.surfaces):
            heights = []
            for r in traced:
                for h in r.history:
                    if h['surface_index'] == s_idx:
                        heights.append(np.sqrt(h['x']**2 + h['y']**2))
            if heights:
                max_h = max(heights)
                status = "OK" if max_h <= s.semi_diameter else f"EXCEEDED ({max_h:.3f} > {s.semi_diameter:.3f})"
                print(f"  Surf {s_idx:2d} ({s.name:<25}): max_height = {max_h:.4f} mm, semi_dia = {s.semi_diameter:.4f} mm -> {status}")
