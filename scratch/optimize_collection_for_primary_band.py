import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from scipy.optimize import minimize
from scratch.find_exact_optical_distances import build_candidate_stack
from optics_app.engine.tracer import OpticalSystem
from verify_layout import generate_fiber_phase_space_rays

wl_list = [1.11130, 1.20000, 1.30000, 1.36200, 1.50000, 1.60000, 1.61330]

rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=wl_list)

print("Optimizing collection relay distances for primary Raman band...")
best_cfgs = []

for relay_name in ['AC254-075-C', 'AC254-050-C']:
    for orient in ['forward', 'reversed']:
        for dmd_r in np.arange(10.0, 40.0, 5.0):
            for r_c in np.arange(1.0, 15.0, 3.0):
                for c_d in np.arange(1.0, 8.0, 0.5):
                    stack = build_candidate_stack(
                        dmd_r=dmd_r, relay_name=relay_name, relay_orient=orient, r_c=r_c, c_d=c_d, pri_wl=1.3620
                    )
                    opt = OpticalSystem(stack)
                    traced = opt.trace_rays(rays)
                    det_idx = len(stack.surfaces) - 1
                    
                    vig = sum(1 for r in traced if r.vignetted and (not r.history or r.history[-1]['surface_index'] < det_idx))
                    if vig > 0:
                        continue
                    hits = [(h['x'], h['y']) for r in traced for h in r.history if h['surface_index'] == det_idx and not (r.vignetted and r.history[-1]['surface_index'] < det_idx)]
                    if len(hits) == len(rays):
                        rads = [np.sqrt(x**2 + y**2) for x, y in hits]
                        max_r = max(rads)
                        rms_r = np.sqrt(np.mean(np.array(rads)**2))
                        best_cfgs.append((max_r, rms_r, relay_name, orient, dmd_r, r_c, c_d))

best_cfgs.sort(key=lambda x: x[0])
print(f"\nTotal zero-vignetting configurations found: {len(best_cfgs)}")
print("Top 10 configurations:")
for max_r, rms_r, rname, orient, dmd_r, r_c, c_d in best_cfgs[:10]:
    print(f"  max_r = {max_r:.4f} mm (dia = {max_r*2:.4f} mm), rms_r = {rms_r:.4f} mm | Relay: {rname} ({orient}) | dmd_r={dmd_r:.1f}, r_c={r_c:.1f}, c_d={c_d:.1f}")
