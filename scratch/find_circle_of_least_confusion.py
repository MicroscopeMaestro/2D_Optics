import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from scratch.find_exact_optical_distances import build_candidate_stack
from optics_app.engine.tracer import OpticalSystem
from verify_layout import generate_fiber_phase_space_rays

# Primary Raman wavelengths (1.1113 to 1.6133 um)
primary_wls = [1.11130, 1.20000, 1.30000, 1.36200, 1.50000, 1.60000, 1.61330]
rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=primary_wls)

print(f"Sweeping circle of least confusion across {len(rays)} rays...")
sys.stdout.flush()

for dmd_r in [15.0, 20.0, 25.0, 30.0]:
    for r_c in [1.0, 3.0, 5.0]:
        for c_d in np.arange(2.0, 5.5, 0.2):
            stack = build_candidate_stack(
                dmd_r=dmd_r, relay_name='AC254-075-C', relay_orient='forward', r_c=r_c, c_d=c_d, pri_wl=1.3620
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
                if max_r <= 1.000:
                    print(f"dmd_r={dmd_r:.1f}, r_c={r_c:.1f}, c_d={c_d:.2f} | max_r={max_r:.4f} mm (dia={max_r*2:.4f} mm), rms_r={rms_r:.4f} mm")
                    sys.stdout.flush()
