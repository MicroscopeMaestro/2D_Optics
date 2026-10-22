import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from scratch.check_surface_clearance import build_system
from optics_app.engine.tracer import OpticalSystem
from verify_layout import generate_fiber_phase_space_rays, RAMAN_WAVELENGTHS_UM

rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)

print("Sweeping c_d from 0.5 to 4.5 mm for AC254-030-C:")
for c_d in np.arange(0.5, 4.6, 0.2):
    stack = build_system(dmd_r=8.0, pri_wl=1.38919, c_d=c_d)
    opt = OpticalSystem(stack)
    traced = opt.trace_rays(rays)
    det_idx = len(stack.surfaces) - 1
    
    hits = [(h['x'], h['y']) for r in traced for h in r.history if h['surface_index'] == det_idx and not r.vignetted]
    vig_optics = sum(1 for r in traced if r.vignetted and (not r.history or r.history[-1]['surface_index'] < det_idx))
    
    if hits:
        rads = [np.sqrt(x**2 + y**2) for x, y in hits]
        max_r = max(rads)
        rms_r = np.sqrt(np.mean(np.array(rads)**2))
        print(f"c_d={c_d:.2f} mm | vig_optics={vig_optics}, det_rays={len(hits)}/{len(rays)} | max_r={max_r:.4f} mm (dia={max_r*2:.4f} mm), rms_r={rms_r:.4f} mm")
