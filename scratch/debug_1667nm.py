import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from scratch.test_verify_harness import build_remediated_stack
from optics_app.engine.tracer import OpticalSystem
from verify_layout import generate_fiber_phase_space_rays

stack = build_remediated_stack()
opt = OpticalSystem(stack)

rays_1667 = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=[1.66708])
traced = opt.trace_rays(rays_1667)

print(f"Tracing 45 rays for 1.66708 um:")
for i, r in enumerate(traced):
    det_h = next((h for h in r.history if h['surface_index'] == 14), None)
    dmd_h = next((h for h in r.history if h['surface_index'] == 8), None)
    relay_h = next((h for h in r.history if h['surface_index'] == 9), None)
    cond_h = next((h for h in r.history if h['surface_index'] == 12), None)
    
    r_det = np.sqrt(det_h['x']**2 + det_h['y']**2) if det_h else 999.0
    r_cond = np.sqrt(cond_h['x']**2 + cond_h['y']**2) if cond_h else 999.0
    r_dmd = np.sqrt(dmd_h['x']**2 + dmd_h['y']**2) if dmd_h else 999.0
    
    if r_det > 0.500:
        print(f"Ray {i:2d}: DMD=({dmd_h['x']:+.3f}, {dmd_h['y']:+.3f}) [r={r_dmd:.3f}], Cond=({cond_h['x']:+.3f}, {cond_h['y']:+.3f}) [r={r_cond:.3f}] -> Det=({det_h['x']:+.4f}, {det_h['y']:+.4f}) [r={r_det:.4f} mm]")
