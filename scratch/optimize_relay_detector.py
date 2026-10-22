import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.optimize import minimize
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import generate_fiber_phase_space_rays, RAMAN_WAVELENGTHS_UM

def test_spacings(d_dmd_relay, d_relay_cond, d_cond_det, d_focus_dmd=44.0):
    stack = SurfaceStack.load_from_json('hybrid_raman_1064_full_range.json')['stack']
    
    # S7 is camera lens back air gap (distance to DMD)
    stack.surfaces[7].thickness = float(d_focus_dmd)
    # S8 is DMD thickness (distance to relay lens)
    stack.surfaces[8].thickness = float(d_dmd_relay)
    # S11 is Relay back air gap (distance to condenser)
    stack.surfaces[11].thickness = float(d_relay_cond)
    # S13 is Condenser back air gap (distance to detector)
    stack.surfaces[13].thickness = float(d_cond_det)
    
    stack._update_accumulated_positions()
    
    rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
    opt = OpticalSystem(stack)
    traced = opt.trace_rays(rays)
    
    vig_count = sum(1 for r in traced if r.vignetted)
    det_hits = [h for r in traced for h in r.history if h['surface_index'] == 14 and not r.vignetted]
    
    if det_hits:
        det_rs = [np.sqrt(h['x']**2 + h['y']**2) for h in det_hits]
        max_r = max(det_rs)
        cx = np.mean([h['x'] for h in det_hits])
        cy = np.mean([h['y'] for h in det_hits])
        cent_off = np.sqrt(cx**2 + cy**2)
    else:
        max_r = 999.0
        cent_off = 999.0
        
    dmd_hits = [h for r in traced for h in r.history if h['surface_index'] == 8 and not r.vignetted]
    if dmd_hits:
        xs = [h['x'] for h in dmd_hits]
        dmd_span = max(xs) - min(xs)
    else:
        dmd_span = 0.0
        
    return vig_count, max_r, cent_off, dmd_span

print("Scanning DMD-to-Relay distance (d_dmd_relay)...")
for d_dr in [10, 20, 30, 40, 50, 60, 70, 75, 80]:
    for d_rc in [5, 10, 20]:
        for d_cd in [4.5, 4.8, 5.0, 5.2, 5.5]:
            v, r, c, span = test_spacings(d_dr, d_rc, d_cd)
            if v == 0 and r < 2.0:
                print(f"d_dr={d_dr:4.1f}, d_rc={d_rc:4.1f}, d_cd={d_cd:4.2f} -> vig={v}, max_r={r:.4f}mm (diam={2*r:.4f}mm), cent={c:.4f}mm, dmd_span={span:.3f}mm")
