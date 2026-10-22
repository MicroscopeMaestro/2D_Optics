import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.optimize import minimize
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import generate_fiber_phase_space_rays, RAMAN_WAVELENGTHS_UM, verify_spectrometer_layout

def build_system(p):
    # p = [d_fiber, d_coll_grating, d_grating_focus, d_focus_dmd, d_dmd_relay, d_relay_cond, d_cond_det]
    t0, t3, t4, t7, t8, t11, t13 = p
    stack = SurfaceStack()
    stack.primary_wavelength = 1.38919
    
    # 0. Fiber
    stack.add_surface(Surface(name="Fiber_FG025LJA", surface_type=SurfaceType.OBJECT, radius=0.0, thickness=float(t0), glass="AIR", semi_diameter=25.0))
    # 1-3. Collimator AC254-050-C
    stack.add_surface(Surface(name="AC254-050-C_Coll_Front", surface_type=SurfaceType.STANDARD, radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Mid", surface_type=SurfaceType.STANDARD, radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Back", surface_type=SurfaceType.STANDARD, radius=-115.0, thickness=float(t3), glass="AIR", semi_diameter=12.7))
    # 4. Grating GR25-0316
    stack.add_surface(Surface(name="Grating_GR25-0316", surface_type=SurfaceType.GRATING, radius=0.0, thickness=float(t4), glass="MIRROR", semi_diameter=12.7, tilt_y=12.0, grating_lines_per_mm=300.0, grating_order=1))
    # 5-7. Camera AC254-050-C
    stack.add_surface(Surface(name="AC254-050-C_Focus_Front", surface_type=SurfaceType.STANDARD, radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Mid", surface_type=SurfaceType.STANDARD, radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Back", surface_type=SurfaceType.STANDARD, radius=-115.0, thickness=float(t7), glass="AIR", semi_diameter=12.7))
    # 8. DMD TI_DMD4500
    stack.add_surface(Surface(name="TI_DMD4500", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=float(t8), glass="MIRROR", semi_diameter=12.7, tilt_y=12.0))
    # 9-11. Relay AC508-075-C / AC254-075-C
    stack.add_surface(Surface(name="AC508-075-C_Relay_Front", surface_type=SurfaceType.STANDARD, radius=49.3, thickness=8.0, glass="N-BK7", semi_diameter=25.4))
    stack.add_surface(Surface(name="AC508-075-C_Relay_Mid", surface_type=SurfaceType.STANDARD, radius=-39.0, thickness=3.0, glass="SF5", semi_diameter=25.4))
    stack.add_surface(Surface(name="AC508-075-C_Relay_Back", surface_type=SurfaceType.STANDARD, radius=-170.0, thickness=float(t11), glass="AIR", semi_diameter=25.4))
    # 12-13. Condenser ACL12708U-C
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Front", surface_type=SurfaceType.ASPHERIC, radius=4.16, thickness=4.5, glass="B270", semi_diameter=12.7, conic=-0.6265, asphere_coeffs=[1.5e-5]))
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Back", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=float(t13), glass="AIR", semi_diameter=12.7))
    # 14. InGaAs Detector
    stack.add_surface(Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.50))
    
    return stack

def eval_system(p, rays):
    stack = build_system(p)
    opt = OpticalSystem(stack)
    traced = opt.trace_rays(rays)
    
    vig = sum(1 for r in traced if r.vignetted)
    det_hits = [h for r in traced for h in r.history if h['surface_index'] == 14 and not r.vignetted]
    
    if det_hits:
        rs = [np.sqrt(h['x']**2 + h['y']**2) for h in det_hits]
        max_r = max(rs)
        cx = np.mean([h['x'] for h in det_hits])
        cy = np.mean([h['y'] for h in det_hits])
        c_off = np.sqrt(cx**2 + cy**2)
    else:
        max_r = 999.0
        c_off = 999.0
        
    return vig, max_r, c_off

if __name__ == '__main__':
    # Generate test rays
    eval_rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
    print(f"Total test rays: {len(eval_rays)}", flush=True)
    
    # Base params
    p0 = [47.5, 20.0, 20.0, 44.0, 20.0, 5.0, 5.2]
    v, r, c = eval_system(p0, eval_rays)
    print(f"Initial: vig={v}, max_r={r:.4f}mm (diam={2*r:.4f}mm), c_off={c:.4f}mm", flush=True)
