import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from scipy.optimize import minimize
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import RAMAN_WAVELENGTHS_UM, generate_fiber_phase_space_rays, verify_spectrometer_layout

def build_system(d_cam_dmd=44.0, d_dmd_relay=73.0, d_relay_cond=70.0, d_cond_det=5.0, relay_orient="forward", cond_orient="forward"):
    stack = SurfaceStack()
    stack.primary_wavelength = 1.38919
    
    # 0. Fiber
    stack.add_surface(Surface(name="Fiber_FG025LJA", surface_type=SurfaceType.OBJECT, radius=0.0, thickness=47.5, glass="AIR", semi_diameter=25.0))
    # 1-3. Collimator AC254-050-C
    stack.add_surface(Surface(name="AC254-050-C_Coll_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Back", radius=-115.0, thickness=20.0, glass="AIR", semi_diameter=12.7))
    # 4. Grating GR25-0316
    stack.add_surface(Surface(name="Grating_GR25-0316", surface_type=SurfaceType.GRATING, radius=0.0, thickness=20.0, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0, grating_lines_per_mm=300.0, grating_order=1))
    # 5-7. Camera AC254-050-C
    stack.add_surface(Surface(name="AC254-050-C_Focus_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Back", radius=-115.0, thickness=float(d_cam_dmd), glass="AIR", semi_diameter=12.7))
    # 8. DMD
    stack.add_surface(Surface(name="TI_DMD4500", radius=0.0, thickness=float(d_dmd_relay), glass="MIRROR", semi_diameter=12.7, tilt_y=12.0))
    
    # 9-11. Relay Lens AC508-075-C / AC254-075-C
    if relay_orient == "forward":
        stack.add_surface(Surface(name="AC508-075-C_Relay_Front", radius=49.3, thickness=8.0, glass="N-BK7", semi_diameter=25.4))
        stack.add_surface(Surface(name="AC508-075-C_Relay_Mid", radius=-39.0, thickness=3.0, glass="SF5", semi_diameter=25.4))
        stack.add_surface(Surface(name="AC508-075-C_Relay_Back", radius=-170.0, thickness=float(d_relay_cond), glass="AIR", semi_diameter=25.4))
    else:
        stack.add_surface(Surface(name="AC508-075-C_Relay_Front", radius=170.0, thickness=3.0, glass="SF5", semi_diameter=25.4))
        stack.add_surface(Surface(name="AC508-075-C_Relay_Mid", radius=39.0, thickness=8.0, glass="N-BK7", semi_diameter=25.4))
        stack.add_surface(Surface(name="AC508-075-C_Relay_Back", radius=-49.3, thickness=float(d_relay_cond), glass="AIR", semi_diameter=25.4))
        
    # 12-13. Condenser ACL12708U-C
    if cond_orient == "forward":
        stack.add_surface(Surface(name="ACL12708U-C_Cond_Front", surface_type=SurfaceType.ASPHERIC, radius=4.16, thickness=4.5, glass="B270", semi_diameter=12.7, conic=-0.6265, asphere_coeffs=[1.5e-5]))
        stack.add_surface(Surface(name="ACL12708U-C_Cond_Back", radius=0.0, thickness=float(d_cond_det), glass="AIR", semi_diameter=12.7))
    else:
        stack.add_surface(Surface(name="ACL12708U-C_Cond_Front", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=4.5, glass="B270", semi_diameter=12.7))
        stack.add_surface(Surface(name="ACL12708U-C_Cond_Back", surface_type=SurfaceType.ASPHERIC, radius=-4.16, thickness=float(d_cond_det), glass="AIR", semi_diameter=12.7, conic=-0.6265, asphere_coeffs=[1.5e-5]))
        
    # 14. Detector
    stack.add_surface(Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.50))
    return stack

rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
print(f"Total rays: {len(rays)}")

# Let's test combinations
for ro in ["forward", "reversed"]:
    for co in ["forward", "reversed"]:
        print(f"\n--- Testing Relay={ro}, Condenser={co} ---")
        for d1 in [60.0, 70.0, 75.0, 80.0]:
            for d2 in [10.0, 30.0, 50.0, 70.0]:
                for d3 in [4.5, 4.85, 5.0, 5.2]:
                    stack = build_system(d_cam_dmd=44.0, d_dmd_relay=d1, d_relay_cond=d2, d_cond_det=d3, relay_orient=ro, cond_orient=co)
                    stack.surfaces[-1].semi_diameter = 10.0
                    opt = OpticalSystem(stack)
                    traced = opt.trace_rays(rays)
                    
                    vig_optics = [r for r in traced if r.vignetted and (not r.history or r.history[-1]['surface_index'] < 14)]
                    det_hits = [h for r in traced for h in r.history if h['surface_index'] == 14 and not r.vignetted]
                    
                    if len(vig_optics) == 0 and len(det_hits) == len(rays):
                        rs = [np.sqrt(h['x']**2 + h['y']**2) for h in det_hits]
                        max_r = max(rs)
                        if max_r < 0.60:
                            print(f"HIT: {ro:7s}/{co:7s} | d1={d1:4.1f}, d2={d2:4.1f}, d3={d3:4.2f} -> max_r={max_r:.4f}mm (diam={2*max_r:.4f}mm)")
