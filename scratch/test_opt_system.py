import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.analysis.spot import SpotDiagram
from verify_layout import RAMAN_WAVELENGTHS_UM, generate_fiber_phase_space_rays

def build_test_stack(
    front_camera_foc=50.0,
    t_fiber_coll=47.5,
    t_coll_grat=20.0,
    grat_tilt=12.0,
    t_grat_cam=20.0,
    t_cam_dmd=44.0,
    dmd_tilt=12.0,
    t_dmd_relay=20.0,
    relay_choice="AC254-050-C",
    t_relay_cond=5.0,
    t_cond_det=5.2,
    relay_tilt=0.0,
    cond_tilt=0.0,
    dmd_decenter_x=0.0,
    relay_decenter_x=0.0,
    cond_decenter_x=0.0,
    det_decenter_x=0.0
):
    stack = SurfaceStack()
    # 0. Fiber
    stack.add_surface(Surface(name="Fiber_FG025LJA", surface_type=SurfaceType.OBJECT, thickness=t_fiber_coll, semi_diameter=25.0))

    # 1-3. Collimator: AC254-050-C
    stack.add_surface(Surface(name="AC254-050-C_Coll_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Back", radius=-115.0, thickness=t_coll_grat, glass="AIR", semi_diameter=12.7))

    # 4. Grating: GR25-0316
    stack.add_surface(Surface(name="Grating_GR25-0316", surface_type=SurfaceType.GRATING, grating_lines_per_mm=300.0, tilt_y=grat_tilt, thickness=t_grat_cam, glass="MIRROR", semi_diameter=12.7))

    # 5-7. Camera
    if front_camera_foc == 50.0:
        stack.add_surface(Surface(name="AC254-050-C_Focus_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-050-C_Focus_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-050-C_Focus_Back", radius=-115.0, thickness=t_cam_dmd, glass="AIR", semi_diameter=12.7))
    elif front_camera_foc == 40.0:
        stack.add_surface(Surface(name="AC254-040-C_Focus_Front", radius=25.1, thickness=6.8, glass="N-BK7", semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-040-C_Focus_Mid", radius=-20.2, thickness=2.0, glass="SF5", semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-040-C_Focus_Back", radius=-84.0, thickness=t_cam_dmd, glass="AIR", semi_diameter=12.7))

    # 8. DMD
    stack.add_surface(Surface(name="TI_DMD4500", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=t_dmd_relay, glass="MIRROR", tilt_y=dmd_tilt, decenter_x=dmd_decenter_x, semi_diameter=12.7))

    # 9-11. Relay doublet (1-inch)
    if relay_choice == "AC254-050-C":
        stack.add_surface(Surface(name="AC254-050-C_Relay_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7, tilt_y=relay_tilt, decenter_x=relay_decenter_x))
        stack.add_surface(Surface(name="AC254-050-C_Relay_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-050-C_Relay_Back", radius=-115.0, thickness=t_relay_cond, glass="AIR", semi_diameter=12.7))
    elif relay_choice == "AC254-075-C":
        stack.add_surface(Surface(name="AC254-075-C_Relay_Front", radius=49.3, thickness=5.2, glass="N-BK7", semi_diameter=12.7, tilt_y=relay_tilt, decenter_x=relay_decenter_x))
        stack.add_surface(Surface(name="AC254-075-C_Relay_Mid", radius=-39.0, thickness=2.0, glass="SF5", semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-075-C_Relay_Back", radius=-170.0, thickness=t_relay_cond, glass="AIR", semi_diameter=12.7))
    elif relay_choice == "AC254-040-C":
        stack.add_surface(Surface(name="AC254-040-C_Relay_Front", radius=25.1, thickness=6.8, glass="N-BK7", semi_diameter=12.7, tilt_y=relay_tilt, decenter_x=relay_decenter_x))
        stack.add_surface(Surface(name="AC254-040-C_Relay_Mid", radius=-20.2, thickness=2.0, glass="SF5", semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-040-C_Relay_Back", radius=-84.0, thickness=t_relay_cond, glass="AIR", semi_diameter=12.7))
    elif relay_choice == "AC254-030-C":
        stack.add_surface(Surface(name="AC254-030-C_Relay_Front", radius=18.5, thickness=7.5, glass="N-BK7", semi_diameter=12.7, tilt_y=relay_tilt, decenter_x=relay_decenter_x))
        stack.add_surface(Surface(name="AC254-030-C_Relay_Mid", radius=-14.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-030-C_Relay_Back", radius=-57.2, thickness=t_relay_cond, glass="AIR", semi_diameter=12.7))

    # 12-13. Condenser ACL12708U-C
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Front", surface_type=SurfaceType.ASPHERIC, radius=4.16, conic=-0.6265, asphere_coeffs=[1.5e-5], thickness=4.5, glass="B270", semi_diameter=12.7, tilt_y=cond_tilt, decenter_x=cond_decenter_x))
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Back", radius=0.0, thickness=t_cond_det, glass="AIR", semi_diameter=12.7))

    # 14. Detector
    stack.add_surface(Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.50, decenter_x=det_decenter_x))

    return stack

def evaluate(stack):
    system = OpticalSystem(stack)
    # Check chief rays on DMD
    chief_wls = [1.11130, 1.20000, 1.30000, 1.38919, 1.50000, 1.60000, 1.61330, 1.66708]
    dmd_xs = []
    for wl in chief_wls:
        cr = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl)
        tr = system.trace_rays([cr])[0]
        h_dmd = next((h for h in tr.history if h["surface_index"] == 8), None)
        if h_dmd:
            dmd_xs.append(h_dmd["x"])
        else:
            dmd_xs.append(None)
    
    span_primary = abs(dmd_xs[6] - dmd_xs[0]) if dmd_xs[6] is not None and dmd_xs[0] is not None else None
    
    # Check full ray bundle
    rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
    traced = system.trace_rays(rays)
    det_idx = 14
    det_hits = []
    vig_count = 0
    for r in traced:
        if r.vignetted:
            vig_count += 1
        for h in r.history:
            if h["surface_index"] == det_idx:
                det_hits.append((h["x"], h["y"]))
                
    if det_hits:
        radii = [np.sqrt(x**2 + y**2) for x, y in det_hits]
        max_r = max(radii)
        cx = np.mean([x for x, y in det_hits])
        cy = np.mean([y for x, y in det_hits])
    else:
        max_r = 999.0
        cx, cy = 999.0, 999.0
        
    return {
        "span_primary": span_primary,
        "dmd_xs": dmd_xs,
        "total_rays": len(rays),
        "det_hits": len(det_hits),
        "vig_count": vig_count,
        "max_r": max_r,
        "centroid": (cx, cy)
    }

# Test baseline with AC254-075-C
stack_75 = build_test_stack(relay_choice="AC254-075-C")
res_75 = evaluate(stack_75)
print("AC254-075-C Baseline:", res_75)

# Test baseline with AC254-050-C
stack_50 = build_test_stack(relay_choice="AC254-050-C")
res_50 = evaluate(stack_50)
print("AC254-050-C Baseline:", res_50)
