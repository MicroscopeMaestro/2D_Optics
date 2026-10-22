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

def test_relay_optimization(relay_name="AC254-075-C", t_dmd_relay=70.0, t_relay_cond=5.0, t_cond_det=5.2, front_cam_t=44.0, grat_tilt=12.0):
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Fiber_FG025LJA", surface_type=SurfaceType.OBJECT, thickness=47.5, semi_diameter=25.0))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Back", radius=-115.0, thickness=20.0, glass="AIR", semi_diameter=12.7))
    stack.add_surface(Surface(name="Grating_GR25-0316", surface_type=SurfaceType.GRATING, grating_lines_per_mm=300.0, tilt_y=grat_tilt, thickness=20.0, glass="MIRROR", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Back", radius=-115.0, thickness=front_cam_t, glass="AIR", semi_diameter=12.7))
    stack.add_surface(Surface(name="TI_DMD4500", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=t_dmd_relay, glass="MIRROR", tilt_y=12.0, semi_diameter=12.7))

    # Relay lens
    if relay_name == "AC254-075-C":
        stack.add_surface(Surface(name="AC254-075-C_Relay_Front", radius=49.3, thickness=5.2, glass="N-BK7", semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-075-C_Relay_Mid", radius=-39.0, thickness=2.0, glass="SF5", semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-075-C_Relay_Back", radius=-170.0, thickness=t_relay_cond, glass="AIR", semi_diameter=12.7))
    elif relay_name == "AC254-050-C":
        stack.add_surface(Surface(name="AC254-050-C_Relay_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-050-C_Relay_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-050-C_Relay_Back", radius=-115.0, thickness=t_relay_cond, glass="AIR", semi_diameter=12.7))
    elif relay_name == "AC254-100-C":
        stack.add_surface(Surface(name="AC254-100-C_Relay_Front", radius=61.2, thickness=5.0, glass="N-BK7", semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-100-C_Relay_Mid", radius=-44.5, thickness=2.0, glass="SF5", semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-100-C_Relay_Back", radius=-198.0, thickness=t_relay_cond, glass="AIR", semi_diameter=12.7))

    # Condenser
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Front", surface_type=SurfaceType.ASPHERIC, radius=4.16, conic=-0.6265, asphere_coeffs=[1.5e-5], thickness=4.5, glass="B270", semi_diameter=12.7))
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Back", radius=0.0, thickness=t_cond_det, glass="AIR", semi_diameter=12.7))

    # Detector
    stack.add_surface(Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.50))

    opt = OpticalSystem(stack)
    rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
    traced = opt.trace_rays(rays)

    det_hits = [h for r in traced for h in r.history if h['surface_index'] == 14]
    vig_rays = [r for r in traced if r.vignetted]
    
    if det_hits:
        radii = [np.sqrt(h['x']**2 + h['y']**2) for h in det_hits]
        max_r = max(radii)
        mean_r = np.mean(radii)
        cx = np.mean([h['x'] for h in det_hits])
        cy = np.mean([h['y'] for h in det_hits])
    else:
        max_r, mean_r, cx, cy = 999.0, 999.0, 999.0, 999.0

    return max_r, len(det_hits), len(vig_rays), cx, cy

print("Testing relay sweep...")
for relay in ["AC254-075-C", "AC254-050-C", "AC254-100-C"]:
    for t_dmd in [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 75.0, 80.0]:
        for t_cond in [1.0, 3.0, 5.0, 8.0, 10.0]:
            for t_det in [3.0, 4.0, 5.0, 5.2, 5.5, 6.0]:
                max_r, hits, vigs, cx, cy = test_relay_optimization(relay_name=relay, t_dmd_relay=t_dmd, t_relay_cond=t_cond, t_cond_det=t_det)
                if max_r < 0.50:
                    print(f"PASS! Relay: {relay}, t_dmd: {t_dmd}, t_cond: {t_cond}, t_det: {t_det} -> max_r: {max_r:.4f} mm (D={2*max_r:.4f} mm), hits: {hits}, vigs: {vigs}, centroid: ({cx:.3f}, {cy:.3f})")
                elif max_r < 1.0:
                    print(f"Close: Relay: {relay}, t_dmd: {t_dmd}, t_cond: {t_cond}, t_det: {t_det} -> max_r: {max_r:.4f} mm, hits: {hits}, vigs: {vigs}")
