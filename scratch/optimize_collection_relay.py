import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import RAMAN_WAVELENGTHS_UM, generate_fiber_phase_space_rays

# Catalog of 1-inch Thorlabs doublets
CATALOG = {
    "AC254-030-C": {"r1": 18.5, "t1": 7.5, "g1": "N-BK7", "r2": -14.8, "t2": 2.0, "g2": "SF5", "r3": -57.2, "sd": 12.7},
    "AC254-040-C": {"r1": 25.1, "t1": 6.8, "g1": "N-BK7", "r2": -20.2, "t2": 2.0, "g2": "SF5", "r3": -84.0, "sd": 12.7},
    "AC254-050-C": {"r1": 34.0, "t1": 6.0, "g1": "N-BK7", "r2": -27.8, "t2": 2.0, "g2": "SF5", "r3": -115.0, "sd": 12.7},
    "AC254-075-C": {"r1": 49.3, "t1": 5.2, "g1": "N-BK7", "r2": -39.0, "t2": 2.0, "g2": "SF5", "r3": -170.0, "sd": 12.7},
    "AC254-100-C": {"r1": 61.2, "t1": 5.0, "g1": "N-BK7", "r2": -44.5, "t2": 2.0, "g2": "SF5", "r3": -198.0, "sd": 12.7},
}

def create_spectrometer(
    relay_name="AC254-075-C",
    reverse_relay=False,
    t_dmd_relay=20.0,
    t_relay_cond=5.0,
    t_cond_det=5.2,
    primary_wl=1.3625,
    camera_t=44.0,
    relay_tilt_y=0.0,
    cond_tilt_y=0.0,
    det_semi_diameter=0.50
):
    stack = SurfaceStack()
    stack.primary_wavelength = primary_wl
    
    # 0. Fiber
    stack.add_surface(Surface(name="Fiber_FG025LJA", surface_type=SurfaceType.OBJECT, thickness=47.5, semi_diameter=25.0))
    # 1-3. Collimator
    stack.add_surface(Surface(name="AC254-050-C_Coll_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Back", radius=-115.0, thickness=20.0, glass="AIR", semi_diameter=12.7))
    # 4. Grating
    stack.add_surface(Surface(name="Grating_GR25-0316", surface_type=SurfaceType.GRATING, grating_lines_per_mm=300.0, tilt_y=12.0, thickness=20.0, glass="MIRROR", semi_diameter=12.7))
    # 5-7. Camera Focus
    stack.add_surface(Surface(name="AC254-050-C_Focus_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Back", radius=-115.0, thickness=camera_t, glass="AIR", semi_diameter=12.7))
    # 8. DMD
    stack.add_surface(Surface(name="TI_DMD4500", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=t_dmd_relay, glass="MIRROR", tilt_y=12.0, semi_diameter=12.7))

    # 9-11. Relay
    cat = CATALOG[relay_name]
    if not reverse_relay:
        stack.add_surface(Surface(name=f"{relay_name}_Relay_Front", radius=cat["r1"], thickness=cat["t1"], glass=cat["g1"], semi_diameter=cat["sd"], tilt_y=relay_tilt_y))
        stack.add_surface(Surface(name=f"{relay_name}_Relay_Mid", radius=cat["r2"], thickness=cat["t2"], glass=cat["g2"], semi_diameter=cat["sd"]))
        stack.add_surface(Surface(name=f"{relay_name}_Relay_Back", radius=cat["r3"], thickness=t_relay_cond, glass="AIR", semi_diameter=cat["sd"]))
    else:
        stack.add_surface(Surface(name=f"{relay_name}_Relay_Front", radius=-cat["r3"], thickness=cat["t2"], glass=cat["g2"], semi_diameter=cat["sd"], tilt_y=relay_tilt_y))
        stack.add_surface(Surface(name=f"{relay_name}_Relay_Mid", radius=-cat["r2"], thickness=cat["t1"], glass=cat["g1"], semi_diameter=cat["sd"]))
        stack.add_surface(Surface(name=f"{relay_name}_Relay_Back", radius=-cat["r1"], thickness=t_relay_cond, glass="AIR", semi_diameter=cat["sd"]))

    # 12-13. Condenser ACL12708U-C
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Front", surface_type=SurfaceType.ASPHERIC, radius=4.16, conic=-0.6265, asphere_coeffs=[1.5e-5], thickness=4.5, glass="B270", semi_diameter=12.7, tilt_y=cond_tilt_y))
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Back", radius=0.0, thickness=t_cond_det, glass="AIR", semi_diameter=12.7))

    # 14. Detector
    stack.add_surface(Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=det_semi_diameter))

    stack._update_accumulated_positions()
    return stack

def trace_and_score(stack, test_wls=RAMAN_WAVELENGTHS_UM):
    opt = OpticalSystem(stack)
    # Fast ray trace: 5 spatial x 9 angular = 45 rays per wavelength
    rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=test_wls, dense_mode=False)
    traced = opt.trace_rays(rays)
    
    det_idx = 14
    det_hits = []
    vignetted_optics = 0
    for r in traced:
        if r.vignetted and (not r.history or r.history[-1]["surface_index"] < det_idx):
            vignetted_optics += 1
        for h in r.history:
            if h["surface_index"] == det_idx:
                det_hits.append((h["x"], h["y"]))
                
    if det_hits and vignetted_optics == 0:
        radii = [np.sqrt(x**2 + y**2) for x, y in det_hits]
        max_r = max(radii)
        cx = np.mean([x for x, y in det_hits])
        cy = np.mean([y for x, y in det_hits])
    else:
        max_r = 999.0
        cx, cy = 999.0, 999.0
        
    return max_r, len(det_hits), vignetted_optics, cx, cy

print("Testing configurations...")
best_score = 999.0
best_cfg = None

for relay in ["AC254-075-C", "AC254-050-C", "AC254-040-C", "AC254-030-C", "AC254-100-C"]:
    for rev in [False, True]:
        for d1 in [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 70.0]:
            for d2 in [1.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]:
                for d3 in [3.0, 3.5, 4.0, 4.5, 4.8, 5.0, 5.2, 5.4, 5.6, 6.0]:
                    stack = create_spectrometer(relay_name=relay, reverse_relay=rev, t_dmd_relay=d1, t_relay_cond=d2, t_cond_det=d3)
                    max_r, hits, vigs, cx, cy = trace_and_score(stack)
                    if max_r < best_score:
                        best_score = max_r
                        best_cfg = (relay, rev, d1, d2, d3, max_r, hits, vigs, cx, cy)
                        print(f"NEW BEST: Relay: {relay} (rev={rev}), d1={d1:4.1f}, d2={d2:4.1f}, d3={d3:4.1f} -> max_r: {max_r:.4f} mm (diam={2*max_r:.4f} mm), vigs: {vigs}, cx: {cx:+.3f}")

print("\nFinal Best Configuration:", best_cfg)
