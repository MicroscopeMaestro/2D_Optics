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

CATALOG = {
    "AC254-030-C": {"r1": 18.5, "t1": 7.5, "g1": "N-BK7", "r2": -14.8, "t2": 2.0, "g2": "SF5", "r3": -57.2, "sd": 12.7},
    "AC254-040-C": {"r1": 25.1, "t1": 6.8, "g1": "N-BK7", "r2": -20.2, "t2": 2.0, "g2": "SF5", "r3": -84.0, "sd": 12.7},
    "AC254-050-C": {"r1": 34.0, "t1": 6.0, "g1": "N-BK7", "r2": -27.8, "t2": 2.0, "g2": "SF5", "r3": -115.0, "sd": 12.7},
    "AC254-075-C": {"r1": 49.3, "t1": 5.2, "g1": "N-BK7", "r2": -39.0, "t2": 2.0, "g2": "SF5", "r3": -170.0, "sd": 12.7},
    "AC254-100-C": {"r1": 61.2, "t1": 5.0, "g1": "N-BK7", "r2": -44.5, "t2": 2.0, "g2": "SF5", "r3": -198.0, "sd": 12.7},
}

def build_system(
    t_coll_grat=20.0,
    grat_tilt=12.0,
    t_grat_cam=20.0,
    t_cam_dmd=44.0,
    relay_name="AC254-075-C",
    reverse_relay=False,
    t_dmd_relay=20.0,
    t_relay_cond=5.0,
    t_cond_det=5.2,
    primary_wl=1.3625
):
    stack = SurfaceStack()
    stack.primary_wavelength = primary_wl
    
    # 0. Fiber
    stack.add_surface(Surface(name="Fiber_FG025LJA", surface_type=SurfaceType.OBJECT, thickness=47.5, semi_diameter=25.0))
    # 1-3. Collimator
    stack.add_surface(Surface(name="AC254-050-C_Coll_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Back", radius=-115.0, thickness=t_coll_grat, glass="AIR", semi_diameter=12.7))
    # 4. Grating
    stack.add_surface(Surface(name="Grating_GR25-0316", surface_type=SurfaceType.GRATING, grating_lines_per_mm=300.0, tilt_y=grat_tilt, thickness=t_grat_cam, glass="MIRROR", semi_diameter=12.7))
    # 5-7. Camera Focus
    stack.add_surface(Surface(name="AC254-050-C_Focus_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Back", radius=-115.0, thickness=t_cam_dmd, glass="AIR", semi_diameter=12.7))
    # 8. DMD
    stack.add_surface(Surface(name="TI_DMD4500", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=t_dmd_relay, glass="MIRROR", tilt_y=12.0, semi_diameter=12.7))

    # 9-11. Relay
    cat = CATALOG[relay_name]
    if not reverse_relay:
        stack.add_surface(Surface(name=f"{relay_name}_Relay_Front", radius=cat["r1"], thickness=cat["t1"], glass=cat["g1"], semi_diameter=cat["sd"]))
        stack.add_surface(Surface(name=f"{relay_name}_Relay_Mid", radius=cat["r2"], thickness=cat["t2"], glass=cat["g2"], semi_diameter=cat["sd"]))
        stack.add_surface(Surface(name=f"{relay_name}_Relay_Back", radius=cat["r3"], thickness=t_relay_cond, glass="AIR", semi_diameter=cat["sd"]))
    else:
        stack.add_surface(Surface(name=f"{relay_name}_Relay_Front", radius=-cat["r3"], thickness=cat["t2"], glass=cat["g2"], semi_diameter=cat["sd"]))
        stack.add_surface(Surface(name=f"{relay_name}_Relay_Mid", radius=-cat["r2"], thickness=cat["t1"], glass=cat["g1"], semi_diameter=cat["sd"]))
        stack.add_surface(Surface(name=f"{relay_name}_Relay_Back", radius=-cat["r1"], thickness=t_relay_cond, glass="AIR", semi_diameter=cat["sd"]))

    # 12-13. Condenser ACL12708U-C
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Front", surface_type=SurfaceType.ASPHERIC, radius=4.16, conic=-0.6265, asphere_coeffs=[1.5e-5], thickness=4.5, glass="B270", semi_diameter=12.7))
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Back", radius=0.0, thickness=t_cond_det, glass="AIR", semi_diameter=12.7))

    # 14. Detector
    stack.add_surface(Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.50))

    stack._update_accumulated_positions()
    return stack

def evaluate_case(stack, rays):
    opt = OpticalSystem(stack)
    traced = opt.trace_rays(rays)
    det_hits = [h for r in traced for h in r.history if h["surface_index"] == 14]
    vig_optics = [r for r in traced if r.vignetted and (not r.history or r.history[-1]["surface_index"] < 14)]
    if det_hits:
        rs = [np.sqrt(h["x"]**2 + h["y"]**2) for h in det_hits]
        max_r = max(rs)
        cx = np.mean([h["x"] for h in det_hits])
        cy = np.mean([h["y"] for h in det_hits])
        return max_r, len(det_hits), len(vig_optics), cx, cy
    return 999.0, 0, len(vig_optics), 999.0, 999.0

if __name__ == "__main__":
    rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
    print("Testing 10 selected configurations:")
    
    test_cases = [
        # (relay, rev, t_dmd_relay, t_relay_cond, t_cond_det)
        ("AC254-075-C", False, 15.0, 5.0, 5.2),
        ("AC254-050-C", False, 15.0, 5.0, 5.2),
        ("AC254-040-C", False, 15.0, 5.0, 5.2),
        ("AC254-030-C", False, 15.0, 5.0, 5.2),
        ("AC254-075-C", True,  15.0, 5.0, 5.2),
        ("AC254-050-C", True,  15.0, 5.0, 5.2),
        ("AC254-040-C", True,  15.0, 5.0, 5.2),
        ("AC254-030-C", True,  15.0, 5.0, 5.2),
        ("AC254-075-C", False, 10.0, 2.0, 5.1),
        ("AC254-030-C", False, 8.0, 2.0, 5.1),
    ]
    
    for tc in test_cases:
        stack = build_system(relay_name=tc[0], reverse_relay=tc[1], t_dmd_relay=tc[2], t_relay_cond=tc[3], t_cond_det=tc[4])
        max_r, hits, vigs, cx, cy = evaluate_case(stack, rays)
        print(f"Relay: {tc[0]} (rev={tc[1]}), d1={tc[2]}, d2={tc[3]}, d3={tc[4]} -> MaxR: {max_r:.4f} mm (D={2*max_r:.4f} mm), Hits: {hits}/{len(rays)}, Vigs: {vigs}, Centroid: ({cx:+.3f}, {cy:+.3f})")
