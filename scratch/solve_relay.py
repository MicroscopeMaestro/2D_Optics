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
    "AC254-030-C": {"r1": 18.5, "t1": 7.5, "g1": "N-BK7", "r2": -14.8, "t2": 2.0, "g2": "SF5", "r3": -57.2, "sd": 12.7, "f_approx": 30.0},
    "AC254-040-C": {"r1": 25.1, "t1": 6.8, "g1": "N-BK7", "r2": -20.2, "t2": 2.0, "g2": "SF5", "r3": -84.0, "sd": 12.7, "f_approx": 40.0},
    "AC254-050-C": {"r1": 34.0, "t1": 6.0, "g1": "N-BK7", "r2": -27.8, "t2": 2.0, "g2": "SF5", "r3": -115.0, "sd": 12.7, "f_approx": 50.0},
    "AC254-075-C": {"r1": 49.3, "t1": 5.2, "g1": "N-BK7", "r2": -39.0, "t2": 2.0, "g2": "SF5", "r3": -170.0, "sd": 12.7, "f_approx": 75.0},
    "AC254-100-C": {"r1": 61.2, "t1": 5.0, "g1": "N-BK7", "r2": -44.5, "t2": 2.0, "g2": "SF5", "r3": -198.0, "sd": 12.7, "f_approx": 100.0},
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
    cond_tilt_y=0.0
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
    stack.add_surface(Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.50))

    stack._update_accumulated_positions()
    return stack

def main():
    rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
    print(f"Total verification rays: {len(rays)}")

    best_result = None
    min_spot_r = 999.0

    # Test physical candidate combinations
    configs = []
    for name, cat in CATALOG.items():
        f_nom = cat["f_approx"]
        # d1 ranges around focal distance
        d1_list = [f_nom - 10.0, f_nom - 5.0, f_nom - 2.0, f_nom, f_nom + 2.0, f_nom + 5.0, f_nom + 10.0, 15.0, 20.0, 25.0]
        d1_list = [d for d in d1_list if d > 2.0]
        for d1 in d1_list:
            for rev in [False, True]:
                for d2 in [1.0, 3.0, 5.0, 8.0]:
                    for d3 in [4.8, 5.0, 5.2, 5.4, 5.6]:
                        configs.append((name, rev, d1, d2, d3))

    print(f"Evaluating {len(configs)} focused configurations...")
    for idx, (name, rev, d1, d2, d3) in enumerate(configs):
        stack = create_spectrometer(relay_name=name, reverse_relay=rev, t_dmd_relay=d1, t_relay_cond=d2, t_cond_det=d3)
        opt = OpticalSystem(stack)
        traced = opt.trace_rays(rays)
        
        vig_optics = [r for r in traced if r.vignetted and (not r.history or r.history[-1]["surface_index"] < 14)]
        if len(vig_optics) > 0:
            continue
            
        det_hits = [h for r in traced for h in r.history if h["surface_index"] == 14]
        if len(det_hits) == len(rays):
            rs = [np.sqrt(h["x"]**2 + h["y"]**2) for h in det_hits]
            max_r = max(rs)
            if max_r < min_spot_r:
                min_spot_r = max_r
                cx = np.mean([h["x"] for h in det_hits])
                cy = np.mean([h["y"] for h in det_hits])
                best_result = (name, rev, d1, d2, d3, max_r, cx, cy)
                print(f"[{idx+1}/{len(configs)}] NEW BEST: {name} (rev={rev}), d1={d1:5.1f}, d2={d2:4.1f}, d3={d3:4.2f} -> max_r={max_r:.4f} mm (D={2*max_r:.4f} mm), cx={cx:+.3f}, cy={cy:+.3f}")

    print("\n" + "="*80)
    if best_result:
        print(f"OVERALL BEST CONFIGURATION:")
        print(f"  Relay Lens:       {best_result[0]} (Reversed: {best_result[1]})")
        print(f"  DMD -> Relay:     {best_result[2]:.2f} mm")
        print(f"  Relay -> Cond:    {best_result[3]:.2f} mm")
        print(f"  Cond -> Detector: {best_result[4]:.2f} mm")
        print(f"  Max Spot Radius:  {best_result[5]:.4f} mm (Diameter: {2*best_result[5]:.4f} mm)")
        print(f"  Centroid Offset:  ({best_result[6]:.4f}, {best_result[7]:.4f}) mm")
    else:
        print("No valid 0%-vignetting configuration found in this grid.")

if __name__ == "__main__":
    main()
