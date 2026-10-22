import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scratch.solve_relay import CATALOG
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import RAMAN_WAVELENGTHS_UM, generate_fiber_phase_space_rays

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

rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
print(f"Total verification rays: {len(rays)}")

# Let's test short d1 (field lens regime: d1 in [2, 15] mm) for all 1-inch doublets
for relay in ["AC254-030-C", "AC254-040-C", "AC254-050-C", "AC254-075-C", "AC254-100-C"]:
    for rev in [False, True]:
        for d1 in [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0]:
            for d2 in [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0]:
                for d3 in [3.5, 4.0, 4.2, 4.5, 4.8, 5.0, 5.2, 5.4, 5.6]:
                    stack = create_spectrometer(relay_name=relay, reverse_relay=rev, t_dmd_relay=d1, t_relay_cond=d2, t_cond_det=d3, det_semi_diameter=25.0)
                    opt = OpticalSystem(stack)
                    traced = opt.trace_rays(rays)
                    
                    vig_optics = [r for r in traced if r.vignetted and (not r.history or r.history[-1]["surface_index"] < 14)]
                    if len(vig_optics) > 0:
                        continue
                        
                    det_hits = [h for r in traced for h in r.history if h["surface_index"] == 14]
                    if len(det_hits) == len(rays):
                        rs = [np.sqrt(h["x"]**2 + h["y"]**2) for h in det_hits]
                        max_r = max(rs)
                        if max_r < 0.60:
                            cx = np.mean([h["x"] for h in det_hits])
                            cy = np.mean([h["y"] for h in det_hits])
                            print(f"PASS: max_r: {max_r:.4f} mm (D={2*max_r:.4f} mm) | Relay: {relay} (rev={rev}) | d1={d1:4.1f}, d2={d2:4.1f}, d3={d3:4.2f} | vigs={len(vig_optics)} | cx={cx:+.3f}")
