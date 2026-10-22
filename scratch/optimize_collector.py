"""
Optimize collector subsystem from DMD (surf 8) to Detector (surf 14).
"""
import sys
import os
import math
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray

WAVELENGTHS = [1.1113, 1.2000, 1.3000, 1.3892, 1.5000, 1.6000, 1.6671]

def build_full_system(d_fiber, d_coll_grating, grating_tilt, d_grating_focus, d_focus_dmd, dmd_tilt, d_dmd_relay, d_relay_cond, d_cond_det, asph_R=4.16, asph_k=-0.6265, asph_A4=1.5e-5):
    stack = SurfaceStack()
    stack.primary_wavelength = 1.3892

    # 0. Fiber
    stack.add_surface(Surface(
        name="Fiber_FG025LJA",
        surface_type=SurfaceType.OBJECT,
        radius=0.0,
        thickness=d_fiber,
        glass="AIR",
        semi_diameter=25.0
    ))

    # 1-3. Collimator AC254-050-C
    stack.add_surface(Surface(
        name="AC254-050-C_Coll_Front",
        surface_type=SurfaceType.STANDARD,
        radius=34.0,
        thickness=6.0,
        glass="N-BK7",
        semi_diameter=12.7
    ))
    stack.add_surface(Surface(
        name="AC254-050-C_Coll_Mid",
        surface_type=SurfaceType.STANDARD,
        radius=-27.8,
        thickness=2.0,
        glass="SF5",
        semi_diameter=12.7
    ))
    stack.add_surface(Surface(
        name="AC254-050-C_Coll_Back",
        surface_type=SurfaceType.STANDARD,
        radius=-115.0,
        thickness=d_coll_grating,
        glass="AIR",
        semi_diameter=12.7
    ))

    # 4. Grating GR25-0316
    stack.add_surface(Surface(
        name="Grating_GR25-0316",
        surface_type=SurfaceType.GRATING,
        radius=0.0,
        thickness=d_grating_focus,
        glass="MIRROR",
        semi_diameter=12.7,
        tilt_y=grating_tilt,
        grating_lines_per_mm=300.0,
        grating_order=1
    ))

    # 5-7. Focusing Doublet AC254-040-C
    stack.add_surface(Surface(
        name="AC254-040-C_Focus_Front",
        surface_type=SurfaceType.STANDARD,
        radius=25.1,
        thickness=6.8,
        glass="N-BK7",
        semi_diameter=12.7
    ))
    stack.add_surface(Surface(
        name="AC254-040-C_Focus_Mid",
        surface_type=SurfaceType.STANDARD,
        radius=-20.2,
        thickness=2.0,
        glass="SF5",
        semi_diameter=12.7
    ))
    stack.add_surface(Surface(
        name="AC254-040-C_Focus_Back",
        surface_type=SurfaceType.STANDARD,
        radius=-84.0,
        thickness=d_focus_dmd,
        glass="AIR",
        semi_diameter=12.7
    ))

    # 8. DMD
    stack.add_surface(Surface(
        name="TI_DMD4500",
        surface_type=SurfaceType.STANDARD,
        radius=0.0,
        thickness=d_dmd_relay,
        glass="MIRROR",
        semi_diameter=12.7,
        tilt_y=dmd_tilt,
        aperture_type="Rectangular",
        aperture_x_half=4.332,
        aperture_y_half=2.4365
    ))

    # 9-11. Relay Doublet AC508-075-C (2" doublet)
    stack.add_surface(Surface(
        name="AC508-075-C_Relay_Front",
        surface_type=SurfaceType.STANDARD,
        radius=49.3,
        thickness=8.0,
        glass="N-BK7",
        semi_diameter=25.4
    ))
    stack.add_surface(Surface(
        name="AC508-075-C_Relay_Mid",
        surface_type=SurfaceType.STANDARD,
        radius=-39.0,
        thickness=3.0,
        glass="SF5",
        semi_diameter=25.4
    ))
    stack.add_surface(Surface(
        name="AC508-075-C_Relay_Back",
        surface_type=SurfaceType.STANDARD,
        radius=-170.0,
        thickness=d_relay_cond,
        glass="AIR",
        semi_diameter=25.4
    ))

    # 12-13. Condenser Lens (ACL12708U-C or doublet)
    stack.add_surface(Surface(
        name="ACL12708U-C_Cond_Front",
        surface_type=SurfaceType.ASPHERIC if (asph_k != 0 or asph_A4 != 0) else SurfaceType.STANDARD,
        radius=asph_R,
        thickness=4.5,
        glass="B270",
        semi_diameter=12.7,
        conic=asph_k,
        asphere_coeffs=[asph_A4] if asph_A4 != 0 else []
    ))
    stack.add_surface(Surface(
        name="ACL12708U-C_Cond_Back",
        surface_type=SurfaceType.STANDARD,
        radius=0.0,
        thickness=d_cond_det,
        glass="AIR",
        semi_diameter=12.7
    ))

    # 14. Detector
    stack.add_surface(Surface(
        name="Detector_InGaAs_1mm",
        surface_type=SurfaceType.DETECTOR,
        radius=0.0,
        thickness=0.0,
        glass="AIR",
        semi_diameter=0.5
    ))

    return stack

def trace_and_score(params):
    d_fiber, d_coll_grating, grating_tilt, d_grating_focus, d_focus_dmd, dmd_tilt, d_dmd_relay, d_relay_cond, d_cond_det = params
    stack = build_full_system(d_fiber, d_coll_grating, grating_tilt, d_grating_focus, d_focus_dmd, dmd_tilt, d_dmd_relay, d_relay_cond, d_cond_det)
    stack._update_accumulated_positions()
    opt = OpticalSystem(stack)

    na = 0.12
    max_theta = math.asin(na)
    fiber_r = 0.0125
    rays = []
    for wl in WAVELENGTHS:
        rays.append(Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl))
        for angle in [max_theta, -max_theta]:
            rays.append(Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([math.sin(angle), 0.0, math.cos(angle)]), wavelength=wl))
            rays.append(Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, math.sin(angle), math.cos(angle)]), wavelength=wl))
        for pos_r in [fiber_r, -fiber_r]:
            rays.append(Ray(pos=np.array([pos_r, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl))
            rays.append(Ray(pos=np.array([0.0, pos_r, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl))

    traced = opt.trace_rays(rays)
    vignetted_count = sum(1 for r in traced if r.vignetted)

    # Det hits
    det_radii = []
    for r in traced:
        for h in r.history:
            if h["surface_index"] == 14:
                det_radii.append(math.sqrt(h["x"]**2 + h["y"]**2))

    max_det_r = max(det_radii) if det_radii else 999.0
    return vignetted_count, max_det_r, len(rays), det_radii

# Let's test a grid of parameters for d_dmd_relay, d_relay_cond, d_cond_det
print("Testing relay and condenser distances...")
best_res = None
best_score = 99999

d_fiber = 64.57
d_coll_grating = 15.0
grating_tilt = 12.0
d_grating_focus = 15.0
d_focus_dmd = 36.5
dmd_tilt = 12.0

for d_dmd_relay in np.linspace(40.0, 80.0, 9):
    for d_relay_cond in np.linspace(2.0, 30.0, 8):
        for d_cond_det in np.linspace(3.0, 8.0, 11):
            params = [d_fiber, d_coll_grating, grating_tilt, d_grating_focus, d_focus_dmd, dmd_tilt, d_dmd_relay, d_relay_cond, d_cond_det]
            vig, max_r, tot, r_list = trace_and_score(params)
            if vig == 0 and max_r < 0.50:
                print(f"FOUND 0% VIGNETTING: d_dmd_relay={d_dmd_relay:.1f}, d_relay_cond={d_relay_cond:.1f}, d_cond_det={d_cond_det:.2f} -> Max Det R = {max_r:.4f} mm")
                if max_r < best_score:
                    best_score = max_r
                    best_res = params

print(f"Grid search done. Best max Det R = {best_score:.4f} mm")
