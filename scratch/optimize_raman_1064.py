"""
Comprehensive multi-stage design and optimization for 15-surface Hybrid Raman 1064nm Spectrometer.
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
from optics_app.analysis.spot import SpotDiagram

WAVELENGTHS = [1.1113, 1.2000, 1.3000, 1.3892, 1.5000, 1.6000, 1.6671]

def build_system(params):
    # params: [d_fiber, d_coll_to_grat, grating_tilt_y, d_grat_to_foc, d_foc_to_dmd, dmd_tilt_y, d_dmd_to_relay, d_relay_to_cond, d_cond_to_det]
    d_fiber = params[0]
    d_coll_to_grat = params[1]
    grating_tilt = params[2]
    d_grat_to_foc = params[3]
    d_foc_to_dmd = params[4]
    dmd_tilt = params[5]
    d_dmd_to_relay = params[6]
    d_relay_to_cond = params[7]
    d_cond_to_det = params[8]

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

    # 1-3. Collimating Doublet (AC254-050-C: R1=34.0, T1=6.0, BK7; R2=-27.8, T2=2.0, SF5; R3=-115.0)
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
        thickness=d_coll_to_grat,
        glass="AIR",
        semi_diameter=12.7
    ))

    # 4. Grating (GR25-0316: 300 l/mm)
    stack.add_surface(Surface(
        name="Grating_GR25-0316",
        surface_type=SurfaceType.GRATING,
        radius=0.0,
        thickness=d_grat_to_foc,
        glass="MIRROR",
        semi_diameter=12.7,
        tilt_y=grating_tilt,
        grating_lines_per_mm=300.0,
        grating_order=1
    ))

    # 5-7. Focusing Doublet (AC254-040-C: R1=25.1, T1=6.8, BK7; R2=-20.2, T2=2.0, SF5; R3=-84.0)
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
        thickness=d_foc_to_dmd,
        glass="AIR",
        semi_diameter=12.7
    ))

    # 8. DMD (Texas Instruments DLP4500)
    stack.add_surface(Surface(
        name="TI_DMD4500",
        surface_type=SurfaceType.STANDARD,
        radius=0.0,
        thickness=d_dmd_to_relay,
        glass="MIRROR",
        semi_diameter=12.7,
        tilt_y=dmd_tilt,
        aperture_type="Rectangular",
        aperture_x_half=4.332,
        aperture_y_half=2.4365
    ))

    # 9-11. Relay Doublet (AC508-075-C: R1=49.3, T1=8.0, BK7; R2=-39.0, T2=3.0, SF5; R3=-170.0)
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
        thickness=d_relay_to_cond,
        glass="AIR",
        semi_diameter=25.4
    ))

    # 12-13. Aspheric Condenser (ACL12708U-C: R1=4.16, k=-0.6265, A4=1.5e-5, T1=4.5, B270; R2=0, T2=d_cond_to_det)
    stack.add_surface(Surface(
        name="ACL12708U-C_Cond_Front",
        surface_type=SurfaceType.ASPHERIC,
        radius=4.16,
        thickness=4.5,
        glass="B270",
        semi_diameter=12.7,
        conic=-0.6265,
        asphere_coeffs=[1.5e-5]
    ))
    stack.add_surface(Surface(
        name="ACL12708U-C_Cond_Back",
        surface_type=SurfaceType.STANDARD,
        radius=0.0,
        thickness=d_cond_to_det,
        glass="AIR",
        semi_diameter=12.7
    ))

    # 14. Single-Element InGaAs Detector
    stack.add_surface(Surface(
        name="Detector_InGaAs_1mm",
        surface_type=SurfaceType.DETECTOR,
        radius=0.0,
        thickness=0.0,
        glass="AIR",
        semi_diameter=0.5
    ))

    return stack

def evaluate(params):
    stack = build_system(params)
    stack._update_accumulated_positions()
    opt = OpticalSystem(stack)

    # Launch rays
    na = 0.12
    max_theta = math.asin(na)
    fiber_r = 0.0125
    rays = []
    for wl in WAVELENGTHS:
        # Chief ray
        rays.append(Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl))
        # Marginal rays
        for angle in [max_theta, -max_theta]:
            rays.append(Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([math.sin(angle), 0.0, math.cos(angle)]), wavelength=wl))
            rays.append(Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, math.sin(angle), math.cos(angle)]), wavelength=wl))
        for pos_r in [fiber_r, -fiber_r]:
            rays.append(Ray(pos=np.array([pos_r, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl))
            rays.append(Ray(pos=np.array([0.0, pos_r, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl))

    traced = opt.trace_rays(rays)
    vignetted_count = sum(1 for r in traced if r.vignetted)

    # DMD metrics (surf 8)
    dmd_xs = []
    dmd_ys = []
    for r in traced:
        for h in r.history:
            if h["surface_index"] == 8:
                dmd_xs.append(h["x"])
                dmd_ys.append(h["y"])

    # Detector metrics (surf 14)
    det_radii = []
    for r in traced:
        for h in r.history:
            if h["surface_index"] == 14:
                det_radii.append(math.sqrt(h["x"]**2 + h["y"]**2))

    dmd_span = (max(dmd_xs) - min(dmd_xs)) if dmd_xs else 999.0
    dmd_max_y = max(map(abs, dmd_ys)) if dmd_ys else 999.0
    max_det_r = max(det_radii) if det_radii else 999.0

    return {
        "vignetted": vignetted_count,
        "total_rays": len(rays),
        "dmd_span": dmd_span,
        "dmd_max_y": dmd_max_y,
        "max_det_r": max_det_r,
        "traced": traced
    }

if __name__ == "__main__":
    init_params = [64.57, 15.0, 12.0, 15.0, 36.5, 12.0, 75.0, 10.0, 4.85]
    res = evaluate(init_params)
    print(f"Init: Vignetted={res['vignetted']}/{res['total_rays']}, DMD Span={res['dmd_span']:.3f} mm, Max Det R={res['max_det_r']:.4f} mm")
