import sys
import os
import math
import numpy as np
import scipy.optimize as opt

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import generate_fiber_phase_space_rays, RAMAN_WAVELENGTHS_UM

def build_stack(params):
    T0, T3, T4, T7, T8, T11, T13, grat_tilt, dmd_tilt = params
    stack = SurfaceStack()
    stack.primary_wavelength = 1.38919
    
    surfaces = [
        # 0. Fiber
        Surface(name="Fiber_FG025LJA", surface_type=SurfaceType.OBJECT, radius=0.0, thickness=float(T0), glass="AIR", semi_diameter=25.0),
        # 1-3. Collimating Doublet (AC254-050-C)
        Surface(name="AC254-050-C_Coll_Front", surface_type=SurfaceType.STANDARD, radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7),
        Surface(name="AC254-050-C_Coll_Mid", surface_type=SurfaceType.STANDARD, radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7),
        Surface(name="AC254-050-C_Coll_Back", surface_type=SurfaceType.STANDARD, radius=-115.0, thickness=float(T3), glass="AIR", semi_diameter=12.7),
        # 4. Grating (GR25-0316)
        Surface(name="Grating_GR25-0316", surface_type=SurfaceType.GRATING, radius=0.0, thickness=float(T4), glass="MIRROR", semi_diameter=12.7, tilt_y=float(grat_tilt), grating_lines_per_mm=300.0, grating_order=1),
        # 5-7. Focusing Doublet (AC254-040-C)
        Surface(name="AC254-040-C_Focus_Front", surface_type=SurfaceType.STANDARD, radius=25.1, thickness=6.8, glass="N-BK7", semi_diameter=12.7),
        Surface(name="AC254-040-C_Focus_Mid", surface_type=SurfaceType.STANDARD, radius=-20.2, thickness=2.0, glass="SF5", semi_diameter=12.7),
        Surface(name="AC254-040-C_Focus_Back", surface_type=SurfaceType.STANDARD, radius=-84.0, thickness=float(T7), glass="AIR", semi_diameter=12.7),
        # 8. DMD (TI DLP4500)
        Surface(name="TI_DMD4500", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=float(T8), glass="MIRROR", semi_diameter=12.7, tilt_y=float(dmd_tilt)),
        # 9-11. 4F Relay Doublet (AC508-075-C)
        Surface(name="AC508-075-C_Relay_Front", surface_type=SurfaceType.STANDARD, radius=49.3, thickness=8.0, glass="N-BK7", semi_diameter=25.4),
        Surface(name="AC508-075-C_Relay_Mid", surface_type=SurfaceType.STANDARD, radius=-39.0, thickness=3.0, glass="SF5", semi_diameter=25.4),
        Surface(name="AC508-075-C_Relay_Back", surface_type=SurfaceType.STANDARD, radius=-170.0, thickness=float(T11), glass="AIR", semi_diameter=25.4),
        # 12-13. Condenser Lens (ACL12708U-C)
        Surface(name="ACL12708U-C_Cond_Front", surface_type=SurfaceType.ASPHERIC, radius=4.16, thickness=4.5, glass="B270", semi_diameter=12.7, conic=-0.6265, asphere_coeffs=[1.5e-5]),
        Surface(name="ACL12708U-C_Cond_Back", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=float(T13), glass="AIR", semi_diameter=12.7),
        # 14. Single-Element InGaAs Detector
        Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.5)
    ]
    for s in surfaces:
        stack.add_surface(s)
    return stack

def evaluate(params, fast=True):
    stack = build_stack(params)
    # Generate subset of rays for fast evaluation (chief + 4 marginal per wl)
    if fast:
        rays = []
        na = 0.12
        max_theta = math.asin(na)
        for wl in RAMAN_WAVELENGTHS_UM:
            rays.append(Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl))
            rays.append(Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([math.sin(max_theta), 0.0, math.cos(max_theta)]), wavelength=wl))
            rays.append(Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([-math.sin(max_theta), 0.0, math.cos(max_theta)]), wavelength=wl))
            rays.append(Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, math.sin(max_theta), math.cos(max_theta)]), wavelength=wl))
            rays.append(Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, -math.sin(max_theta), math.cos(max_theta)]), wavelength=wl))
    else:
        rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
        
    opt_sys = OpticalSystem(stack)
    traced = opt_sys.trace_rays(rays)
    
    vig_count = sum(1 for r in traced if r.vignetted)
    
    det_hits = []
    dmd_xs = []
    dmd_ys = []
    for r in traced:
        for h in r.history:
            if h['surface_index'] == 8:
                dmd_xs.append(h['x'])
                dmd_ys.append(h['y'])
            if h['surface_index'] == 14:
                det_hits.append((h['x'], h['y']))
                
    dmd_span = (max(dmd_xs) - min(dmd_xs)) if dmd_xs else 999.0
    dmd_max_y = max(map(abs, dmd_ys)) if dmd_ys else 999.0
    
    if det_hits:
        det_r = [math.sqrt(x**2 + y**2) for x, y in det_hits]
        max_det_r = max(det_r)
        cx = np.mean([x for x, y in det_hits])
        cy = np.mean([y for x, y in det_hits])
        centroid_offset = math.sqrt(cx**2 + cy**2)
    else:
        max_det_r = 999.0
        centroid_offset = 999.0
        
    return {
        "vig": vig_count,
        "total": len(rays),
        "dmd_span": dmd_span,
        "dmd_max_y": dmd_max_y,
        "max_det_r": max_det_r,
        "centroid_offset": centroid_offset,
        "stack": stack,
        "traced": traced
    }

def merit_function(params):
    res = evaluate(params, fast=True)
    cost = 0.0
    # Penalty for vignetting
    cost += res["vig"] * 1000.0
    # Penalty for DMD span exceeding 8.664 mm
    if res["dmd_span"] > 8.664:
        cost += (res["dmd_span"] - 8.664) * 200.0
    # Penalty for detector spot exceeding 0.50 mm
    if res["max_det_r"] > 0.50:
        cost += (res["max_det_r"] - 0.50) * 1000.0
    cost += res["max_det_r"] * 50.0
    cost += res["centroid_offset"] * 50.0
    return cost

def main():
    # Initial parameters: [T0, T3, T4, T7, T8, T11, T13, grat_tilt, dmd_tilt]
    p0 = [64.57, 15.0, 10.0, 36.5, 75.0, 20.0, 4.85, 12.0, 12.0]
    
    print("Initial evaluation:")
    res0 = evaluate(p0, fast=False)
    print(f"Vig: {res0['vig']}/{res0['total']}, DMD Span: {res0['dmd_span']:.3f} mm, Max Det R: {res0['max_det_r']:.4f} mm, Centroid: {res0['centroid_offset']:.4f} mm")

if __name__ == "__main__":
    main()
