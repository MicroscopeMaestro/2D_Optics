"""
Hybrid DMD4500 Compressive Raman Spectrometer Design.
Uses Thorlabs Zemax Optical Components (AC254 doublets and ACL aspheres).
Optimized for 10:1 demagnification to fit the 1mm detector.
"""

import os
import sys
import numpy as np
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.engine.optimizer import OpticalOptimizer

def build_hybrid_spectrometer() -> SurfaceStack:
    stack = SurfaceStack()
    stack.primary_wavelength = 0.785

    # 0. Fiber Input
    stack.add_surface(Surface(
        name="Fiber_Head_25um_Input",
        surface_type=SurfaceType.OBJECT,
        thickness=28.5, # Initial guess, will be optimized
        glass="AIR",
        semi_diameter=0.0125,
        aperture_type="Circular",
        aperture_r_outer=1.5875
    ))

    # 1. L1 (Collimator): AC254-030-B (f=30 mm) Front
    stack.add_surface(Surface(
        name="L1_AC254-030-B_Front",
        surface_type=SurfaceType.STANDARD,
        radius=18.5,
        thickness=7.5,
        glass="N-BK7",
        semi_diameter=12.7,
        aperture_r_outer=12.7
    ))

    # 2. L1 Mid
    stack.add_surface(Surface(
        name="L1_AC254-030-B_Mid",
        surface_type=SurfaceType.STANDARD,
        radius=-14.8,
        thickness=2.0,
        glass="SF5",
        semi_diameter=12.7,
        aperture_r_outer=12.7
    ))

    # 3. L1 Back
    stack.add_surface(Surface(
        name="L1_AC254-030-B_Back",
        surface_type=SurfaceType.STANDARD,
        radius=-57.2,
        thickness=30.0, # Collimated space to grating
        glass="AIR",
        semi_diameter=12.7,
        aperture_r_outer=12.7
    ))

    # 4. Diffraction Grating (Thorlabs GR25-0608)
    stack.add_surface(Surface(
        name="Grating_Thorlabs_GR25-0608",
        surface_type=SurfaceType.GRATING,
        radius=0.0,
        thickness=30.0, # Space to L2
        glass="MIRROR",
        semi_diameter=12.5,
        tilt_x=18.2,
        grating_lines_per_mm=600.0,
        grating_order=1,
        aperture_type="Rectangular",
        aperture_x_half=12.5,
        aperture_y_half=12.5
    ))

    # 5. L2 (Focusing): AC254-040-B (f=40 mm) Front
    stack.add_surface(Surface(
        name="L2_AC254-040-B_Front",
        surface_type=SurfaceType.STANDARD,
        radius=25.1,
        thickness=6.8,
        glass="N-BK7",
        semi_diameter=12.7,
        aperture_r_outer=12.7
    ))

    # 6. L2 Mid
    stack.add_surface(Surface(
        name="L2_AC254-040-B_Mid",
        surface_type=SurfaceType.STANDARD,
        radius=-20.2,
        thickness=2.0,
        glass="SF5",
        semi_diameter=12.7,
        aperture_r_outer=12.7
    ))

    # 7. L2 Back
    stack.add_surface(Surface(
        name="L2_AC254-040-B_Back",
        surface_type=SurfaceType.STANDARD,
        radius=-84.0,
        thickness=36.8, # Initial guess, will be optimized
        glass="AIR",
        semi_diameter=12.7,
        aperture_r_outer=12.7
    ))

    # 8. TI DLP4500 DMD Array
    stack.add_surface(Surface(
        name="DMD4500_WXGA_Array",
        surface_type=SurfaceType.STANDARD,
        radius=0.0,
        thickness=300.0, # L3 is f=300, so DMD must be 300mm away
        glass="MIRROR",
        semi_diameter=4.332,
        tilt_x=12.0,
        aperture_type="Rectangular",
        aperture_x_half=4.332,
        aperture_y_half=3.465
    ))

    # 9. L3 (Relay Collector): AC254-300-B (f=300 mm) FLIPPED (flatter side faces DMD)
    stack.add_surface(Surface(
        name="L3_AC254-300-B_Back_Flipped",
        surface_type=SurfaceType.STANDARD,
        radius=198.8,
        thickness=4.2,
        glass="N-BK7",
        semi_diameter=12.7,
        aperture_r_outer=12.7
    ))

    # 10. L3 Mid
    stack.add_surface(Surface(
        name="L3_AC254-300-B_Mid_Flipped",
        surface_type=SurfaceType.STANDARD,
        radius=-72.8,
        thickness=2.0,
        glass="SF5",
        semi_diameter=12.7,
        aperture_r_outer=12.7
    ))

    # 11. L3 Front
    stack.add_surface(Surface(
        name="L3_AC254-300-B_Front_Flipped",
        surface_type=SurfaceType.STANDARD,
        radius=-177.3,
        thickness=50.0, # Collimated relay space
        glass="AIR",
        semi_diameter=12.7,
        aperture_r_outer=12.7
    ))

    # 12. L4 (Relay Condenser): AC254-030-B (f=30 mm) Front
    stack.add_surface(Surface(
        name="L4_AC254-030-B_Front",
        surface_type=SurfaceType.STANDARD,
        radius=18.5,
        thickness=7.5,
        glass="N-BK7",
        semi_diameter=12.7,
        aperture_r_outer=12.7
    ))

    # 13. L4 Mid
    stack.add_surface(Surface(
        name="L4_AC254-030-B_Mid",
        surface_type=SurfaceType.STANDARD,
        radius=-14.8,
        thickness=2.0,
        glass="SF5",
        semi_diameter=12.7,
        aperture_r_outer=12.7
    ))

    # 13b. L4 Back
    stack.add_surface(Surface(
        name="L4_AC254-030-B_Back",
        surface_type=SurfaceType.STANDARD,
        radius=-57.2,
        thickness=30.0, # L4 is f=30, so detector is 30mm away
        glass="AIR",
        semi_diameter=12.7,
        aperture_r_outer=12.7
    ))

    # 14. 1.0mm Detector
    stack.add_surface(Surface(
        name="Detector_1mm",
        surface_type=SurfaceType.DETECTOR,
        thickness=0.0,
        glass="AIR",
        semi_diameter=0.50,
        aperture_r_outer=4.05
    ))

    stack._update_accumulated_positions()
    return stack

def generate_fiber_rays(wavelengths, num_rays_per_ring=12, fiber_na=0.10):
    rays = []
    theta_max = math.asin(fiber_na)
    z_dist = 28.5
    for wl in wavelengths:
        rays.append(Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=wl))
        for ring_frac in [0.5, 1.0]:
            theta_cone = ring_frac * theta_max
            r_pupil = z_dist * math.tan(theta_cone)
            n_pts = int(num_rays_per_ring * ring_frac)
            for p in range(max(6, n_pts)):
                phi = 2.0 * np.pi * p / max(6, n_pts)
                px = r_pupil * np.cos(phi)
                py = r_pupil * np.sin(phi)
                dir_v = np.array([px, py, z_dist])
                dir_v /= np.linalg.norm(dir_v)
                rays.append(Ray(pos=np.array([0., 0., 0.]), dir=dir_v, wavelength=wl))
    return rays

def evaluate_hybrid_spectrometer():
    stack = build_hybrid_spectrometer()
    opt_sys = OpticalSystem(stack)
    
    print("Optimizing air gap thicknesses for hybrid thorlabs design...")
    optimizer = OpticalOptimizer(opt_sys, wavelengths=0.785, method="Nelder-Mead", max_iter=100)
    optimizer.optimize(variable_indices={"thickness": [0, 7, 8, 14]})
    print(f"Optimization Complete!")

    wavelengths = [0.785, 0.808, 0.830, 0.850, 0.880]
    rays = generate_fiber_rays(wavelengths)
    traced_rays = opt_sys.trace_rays(rays)

    print("=" * 80)
    print("HYBRID ZEMAX SPECTROMETER DESIGN AUDIT (THORLABS ACHROMATS)")
    print("=" * 80)

    det_hits_x, det_hits_y = [], []
    valid_rays = 0
    for tr in traced_rays:
        for h in tr.history:
            if h["surface_index"] == 15:
                valid_rays += 1
                det_hits_x.append(h["x"])
                det_hits_y.append(h["y"])

    det_span_x = np.ptp(det_hits_x) if det_hits_x else 0.0
    det_span_y = np.ptp(det_hits_y) if det_hits_y else 0.0
    det_max_radius = max(np.max(np.abs(det_hits_x)), np.max(np.abs(det_hits_y))) if det_hits_x else 0.0
    encircled_pct = (valid_rays / len(rays)) * 100.0 if rays else 0.0

    print(f"Detector Plane (Surface 15, Ø1.0 mm Active Area):")
    print(f"  Spot Span:             X = {det_span_x:.3f} mm, Y = {det_span_y:.3f} mm")
    print(f"  Max Radial Extent:     {det_max_radius:.3f} mm (Must be under 0.50 mm)")
    print(f"  Detector Encircled E:  {encircled_pct:.1f}%")

    base_dir = r"c:\Users\Juan\Documents\GitHub\2D_Optics"
    app_model_path = os.path.join(base_dir, "hybrid_zemax_spectrometer_app_model.json")
    stack.save_to_json(app_model_path, wavelengths=wavelengths)
    print(f"\nNative App Model exported: {app_model_path}")

if __name__ == "__main__":
    evaluate_hybrid_spectrometer()
