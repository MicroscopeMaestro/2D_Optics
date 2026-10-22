import sys, os, math
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, r"c:\Users\Juan\Documents\GitHub\2D_Optics")
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem, Ray
from v2.engine.surface import Surface, SurfaceType

def build_system(params):
    tilt_m1, tilt_m2, tilt_m3, tilt_m4, d_m2_dmd, d_dmd_m3, d_m3_m4, d_m4_det = params
    
    stack = SurfaceStack()
    wls = [0.800, 0.815, 0.830, 0.845, 0.860]
    stack.primary_wavelength = 0.830
    stack.wavelengths = wls

    stack.add_surface(Surface(
        name="Input_Slit_30um",
        surface_type=SurfaceType.OBJECT,
        thickness=48.5,
        glass="AIR",
        semi_diameter=12.7,
        aperture_type="Rectangular",
        aperture_x_half=0.250,
        aperture_y_half=0.015
    ))

    stack.add_surface(Surface(
        name="Thorlabs_SM1D12_Iris_Stop",
        surface_type=SurfaceType.STOP,
        thickness=1.5,
        glass="AIR",
        semi_diameter=5.5,
        aperture_type="Circular",
        aperture_r_outer=5.5
    ))

    stack.add_surface(Surface(
        name="M1_Collimator_CM254-050-P01",
        surface_type=SurfaceType.STANDARD,
        radius=-100.0,
        thickness=50.0,
        glass="MIRROR",
        semi_diameter=12.7,
        tilt_x=tilt_m1,
        aperture_type="Circular",
        aperture_r_outer=12.5
    ))

    stack.add_surface(Surface(
        name="Grating_GR25-0608_600lpmm",
        surface_type=SurfaceType.GRATING,
        radius=0.0,
        thickness=60.0,
        glass="MIRROR",
        semi_diameter=12.5,
        tilt_x=18.2,
        grating_lines_per_mm=600.0,
        grating_order=1,
        aperture_type="Rectangular",
        aperture_x_half=12.5,
        aperture_y_half=12.5
    ))

    stack.add_surface(Surface(
        name="M2_Camera_CM254-075-P01",
        surface_type=SurfaceType.STANDARD,
        radius=-150.0,
        thickness=d_m2_dmd,
        glass="MIRROR",
        semi_diameter=12.7,
        tilt_x=tilt_m2,
        aperture_type="Circular",
        aperture_r_outer=12.5
    ))

    stack.add_surface(Surface(
        name="DMD4500_Spatial_Modulator",
        surface_type=SurfaceType.STANDARD,
        radius=0.0,
        thickness=d_dmd_m3,
        glass="MIRROR",
        semi_diameter=6.0,
        tilt_x=12.0,
        aperture_type="Rectangular",
        aperture_x_half=4.9,
        aperture_y_half=3.0
    ))

    stack.add_surface(Surface(
        name="M3_Relay_Collector_CM254-150-P01",
        surface_type=SurfaceType.STANDARD,
        radius=-300.0,
        thickness=d_m3_m4,
        glass="MIRROR",
        semi_diameter=12.7,
        tilt_x=tilt_m3,
        aperture_type="Circular",
        aperture_r_outer=12.5
    ))

    stack.add_surface(Surface(
        name="M4_Relay_Condenser_CM254-019-P01",
        surface_type=SurfaceType.STANDARD,
        radius=-38.0,
        thickness=d_m4_det,
        glass="MIRROR",
        semi_diameter=12.7,
        tilt_x=tilt_m4,
        aperture_type="Circular",
        aperture_r_outer=12.5
    ))

    stack.add_surface(Surface(
        name="Detector_0.8mm_Photodiode",
        surface_type=SurfaceType.DETECTOR,
        thickness=0.0,
        glass="AIR",
        semi_diameter=1.0, # allow tracing to record all hits without early clipping
        aperture_type="Circular",
        aperture_r_outer=1.0
    ))

    stack._update_accumulated_positions()
    return stack

def eval_params(params):
    stack = build_system(params)
    opt = OpticalSystem(stack)
    wls = stack.wavelengths
    theta_max = math.asin(0.10)
    rays = []
    for wl in wls:
        for x in [-0.25, 0.0, 0.25]:
            for y in [-0.015, 0.0, 0.015]:
                rays.append(Ray(pos=np.array([x, y, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl))
                for ring in [0.5, 1.0]:
                    for p in range(8):
                        phi = 2 * math.pi * p / 8
                        kx = math.sin(ring * theta_max) * math.cos(phi)
                        ky = math.sin(ring * theta_max) * math.sin(phi)
                        kz = math.cos(ring * theta_max)
                        rays.append(Ray(pos=np.array([x, y, 0.0]), dir=np.array([kx, ky, kz]), wavelength=wl))

    traced = [opt.trace_ray(r) for r in rays]
    vignetted = [tr for tr in traced if tr.vignetted]
    valid = [tr for tr in traced if not tr.vignetted]

    det_hits = []
    for tr in valid:
        for h in tr.history:
            if h["surface_index"] == 8:
                det_hits.append((h["x_local"], h["y_local"]))

    if not det_hits or len(valid) < len(rays) * 0.95:
        return 999.0, 999.0, len(vignetted), len(rays)

    det_xs = np.array([p[0] for p in det_hits])
    det_ys = np.array([p[1] for p in det_hits])
    det_radii = np.sqrt(det_xs**2 + det_ys**2)
    max_r = float(np.max(det_radii))
    rms_r = float(np.sqrt(np.mean(det_radii**2)))
    return max_r, rms_r, len(vignetted), len(rays)

print("--- Testing Confocal Relay Spacing (d_m3_m4 = 169 mm) ---")
# initial params: tilt_m1, tilt_m2, tilt_m3, tilt_m4, d_m2_dmd, d_dmd_m3, d_m3_m4, d_m4_det
for d_m3_m4 in [50.0, 100.0, 150.0, 169.0, 180.0]:
    for d_det in np.linspace(17.5, 19.5, 9):
        p = [8.5, -8.5, 8.0, -8.0, 73.5, 150.0, d_m3_m4, d_det]
        max_r, rms_r, vig, total = eval_params(p)
        if max_r < 0.400:
            print(f"PASS: d_m3_m4={d_m3_m4:.1f}, d_det={d_det:.2f} -> max_r={max_r:.4f} mm, rms_r={rms_r:.4f} mm, vig={vig}/{total}")
