import sys, os, math
import numpy as np

sys.path.insert(0, r"c:\Users\Juan\Documents\GitHub\2D_Optics")
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem, Ray
from v2.engine.surface import Surface, SurfaceType

def test_relay(f4=19.0, tilt_m3=8.0, tilt_m4=-8.0, dist_m3_m4=50.0, dist_m4_det=18.8, d_dmd_m3=150.0):
    stack = SurfaceStack()
    wls = [0.785, 0.808, 0.830, 0.850, 0.880]
    stack.primary_wavelength = 0.830
    stack.wavelengths = wls

    # S0: Slit (30 um width, 500 um height)
    stack.add_surface(Surface(
        name="Input_Slit_30um",
        surface_type=SurfaceType.OBJECT,
        thickness=48.5,
        glass="AIR",
        semi_diameter=12.7,
        aperture_type="Rectangular",
        aperture_x_half=0.250, # 500 um slit length along X
        aperture_y_half=0.015, # 30 um slit width along Y (dispersion direction)
        substrate_thickness=6.0
    ))

    # S1: Aperture Stop (SM1D12, r=5.2 mm <= 6.0 mm)
    stack.add_surface(Surface(
        name="Thorlabs_SM1D12_Iris_Stop",
        surface_type=SurfaceType.STOP,
        thickness=1.5,
        glass="AIR",
        semi_diameter=5.5,
        aperture_type="Circular",
        aperture_r_outer=5.5
    ))

    # S2: Collimator M1 (f1=50mm, ROC=-100mm)
    stack.add_surface(Surface(
        name="M1_Collimator_CM254-050-P01",
        surface_type=SurfaceType.STANDARD,
        radius=-100.0,
        thickness=50.0,
        glass="MIRROR",
        semi_diameter=12.7,
        tilt_x=8.5,
        aperture_type="Circular",
        aperture_r_outer=12.5
    ))

    # S3: Grating GR25-0608 (600 l/mm)
    stack.add_surface(Surface(
        name="Grating_GR25-0608_600lpmm",
        surface_type=SurfaceType.GRATING,
        radius=0.0,
        thickness=60.0,
        glass="MIRROR",
        semi_diameter=12.7,
        tilt_x=18.2,
        grating_lines_per_mm=600.0,
        grating_order=1,
        aperture_type="Rectangular",
        aperture_x_half=12.5,
        aperture_y_half=12.5
    ))

    # S4: Camera Mirror M2 (f2=75mm, ROC=-150mm)
    stack.add_surface(Surface(
        name="M2_Camera_CM254-075-P01",
        surface_type=SurfaceType.STANDARD,
        radius=-150.0,
        thickness=73.5,
        glass="MIRROR",
        semi_diameter=12.7,
        tilt_x=-8.5,
        aperture_type="Circular",
        aperture_r_outer=12.5
    ))

    # S5: DMD4500 (5mm active area)
    stack.add_surface(Surface(
        name="DMD4500_Spatial_Modulator",
        surface_type=SurfaceType.STANDARD,
        radius=0.0,
        thickness=d_dmd_m3,
        glass="MIRROR",
        semi_diameter=6.0,
        tilt_x=12.0,
        aperture_type="Rectangular",
        aperture_x_half=3.5,
        aperture_y_half=3.5
    ))

    # S6: Relay Collector M3 (f3=150mm, ROC=-300mm)
    stack.add_surface(Surface(
        name="M3_Relay_Collector_CM254-150-P01",
        surface_type=SurfaceType.STANDARD,
        radius=-300.0,
        thickness=dist_m3_m4,
        glass="MIRROR",
        semi_diameter=12.7,
        tilt_x=tilt_m3,
        aperture_type="Circular",
        aperture_r_outer=12.5
    ))

    # S7: Relay Condenser M4
    roc4 = -2.0 * f4
    stack.add_surface(Surface(
        name=f"M4_Relay_Condenser_{f4}mm",
        surface_type=SurfaceType.STANDARD,
        radius=roc4,
        thickness=dist_m4_det,
        glass="MIRROR",
        semi_diameter=12.7 if f4 >= 19.0 else 6.35,
        tilt_x=tilt_m4,
        aperture_type="Circular",
        aperture_r_outer=12.5 if f4 >= 19.0 else 6.0
    ))

    # S8: Detector (0.8 mm diameter -> 0.400 mm radius)
    stack.add_surface(Surface(
        name="Detector_0.8mm_Photodiode",
        surface_type=SurfaceType.DETECTOR,
        thickness=0.0,
        glass="AIR",
        semi_diameter=0.400,
        aperture_type="Circular",
        aperture_r_outer=0.400
    ))

    stack._update_accumulated_positions()
    opt = OpticalSystem(stack)

    # Trace dense rays
    theta_max = math.asin(0.10)
    rays = []
    for wl in wls:
        for x in [-0.25, 0.0, 0.25]:
            for y in [-0.015, 0.0, 0.015]:
                rays.append(Ray(pos=np.array([x, y, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl))
                for ring in [0.4, 0.7, 1.0]:
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
    dmd_hits = []
    for tr in valid:
        for h in tr.history:
            if h["surface_index"] == 8:
                det_hits.append((h["x_local"], h["y_local"]))
            elif h["surface_index"] == 5:
                dmd_hits.append((h["x_local"], h["y_local"]))

    if det_hits:
        det_xs = np.array([p[0] for p in det_hits])
        det_ys = np.array([p[1] for p in det_hits])
        det_radii = np.sqrt(det_xs**2 + det_ys**2)
        max_r = np.max(det_radii)
        rms_r = np.sqrt(np.mean(det_radii**2))
    else:
        max_r = 999.0
        rms_r = 999.0

    if dmd_hits:
        dmd_xs = [p[0] for p in dmd_hits]
        dmd_ys = [p[1] for p in dmd_hits]
        dmd_span_x = max(dmd_xs) - min(dmd_xs)
        dmd_span_y = max(dmd_ys) - min(dmd_ys)
    else:
        dmd_span_x = dmd_span_y = 0.0

    print(f"Results for f4={f4}mm, tilt_m3={tilt_m3}, tilt_m4={tilt_m4}, dist_m3_m4={dist_m3_m4}, dist_m4_det={dist_m4_det}:")
    print(f"  Valid: {len(valid)}/{len(rays)} ({len(valid)/len(rays)*100:.1f}%), Vignetted: {len(vignetted)}")
    print(f"  DMD span: X={dmd_span_x:.3f} mm, Y={dmd_span_y:.3f} mm (Dispersion)")
    print(f"  Detector: max_r = {max_r:.4f} mm (<=0.400mm? {max_r <= 0.400}), rms_r = {rms_r:.4f} mm")
    return max_r, rms_r, len(vignetted)

print("--- Testing CM254-019 (f4=19mm) ---")
test_relay(f4=19.0, dist_m4_det=18.8)
print("\n--- Testing CM127-012 (f4=12mm) ---")
test_relay(f4=12.0, dist_m4_det=11.8)
