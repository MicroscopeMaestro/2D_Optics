import sys, os, math
import numpy as np

sys.path.insert(0, r"c:\Users\Juan\Documents\GitHub\2D_Optics")
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem, Ray
from v2.engine.surface import Surface, SurfaceType

def test_optimized_all_mirror(f4=12.0, wls=None):
    if wls is None:
        wls = [0.800, 0.815, 0.830, 0.845, 0.860]

    stack = SurfaceStack()
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
        aperture_x_half=0.250,
        aperture_y_half=0.015,
        substrate_thickness=6.0
    ))

    # S1: Aperture Stop (Thorlabs SM1D12 Iris, r=5.5mm <= 6.0mm)
    stack.add_surface(Surface(
        name="Thorlabs_SM1D12_Iris_Stop",
        surface_type=SurfaceType.STOP,
        thickness=1.5,
        glass="AIR",
        semi_diameter=5.5,
        aperture_type="Circular",
        aperture_r_outer=5.5
    ))

    # S2: Collimator M1 (CM254-050-P01, f1=50mm, ROC=-100mm)
    stack.add_surface(Surface(
        name="M1_Collimator_CM254-050-P01",
        surface_type=SurfaceType.STANDARD,
        radius=-100.0,
        thickness=50.0,
        glass="MIRROR",
        semi_diameter=12.7,
        tilt_x=8.5,
        aperture_type="Circular",
        aperture_r_outer=12.0
    ))

    # S3: Grating GR25-0608 (600 l/mm)
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

    # S4: Camera Mirror M2 (CM254-075-P01, f2=75mm, ROC=-150mm)
    stack.add_surface(Surface(
        name="M2_Camera_CM254-075-P01",
        surface_type=SurfaceType.STANDARD,
        radius=-150.0,
        thickness=73.5,
        glass="MIRROR",
        semi_diameter=12.7,
        tilt_x=-8.5,
        aperture_type="Circular",
        aperture_r_outer=12.0
    ))

    # S5: DMD4500 (TI DLP4500 5 mm active area)
    stack.add_surface(Surface(
        name="DMD4500_Spatial_Modulator",
        surface_type=SurfaceType.STANDARD,
        radius=0.0,
        thickness=150.0,
        glass="MIRROR",
        semi_diameter=6.0,
        tilt_x=12.0,
        aperture_type="Rectangular",
        aperture_x_half=4.9,
        aperture_y_half=3.0
    ))

    # S6: Relay Collector M3 (CM254-150-P01, f3=150mm, ROC=-300mm)
    stack.add_surface(Surface(
        name="M3_Relay_Collector_CM254-150-P01",
        surface_type=SurfaceType.STANDARD,
        radius=-300.0,
        thickness=50.0,
        glass="MIRROR",
        semi_diameter=12.7,
        tilt_x=8.0,
        aperture_type="Circular",
        aperture_r_outer=12.0
    ))

    # S7: Relay Condenser M4 (f4=12mm: CM127-012-P01, ROC=-24mm OR f4=19mm: CM254-019-P01, ROC=-38mm)
    roc4 = -2.0 * f4
    dist_det = f4 * 0.985
    stack.add_surface(Surface(
        name=f"M4_Relay_Condenser_{f4}mm",
        surface_type=SurfaceType.STANDARD,
        radius=roc4,
        thickness=dist_det,
        glass="MIRROR",
        semi_diameter=12.7 if f4 >= 19.0 else 6.35,
        tilt_x=-8.0,
        aperture_type="Circular",
        aperture_r_outer=12.0 if f4 >= 19.0 else 6.0
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
        max_r = float(np.max(det_radii))
        rms_r = float(np.sqrt(np.mean(det_radii**2)))
    else:
        max_r = rms_r = 999.0

    dmd_xs = [p[0] for p in dmd_hits]
    dmd_ys = [p[1] for p in dmd_hits]
    dmd_x_span = max(dmd_xs) - min(dmd_xs) if dmd_xs else 0.0
    dmd_y_span = max(dmd_ys) - min(dmd_ys) if dmd_ys else 0.0

    print(f"--- Results for f4={f4}mm ---")
    print(f"  Total Rays:       {len(rays)}")
    print(f"  Transmitted:      {len(valid)} ({len(valid)/len(rays)*100:.1f}%)")
    print(f"  Vignetted:        {len(vignetted)}")
    print(f"  DMD span:         X={dmd_x_span:.3f} mm, Y={dmd_y_span:.3f} mm (Dispersion)")
    print(f"  Detector Spot:    max_r = {max_r:.4f} mm, rms_r = {rms_r:.4f} mm")
    print(f"  100% Contained:   {max_r <= 0.400 and len(vignetted) == 0}")

    # Clearance checks:
    m1_orig = stack.surfaces[2].global_origin
    grating_orig = stack.surfaces[3].global_origin
    m1_to_grating = grating_orig - m1_orig
    t_z0 = (0.0 - m1_orig[2]) / m1_to_grating[2]
    pos_at_z0 = m1_orig + t_z0 * m1_to_grating
    slit_clearance = float(np.linalg.norm(pos_at_z0[:2]))

    m2_orig = stack.surfaces[4].global_origin
    dmd_orig = stack.surfaces[5].global_origin
    m2_to_dmd = dmd_orig - m2_orig
    t_z50 = (50.0 - m2_orig[2]) / m2_to_dmd[2]
    pos_at_z50 = m2_orig + t_z50 * m2_to_dmd
    m1_clearance = float(np.linalg.norm(pos_at_z50[:2] - m1_orig[:2]))

    print(f"  Slit Clearance (Z=0):        {slit_clearance:.2f} mm (> 12.7 mm: {slit_clearance > 12.7})")
    print(f"  Collimator Clearance (Z=50): {m1_clearance:.2f} mm (> 12.7 mm: {m1_clearance > 12.7})")

print("Testing with f4 = 12.0 mm (CM127-012-P01):")
test_optimized_all_mirror(f4=12.0)
print("\nTesting with f4 = 19.0 mm (CM254-019-P01):")
test_optimized_all_mirror(f4=19.0)
