import sys, os, math
import numpy as np

sys.path.insert(0, r"c:\Users\Juan\Documents\GitHub\2D_Optics")
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem, Ray
from v2.engine.surface import Surface, SurfaceType

def test_centering(tilt_m1=8.5, tilt_g=18.2, tilt_m2=-8.5, tilt_dmd=12.0, tilt_m3=8.0, tilt_m4=-8.0, d_m2_dmd=73.5, d_dmd_m3=150.0, d_m3_m4=50.0, d_m4_det=18.8):
    stack = SurfaceStack()
    wls = [0.785, 0.808, 0.830, 0.850, 0.880]
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
        semi_diameter=12.7,
        tilt_x=tilt_g,
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
        tilt_x=tilt_dmd,
        aperture_type="Rectangular",
        aperture_x_half=4.9, # DLP4500 active half-width
        aperture_y_half=3.0  # DLP4500 active half-height
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
        semi_diameter=0.400,
        aperture_type="Circular",
        aperture_r_outer=0.400
    ))

    stack._update_accumulated_positions()
    opt = OpticalSystem(stack)

    # First check chief rays at all wavelengths
    print("--- CHIEF RAYS AT DMD & DETECTOR ---")
    for wl in wls:
        r = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl)
        tr = opt.trace_ray(r)
        dmd_h = [h for h in tr.history if h["surface_index"] == 5][0]
        det_h = [h for h in tr.history if h["surface_index"] == 8]
        det_str = f"local=({det_h[0]['x_local']:.4f}, {det_h[0]['y_local']:.4f})" if det_h else "VIGNETTED"
        print(f"  WL {wl*1000:.0f} nm: DMD local=({dmd_h['x_local']:.3f}, {dmd_h['y_local']:.3f}) | Det: {det_str}")

    # Now trace full ray bundle
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

    dmd_xs = [p[0] for p in dmd_hits]
    dmd_ys = [p[1] for p in dmd_hits]
    print(f"\nFULL BUNDLE TRACE ({len(rays)} rays):")
    print(f"  Valid: {len(valid)} / {len(rays)} ({len(valid)/len(rays)*100:.1f}%), Vignetted: {len(vignetted)}")
    print(f"  DMD span: X=[{min(dmd_xs):.3f}, {max(dmd_xs):.3f}] (span={max(dmd_xs)-min(dmd_xs):.3f} mm), Y=[{min(dmd_ys):.3f}, {max(dmd_ys):.3f}] (span={max(dmd_ys)-min(dmd_ys):.3f} mm)")
    print(f"  Detector: max_radius = {max_r:.4f} mm, rms_radius = {rms_r:.4f} mm, <= 0.400 mm? {max_r <= 0.400}")

test_centering()
