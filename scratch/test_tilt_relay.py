import sys, os, math
import numpy as np

sys.path.insert(0, r"c:\Users\Juan\Documents\GitHub\2D_Optics")
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem, Ray
from v2.engine.surface import Surface, SurfaceType

def test_tilt_and_relay(tilt_m3=4.0, tilt_m4=-4.0, f4=12.0, d_m3_m4=50.0):
    wls = [0.800, 0.815, 0.830, 0.845, 0.860]

    stack = SurfaceStack()
    stack.primary_wavelength = 0.830
    stack.wavelengths = wls

    stack.add_surface(Surface(name="Slit", surface_type=SurfaceType.OBJECT, thickness=48.5, semi_diameter=12.7, aperture_type="Rectangular", aperture_x_half=0.250, aperture_y_half=0.015))
    stack.add_surface(Surface(name="Stop", surface_type=SurfaceType.STOP, thickness=1.5, semi_diameter=5.5, aperture_type="Circular", aperture_r_outer=5.5))
    stack.add_surface(Surface(name="M1", surface_type=SurfaceType.STANDARD, radius=-100.0, thickness=50.0, glass="MIRROR", semi_diameter=12.7, tilt_x=8.5, aperture_type="Circular", aperture_r_outer=12.5))
    stack.add_surface(Surface(name="Grating", surface_type=SurfaceType.GRATING, radius=0.0, thickness=60.0, glass="MIRROR", semi_diameter=12.5, tilt_x=18.2, grating_lines_per_mm=600.0, grating_order=1, aperture_type="Rectangular", aperture_x_half=12.5, aperture_y_half=12.5))
    stack.add_surface(Surface(name="M2", surface_type=SurfaceType.STANDARD, radius=-150.0, thickness=73.5, glass="MIRROR", semi_diameter=12.7, tilt_x=-8.5, aperture_type="Circular", aperture_r_outer=12.5))
    stack.add_surface(Surface(name="DMD", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=150.0, glass="MIRROR", semi_diameter=6.0, tilt_x=12.0, aperture_type="Rectangular", aperture_x_half=4.9, aperture_y_half=3.0))
    stack.add_surface(Surface(name="M3", surface_type=SurfaceType.STANDARD, radius=-300.0, thickness=d_m3_m4, glass="MIRROR", semi_diameter=12.7, tilt_x=tilt_m3, aperture_type="Circular", aperture_r_outer=12.5))
    
    # We will search best focus distance d_m4_det
    best_max_r = 999.0
    best_ddet = 0.0
    best_rms = 999.0

    for d_det in np.linspace(f4 * 0.80, f4 * 1.10, 31):
        stack_test = SurfaceStack()
        stack_test.primary_wavelength = 0.830
        stack_test.wavelengths = wls
        for s in stack.surfaces:
            stack_test.add_surface(s)

        stack_test.add_surface(Surface(name="M4", surface_type=SurfaceType.STANDARD, radius=-2.0*f4, thickness=d_det, glass="MIRROR", semi_diameter=12.7 if f4>=19 else 6.35, tilt_x=tilt_m4, aperture_type="Circular", aperture_r_outer=12.5 if f4>=19 else 6.0))
        stack_test.add_surface(Surface(name="Det", surface_type=SurfaceType.DETECTOR, thickness=0.0, glass="AIR", semi_diameter=2.0, aperture_type="Circular", aperture_r_outer=2.0))

        stack_test._update_accumulated_positions()
        opt = OpticalSystem(stack_test)

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
        det_hits = [h for tr in traced if not tr.vignetted for h in tr.history if h["surface_index"] == 8]
        if len(det_hits) == len(rays):
            xs = np.array([h["x_local"] for h in det_hits])
            ys = np.array([h["y_local"] for h in det_hits])
            radii = np.sqrt(xs**2 + ys**2)
            max_r = float(np.max(radii))
            rms_r = float(np.sqrt(np.mean(radii**2)))
            if max_r < best_max_r:
                best_max_r = max_r
                best_rms = rms_r
                best_ddet = d_det

    print(f"tilt_m3={tilt_m3:4.1f}, tilt_m4={tilt_m4:4.1f}, f4={f4:4.1f}mm, d_m3_m4={d_m3_m4:4.1f}mm -> best_ddet={best_ddet:.3f}mm: max_r={best_max_r:.4f} mm, rms_r={best_rms:.4f} mm")
    return best_max_r

print("Testing f4 = 12.0 mm (CM127-012-P01):")
for t in [2.0, 4.0, 6.0, 8.0, 10.0]:
    test_tilt_and_relay(tilt_m3=t, tilt_m4=-t, f4=12.0, d_m3_m4=50.0)

print("\nTesting f4 = 19.0 mm (CM254-019-P01):")
for t in [2.0, 4.0, 6.0, 8.0, 10.0]:
    test_tilt_and_relay(tilt_m3=t, tilt_m4=-t, f4=19.0, d_m3_m4=50.0)
