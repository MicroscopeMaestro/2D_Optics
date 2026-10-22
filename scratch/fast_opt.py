import sys, os, math
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, r"c:\Users\Juan\Documents\GitHub\2D_Optics")
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem, Ray
from v2.engine.surface import Surface, SurfaceType

def fast_optimize():
    wls = [0.785, 0.808, 0.830, 0.850, 0.880]
    
    # 5 wavelengths x 3x3 slit points x 9 rays = 135 rays
    theta_max = math.asin(0.10)
    eval_rays = []
    for wl in wls:
        for x in [-0.25, 0.0, 0.25]:
            for y in [-0.015, 0.0, 0.015]:
                eval_rays.append(Ray(pos=np.array([x, y, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl))
                for ring in [0.5, 1.0]:
                    for p in range(4):
                        phi = 2 * math.pi * p / 4 + math.pi / 4
                        kx = math.sin(ring * theta_max) * math.cos(phi)
                        ky = math.sin(ring * theta_max) * math.sin(phi)
                        kz = math.cos(ring * theta_max)
                        eval_rays.append(Ray(pos=np.array([x, y, 0.0]), dir=np.array([kx, ky, kz]), wavelength=wl))

    print(f"Tracing {len(eval_rays)} rays per evaluation.")

    def objective(x_params):
        d_m2_dmd, d_m4_det, tilt_m3, tilt_m4 = x_params
        stack = SurfaceStack()
        stack.primary_wavelength = 0.830
        stack.wavelengths = wls

        stack.add_surface(Surface(name="Slit", surface_type=SurfaceType.OBJECT, thickness=48.5, semi_diameter=12.7, aperture_type="Rectangular", aperture_x_half=0.250, aperture_y_half=0.015))
        stack.add_surface(Surface(name="Stop", surface_type=SurfaceType.STOP, thickness=1.5, semi_diameter=5.5, aperture_type="Circular", aperture_r_outer=5.5))
        stack.add_surface(Surface(name="M1", surface_type=SurfaceType.STANDARD, radius=-100.0, thickness=50.0, glass="MIRROR", semi_diameter=12.7, tilt_x=8.5, aperture_type="Circular", aperture_r_outer=12.5))
        stack.add_surface(Surface(name="Grating", surface_type=SurfaceType.GRATING, radius=0.0, thickness=60.0, glass="MIRROR", semi_diameter=12.5, tilt_x=18.2, grating_lines_per_mm=600.0, grating_order=1, aperture_type="Rectangular", aperture_x_half=12.5, aperture_y_half=12.5))
        stack.add_surface(Surface(name="M2", surface_type=SurfaceType.STANDARD, radius=-150.0, thickness=d_m2_dmd, glass="MIRROR", semi_diameter=12.7, tilt_x=-8.5, aperture_type="Circular", aperture_r_outer=12.5))
        stack.add_surface(Surface(name="DMD", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=150.0, glass="MIRROR", semi_diameter=6.0, tilt_x=12.0, aperture_type="Rectangular", aperture_x_half=5.0, aperture_y_half=5.0))
        stack.add_surface(Surface(name="M3", surface_type=SurfaceType.STANDARD, radius=-300.0, thickness=50.0, glass="MIRROR", semi_diameter=12.7, tilt_x=tilt_m3, aperture_type="Circular", aperture_r_outer=12.5))
        stack.add_surface(Surface(name="M4", surface_type=SurfaceType.STANDARD, radius=-38.0, thickness=d_m4_det, glass="MIRROR", semi_diameter=12.7, tilt_x=tilt_m4, aperture_type="Circular", aperture_r_outer=12.5))
        stack.add_surface(Surface(name="Det", surface_type=SurfaceType.DETECTOR, thickness=0.0, glass="AIR", semi_diameter=2.0, aperture_type="Circular", aperture_r_outer=2.0))

        stack._update_accumulated_positions()
        opt = OpticalSystem(stack)
        traced = [opt.trace_ray(r) for r in eval_rays]
        valid = [tr for tr in traced if not tr.vignetted]
        if len(valid) < len(eval_rays):
            return 100.0 + (len(eval_rays) - len(valid))

        det_hits = [h for tr in valid for h in tr.history if h["surface_index"] == 8]
        det_xs = np.array([h["x_local"] for h in det_hits])
        det_ys = np.array([h["y_local"] for h in det_hits])
        radii = np.sqrt(det_xs**2 + det_ys**2)
        max_r = float(np.max(radii))
        rms_r = float(np.sqrt(np.mean(radii**2)))
        return max_r * 2.0 + rms_r

    init_x = [73.5, 18.8, 8.0, -8.0]
    res = minimize(objective, init_x, method="Nelder-Mead", options={"maxiter": 150, "disp": True})
    print("\nOptimization result:", res.x)
    print("Final objective value:", res.fun)

    # Evaluate full system with optimized parameters
    d_m2_dmd, d_m4_det, tilt_m3, tilt_m4 = res.x
    print(f"Optimal parameters:\n  d_m2_dmd = {d_m2_dmd:.3f} mm\n  d_m4_det = {d_m4_det:.3f} mm\n  tilt_m3  = {tilt_m3:.3f} deg\n  tilt_m4  = {tilt_m4:.3f} deg")

if __name__ == "__main__":
    fast_optimize()
