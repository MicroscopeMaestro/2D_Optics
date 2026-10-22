import sys, os, math
import numpy as np

sys.path.insert(0, r"c:\Users\Juan\Documents\GitHub\2D_Optics")
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem, Ray
from v2.engine.surface import Surface, SurfaceType

def run_optimizer():
    wls = [0.800, 0.815, 0.830, 0.845, 0.860]
    
    # We want to optimize:
    # 1. tilt_m1 (around 8.5 to 10 deg)
    # 2. tilt_m2 (around -8.5 to -10 deg)
    # 3. tilt_m3 (around 6 to 12 deg)
    # 4. tilt_m4 (around -6 to -12 deg)
    # 5. d_m2_dmd (around 70 to 76 mm)
    # 6. d_dmd_m3 (around 140 to 160 mm)
    # 7. d_m3_m4 (around 40 to 175 mm)
    # 8. d_m4_det (around 10 to 20 mm)
    
    # Let's write a fast evaluator tracing 5 wavelengths x 3x3 slit points x 9 pupil rays = 135 rays
    theta_max = math.asin(0.10)
    eval_rays = []
    for wl in wls:
        for x in [-0.25, 0.0, 0.25]:
            for y in [-0.015, 0.0, 0.015]:
                eval_rays.append(Ray(pos=np.array([x, y, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl))
                for ring in [0.6, 1.0]:
                    for p in range(4):
                        phi = 2 * math.pi * p / 4 + math.pi / 4
                        kx = math.sin(ring * theta_max) * math.cos(phi)
                        ky = math.sin(ring * theta_max) * math.sin(phi)
                        kz = math.cos(ring * theta_max)
                        eval_rays.append(Ray(pos=np.array([x, y, 0.0]), dir=np.array([kx, ky, kz]), wavelength=wl))

    print(f"Fast evaluator uses {len(eval_rays)} rays.")

    def evaluate(p, f4=19.0):
        tilt_m1, tilt_m2, tilt_m3, tilt_m4, d_m2_dmd, d_dmd_m3, d_m3_m4, d_m4_det = p
        stack = SurfaceStack()
        stack.primary_wavelength = 0.830
        stack.wavelengths = wls

        stack.add_surface(Surface(name="Slit", surface_type=SurfaceType.OBJECT, thickness=48.5, semi_diameter=12.7, aperture_type="Rectangular", aperture_x_half=0.250, aperture_y_half=0.015))
        stack.add_surface(Surface(name="Stop", surface_type=SurfaceType.STOP, thickness=1.5, semi_diameter=5.5, aperture_type="Circular", aperture_r_outer=5.5))
        stack.add_surface(Surface(name="M1", surface_type=SurfaceType.STANDARD, radius=-100.0, thickness=50.0, glass="MIRROR", semi_diameter=12.7, tilt_x=tilt_m1, aperture_type="Circular", aperture_r_outer=12.5))
        stack.add_surface(Surface(name="Grating", surface_type=SurfaceType.GRATING, radius=0.0, thickness=60.0, glass="MIRROR", semi_diameter=12.5, tilt_x=18.2, grating_lines_per_mm=600.0, grating_order=1, aperture_type="Rectangular", aperture_x_half=12.5, aperture_y_half=12.5))
        stack.add_surface(Surface(name="M2", surface_type=SurfaceType.STANDARD, radius=-150.0, thickness=d_m2_dmd, glass="MIRROR", semi_diameter=12.7, tilt_x=tilt_m2, aperture_type="Circular", aperture_r_outer=12.5))
        stack.add_surface(Surface(name="DMD", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=d_dmd_m3, glass="MIRROR", semi_diameter=6.0, tilt_x=12.0, aperture_type="Rectangular", aperture_x_half=4.9, aperture_y_half=3.0))
        stack.add_surface(Surface(name="M3", surface_type=SurfaceType.STANDARD, radius=-300.0, thickness=d_m3_m4, glass="MIRROR", semi_diameter=12.7, tilt_x=tilt_m3, aperture_type="Circular", aperture_r_outer=12.5))
        stack.add_surface(Surface(name="M4", surface_type=SurfaceType.STANDARD, radius=-2.0*f4, thickness=d_m4_det, glass="MIRROR", semi_diameter=12.7 if f4>=19 else 6.35, tilt_x=tilt_m4, aperture_type="Circular", aperture_r_outer=12.5 if f4>=19 else 6.0))
        stack.add_surface(Surface(name="Det", surface_type=SurfaceType.DETECTOR, thickness=0.0, glass="AIR", semi_diameter=2.0, aperture_type="Circular", aperture_r_outer=2.0))

        stack._update_accumulated_positions()
        opt = OpticalSystem(stack)
        traced = [opt.trace_ray(r) for r in eval_rays]
        valid = [tr for tr in traced if not tr.vignetted]
        if len(valid) < len(eval_rays) * 0.90:
            return 999.0, 999.0, len(traced) - len(valid), stack

        det_hits = [h for tr in valid for h in tr.history if h["surface_index"] == 8]
        det_xs = np.array([h["x_local"] for h in det_hits])
        det_ys = np.array([h["y_local"] for h in det_hits])
        radii = np.sqrt(det_xs**2 + det_ys**2)
        max_r = float(np.max(radii))
        rms_r = float(np.sqrt(np.mean(radii**2)))
        return max_r, rms_r, len(traced) - len(valid), stack

    # Search best parameters for f4 = 19mm and f4 = 12mm
    for f4 in [19.0, 12.0]:
        print(f"\n==========================================")
        print(f"SEARCHING OPTIMAL DESIGN FOR f4 = {f4} mm")
        print(f"==========================================")
        best_max_r = 999.0
        best_p = None
        best_stack = None

        # Step 1: coarse grid over tilts and distances
        for tilt_m1 in [8.0, 8.5, 9.0, 9.5]:
            tilt_m2 = -tilt_m1 # Shafer coma compensation
            for tilt_m3 in [6.0, 7.0, 8.0, 9.0, 10.0]:
                for tilt_m4 in [-6.0, -7.0, -8.0, -9.0, -10.0]:
                    for d_m2_dmd in [73.0, 73.5, 74.0]:
                        for d_dmd_m3 in [148.0, 150.0, 152.0]:
                            for d_m3_m4 in [45.0, 50.0, 55.0, 60.0, 100.0, 150.0, 169.0]:
                                d_det_nominal = f4 * (1.0 - (f4/150.0))
                                for d_det in np.linspace(f4 * 0.85, f4 * 1.05, 7):
                                    p = [tilt_m1, tilt_m2, tilt_m3, tilt_m4, d_m2_dmd, d_dmd_m3, d_m3_m4, d_det]
                                    max_r, rms_r, vig, st = evaluate(p, f4=f4)
                                    if vig == 0 and max_r < best_max_r:
                                        best_max_r = max_r
                                        best_p = p
                                        best_stack = st
                                        print(f"  New best max_r = {max_r:.4f} mm (RMS={rms_r:.4f} mm) with tilts=({tilt_m1}, {tilt_m2}, {tilt_m3}, {tilt_m4}), dists=({d_m2_dmd}, {d_dmd_m3}, {d_m3_m4}, {d_det:.2f})")

        print(f"\nCoarse Search Best for f4={f4}mm: max_r = {best_max_r:.4f} mm")
        print(f"Best parameters: {best_p}")

        # Step 2: Fine tune around best_p
        if best_p is not None:
            t1, t2, t3, t4, dm2, ddmd, dm3, ddet = best_p
            for fine_t3 in np.linspace(t3 - 1.5, t3 + 1.5, 7):
                for fine_t4 in np.linspace(t4 - 1.5, t4 + 1.5, 7):
                    for fine_dm3 in np.linspace(dm3 - 5.0, dm3 + 5.0, 7):
                        for fine_ddet in np.linspace(ddet - 0.8, ddet + 0.8, 17):
                            p = [t1, t2, fine_t3, fine_t4, dm2, ddmd, fine_dm3, fine_ddet]
                            max_r, rms_r, vig, st = evaluate(p, f4=f4)
                            if vig == 0 and max_r < best_max_r:
                                best_max_r = max_r
                                best_p = p
                                best_stack = st
                                print(f"  Fine tuned best max_r = {max_r:.4f} mm (RMS={rms_r:.4f} mm) with tilts=({t1}, {t2}, {fine_t3:.2f}, {fine_t4:.2f}), dists=({dm2}, {ddmd}, {fine_dm3:.1f}, {fine_ddet:.3f})")

            print(f"\nFinal Best for f4={f4}mm: max_r = {best_max_r:.4f} mm (Limit <= 0.400 mm)")

if __name__ == "__main__":
    run_optimizer()
