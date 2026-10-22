"""
All-Mirror Thorlabs Spectrometer Design
"""

import os, sys, numpy as np, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.engine.optimizer import OpticalOptimizer

def build_system() -> SurfaceStack:
    stack = SurfaceStack()
    stack.primary_wavelength = 0.785

    # 0. Fiber
    stack.add_surface(Surface(
        name="Fiber_25um",
        surface_type=SurfaceType.OBJECT,
        thickness=50.0, # guess
        glass="AIR",
        semi_diameter=0.0125
    ))

    # 1. M1 (Collimator): CM254-050-E03 (f=50, R=-100)
    stack.add_surface(Surface(
        name="M1_CM254-050-E03",
        surface_type=SurfaceType.STANDARD,
        radius=-100.0,
        thickness=50.0, # guess
        glass="MIRROR",
        tilt_y=10.0,
        semi_diameter=12.7
    ))

    # 2. Grating (600 l/mm)
    stack.add_surface(Surface(
        name="Grating_600lmm",
        surface_type=SurfaceType.GRATING,
        radius=0.0,
        thickness=50.0, # guess
        glass="MIRROR",
        tilt_y=18.2,
        grating_lines_per_mm=600.0,
        grating_order=1,
        semi_diameter=12.5
    ))

    # 3. M2 (Kinematic Focusing): CM254-075-E03 (f=75, R=-150)
    stack.add_surface(Surface(
        name="M2_CM254-075-E03",
        surface_type=SurfaceType.STANDARD,
        radius=-150.0,
        thickness=75.0, # guess
        glass="MIRROR",
        tilt_y=10.0,
        semi_diameter=12.7
    ))

    # 4. DMD
    stack.add_surface(Surface(
        name="DMD4500",
        surface_type=SurfaceType.STANDARD,
        radius=0.0,
        thickness=150.0, # guess
        glass="MIRROR",
        tilt_y=12.0,
        semi_diameter=4.332
    ))

    # 5. M3 (Spectrum Collector): CM254-150-E03 (f=150, R=1-300)
    stack.add_surface(Surface(
        name="M3_CM254-150-E03",
        surface_type=SurfaceType.STANDARD,
        radius=-300.0,
        thickness=100.0, # guess
        glass="MIRROR",
        tilt_y=10.0,
        semi_diameter=12.7
    ))

    # 6. M4 (Condenser): CM254-025-E03 (f=25, R=-50)
    stack.add_surface(Surface(
        name="M4_CM254-025-E03",
        surface_type=SurfaceType.STANDARD,
        radius=-50.0,
        thickness=25.0, # guess
        glass="MIRROR",
        tilt_y=10.0,
        semi_diameter=12.7
    ))

    # 7. Detector
    stack.add_surface(Surface(
        name="Detector_1mm",
        surface_type=SurfaceType.DETECTOR,
        thickness=0.0,
        semi_diameter=0.5
    ))

    stack._update_accumulated_positions()
    return stack

def generate_fiber_rays(wavelengths, num_rays_per_ring=12, fiber_na=0.10):
    rays = []
    theta_max = math.asin(fiber_na)
    zdist = 50.0
    for wl in wavelengths:
        rays.append(Ray(pos=np.array([0.,0.,0.]), dir=np.array([0.,0.,1.]), wavelength=wl))
        for ring_frac in [0.5, 1.0]:
            r_pupil = zdist * math.tan(theta_max * ring_frac)
            npts = int(num_rays_per_ring * ring_frac)
            for p in range(max(6, npts)):
                phi = 2.0 * np.pi * p / max(6, npts)
                dir_v = np.array([r_pupil * np.cos(phi), r_pupil * np.sin(phi), zdist])
                dir_v /= np.linalg.norm(dir_v)
                rays.append(Ray(pos=np.array([0.,0.,0.]), dir=dir_v, wavelength=wl))
    return rays

def main():
    stack = build_system()
    opt_sys = OpticalSystem(stack)

    print("Optimizing All-Mirror 10:1 Design ...")
    optimizer = OpticalOptimizer(opt_sys, wavelengths=0.785, method="L-BFGS-B", max_iter=200)
    optimizer.optimize(variable_indices={"thickness": [0, 1, 2, 3, 4, 5, 6]})
    
    wls = [0.785, 0.808, 0.830, 0.850, 0.880]
    rays = generate_fiber_rays(wls)
    traced_rays = opt_sys.trace_rays(rays)
    det_hits_x, det_hits_y = [], []
    for tr in traced_rays:
        for h in tr.history:
            if h["surface_index"] == 7:
                det_hits_x.append(h["x"])
                det_hits_y.append(h["y"])
    max_rad = max(np.max(np.abs(det_hits_x)), np.max(np.abs(det_hits_y)))
    print(f"Hybrid All-Mirror MaxR: {max_rad:.3f} mm")

    base_dir = r"C:\Users\Juan\Documents\GitHub\2D_Optics"
    path = os.path.join(base_dir, "thorlabs_all_mirror_spectrometer.json")
    stack.save_to_json(path, wavelengths=wls)
    print(f"Exported to {path}")

if __name__ == "__main__":
    main()
