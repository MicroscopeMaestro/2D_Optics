"""
Hybrid (Lens-based) Spectrometer with tilt_y fix
"""

import os, sys, numpy as np, math
sys.path.insert(0, os.path.abspath('.'))
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.engine.optimizer import OpticalOptimizer

def build_system() -> SurfaceStack:
    stack = SurfaceStack()
    stack.primary_wavelength = 0.785

    # 0. Fiber
    stack.add_surface(Surface(name="Fiber", surface_type=SurfaceType.OBJECT, thickness=25.0, glass="AIR"))

    # 1. L1: AC254-030-B (oriented flatter side to fiber, steep side to collimated)
    # Thorlabs AC254-030-B: R1=18.5, R2=-14.8, R3=-57.2.
    # Flipped: R1=57.2 (t=2.0, SF5), R2=14.8 (t=7.5, N-BK7), R3=-18.5
    stack.add_surface(Surface(name="L1_F", surface_type=SurfaceType.STANDARD, radius=57.2, thickness=2.0, glass="SF5"))
    stack.add_surface(Surface(name="L1_M", surface_type=SurfaceType.STANDARD, radius=14.8, thickness=7.5, glass="N-BK7"))
    stack.add_surface(Surface(name="L1_B", surface_type=SurfaceType.STANDARD, radius=-18.5, thickness=30.0, glass="AIR"))

    # 2. Grating 600 l/mm. Use tilt_y to keep dispersion in XZ plane.
    stack.add_surface(Surface(name="Grating", surface_type=SurfaceType.GRATING, radius=0.0, thickness=30.0, glass="MIRROR", tilt_y=18.2, grating_lines_per_mm=600.0, grating_order=1))

    # 3. L2: AC254-040-B (focusing to DMD. Steep side to collimated, flatter to DMD)
    # Thorlabs AC254-040-B: R1=25.1, R2=-20.2, R3=-84.0
    stack.add_surface(Surface(name="L2_F", surface_type=SurfaceType.STANDARD, radius=25.1, thickness=6.8, glass="N-BK7"))
    stack.add_surface(Surface(name="L2_M", surface_type=SurfaceType.STANDARD, radius=-20.2, thickness=2.0, glass="SF5"))
    stack.add_surface(Surface(name="L2_B", surface_type=SurfaceType.STANDARD, radius=-84.0, thickness=38.0, glass="AIR"))

    # 4. DMD - Use tilt_y
    stack.add_surface(Surface(name="DMD", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=150.0, glass="MIRROR", tilt_y=12.0))

    # 5. L3: AC254-150-B (collimating from DMD. Flatter side to DMD, Steep side to collimated)
    # AC254-150-B: R1=92.5, R2=-66.8, R3=-298.0
    # Flipped: R1=298.0 (t=2.0, SF5), R2=66.8 (t=4.5, N-BK7), R3=-92.5
    stack.add_surface(Surface(name="L3_F", surface_type=SurfaceType.STANDARD, radius=298.0, thickness=2.0, glass="SF5"))
    stack.add_surface(Surface(name="L3_M", surface_type=SurfaceType.STANDARD, radius=66.8, thickness=4.5, glass="N-BK7"))
    stack.add_surface(Surface(name="L3_B", surface_type=SurfaceType.STANDARD, radius=-92.5, thickness=25.0, glass="AIR"))

    # 6. L4: AC254-030-B (focusing to detector. Steep side to collimated, flatter to detector)
    # AC254-030-B: R1=18.5, R2=-14.8, R3=-57.2
    stack.add_surface(Surface(name="L4_F", surface_type=SurfaceType.STANDARD, radius=18.5, thickness=7.5, glass="N-BK7"))
    stack.add_surface(Surface(name="L4_M", surface_type=SurfaceType.STANDARD, radius=-14.8, thickness=2.0, glass="SF5"))
    stack.add_surface(Surface(name="L4_B", surface_type=SurfaceType.STANDARD, radius=-57.2, thickness=25.0, glass="AIR"))

    # 7. Detector
    stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, thickness=0.0))

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

    print("Optimizing Hybrid Lens Design with Tilt Y ...")
    optimizer = OpticalOptimizer(opt_sys, wavelengths=[0.785, 0.880], method="L-BFGS-B", max_iter=200)
    # Optimize air gaps: 0, 3, 6, 7, 10, 13 (Detector gap)
    optimizer.optimize(variable_indices={"thickness": [0, 3, 6, 7, 10, 13]})
    
    wls = [0.785, 0.808, 0.830, 0.850, 0.880]
    rays = generate_fiber_rays(wls)
    traced_rays = opt_sys.trace_rays(rays)
    det_hits_x, det_hits_y = [], []
    for tr in traced_rays:
        for h in tr.history:
            if h["surface_index"] == 14: # Detector
                det_hits_x.append(h["x"])
                det_hits_y.append(h["y"])
    if det_hits_x:
        max_rad = max(np.max(np.abs(det_hits_x)), np.max(np.abs(det_hits_y)))
        print(f"Hybrid Lens MaxR: {max_rad:.3f} mm")
    else:
        print("Rays did not reach detector!")

if __name__ == "__main__":
    main()
