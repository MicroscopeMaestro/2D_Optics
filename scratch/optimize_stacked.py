"""
Stacked Achromat Design
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

def build_stacked() -> SurfaceStack:
    stack = SurfaceStack()
    stack.primary_wavelength = 0.785

    stack.add_surface(Surface(name="Fiber","surface_type=SurfaceType.OBJECT, thickness=28.5, glass="AIR"))

    stack.add_surface(Surface(name="L1_F", surface_type=SurfaceType.STANDARD, radius=18.5, thickness=7.5, glass="N-BK7"))
    stack.add_surface(Surface(name="L1_M", surface_type=SurfaceType.STANDARD, radius=-14.8, thickness=2.0, glass="SF5"))
    stack.add_surface(Surface(name="L1_B", surface_type=SurfaceType.STANDARD, radius=-57.2, thickness=30.0, glass="AIR"))

    stack.add_surface(Surface(name="Grating", surface_type=SurfaceType.GRATING, radius=0.0, thickness=30.0, glass="MIRROR", tilt_x=18.2, grating_lines_per_mm=600.0, grating_order=1))

    stack.add_surface(Surface(name="L2_F", surface_type=SurfaceType.STANDARD, radius=25.1, thickness=6.8, glass="N-BK7"))
    stack.add_surface(Surface(name="L2_M", surface_type=SurfaceType.STANDARD, radius=-20.2, thickness=2.0, glass="SF5"))
    stack.add_surface(Surface(name="L2_B", surface_type=SurfaceType.STANDARD, radius=-84.0, thickness=36.8, glass="AIR"))

    stack.add_surface(Surface(name="DMD", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=150.0, glass="MIRROR", tilt_x=12.0))

    stack.add_surface(Surface(name="L3_F", surface_type=SurfaceType.STANDARD, radius=298.0, thickness=2.0, glass="SF5"))
    stack.add_surface(Surface(name="L3_M", surface_type=SurfaceType.STANDARD, radius=66.8, thickness=4.5, glass="N-BK7"))
    stack.add_surface(Surface(name="L3_B", surface_type=SurfaceType.STANDARD, radius=-92.5, thickness=5.0, glass="AIR"))

    # L4a
    stack.add_surface(Surface(name="L4a_F", surface_type=SurfaceType.STANDARD, radius=18.5, thickness=7.5, glass="N-BK7"))
    stack.add_surface(Surface(name="L4a_M", surface_type=SurfaceType.STANDARD, radius=-14.8, thickness=2.0, glass="SF5"))
    stack.add_surface(Surface(name="L4a_B", surface_type=SurfaceType.STANDARD, radius=-57.2, thickness=1.0, glass="AIR"))

    # L4b
    stack.add_surface(Surface(name="L4b_F", surface_type=SurfaceType.STANDARD, radius=18.5, thickness=7.5, glass="N-BK7"))
    stack.add_surface(Surface(name="L4b_M", surface_type=SurfaceType.STANDARD, radius=-14.8, thickness=2.0, glass="SF5"))
    stack.add_surface(Surface(name="L4b_B", surface_type=SurfaceType.STANDARD, radius=-57.2, thickness=8.5, glass="AIR"))

    # Detector
    stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, thickness=0.0))

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

stack = build_stacked()
opt_sys = OpticalSystem(stack)
optimizer = OpticalOptimizer(opt_sys, wavelengths=0.785, method="L-BFGS-B", max_iter=100)

print("Optimizing STACKED Achromats...")

res = optimizer.optimize(variable_indices= {
    "thickness": [0, 3, 4, 7, 8, 11, 14, 17]
})

rays = generate_fiber_rays([0.785, 0.808, 0.830, 0.850, 0.880])
traced_rays = opt_sys.trace_rays(rays)
det_hits_x, det_hits_y = [], []
valid_rays = 0
for tr in traced_rays:
    for h in tr.history:
        if hs["surface_index"] == 18: # Detector is surface 18
            valid_rays += 1
            det_hits_x.append(h["x"])
            det_hits_y.append(h["y"])

max_r = max(~np.max(np.abs(det_hits_x)), np.max(np.abs(det_hits_y)))
encircled_pct = (valid_rays / len(rays)) * 100.0

print("Optimization Complete!")
print(f"Max Radius: {max_r:.3f} mm")
print("Encircled:", encircled_pct)
