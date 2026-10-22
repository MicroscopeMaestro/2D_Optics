import sys, os, math
import numpy as np

sys.path.insert(0, r"c:\Users\Juan\Documents\GitHub\2D_Optics")
from scratch.test_zero_vignetting import test_complete_zero_vignetting
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem, Ray

stack, _, _ = test_complete_zero_vignetting(f4=19.0)
# Make detector semi_diameter large so all rays hit detector and record (x,y)
stack.surfaces[8].semi_diameter = 5.0
stack.surfaces[8].aperture_r_outer = 5.0
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
det_hits = [h for tr in traced for h in tr.history if h["surface_index"] == 8]
xs = np.array([h["x_local"] for h in det_hits])
ys = np.array([h["y_local"] for h in det_hits])

print(f"Total rays hitting detector: {len(xs)} / {len(rays)}")
print(f"X bounds: [{np.min(xs):.4f}, {np.max(xs):.4f}], mean = {np.mean(xs):.4f}")
print(f"Y bounds: [{np.min(ys):.4f}, {np.max(ys):.4f}], mean = {np.mean(ys):.4f}")

centered_radii = np.sqrt((xs - np.mean(xs))**2 + (ys - np.mean(ys))**2)
print(f"Centered Max Spot Radius: {np.max(centered_radii):.4f} mm")
print(f"Centered RMS Spot Radius: {np.sqrt(np.mean(centered_radii**2)):.4f} mm")

# If we decenter detector by decenter_y = np.mean(ys):
print(f"Required detector decenter_y offset: {np.mean(ys):.4f} mm")
