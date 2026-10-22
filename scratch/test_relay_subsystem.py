"""
Test and optimize relay and condenser subsystem from DMD to Detector.
"""
import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray

def test_relay_subsystem(d_dmd_relay, d_relay_cond, d_cond_det, asph_R=4.16, asph_k=-0.6265, asph_A4=1.5e-5):
    stack = SurfaceStack()
    stack.primary_wavelength = 1.3892

    # DMD plane (Object for this sub-system)
    stack.add_surface(Surface(name="DMD", surface_type=SurfaceType.OBJECT, thickness=d_dmd_relay, glass="AIR", semi_diameter=12.7))

    # Relay Lens AC508-075-C (f=75mm)
    stack.add_surface(Surface(name="Relay_F", radius=49.3, thickness=8.0, glass="N-BK7", semi_diameter=25.4))
    stack.add_surface(Surface(name="Relay_M", radius=-39.0, thickness=3.0, glass="SF5", semi_diameter=25.4))
    stack.add_surface(Surface(name="Relay_B", radius=-170.0, thickness=d_relay_cond, glass="AIR", semi_diameter=25.4))

    # Condenser ACL12708U-C (f=8mm)
    stack.add_surface(Surface(
        name="Cond_F",
        surface_type=SurfaceType.ASPHERIC if (asph_k != 0 or asph_A4 != 0) else SurfaceType.STANDARD,
        radius=asph_R,
        thickness=4.5,
        glass="B270",
        semi_diameter=12.7,
        conic=asph_k,
        asphere_coeffs=[asph_A4] if asph_A4 != 0 else []
    ))
    stack.add_surface(Surface(name="Cond_B", radius=0.0, thickness=d_cond_det, glass="AIR", semi_diameter=12.7))

    # Detector
    stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, thickness=0.0, glass="AIR", semi_diameter=0.5))

    stack._update_accumulated_positions()
    opt = OpticalSystem(stack)

    # Launch rays from different DMD field points (x = -3.9 mm, 0.0 mm, +3.9 mm)
    # with NA cone = 0.12
    rays = []
    dmd_x_positions = [-3.9, -2.0, 0.0, 2.0, 3.9]
    na = 0.12
    max_theta = math.asin(na)

    for x_pos in dmd_x_positions:
        # Chief ray
        rays.append(Ray(pos=np.array([x_pos, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.3892))
        # Marginal rays
        for angle in [max_theta, -max_theta]:
            rays.append(Ray(pos=np.array([x_pos, 0.0, 0.0]), dir=np.array([math.sin(angle), 0.0, math.cos(angle)]), wavelength=1.3892))
            rays.append(Ray(pos=np.array([x_pos, 0.0, 0.0]), dir=np.array([0.0, math.sin(angle), math.cos(angle)]), wavelength=1.3892))

    tr = opt.trace_rays(rays)
    vignetted = [r for r in tr if r.vignetted]
    det_hits = [h for r in tr for h in r.history if h['surface_index'] == 6] # index 6 is detector in this sub-stack

    if len(det_hits) == len(rays):
        det_radii = [math.sqrt(h['x']**2 + h['y']**2) for h in det_hits]
        max_r = max(det_radii)
        return True, max_r, det_hits
    else:
        return False, 999.0, det_hits

# Search for the best relay distance and condenser distance
print("Searching relay parameters...")
best_max_r = 999.0
best_config = None

for d1 in np.linspace(50.0, 100.0, 51):
    for d2 in np.linspace(1.0, 25.0, 25):
        for d3 in np.linspace(2.0, 8.0, 31):
            ok, max_r, hits = test_relay_subsystem(d1, d2, d3)
            if ok and max_r < 0.50:
                print(f"PASS: d_dmd_relay={d1:.2f}, d_relay_cond={d2:.2f}, d_cond_det={d3:.2f} -> Max Det Radius = {max_r:.4f} mm")
                if max_r < best_max_r:
                    best_max_r = max_r
                    best_config = (d1, d2, d3)

print(f"\nBest configuration: {best_config} with max detector radius = {best_max_r:.4f} mm")
