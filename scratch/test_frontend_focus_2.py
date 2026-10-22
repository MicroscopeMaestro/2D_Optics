import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray

# Let's find the exact front-end focus distance so that NA=0.12 rays focus to X=0 on DMD
for f_coll in [47.5, 50.0, 52.0, 55.0, 58.0, 60.0, 62.0, 64.5]:
    for f_dmd in [40.0, 44.0, 48.0, 52.0, 56.0, 60.0, 64.5]:
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Fiber", thickness=f_coll, semi_diameter=25.0))
        stack.add_surface(Surface(name="Coll_F", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
        stack.add_surface(Surface(name="Coll_M", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
        stack.add_surface(Surface(name="Coll_B", radius=-115.0, thickness=20.0, glass="AIR", semi_diameter=12.7))
        stack.add_surface(Surface(name="Grating", surface_type=SurfaceType.GRATING, radius=0.0, thickness=20.0, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0, grating_lines_per_mm=300.0, grating_order=1))
        stack.add_surface(Surface(name="Focus_F", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
        stack.add_surface(Surface(name="Focus_M", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
        stack.add_surface(Surface(name="Focus_B", radius=-115.0, thickness=f_dmd, glass="AIR", semi_diameter=12.7))
        stack.add_surface(Surface(name="DMD", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=20.0, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0))
        # Relay
        stack.add_surface(Surface(name="Relay_F", radius=49.3, thickness=5.2, glass="N-BK7", semi_diameter=12.7))
        stack.add_surface(Surface(name="Relay_M", radius=-39.0, thickness=2.0, glass="SF5", semi_diameter=12.7))
        stack.add_surface(Surface(name="Relay_B", radius=-170.0, thickness=5.0, glass="AIR", semi_diameter=12.7))
        # Condenser
        stack.add_surface(Surface(name="Cond_F", surface_type=SurfaceType.ASPHERIC, radius=4.16, thickness=4.5, glass="B270", semi_diameter=12.7, conic=-0.6265, asphere_coeffs=[1.5e-5]))
        stack.add_surface(Surface(name="Cond_B", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=5.2, glass="AIR", semi_diameter=12.7))
        stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=25.0))
        
        stack.primary_wavelength = 1.3620
        stack._update_accumulated_positions()
        opt = OpticalSystem(stack)
        
        # Test marginal ray
        max_theta = np.arcsin(0.12)
        r_na = Ray(pos=np.array([0., 0., 0.]), dir=np.array([np.sin(max_theta), 0., np.cos(max_theta)]), wavelength=1.3620)
        tr = opt.trace_ray(r_na)
        dmd_x = tr.history[8]['x']
        det_x = tr.history[14]['x']
        if abs(dmd_x) < 0.20:
            print(f"FOCUSED FRONTEND: f_coll={f_coll:.1f}, f_dmd={f_dmd:.1f} -> DMD_x={dmd_x*1000:+.1f} um, Det_x={det_x*1000:+.1f} um")
