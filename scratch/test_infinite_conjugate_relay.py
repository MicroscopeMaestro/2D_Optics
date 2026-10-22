import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import generate_fiber_phase_space_rays, RAMAN_WAVELENGTHS_UM

# Let's test infinite conjugate relay with AC254-075-C and AC254-100-C
for relay_name, r1, t1, g1, r2, t2, g2, r3, fl in [
    ('AC254-075-C', 49.3, 5.2, 'N-BK7', -39.0, 2.0, 'SF5', -170.0, 75.0),
    ('AC254-100-C', 61.2, 5.0, 'N-BK7', -44.5, 2.0, 'SF5', -198.0, 100.0),
    ('AC254-050-C', 34.0, 6.0, 'N-BK7', -27.8, 2.0, 'SF5', -115.0, 50.0),
]:
    for dmd_r in np.arange(fl - 10.0, fl + 15.0, 2.0):
        for r_c in [2.0, 5.0, 10.0]:
            for c_d in np.arange(4.8, 5.8, 0.1):
                stack = SurfaceStack()
                stack.add_surface(Surface(name="Fiber", thickness=47.5, semi_diameter=25.0))
                stack.add_surface(Surface(name="Coll_F", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
                stack.add_surface(Surface(name="Coll_M", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
                stack.add_surface(Surface(name="Coll_B", radius=-115.0, thickness=20.0, glass="AIR", semi_diameter=12.7))
                stack.add_surface(Surface(name="Grating", surface_type=SurfaceType.GRATING, radius=0.0, thickness=20.0, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0, grating_lines_per_mm=300.0, grating_order=1))
                stack.add_surface(Surface(name="Focus_F", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
                stack.add_surface(Surface(name="Focus_M", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
                stack.add_surface(Surface(name="Focus_B", radius=-115.0, thickness=44.0, glass="AIR", semi_diameter=12.7))
                stack.add_surface(Surface(name="DMD", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=dmd_r, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0))
                
                # Relay
                stack.add_surface(Surface(name="Relay_F", radius=r1, thickness=t1, glass=g1, semi_diameter=12.7))
                stack.add_surface(Surface(name="Relay_M", radius=r2, thickness=t2, glass=g2, semi_diameter=12.7))
                stack.add_surface(Surface(name="Relay_B", radius=r3, thickness=r_c, glass="AIR", semi_diameter=12.7))
                # Condenser ACL12708U-C
                stack.add_surface(Surface(name="Cond_F", surface_type=SurfaceType.ASPHERIC, radius=4.16, thickness=4.5, glass="B270", semi_diameter=12.7, conic=-0.6265, asphere_coeffs=[1.5e-5]))
                stack.add_surface(Surface(name="Cond_B", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=c_d, glass="AIR", semi_diameter=12.7))
                # Detector
                stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=25.0))
                
                stack.primary_wavelength = 1.3620
                stack._update_accumulated_positions()
                opt = OpticalSystem(stack)
                
                # Trace 7 chief rays
                chief_rays = [Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=wl) for wl in RAMAN_WAVELENGTHS_UM]
                tr = opt.trace_rays(chief_rays)
                hits = [(h['x'], h['y']) for r in tr for h in r.history if h['surface_index'] == 14 and not r.vignetted]
                if len(hits) == 7:
                    rads = [np.sqrt(x**2 + y**2) for x, y in hits]
                    max_r = max(rads)
                    if max_r <= 0.400:
                        print(f"PASS CHIEF: Relay={relay_name}, dmd_r={dmd_r:.1f}, r_c={r_c:.1f}, c_d={c_d:.1f} -> max_r={max_r:.4f} mm (dia={max_r*2:.4f} mm)")
                        sys.stdout.flush()
