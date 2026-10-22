import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import generate_fiber_phase_space_rays, RAMAN_WAVELENGTHS_UM

# Let's test true focal planes
for f_coll in [47.5, 50.0, 55.0, 60.0, 64.5]:
    for f_dmd in [44.0, 50.0, 55.0, 60.0, 64.5]:
        for dmd_r in [15.0, 20.0, 30.0, 40.0, 50.0, 60.0]:
            for r_c in [2.0, 5.0, 10.0]:
                for c_d in [4.5, 4.8, 5.0, 5.2, 5.5, 5.8]:
                    stack = SurfaceStack()
                    stack.add_surface(Surface(name="Fiber", thickness=f_coll, semi_diameter=25.0))
                    stack.add_surface(Surface(name="Coll_F", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
                    stack.add_surface(Surface(name="Coll_M", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
                    stack.add_surface(Surface(name="Coll_B", radius=-115.0, thickness=20.0, glass="AIR", semi_diameter=12.7))
                    stack.add_surface(Surface(name="Grating", surface_type=SurfaceType.GRATING, radius=0.0, thickness=20.0, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0, grating_lines_per_mm=300.0, grating_order=1))
                    stack.add_surface(Surface(name="Focus_F", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
                    stack.add_surface(Surface(name="Focus_M", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
                    stack.add_surface(Surface(name="Focus_B", radius=-115.0, thickness=f_dmd, glass="AIR", semi_diameter=12.7))
                    stack.add_surface(Surface(name="DMD", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=dmd_r, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0))
                    # Relay AC254-075-C
                    stack.add_surface(Surface(name="Relay_F", radius=49.3, thickness=5.2, glass="N-BK7", semi_diameter=12.7))
                    stack.add_surface(Surface(name="Relay_M", radius=-39.0, thickness=2.0, glass="SF5", semi_diameter=12.7))
                    stack.add_surface(Surface(name="Relay_B", radius=-170.0, thickness=r_c, glass="AIR", semi_diameter=12.7))
                    # Condenser ACL12708U-C
                    stack.add_surface(Surface(name="Cond_F", surface_type=SurfaceType.ASPHERIC, radius=4.16, thickness=4.5, glass="B270", semi_diameter=12.7, conic=-0.6265, asphere_coeffs=[1.5e-5]))
                    stack.add_surface(Surface(name="Cond_B", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=c_d, glass="AIR", semi_diameter=12.7))
                    # Detector
                    stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=25.0))
                    
                    stack.primary_wavelength = 1.3620
                    stack._update_accumulated_positions()
                    opt = OpticalSystem(stack)
                    
                    # Trace 5 rays at center wavelength 1.3620 (center ray + 4 marginal rays)
                    max_theta = np.arcsin(0.12)
                    test_5 = [
                        Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=1.3620),
                        Ray(pos=np.array([0.0125, 0., 0.]), dir=np.array([np.sin(max_theta), 0., np.cos(max_theta)]), wavelength=1.3620),
                        Ray(pos=np.array([-0.0125, 0., 0.]), dir=np.array([-np.sin(max_theta), 0., np.cos(max_theta)]), wavelength=1.3620),
                        Ray(pos=np.array([0., 0.0125, 0.]), dir=np.array([0., np.sin(max_theta), np.cos(max_theta)]), wavelength=1.3620),
                        Ray(pos=np.array([0., -0.0125, 0.]), dir=np.array([0., -np.sin(max_theta), np.cos(max_theta)]), wavelength=1.3620),
                    ]
                    tr = opt.trace_rays(test_5)
                    hits = [(h['x'], h['y']) for r in tr for h in r.history if h['surface_index'] == 14]
                    if len(hits) == 5:
                        rads = [np.sqrt(x**2 + y**2) for x, y in hits]
                        max_r = max(rads)
                        if max_r < 0.100:
                            print(f"SHARP SPOT AT DETECTOR: f_coll={f_coll:.1f}, f_dmd={f_dmd:.1f}, dmd_r={dmd_r:.1f}, r_c={r_c:.1f}, c_d={c_d:.1f} -> max_r = {max_r*1000:.1f} um (dia = {max_r*2000:.1f} um)")
                            sys.stdout.flush()
