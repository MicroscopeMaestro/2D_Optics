import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import generate_fiber_phase_space_rays, RAMAN_WAVELENGTHS_UM

# Let's test the Front-End (Fiber -> Collimator -> Grating -> Focus -> DMD)
for f_coll in [47.5, 50.0, 55.0, 60.0, 64.5]:
    for f_dmd in [44.0, 48.0, 55.0, 60.0, 64.5]:
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Fiber", thickness=f_coll, semi_diameter=25.0))
        stack.add_surface(Surface(name="Coll_F", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
        stack.add_surface(Surface(name="Coll_M", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
        stack.add_surface(Surface(name="Coll_B", radius=-115.0, thickness=20.0, glass="AIR", semi_diameter=12.7))
        stack.add_surface(Surface(name="Grating", surface_type=SurfaceType.GRATING, radius=0.0, thickness=20.0, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0, grating_lines_per_mm=300.0, grating_order=1))
        # Note: Focus lens is usually oriented with curved side facing collimated beam!
        # AC254-050-C: R1=115, R2=27.8, R3=-34 (reversed) or R1=34, R2=-27.8, R3=-115
        stack.add_surface(Surface(name="Focus_F", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
        stack.add_surface(Surface(name="Focus_M", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
        stack.add_surface(Surface(name="Focus_B", radius=-115.0, thickness=f_dmd, glass="AIR", semi_diameter=12.7))
        stack.add_surface(Surface(name="DMD", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=0.0, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0))
        
        stack.primary_wavelength = 1.3620
        stack._update_accumulated_positions()
        opt = OpticalSystem(stack)
        
        # Trace rays at center wavelength 1.3620
        rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=[1.3620])
        traced = opt.trace_rays(rays)
        
        hits = [(h['x'], h['y']) for r in traced for h in r.history if h['surface_index'] == 8 and not r.vignetted]
        if len(hits) == len(rays):
            rads = [np.sqrt(x**2 + y**2) for x, y in hits]
            max_r = max(rads)
            rms_r = np.sqrt(np.mean(np.array(rads)**2))
            if max_r < 0.20:
                print(f"f_coll={f_coll:.1f}, f_dmd={f_dmd:.1f} -> DMD spot max_r = {max_r*1000:.1f} um, rms_r = {rms_r*1000:.1f} um")
