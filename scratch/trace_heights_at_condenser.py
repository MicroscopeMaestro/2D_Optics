import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from verify_layout import generate_fiber_phase_space_rays, RAMAN_WAVELENGTHS_UM

rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)

for dmd_r in [5.0, 8.0, 10.0, 12.0, 15.0, 18.0, 20.0]:
    for r_c in [1.0, 2.0, 3.0, 5.0]:
        for c_d in [4.5, 4.8, 5.0, 5.2, 5.5]:
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
            
            # AC254-075-C
            stack.add_surface(Surface(name="Relay_F", radius=49.3, thickness=5.2, glass="N-BK7", semi_diameter=12.7))
            stack.add_surface(Surface(name="Relay_M", radius=-39.0, thickness=2.0, glass="SF5", semi_diameter=12.7))
            stack.add_surface(Surface(name="Relay_B", radius=-170.0, thickness=r_c, glass="AIR", semi_diameter=12.7))
            
            # ACL12708U-C
            stack.add_surface(Surface(name="Cond_F", surface_type=SurfaceType.ASPHERIC, radius=4.16, thickness=4.5, glass="B270", semi_diameter=12.7, conic=-0.6265, asphere_coeffs=[1.5e-5]))
            stack.add_surface(Surface(name="Cond_B", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=c_d, glass="AIR", semi_diameter=12.7))
            stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.50))
            
            stack.primary_wavelength = 1.3620
            stack._update_accumulated_positions()
            
            opt = OpticalSystem(stack)
            traced = opt.trace_rays(rays)
            det_idx = len(stack.surfaces) - 1
            
            # Check for zero vignetting
            vig = sum(1 for r in traced if r.vignetted and (not r.history or r.history[-1]['surface_index'] < det_idx))
            hits = [(h['x'], h['y']) for r in traced for h in r.history if h['surface_index'] == det_idx and not (r.vignetted and r.history[-1]['surface_index'] < det_idx)]
            
            if vig == 0 and len(hits) == len(rays):
                rads = [np.sqrt(x**2 + y**2) for x, y in hits]
                max_r = max(rads)
                rms_r = np.sqrt(np.mean(np.array(rads)**2))
                if max_r <= 0.500:
                    print(f"*** STRICT PASS ALL 7 WAVELENGTHS *** dmd_r={dmd_r:.1f}, r_c={r_c:.1f}, c_d={c_d:.1f} | max_r={max_r:.4f} mm (dia={max_r*2:.4f} mm), rms_r={rms_r:.4f} mm")
                    sys.stdout.flush()
                elif max_r <= 1.500:
                    print(f"  Candidate: dmd_r={dmd_r:.1f}, r_c={r_c:.1f}, c_d={c_d:.1f} | max_r={max_r:.4f} mm (dia={max_r*2:.4f} mm), rms_r={rms_r:.4f} mm")
                    sys.stdout.flush()
