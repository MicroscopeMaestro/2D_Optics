import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from verify_layout import generate_fiber_phase_space_rays, RAMAN_WAVELENGTHS_UM

rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)

def evaluate(f_coll=47.5, f_dmd=44.0, dmd_r=20.0, r_c=5.0, c_d=5.2, pri_wl=1.38919, relay_f=50.0):
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
    
    if relay_f == 75.0:
        stack.add_surface(Surface(name="AC254-075-C_F", radius=49.3, thickness=5.2, glass="N-BK7", semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-075-C_M", radius=-39.0, thickness=2.0, glass="SF5", semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-075-C_B", radius=-170.0, thickness=r_c, glass="AIR", semi_diameter=12.7))
    else:
        stack.add_surface(Surface(name="AC254-050-C_F", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-050-C_M", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-050-C_B", radius=-115.0, thickness=r_c, glass="AIR", semi_diameter=12.7))
        
    stack.add_surface(Surface(name="ACL12708U-C_F", surface_type=SurfaceType.ASPHERIC, radius=4.16, thickness=4.5, glass="B270", semi_diameter=12.7, conic=-0.6265, asphere_coeffs=[1.5e-5]))
    stack.add_surface(Surface(name="ACL12708U-C_B", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=c_d, glass="AIR", semi_diameter=12.7))
    stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.50))
    
    stack.primary_wavelength = pri_wl
    stack._update_accumulated_positions()
    
    opt = OpticalSystem(stack)
    traced = opt.trace_rays(rays)
    det_idx = len(stack.surfaces) - 1
    
    det_hits = [(h['x'], h['y'], r.wavelength) for r in traced for h in r.history if h['surface_index'] == det_idx]
    vig_count = sum(1 for r in traced if r.vignetted and (not r.history or r.history[-1]['surface_index'] < det_idx))
    
    return det_hits, vig_count

print("Testing configs:")
for relay_f in [75.0, 50.0]:
    for dmd_r in [10.0, 15.0, 20.0]:
        for c_d in [4.8, 5.0, 5.2, 5.4]:
            hits, vig = evaluate(dmd_r=dmd_r, c_d=c_d, relay_f=relay_f)
            if hits:
                rads = [np.sqrt(x**2 + y**2) for x, y, wl in hits]
                # Breakdown by wavelength
                max_per_wl = {}
                for x, y, wl in hits:
                    r = np.sqrt(x**2 + y**2)
                    max_per_wl[wl] = max(max_per_wl.get(wl, 0.0), r)
                wl_str = ", ".join([f"{wl}:{max_per_wl.get(wl, 0):.2f}" for wl in RAMAN_WAVELENGTHS_UM])
                print(f"Relay f={relay_f:.0f}, dmd_r={dmd_r:.0f}, cd={c_d:.1f} | vig={vig}, max_r={max(rads):.3f} mm (rms={np.sqrt(np.mean(np.array(rads)**2)):.3f}) | {wl_str}")
