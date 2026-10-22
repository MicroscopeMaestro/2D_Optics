import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import generate_fiber_phase_space_rays, RAMAN_WAVELENGTHS_UM

def build_system(dmd_r=8.0, pri_wl=1.38919, c_d=5.2):
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Fiber_FG025LJA", surface_type=SurfaceType.OBJECT, thickness=47.5, semi_diameter=25.0))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Back", radius=-115.0, thickness=20.0, glass="AIR", semi_diameter=12.7))
    stack.add_surface(Surface(name="Grating_GR25-0316", surface_type=SurfaceType.GRATING, radius=0.0, thickness=20.0, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0, grating_lines_per_mm=300.0, grating_order=1))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Back", radius=-115.0, thickness=38.5, glass="AIR", semi_diameter=12.7))
    stack.add_surface(Surface(name="TI_DMD4500", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=dmd_r, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0))
    
    # Relay AC254-030-C (reversed)
    stack.add_surface(Surface(name="AC254-030-C_Relay_Front", radius=57.2, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-030-C_Relay_Mid", radius=14.8, thickness=7.5, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-030-C_Relay_Back", radius=-18.5, thickness=2.0, glass="AIR", semi_diameter=12.7))
    
    # Condenser (ACL12708U-C)
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Front", surface_type=SurfaceType.ASPHERIC, radius=4.16, thickness=4.5, glass="B270", semi_diameter=12.7, conic=-0.6265, asphere_coeffs=[1.5e-5]))
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Back", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=c_d, glass="AIR", semi_diameter=12.7))
    # Detector
    stack.add_surface(Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=25.0))
    
    stack.primary_wavelength = pri_wl
    stack._update_accumulated_positions()
    return stack

if __name__ == '__main__':
    rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
    print(f"Generated {len(rays)} rays.")
    
    for c_d in [4.8, 5.0, 5.2, 5.4, 5.6]:
        stack = build_system(dmd_r=8.0, pri_wl=1.38919, c_d=c_d)
        opt = OpticalSystem(stack)
        traced = opt.trace_rays(rays)
        det_idx = len(stack.surfaces) - 1
        
        hits = [(h['x'], h['y']) for r in traced for h in r.history if h['surface_index'] == det_idx and not r.vignetted]
        vig_optics = sum(1 for r in traced if r.vignetted and (not r.history or r.history[-1]['surface_index'] < det_idx))
        
        if hits:
            rads = [np.sqrt(x**2 + y**2) for x, y in hits]
            max_r = max(rads)
            rms_r = np.sqrt(np.mean(np.array(rads)**2))
            print(f"c_d={c_d:.2f} mm | vig_optics={vig_optics}, det_rays={len(hits)}/{len(rays)} | max_r={max_r:.4f} mm (dia={max_r*2:.4f} mm), rms_r={rms_r:.4f} mm")
        else:
            print(f"c_d={c_d:.2f} mm | NO HITS (vig_optics={vig_optics})")
