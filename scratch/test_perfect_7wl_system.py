import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import generate_fiber_phase_space_rays, RAMAN_WAVELENGTHS_UM

def build_spectrometer(f_coll, f_dmd, dmd_r, relay_name, orient, r_c, c_d, pri_wl=1.38919):
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Fiber_FG025LJA", surface_type=SurfaceType.OBJECT, thickness=f_coll, semi_diameter=25.0))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Back", radius=-115.0, thickness=20.0, glass="AIR", semi_diameter=12.7))
    stack.add_surface(Surface(name="Grating_GR25-0316", surface_type=SurfaceType.GRATING, radius=0.0, thickness=20.0, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0, grating_lines_per_mm=300.0, grating_order=1))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Back", radius=-115.0, thickness=f_dmd, glass="AIR", semi_diameter=12.7))
    stack.add_surface(Surface(name="TI_DMD4500", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=dmd_r, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0))
    
    # Relay Lens
    if relay_name == 'AC254-075-C':
        if orient == 'forward':
            stack.add_surface(Surface(name="AC254-075-C_Relay_Front", radius=49.3, thickness=5.2, glass="N-BK7", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-075-C_Relay_Mid", radius=-39.0, thickness=2.0, glass="SF5", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-075-C_Relay_Back", radius=-170.0, thickness=r_c, glass="AIR", semi_diameter=12.7))
        else:
            stack.add_surface(Surface(name="AC254-075-C_Relay_Front", radius=170.0, thickness=2.0, glass="SF5", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-075-C_Relay_Mid", radius=39.0, thickness=5.2, glass="N-BK7", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-075-C_Relay_Back", radius=-49.3, thickness=r_c, glass="AIR", semi_diameter=12.7))
    elif relay_name == 'AC254-050-C':
        if orient == 'forward':
            stack.add_surface(Surface(name="AC254-050-C_Relay_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-050-C_Relay_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-050-C_Relay_Back", radius=-115.0, thickness=r_c, glass="AIR", semi_diameter=12.7))
        else:
            stack.add_surface(Surface(name="AC254-050-C_Relay_Front", radius=115.0, thickness=2.0, glass="SF5", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-050-C_Relay_Mid", radius=27.8, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-050-C_Relay_Back", radius=-34.0, thickness=r_c, glass="AIR", semi_diameter=12.7))
            
    # Condenser (ACL12708U-C)
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Front", surface_type=SurfaceType.ASPHERIC, radius=4.16, thickness=4.5, glass="B270", semi_diameter=12.7, conic=-0.6265, asphere_coeffs=[1.5e-5]))
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Back", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=c_d, glass="AIR", semi_diameter=12.7))
    # Detector
    stack.add_surface(Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.50))
    
    stack.primary_wavelength = pri_wl
    stack._update_accumulated_positions()
    return stack

rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)

for f_dmd in [38.5, 39.0, 39.5, 40.0]:
    for dmd_r in [15.0, 18.0, 20.0]:
        for r_c in [2.0, 3.0, 5.0]:
            for c_d in [5.0, 5.2, 5.4]:
                stack = build_spectrometer(47.5, f_dmd, dmd_r, 'AC254-075-C', 'forward', r_c, c_d, pri_wl=1.38919)
                opt = OpticalSystem(stack)
                traced = opt.trace_rays(rays)
                det_idx = len(stack.surfaces) - 1
                
                vig_optics = sum(1 for r in traced if r.vignetted and (not r.history or r.history[-1]['surface_index'] < det_idx))
                det_hits = [(h['x'], h['y']) for r in traced for h in r.history if h['surface_index'] == det_idx and not (r.vignetted and r.history[-1]['surface_index'] < det_idx)]
                
                # Check DMD chief span
                r1 = opt.trace_ray(Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=1.1113))
                r2 = opt.trace_ray(Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=1.6133))
                r3 = opt.trace_ray(Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=1.66708))
                x1 = r1.history[8]['x']
                x2 = r2.history[8]['x']
                x3 = r3.history[8]['x']
                span = abs(x2 - x1)
                
                if det_hits:
                    rads = [np.sqrt(x**2 + y**2) for x, y in det_hits]
                    max_r = max(rads)
                    rms_r = np.sqrt(np.mean(np.array(rads)**2))
                    print(f"f_dmd={f_dmd:.1f}, dmd_r={dmd_r:.1f}, r_c={r_c:.1f}, c_d={c_d:.1f} | vig_optics={vig_optics}, det_rays={len(det_hits)}/{len(rays)} | span(1.11-1.61)={span:.3f} mm [x1={x1:+.3f}, x2={x2:+.3f}, x3={x3:+.3f}] | max_r={max_r:.4f} mm (dia={max_r*2:.4f} mm), rms_r={rms_r:.4f} mm")
                    sys.stdout.flush()
