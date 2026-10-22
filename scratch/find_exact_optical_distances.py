import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import generate_fiber_phase_space_rays, RAMAN_WAVELENGTHS_UM

def build_candidate_stack(
    f_coll=47.5,
    c_g=20.0,
    g_f=20.0,
    f_dmd=44.0,
    dmd_r=20.0,
    relay_name='AC254-075-C',
    relay_orient='forward',
    r_c=5.0,
    c_d=5.2,
    pri_wl=1.3620
):
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Fiber_FG025LJA", surface_type=SurfaceType.OBJECT, thickness=f_coll, semi_diameter=25.0))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Back", radius=-115.0, thickness=c_g, glass="AIR", semi_diameter=12.7))
    stack.add_surface(Surface(name="Grating_GR25-0316", surface_type=SurfaceType.GRATING, radius=0.0, thickness=g_f, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0, grating_lines_per_mm=300.0, grating_order=1))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Back", radius=-115.0, thickness=f_dmd, glass="AIR", semi_diameter=12.7))
    stack.add_surface(Surface(name="TI_DMD4500", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=dmd_r, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0))
    
    # Relay Lens (1-inch Thorlabs Doublet)
    if relay_name == 'AC254-075-C':
        if relay_orient == 'forward':
            stack.add_surface(Surface(name="AC254-075-C_Relay_Front", radius=49.3, thickness=5.2, glass="N-BK7", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-075-C_Relay_Mid", radius=-39.0, thickness=2.0, glass="SF5", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-075-C_Relay_Back", radius=-170.0, thickness=r_c, glass="AIR", semi_diameter=12.7))
        else:
            stack.add_surface(Surface(name="AC254-075-C_Relay_Front", radius=170.0, thickness=2.0, glass="SF5", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-075-C_Relay_Mid", radius=39.0, thickness=5.2, glass="N-BK7", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-075-C_Relay_Back", radius=-49.3, thickness=r_c, glass="AIR", semi_diameter=12.7))
    elif relay_name == 'AC254-050-C':
        if relay_orient == 'forward':
            stack.add_surface(Surface(name="AC254-050-C_Relay_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-050-C_Relay_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-050-C_Relay_Back", radius=-115.0, thickness=r_c, glass="AIR", semi_diameter=12.7))
        else:
            stack.add_surface(Surface(name="AC254-050-C_Relay_Front", radius=115.0, thickness=2.0, glass="SF5", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-050-C_Relay_Mid", radius=27.8, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
            stack.add_surface(Surface(name="AC254-050-C_Relay_Back", radius=-34.0, thickness=r_c, glass="AIR", semi_diameter=12.7))
            
    # Condenser (ACL12708U-C) - 12.7 mm diameter
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Front", surface_type=SurfaceType.ASPHERIC, radius=4.16, thickness=4.5, glass="B270", semi_diameter=12.7, conic=-0.6265, asphere_coeffs=[1.5e-5]))
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Back", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=c_d, glass="AIR", semi_diameter=12.7))
    # Detector (1.0 mm InGaAs detector, semi_diameter=0.50 mm)
    stack.add_surface(Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.50))

    stack.primary_wavelength = pri_wl
    stack._update_accumulated_positions()
    return stack

if __name__ == '__main__':
    # Test primary Raman wavelengths (400 to 3200 cm-1)
    wl_list = [1.11130, 1.20000, 1.30000, 1.36200, 1.50000, 1.60000, 1.61330]
    rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=wl_list)
    print(f"Generated {len(rays)} rays for Raman band 1.1113 - 1.6133 um.")
    
    stack = build_candidate_stack(
        f_coll=47.5, c_g=20.0, g_f=20.0, f_dmd=44.0,
        dmd_r=20.0, relay_name='AC254-075-C', relay_orient='forward', r_c=5.0, c_d=5.2, pri_wl=1.3620
    )
    opt = OpticalSystem(stack)
    traced = opt.trace_rays(rays)
    det_idx = len(stack.surfaces) - 1
    
    print("\n--- Ray trace results on 1.1113 - 1.6133 um band ---")
    vig_optics = sum(1 for r in traced if r.vignetted and (not r.history or r.history[-1]['surface_index'] < det_idx))
    det_hits = [(h['x'], h['y']) for r in traced for h in r.history if h['surface_index'] == det_idx and not (r.vignetted and r.history[-1]['surface_index'] < det_idx)]
    
    print(f"Vignetted in optics: {vig_optics} / {len(rays)} (Transmission: {(len(rays)-vig_optics)/len(rays)*100:.2f}%)")
    print(f"Reaching detector: {len(det_hits)} / {len(rays)}")
    
    # Check DMD span
    r1 = opt.trace_ray(Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=1.1113))
    r2 = opt.trace_ray(Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=1.6133))
    x1 = r1.history[8]['x']
    x2 = r2.history[8]['x']
    span = abs(x2 - x1)
    print(f"DMD Chief Ray Span (1.1113 to 1.6133 um): {span:.4f} mm [x1={x1:+.4f} mm, x2={x2:+.4f} mm]")
    print(f"Active DMD width utilization: {span / 8.664 * 100:.2f}%")
    
    if det_hits:
        radii = [np.sqrt(x**2 + y**2) for x, y in det_hits]
        print(f"Max Detector Spot Radius: {max(radii):.4f} mm (Max Diameter: {max(radii)*2:.4f} mm)")
        print(f"RMS Detector Spot Radius: {np.sqrt(np.mean(np.array(radii)**2)):.4f} mm")
        print(f"Detector Centroid: ({np.mean([x for x, y in det_hits]):+.4f}, {np.mean([y for x, y in det_hits]):+.4f}) mm")
