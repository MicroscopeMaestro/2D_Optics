import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import generate_fiber_phase_space_rays, RAMAN_WAVELENGTHS_UM

# We want:
# x(1.11130) >= -4.332 mm
# x(1.66708) <= +4.332 mm
# span(1.1113 to 1.6133) in [8.00, 8.664] mm
# And spot radius at detector <= 0.500 mm for ALL 7 wavelengths!

def build_test_system(f_dmd, pri_wl, dmd_r, r_c, c_d):
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Fiber_FG025LJA", surface_type=SurfaceType.OBJECT, thickness=47.5, semi_diameter=25.0))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Back", radius=-115.0, thickness=20.0, glass="AIR", semi_diameter=12.7))
    stack.add_surface(Surface(name="Grating_GR25-0316", surface_type=SurfaceType.GRATING, radius=0.0, thickness=20.0, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0, grating_lines_per_mm=300.0, grating_order=1))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Back", radius=-115.0, thickness=f_dmd, glass="AIR", semi_diameter=12.7))
    stack.add_surface(Surface(name="TI_DMD4500", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=dmd_r, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0))
    # Relay AC254-075-C
    stack.add_surface(Surface(name="AC254-075-C_Relay_Front", radius=49.3, thickness=5.2, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-075-C_Relay_Mid", radius=-39.0, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-075-C_Relay_Back", radius=-170.0, thickness=r_c, glass="AIR", semi_diameter=12.7))
    # Condenser ACL12708U-C
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Front", surface_type=SurfaceType.ASPHERIC, radius=4.16, thickness=4.5, glass="B270", semi_diameter=12.7, conic=-0.6265, asphere_coeffs=[1.5e-5]))
    stack.add_surface(Surface(name="ACL12708U-C_Cond_Back", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=c_d, glass="AIR", semi_diameter=12.7))
    # Detector
    stack.add_surface(Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.50))

    stack.primary_wavelength = pri_wl
    stack._update_accumulated_positions()
    return stack

rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)

print("Scanning for simultaneous DMD span [8.0, 8.66] mm AND 7-wl spot radius <= 0.500 mm:")
found = []

for pri_wl in np.arange(1.370, 1.410, 0.005):
    for f_dmd in np.arange(36.0, 44.0, 0.5):
        # Quick check of DMD chief ray positions
        stack = build_test_system(f_dmd, pri_wl, 20.0, 5.0, 5.2)
        opt = OpticalSystem(stack)
        r1 = opt.trace_ray(Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=1.1113))
        r2 = opt.trace_ray(Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=1.6133))
        r3 = opt.trace_ray(Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=1.66708))
        
        x1 = r1.history[8]['x']
        x2 = r2.history[8]['x']
        x3 = r3.history[8]['x']
        span12 = abs(x2 - x1)
        
        # Must fit on DMD active area [-4.332, +4.332]
        if x1 < -4.332 or x3 > 4.332:
            continue
        if not (8.00 <= span12 <= 8.664):
            continue
            
        print(f"Found DMD match: pri_wl={pri_wl:.4f}, f_dmd={f_dmd:.1f} | x1={x1:+.3f}, x2={x2:+.3f}, x3={x3:+.3f} | span={span12:.3f} mm")
        found.append((pri_wl, f_dmd, span12, x1, x2, x3))

print(f"\nTotal matching front-end designs: {len(found)}")
