import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import generate_fiber_phase_space_rays, RAMAN_WAVELENGTHS_UM

# Let's test with Focus Lens = AC254-075-C (f=75mm) or AC254-100-C (f=100mm)
# and Grating lines/mm = 150 to 300
rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)

def build_scaled_stack(f_coll=47.5, grating_lpmm=300.0, f_focus_name='AC254-075-C', f_dmd=70.0, dmd_r=20.0, r_c=5.0, c_d=5.2, pri_wl=1.3620):
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Fiber", thickness=f_coll, semi_diameter=25.0))
    stack.add_surface(Surface(name="Coll_F", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="Coll_M", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="Coll_B", radius=-115.0, thickness=20.0, glass="AIR", semi_diameter=12.7))
    stack.add_surface(Surface(name="Grating", surface_type=SurfaceType.GRATING, radius=0.0, thickness=20.0, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0, grating_lines_per_mm=grating_lpmm, grating_order=1))
    
    if f_focus_name == 'AC254-075-C':
        stack.add_surface(Surface(name="Focus_F", radius=49.3, thickness=5.2, glass="N-BK7", semi_diameter=12.7))
        stack.add_surface(Surface(name="Focus_M", radius=-39.0, thickness=2.0, glass="SF5", semi_diameter=12.7))
        stack.add_surface(Surface(name="Focus_B", radius=-170.0, thickness=f_dmd, glass="AIR", semi_diameter=12.7))
    elif f_focus_name == 'AC254-050-C':
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
    stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.50))
    
    stack.primary_wavelength = pri_wl
    stack._update_accumulated_positions()
    return stack

print("Testing f_focus scaling:")
for grating_lpmm in [300.0]:
    for f_focus_name, f_dmd in [('AC254-050-C', 44.0), ('AC254-075-C', 70.0)]:
        stack = build_scaled_stack(grating_lpmm=grating_lpmm, f_focus_name=f_focus_name, f_dmd=f_dmd)
        opt = OpticalSystem(stack)
        r1 = opt.trace_ray(Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=1.1113))
        r2 = opt.trace_ray(Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=1.6133))
        x1 = r1.history[8]['x']
        x2 = r2.history[8]['x']
        span = abs(x2 - x1)
        print(f"Focus={f_focus_name} (f_dmd={f_dmd:.1f}) | span(1.11-1.61)={span:.3f} mm [x1={x1:+.3f}, x2={x2:+.3f}]")
