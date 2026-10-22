import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray

stack = SurfaceStack()
stack.add_surface(Surface(name="Fiber_FG025LJA", surface_type=SurfaceType.OBJECT, thickness=47.5, semi_diameter=25.0))
stack.add_surface(Surface(name="AC254-050-C_Coll_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
stack.add_surface(Surface(name="AC254-050-C_Coll_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
stack.add_surface(Surface(name="AC254-050-C_Coll_Back", radius=-115.0, thickness=20.0, glass="AIR", semi_diameter=12.7))
stack.add_surface(Surface(name="Grating_GR25-0316", surface_type=SurfaceType.GRATING, radius=0.0, thickness=20.0, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0, grating_lines_per_mm=300.0, grating_order=1))
stack.add_surface(Surface(name="AC254-050-C_Focus_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
stack.add_surface(Surface(name="AC254-050-C_Focus_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
stack.add_surface(Surface(name="AC254-050-C_Focus_Back", radius=-115.0, thickness=44.0, glass="AIR", semi_diameter=12.7))
stack.add_surface(Surface(name="TI_DMD4500", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=20.0, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0))
# Relay AC254-075-C
stack.add_surface(Surface(name="AC254-075-C_Relay_Front", radius=49.3, thickness=5.2, glass="N-BK7", semi_diameter=12.7))
stack.add_surface(Surface(name="AC254-075-C_Relay_Mid", radius=-39.0, thickness=2.0, glass="SF5", semi_diameter=12.7))
stack.add_surface(Surface(name="AC254-075-C_Relay_Back", radius=-170.0, thickness=5.0, glass="AIR", semi_diameter=12.7))
# Condenser ACL12708U-C
stack.add_surface(Surface(name="ACL12708U-C_Cond_Front", surface_type=SurfaceType.ASPHERIC, radius=4.16, thickness=4.5, glass="B270", semi_diameter=12.7, conic=-0.6265, asphere_coeffs=[1.5e-5]))
stack.add_surface(Surface(name="ACL12708U-C_Cond_Back", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=5.2, glass="AIR", semi_diameter=12.7))
# Detector (set semi_diameter=25.0 so we see where all rays land without artificial clipping)
stack.add_surface(Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=25.0))

stack.primary_wavelength = 1.3620
stack._update_accumulated_positions()
opt = OpticalSystem(stack)

# Trace on-axis rays at various NA angles
max_theta = np.arcsin(0.12)
print(f"Tracing rays at lambda = 1.3620 um from center (0,0):")
for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
    theta = frac * max_theta
    r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([np.sin(theta), 0., np.cos(theta)]), wavelength=1.3620)
    tr = opt.trace_ray(r)
    dmd_h = tr.history[8]
    relay_h = tr.history[9]
    cond_h = tr.history[12]
    det_h = tr.history[14]
    print(f"  NA_frac={frac:.2f} (theta={np.degrees(theta):.2f} deg): DMD_x={dmd_h['x']:+.3f}, Relay_x={relay_h['x']:+.3f}, Cond_x={cond_h['x']:+.3f} -> Det_x={det_h['x']:+.4f} mm, Det_r={abs(det_h['x']):.4f} mm")
