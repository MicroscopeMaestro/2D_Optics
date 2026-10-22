import sys, os, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray

for d_fiber in np.linspace(45.0, 70.0, 26):
    stack = SurfaceStack()
    stack.primary_wavelength = 1.3892
    stack.add_surface(Surface(name="Fiber", surface_type=SurfaceType.OBJECT, thickness=d_fiber, glass="AIR"))
    stack.add_surface(Surface(name="L1_F", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="L1_M", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="L1_B", radius=-115.0, thickness=50.0, glass="AIR", semi_diameter=12.7))
    stack._update_accumulated_positions()

    theta = math.asin(0.12)
    r2 = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([math.sin(theta), 0.0, math.cos(theta)]), wavelength=1.3892)
    opt = OpticalSystem(stack)
    tr = opt.trace_rays([r2])
    # The direction of the ray in global or local frame after L1_B
    vx, vy, vz = tr[0].dir
    div_deg = math.degrees(math.atan2(vx, vz))
    print(f"d_fiber = {d_fiber:.2f} mm -> ray exit dir vx = {vx:.6f}, vz = {vz:.6f}, angle = {div_deg:.4f} deg")
