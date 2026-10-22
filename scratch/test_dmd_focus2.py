import sys, os, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray

for d_foc in np.linspace(45.0, 55.0, 21):
    stack = SurfaceStack()
    stack.primary_wavelength = 1.3892
    stack.add_surface(Surface(name="Fiber", surface_type=SurfaceType.OBJECT, thickness=64.57, glass="AIR"))
    stack.add_surface(Surface(name="Coll1", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="Coll2", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="Coll3", radius=-115.0, thickness=15.0, glass="AIR", semi_diameter=12.7))
    stack.add_surface(Surface(name="Grating", surface_type=SurfaceType.GRATING, radius=0.0, thickness=15.0, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0, grating_lines_per_mm=300.0, grating_order=1))
    stack.add_surface(Surface(name="Foc1", radius=25.1, thickness=6.8, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="Foc2", radius=-20.2, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="Foc3", radius=-84.0, thickness=d_foc, glass="AIR", semi_diameter=12.7))
    stack.add_surface(Surface(name="DMD", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=0.0, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0))
    stack._update_accumulated_positions()
    opt = OpticalSystem(stack)

    na = 0.12
    max_theta = math.asin(na)
    rays = [
        Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.3892),
        Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([math.sin(max_theta), 0.0, math.cos(max_theta)]), wavelength=1.3892),
        Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([-math.sin(max_theta), 0.0, math.cos(max_theta)]), wavelength=1.3892),
        Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, math.sin(max_theta), math.cos(max_theta)]), wavelength=1.3892),
        Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, -math.sin(max_theta), math.cos(max_theta)]), wavelength=1.3892),
    ]
    tr = opt.trace_rays(rays)
    dmd_hits = [h for r in tr for h in r.history if h['surface_index'] == 8]
    if dmd_hits:
        xs = [h['x'] for h in dmd_hits]
        ys = [h['y'] for h in dmd_hits]
        spot_size_x = max(xs) - min(xs)
        spot_size_y = max(ys) - min(ys)
        print(f"d_foc = {d_foc:.2f} mm -> Spot on DMD at 1.3892 um: delta_x = {spot_size_x*1000:.1f} um, delta_y = {spot_size_y*1000:.1f} um")
