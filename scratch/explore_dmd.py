import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray

def check_dmd(grat_tilt=12.0, t_cam=44.0, t_coll_grat=20.0, t_grat_cam=20.0, dmd_tilt=12.0):
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Fiber", surface_type=SurfaceType.OBJECT, thickness=47.5, semi_diameter=25.0))
    stack.add_surface(Surface(name="Coll_F", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="Coll_M", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="Coll_B", radius=-115.0, thickness=t_coll_grat, glass="AIR", semi_diameter=12.7))
    stack.add_surface(Surface(name="Grating", surface_type=SurfaceType.GRATING, grating_lines_per_mm=300.0, tilt_y=grat_tilt, thickness=t_grat_cam, glass="MIRROR", semi_diameter=12.7))
    stack.add_surface(Surface(name="Cam_F", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="Cam_M", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="Cam_B", radius=-115.0, thickness=t_cam, glass="AIR", semi_diameter=12.7))
    stack.add_surface(Surface(name="DMD", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=20.0, glass="MIRROR", tilt_y=dmd_tilt, semi_diameter=12.7))

    opt = OpticalSystem(stack)
    res = {}
    for wl in [1.11130, 1.20000, 1.30000, 1.38919, 1.50000, 1.60000, 1.61330, 1.66708]:
        cr = opt.trace_rays([Ray(pos=[0,0,0], dir=[0,0,1], wavelength=wl)])[0]
        h_dmd = next((h for h in cr.history if h['surface_index'] == 8), None)
        res[wl] = h_dmd['x'] if h_dmd else None
    
    span_8 = abs(res[1.61330] - res[1.11130]) if res[1.61330] is not None and res[1.11130] is not None else 0
    center_8 = (res[1.61330] + res[1.11130]) / 2.0 if res[1.61330] is not None and res[1.11130] is not None else 0
    return span_8, center_8, res

print("Baseline (grat_tilt=12.0, t_cam=44.0):")
s, c, r = check_dmd()
print(f"Span(1.11-1.61): {s:.4f} mm, Center: {c:.4f} mm")
for wl, x in r.items():
    print(f"  {wl:.5f} um -> X = {x:+.4f} mm")

print("\nTesting camera decenter / tilts / spacings...")
for t_cam in np.linspace(40.0, 48.0, 9):
    for g_tilt in np.linspace(11.0, 13.0, 9):
        s, c, r = check_dmd(grat_tilt=g_tilt, t_cam=t_cam)
        if 8.00 <= s <= 8.66:
            print(f"t_cam={t_cam:4.1f}, g_tilt={g_tilt:5.2f} -> Span={s:6.3f} mm, Center={c:+6.3f} mm, 1.111={r[1.11130]:+6.3f}, 1.613={r[1.61330]:+6.3f}, 1.667={r[1.66708]:+6.3f}")
