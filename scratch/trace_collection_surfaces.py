import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import RAMAN_WAVELENGTHS_UM

stack = SurfaceStack()
# 0. Fiber
stack.add_surface(Surface(name="Fiber_FG025LJA", surface_type=SurfaceType.OBJECT, thickness=47.5, semi_diameter=25.0))
# 1-3. Collimator AC254-050-C
stack.add_surface(Surface(name="AC254-050-C_Coll_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
stack.add_surface(Surface(name="AC254-050-C_Coll_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
stack.add_surface(Surface(name="AC254-050-C_Coll_Back", radius=-115.0, thickness=20.0, glass="AIR", semi_diameter=12.7))
# 4. Grating GR25-0316
stack.add_surface(Surface(name="Grating_GR25-0316", surface_type=SurfaceType.GRATING, radius=0.0, thickness=20.0, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0, grating_lines_per_mm=300.0, grating_order=1))
# 5-7. Focus lens AC254-050-C
stack.add_surface(Surface(name="AC254-050-C_Focus_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
stack.add_surface(Surface(name="AC254-050-C_Focus_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
stack.add_surface(Surface(name="AC254-050-C_Focus_Back", radius=-115.0, thickness=44.0, glass="AIR", semi_diameter=12.7))
# 8. DMD4500
stack.add_surface(Surface(name="TI_DMD4500", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=20.0, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0))
# 9-11. Relay Lens AC254-075-C
stack.add_surface(Surface(name="AC254-075-C_Relay_Front", radius=49.3, thickness=5.2, glass="N-BK7", semi_diameter=12.7))
stack.add_surface(Surface(name="AC254-075-C_Relay_Mid", radius=-39.0, thickness=2.0, glass="SF5", semi_diameter=12.7))
stack.add_surface(Surface(name="AC254-075-C_Relay_Back", radius=-170.0, thickness=5.0, glass="AIR", semi_diameter=12.7))
# 12-13. Condenser ACL12708U-C
stack.add_surface(Surface(name="ACL12708U-C_Cond_Front", surface_type=SurfaceType.ASPHERIC, radius=4.16, thickness=4.5, glass="B270", semi_diameter=12.7, conic=-0.6265, asphere_coeffs=[1.5e-5]))
stack.add_surface(Surface(name="ACL12708U-C_Cond_Back", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=5.2, glass="AIR", semi_diameter=12.7))
# 14. Detector
stack.add_surface(Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.50))

stack.primary_wavelength = 1.3620
stack._update_accumulated_positions()
opt = OpticalSystem(stack)

print("Surface Origins & Orientations:")
for i, s in enumerate(stack.surfaces):
    print(f"Surface {i:2d} ({s.name:<25}): origin={s.global_origin}, R_z_axis={s.global_R[:,2]}")

print("\nChief ray trace per surface (local x, y):")
for wl in [1.1113, 1.3620, 1.6133, 1.66708]:
    r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=wl)
    tr = opt.trace_ray(r)
    print(f"\n--- Wavelength {wl:.5f} um ---")
    for h in tr.history:
        s_idx = h['surface_index']
        s_name = stack.surfaces[s_idx].name
        print(f"  Surf {s_idx:2d} ({s_name:<25}): local=({h['x']:+.4f}, {h['y']:+.4f}) mm | vignetted={h['vignetted']}")
