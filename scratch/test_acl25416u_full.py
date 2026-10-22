import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import RAMAN_WAVELENGTHS_UM, generate_fiber_phase_space_rays, verify_spectrometer_layout

def build_system_acl25416(d_dmd_relay=50.0, d_relay_cond=10.0, d_cond_det=8.0):
    stack = SurfaceStack()
    stack.primary_wavelength = 1.38919
    
    # 0. Fiber
    stack.add_surface(Surface(name="Fiber_FG025LJA", surface_type=SurfaceType.OBJECT, radius=0.0, thickness=47.5, glass="AIR", semi_diameter=25.0))
    # 1-3. Collimator AC254-050-C
    stack.add_surface(Surface(name="AC254-050-C_Coll_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Back", radius=-115.0, thickness=20.0, glass="AIR", semi_diameter=12.7))
    # 4. Grating GR25-0316
    stack.add_surface(Surface(name="Grating_GR25-0316", surface_type=SurfaceType.GRATING, radius=0.0, thickness=20.0, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0, grating_lines_per_mm=300.0, grating_order=1))
    # 5-7. Camera AC254-050-C
    stack.add_surface(Surface(name="AC254-050-C_Focus_Front", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Mid", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Back", radius=-115.0, thickness=44.0, glass="AIR", semi_diameter=12.7))
    # 8. DMD
    stack.add_surface(Surface(name="TI_DMD4500", radius=0.0, thickness=float(d_dmd_relay), glass="MIRROR", semi_diameter=12.7, tilt_y=12.0))
    # 9-11. Relay AC508-075-C / AC254-075-C
    stack.add_surface(Surface(name="AC508-075-C_Relay_Front", radius=49.3, thickness=8.0, glass="N-BK7", semi_diameter=25.4))
    stack.add_surface(Surface(name="AC508-075-C_Relay_Mid", radius=-39.0, thickness=3.0, glass="SF5", semi_diameter=25.4))
    stack.add_surface(Surface(name="AC508-075-C_Relay_Back", radius=-170.0, thickness=float(d_relay_cond), glass="AIR", semi_diameter=25.4))
    # 12-13. Condenser ACL25416U-C (1-inch aspheric condenser f=16mm, NA=0.68)
    stack.add_surface(Surface(name="ACL25416U-C_Cond_Front", surface_type=SurfaceType.ASPHERIC, radius=8.32, thickness=12.0, glass="B270", semi_diameter=12.7, conic=-0.6265, asphere_coeffs=[1.8e-6]))
    stack.add_surface(Surface(name="ACL25416U-C_Cond_Back", radius=0.0, thickness=float(d_cond_det), glass="AIR", semi_diameter=12.7))
    # 14. Detector InGaAs 1mm
    stack.add_surface(Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.50))
    
    return stack

rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
print(f"Total test rays: {len(rays)}")

stack = build_system_acl25416(d_dmd_relay=50.0, d_relay_cond=10.0, d_cond_det=8.0)
opt = OpticalSystem(stack)
traced = opt.trace_rays(rays)

print("\nSurface footprints for ACL25416U-C configuration:")
for s_idx, s in enumerate(stack.surfaces):
    hits = [h for r in traced for h in r.history if h['surface_index'] == s_idx]
    if hits:
        xs = [h['x'] for h in hits]
        ys = [h['y'] for h in hits]
        rs = [np.sqrt(h['x']**2 + h['y']**2) for h in hits]
        print(f"S{s_idx:2d} ({s.name:24s}): hits={len(hits):3d}/315 | x=[{min(xs):7.3f}, {max(xs):7.3f}] | y=[{min(ys):7.3f}, {max(ys):7.3f}] | max_r={max(rs):6.3f} (SD={s.semi_diameter:5.2f})")

det_hits = [h for r in traced for h in r.history if h['surface_index'] == 14]
rs_det = [np.sqrt(h['x']**2 + h['y']**2) for h in det_hits]
print(f"\nFinal Detector spot radius: {max(rs_det):.4f} mm (Diameter: {2*max(rs_det):.4f} mm)")
print(f"Vignetted rays: {sum(1 for r in traced if r.vignetted)} / {len(rays)} (Transmission: {100.0 * (1.0 - sum(1 for r in traced if r.vignetted)/len(rays)):.2f}%)")
