import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import generate_fiber_phase_space_rays, RAMAN_WAVELENGTHS_UM

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
# Detector
stack.add_surface(Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=25.0))

stack.primary_wavelength = 1.3620
stack._update_accumulated_positions()
opt = OpticalSystem(stack)

rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
traced = opt.trace_rays(rays)
det_idx = len(stack.surfaces) - 1

print("Analyzing all 315 rays at detector:")
all_hits_per_wl = {wl: [] for wl in RAMAN_WAVELENGTHS_UM}

for r in traced:
    for h in r.history:
        if h['surface_index'] == det_idx:
            all_hits_per_wl[r.wavelength].append((h['x'], h['y']))

all_radii = []
for wl in RAMAN_WAVELENGTHS_UM:
    hits = all_hits_per_wl[wl]
    radii = [np.sqrt(x**2 + y**2) for x, y in hits]
    all_radii.extend(radii)
    max_r = max(radii)
    rms_r = np.sqrt(np.mean(np.array(radii)**2))
    inside_05 = sum(1 for r in radii if r <= 0.500)
    print(f"wl={wl:.5f} um: total={len(hits)} | max_r={max_r:.4f} mm (dia={max_r*2:.4f} mm) | rms_r={rms_r:.4f} mm | <=0.5mm: {inside_05}/{len(hits)}")

print(f"\nOVERALL across all 7 wavelengths:")
print(f"Max Spot Radius: {max(all_radii):.4f} mm (Max Spot Diameter: {max(all_radii)*2:.4f} mm)")
print(f"RMS Spot Radius: {np.sqrt(np.mean(np.array(all_radii)**2)):.4f} mm")
print(f"Rays <= 0.500 mm radius: {sum(1 for r in all_radii if r <= 0.500)} / {len(all_radii)} ({sum(1 for r in all_radii if r <= 0.500)/len(all_radii)*100:.1f}%)")
