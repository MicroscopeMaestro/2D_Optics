import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from scratch.test_verify_harness import build_remediated_stack
from optics_app.engine.tracer import OpticalSystem
from verify_layout import generate_fiber_phase_space_rays, RAMAN_WAVELENGTHS_UM

stack = build_remediated_stack()
stack.primary_wavelength = 1.38919
stack._update_accumulated_positions()

opt = OpticalSystem(stack)
rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
traced = opt.trace_rays(rays)
det_idx = len(stack.surfaces) - 1

print(f"{'Wavelength':<15} {'Total':<8} {'Vignetted':<10} {'Max Radius (mm)':<18} {'RMS Radius (mm)':<18} {'Centroid (mm)':<20}")
print("-" * 90)

all_radii = []
for wl in RAMAN_WAVELENGTHS_UM:
    wl_rays = [r for r in traced if abs(r.wavelength - wl) < 1e-4]
    hits = [(h['x'], h['y']) for r in wl_rays for h in r.history if h['surface_index'] == det_idx]
    vignetted = sum(1 for r in wl_rays if r.vignetted)
    if hits:
        xs = [x for x, y in hits]
        ys = [y for x, y in hits]
        radii = [np.sqrt(x**2 + y**2) for x, y in hits]
        all_radii.extend(radii)
        max_r = max(radii)
        rms_r = np.sqrt(np.mean(np.array(radii)**2))
        cx = np.mean(xs)
        cy = np.mean(ys)
        print(f"{wl:<15.5f} {len(wl_rays):<8} {vignetted:<10} {max_r:<18.4f} {rms_r:<18.4f} ({cx:+.4f}, {cy:+.4f})")
    else:
        print(f"{wl:<15.5f} {len(wl_rays):<8} {vignetted:<10} {'NO HITS':<18}")

if all_radii:
    print(f"\nOVERALL MAX RADIUS: {max(all_radii):.4f} mm (MAX DIAMETER: {max(all_radii)*2:.4f} mm)")
    print(f"OVERALL RMS RADIUS: {np.sqrt(np.mean(np.array(all_radii)**2)):.4f} mm")
