import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from scratch.find_true_detector_focus import build_test_stack
from optics_app.engine.tracer import OpticalSystem
from verify_layout import generate_fiber_phase_space_rays, RAMAN_WAVELENGTHS_UM

for c_d in [1.90, 1.95, 2.00, 2.05, 2.10]:
    stack = build_test_stack(c_d=c_d, pri_wl=1.3620)
    opt = OpticalSystem(stack)
    rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
    traced = opt.trace_rays(rays)
    det_idx = len(stack.surfaces) - 1
    
    print(f"\n=======================================================")
    print(f"Testing All 7 Wavelengths at c_d = {c_d:.2f} mm:")
    print(f"=======================================================")
    
    all_radii = []
    det_hits = 0
    vig_count = sum(1 for r in traced if r.vignetted and (not r.history or r.history[-1]['surface_index'] < det_idx))
    
    for wl in RAMAN_WAVELENGTHS_UM:
        wl_rays = [r for r in traced if abs(r.wavelength - wl) < 1e-4]
        hits = [(h['x'], h['y']) for r in wl_rays for h in r.history if h['surface_index'] == det_idx]
        if hits:
            radii = [np.sqrt(x**2 + y**2) for x, y in hits]
            all_radii.extend(radii)
            det_hits += len(hits)
            max_r = max(radii)
            rms_r = np.sqrt(np.mean(np.array(radii)**2))
            cx = np.mean([x for x, y in hits])
            cy = np.mean([y for x, y in hits])
            print(f"  wl={wl:.5f} um ({len(hits)} rays): max_r = {max_r:.4f} mm (dia = {max_r*2:.4f} mm), rms_r = {rms_r:.4f} mm | centroid = ({cx:+.4f}, {cy:+.4f})")
            
    print(f"\nTotal Rays: {len(rays)}, Det Rays: {det_hits}, Vignetted before Det: {vig_count}")
    if all_radii:
        print(f"OVERALL MAX SPOT RADIUS: {max(all_radii):.4f} mm (OVERALL MAX DIAMETER: {max(all_radii)*2:.4f} mm)")
        print(f"OVERALL RMS SPOT RADIUS: {np.sqrt(np.mean(np.array(all_radii)**2)):.4f} mm")
