import json
import math
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem, Ray

def get_rays(start_pos, NA, wavelengths, num_rings=1):
    rays = []
    max_theta = np.arcsin(NA)
    for wl in wavelengths:
        rays.append(Ray(pos=np.array(start_pos, dtype=float), dir=np.array([0.0, 0.0, 1.0], dtype=float), wavelength=wl))
        for ring in range(1, num_rings + 1):
            theta = ring / num_rings * max_theta
            n_pts = 6 * ring
            for p in range(n_pts):
                phi = 2.0 * np.pi * p / n_pts
                dx = np.sin(theta) * np.cos(phi)
                dy = np.sin(theta) * np.sin(phi)
                dz = np.cos(theta)
                rays.append(Ray(pos=np.array(start_pos, dtype=float), dir=np.array([dx, dy, dz], dtype=float), wavelength=wl))
    return rays

def main():
    json_path = 'spectrometer_raman_50mm_100mm_manual.json'
    load_res = SurfaceStack.load_from_json(json_path)
    stack = load_res['stack']
    wavelengths = load_res['wavelengths']
    stack.primary_wavelength = wavelengths[1] if len(wavelengths) > 1 else wavelengths[0]
    
    for s in stack.surfaces:
        s.semi_diameter = 25.4
        s.aperture_type = 'None'
        
    sys_opt = OpticalSystem(stack)
    
    # 1. Optimize T1: AC254-030-C_S3 -> Aperture Stop (index 3)
    best_err = 1e9
    best_t1 = stack.surfaces[3].thickness
    for t in np.linspace(5, 20, 31):
        stack.surfaces[3].thickness = t
        stack._update_accumulated_positions()
        rays = get_rays([0,0,0], 0.1, [stack.primary_wavelength])
        traced = sys_opt.trace_rays(rays)
        err = sum((h['x']**2 + h['y']**2) for r in traced for h in r.history if h['surface_index'] == 8)
        if err < best_err:
            best_err = err
            best_t1 = t
    stack.surfaces[3].thickness = best_t1
    print(f'Best T1 (surf 3): {best_t1:.2f} (err: {best_err:.4f})')
    
    # 2. Optimize T2: AC254-100-C_S3 -> Entrance Slit (index 7)
    best_err = 1e9
    best_t2 = stack.surfaces[7].thickness
    for t in np.linspace(80, 110, 31):
        stack.surfaces[7].thickness = t
        stack._update_accumulated_positions()
        rays = get_rays([0,0,0], 0.1, [stack.primary_wavelength])
        traced = sys_opt.trace_rays(rays)
        err = sum((h['x']**2 + h['y']**2) for r in traced for h in r.history if h['surface_index'] == 8)
        if err < best_err:
            best_err = err
            best_t2 = t
    stack.surfaces[7].thickness = best_t2
    print(f'Best T2 (surf 7): {best_t2:.2f} (err: {best_err:.4f})')
    
    # 3. Optimize T3: AC254-030-C_S3 -> Filter (index 12)
    # The source is Entrance Slit. We trace from start, but just check DMD plane (index 20)
    best_err = 1e9
    best_t3 = stack.surfaces[12].thickness
    for t in np.linspace(0, 15, 31):
        stack.surfaces[12].thickness = t
        stack._update_accumulated_positions()
        rays = get_rays([0,0,0], 0.05, wavelengths)
        traced = sys_opt.trace_rays(rays)
        err = sum((h['x']**2 + h['y']**2) for r in traced for h in r.history if h['surface_index'] == 20)
        if err < best_err:
            best_err = err
            best_t3 = t
    stack.surfaces[12].thickness = best_t3
    print(f'Best T3 (surf 12): {best_t3:.2f} (err: {best_err:.4f})')

    # 4. Optimize T4: AC254-050-C_S3 -> Aperture 3 (index 18)
    best_err = 1e9
    best_t4 = stack.surfaces[18].thickness
    for t in np.linspace(0, 20, 21):
        stack.surfaces[18].thickness = t
        stack._update_accumulated_positions()
        rays = get_rays([0,0,0], 0.05, wavelengths)
        traced = sys_opt.trace_rays(rays)
        err = sum((h['x']**2 + h['y']**2) for r in traced for h in r.history if h['surface_index'] == 20)
        if err < best_err:
            best_err = err
            best_t4 = t
    stack.surfaces[18].thickness = best_t4
    print(f'Best T4 (surf 18): {best_t4:.2f} (err: {best_err:.4f})')

    # 5. Optimize T5: ACL25416U_S2 -> Detector (index 22)
    best_err = 1e9
    best_t5 = stack.surfaces[22].thickness
    for t in np.linspace(5, 20, 31):
        stack.surfaces[22].thickness = t
        stack._update_accumulated_positions()
        rays = get_rays([0,0,0], 0.05, wavelengths)
        traced = sys_opt.trace_rays(rays)
        err = sum((h['x']**2 + h['y']**2) for r in traced for h in r.history if h['surface_index'] == 23)
        if err < best_err:
            best_err = err
            best_t5 = t
    stack.surfaces[22].thickness = best_t5
    print(f'Best T5 (surf 22): {best_t5:.2f} (err: {best_err:.4f})')
    
    # Re-evaluate with best params
    print("Final validation:")
    rays = get_rays([0,0,0], 0.05, wavelengths)
    traced = sys_opt.trace_rays(rays)
    err = sum((h['x']**2 + h['y']**2) for r in traced for h in r.history if h['surface_index'] == 23)
    print(f"Final detector err: {err:.4f}")
    
    # Save 
    orig_stack = SurfaceStack.load_from_json(json_path)['stack']
    for i in range(len(stack.surfaces)):
        stack.surfaces[i].semi_diameter = orig_stack.surfaces[i].semi_diameter
        stack.surfaces[i].aperture_type = orig_stack.surfaces[i].aperture_type
        stack.surfaces[i].aperture_x_half = orig_stack.surfaces[i].aperture_x_half
        stack.surfaces[i].aperture_y_half = orig_stack.surfaces[i].aperture_y_half
        
    stack.save_to_json('spectrometer_raman_50mm_100mm_manual_optimized.json', wavelengths=wavelengths)

if __name__ == '__main__':
    main()
