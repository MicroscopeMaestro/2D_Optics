import json, math, numpy as np, sys, os
sys.path.insert(0, os.path.abspath('.'))
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem, Ray

def get_rays(start_pos, NA, wavelengths):
    rays = []
    max_theta = np.arcsin(NA)
    for wl in wavelengths:
        rays.append(Ray(pos=np.array(start_pos, dtype=float), dir=np.array([0.0, 0.0, 1.0], dtype=float), wavelength=wl))
        for ring in [1, 2]:
            theta = ring / 2 * max_theta
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
    
    orig_apertures = []
    for s in stack.surfaces:
        orig_apertures.append((s.semi_diameter, s.aperture_type))
        s.semi_diameter = 25.4
        s.aperture_type = 'None'
        
    sys_opt = OpticalSystem(stack)
    
    # STAGE 1: T1 (idx 3) [5 to 25], T2 (idx 7) [80 to 110]
    print('Stage 1...')
    best_e = 1e9; best_t1=12.2; best_t2=91.0
    for t1 in np.linspace(5, 25, 21):
        for t2 in np.linspace(80, 110, 31):
            stack.surfaces[3].thickness = t1
            stack.surfaces[7].thickness = t2
            stack._update_accumulated_positions()
            rays = get_rays([0,0,0], 0.05, [stack.primary_wavelength])
            traced = sys_opt.trace_rays(rays)
            err = 0.0; n = 0
            for r in traced:
                for h in r.history:
                    if h['surface_index'] == 8:
                        err += h['x']**2 + h['y']**2
                        n += 1
            if n > 0:
                e = err/n
                if e < best_e: best_e=e; best_t1=t1; best_t2=t2
    stack.surfaces[3].thickness = best_t1
    stack.surfaces[7].thickness = best_t2
    print(f'Stage 1: T1={best_t1:.2f}, T2={best_t2:.2f}, Err={best_e:.6f}')
    
    # STAGE 2: T3 (idx 12) [0 to 20], T4 (idx 18) [0 to 20]
    print('Stage 2...')
    best_e = 1e9; best_t3=5.0; best_t4=0.0
    for t3 in np.linspace(0, 20, 21):
        for t4 in np.linspace(0, 20, 21):
            stack.surfaces[12].thickness = t3
            stack.surfaces[18].thickness = t4
            stack._update_accumulated_positions()
            rays = get_rays([0,0,0], 0.05, wavelengths)
            traced = sys_opt.trace_rays(rays)
            err = 0.0; n = 0
            for r in traced:
                for h in r.history:
                    if h['surface_index'] == 20:
                        err += h['y']**2 # Y only
                        n += 1
            if n > 0:
                e = err/n
                if e < best_e: best_e=e; best_t3=t3; best_t4=t4
    stack.surfaces[12].thickness = best_t3
    stack.surfaces[18].thickness = best_t4
    print(f'Stage 2: T3={best_t3:.2f}, T4={best_t4:.2f}, Err={best_e:.6f}')
    
    # STAGE 3: T5 (idx 22) [0 to 20]
    print('Stage 3...')
    best_e = 1e9; best_t5=9.0
    for t5 in np.linspace(0, 20, 41):
        stack.surfaces[22].thickness = t5
        stack._update_accumulated_positions()
        rays = get_rays([0,0,0], 0.05, wavelengths)
        traced = sys_opt.trace_rays(rays)
        err = 0.0; n = 0
        for r in traced:
            for h in r.history:
                if h['surface_index'] == 23:
                    err += h['x']**2 + h['y']**2 # spot size
                    n += 1
        if n > 0:
            e = err/n
            if e < best_e: best_e=e; best_t5=t5
    stack.surfaces[22].thickness = best_t5
    print(f'Stage 3: T5={best_t5:.2f}, Err={best_e:.6f}')
    
    # Restore apertures
    for i, s in enumerate(stack.surfaces):
        s.semi_diameter = orig_apertures[i][0]
        s.aperture_type = orig_apertures[i][1]
        
    stack.save_to_json('spectrometer_raman_50mm_100mm_manual_optimized.json', wavelengths=wavelengths)
    
if __name__ == '__main__':
    main()
