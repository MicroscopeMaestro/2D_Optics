import json, math, numpy as np, sys, os
from scipy.optimize import minimize
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
    
    # Remove apertures
    orig_apertures = []
    for s in stack.surfaces:
        orig_apertures.append((s.semi_diameter, s.aperture_type))
        s.semi_diameter = 25.4
        s.aperture_type = 'None'
        
    sys_opt = OpticalSystem(stack)
    
    # STAGE 1: Focus on Entrance Slit (surf 8)
    # Vars: T1 (idx 3), T2 (idx 7)
    def obj1(x):
        stack.surfaces[3].thickness = x[0]
        stack.surfaces[7].thickness = x[1]
        stack._update_accumulated_positions()
        rays = get_rays([0,0,0], 0.05, [stack.primary_wavelength])
        traced = sys_opt.trace_rays(rays)
        err = 0.0
        n = 0
        for r in traced:
            for h in r.history:
                if h['surface_index'] == 8:
                    err += h['x']**2 + h['y']**2
                    n += 1
        return (err / max(n, 1)) + (0 if n > 0 else 1000)

    res1 = minimize(obj1, [12.2, 91.0], method='Nelder-Mead')
    stack.surfaces[3].thickness = res1.x[0]
    stack.surfaces[7].thickness = res1.x[1]
    print(f'Stage 1 (Slit Focus): T1={res1.x[0]:.2f}, T2={res1.x[1]:.2f}, Error={res1.fun:.6f}')
    
    # STAGE 2: Focus on DMD (surf 20)
    # Vars: T3 (idx 12), T4 (idx 18)
    def obj2(x):
        stack.surfaces[12].thickness = x[0]
        stack.surfaces[18].thickness = x[1]
        stack._update_accumulated_positions()
        rays = get_rays([0,0,0], 0.05, wavelengths)
        traced = sys_opt.trace_rays(rays)
        err = 0.0
        n = 0
        for r in traced:
            for h in r.history:
                if h['surface_index'] == 20:
                    err += h['y']**2 # Only optimize Y for DMD focus? Or X and Y? Wait, grating spreads in X!
                    # If grating spreads in X, we can't minimize X for all wavelengths!
                    # So we ONLY minimize Y for focus!
                    err += h['y']**2
                    n += 1
        return (err / max(n, 1)) + (0 if n > 0 else 1000)

    res2 = minimize(obj2, [5.0, 0.0], method='Nelder-Mead')
    stack.surfaces[12].thickness = res2.x[0]
    stack.surfaces[18].thickness = res2.x[1]
    print(f'Stage 2 (DMD Focus): T3={res2.x[0]:.2f}, T4={res2.x[1]:.2f}, Error={res2.fun:.6f}')
    
    # STAGE 3: Focus on Detector (surf 23)
    # Vars: T5 (idx 22)
    def obj3(x):
        stack.surfaces[22].thickness = x[0]
        stack._update_accumulated_positions()
        rays = get_rays([0,0,0], 0.05, wavelengths)
        traced = sys_opt.trace_rays(rays)
        err = 0.0
        n = 0
        for r in traced:
            for h in r.history:
                if h['surface_index'] == 23:
                    # Detector should collect all light to a single spot (if possible) or at least minimize overall spot
                    err += h['x']**2 + h['y']**2
                    n += 1
        return (err / max(n, 1)) + (0 if n > 0 else 1000)

    res3 = minimize(obj3, [9.0], method='Nelder-Mead')
    stack.surfaces[22].thickness = res3.x[0]
    print(f'Stage 3 (Detector Focus): T5={res3.x[0]:.2f}, Error={res3.fun:.6f}')
    
    # Restore apertures
    for i, s in enumerate(stack.surfaces):
        s.semi_diameter = orig_apertures[i][0]
        s.aperture_type = orig_apertures[i][1]
        
    stack.save_to_json('spectrometer_raman_50mm_100mm_manual_optimized.json', wavelengths=wavelengths)

if __name__ == '__main__':
    main()
