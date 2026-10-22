
import json
import numpy as np
from scipy.optimize import minimize
import sys
import os
sys.path.insert(0, os.path.abspath('.'))
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem, Ray

def get_rays(start_pos, NA, wavelengths, num_rings=3):
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
    
    orig_apertures = []
    for s in stack.surfaces:
        orig_apertures.append((s.semi_diameter, s.aperture_type))
        s.semi_diameter = 25.4
        s.aperture_type = 'None'
        
    sys_opt = OpticalSystem(stack)
    
    def obj1(x):
        stack.surfaces[3].thickness = x[0]
        stack.surfaces[7].thickness = x[1]
        stack._update_accumulated_positions()
        rays = get_rays([0,0,0], 0.1, [stack.primary_wavelength])
        traced = sys_opt.trace_rays(rays)
        
        err = 0.0
        n = 0
        for r in traced:
            for h in r.history:
                if h['surface_index'] == 8:
                    err += h['x']**2 + h['y']**2
                    n += 1
        return err / max(n, 1)

    print('Optimizing Stage 1 (Sample to Slit)...')
    res1 = minimize(obj1, [12.2, 91.0], method='Nelder-Mead')
    print('Stage 1:', res1.x)
    stack.surfaces[3].thickness = res1.x[0]
    stack.surfaces[7].thickness = res1.x[1]
    
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
                    err += h['x']**2 + h['y']**2
                    n += 1
        return err / max(n, 1)

    print('Optimizing Stage 2 (Slit to DMD)...')
    res2 = minimize(obj2, [5.0, 0.0], method='Nelder-Mead')
    print('Stage 2:', res2.x)
    stack.surfaces[12].thickness = res2.x[0]
    stack.surfaces[18].thickness = res2.x[1]
    
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
                    err += h['x']**2 + h['y']**2
                    n += 1
        return err / max(n, 1)

    print('Optimizing Stage 3 (DMD to Detector)...')
    res3 = minimize(obj3, [9.0], method='Nelder-Mead')
    print('Stage 3:', res3.x)
    stack.surfaces[22].thickness = res3.x[0]
    
    for i, s in enumerate(stack.surfaces):
        s.semi_diameter = orig_apertures[i][0]
        s.aperture_type = orig_apertures[i][1]
        
    stack.save_to_json('spectrometer_raman_50mm_100mm_manual_optimized.json', wavelengths=wavelengths)
    print('Done!')

if __name__ == '__main__':
    main()
