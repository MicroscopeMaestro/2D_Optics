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
    
    # We will ONLY trace the primary wavelength (1.225) to avoid the other wavelengths missing the lenses!
    wl = 1.225 
    wavelengths = [wl]
    stack.primary_wavelength = wl
    
    orig_apertures = []
    for s in stack.surfaces:
        orig_apertures.append((s.semi_diameter, s.aperture_type))
        s.semi_diameter = 25.4
        s.aperture_type = 'None'
        
    sys_opt = OpticalSystem(stack)
    
    var_indices = [3, 7, 12, 18, 22]
    x0 = [stack.surfaces[i].thickness for i in var_indices]
    
    def obj(x):
        for i, idx in enumerate(var_indices):
            stack.surfaces[idx].thickness = x[i]
        stack._update_accumulated_positions()
        
        rays = get_rays([0,0,0], 0.1, wavelengths)
        traced = sys_opt.trace_rays(rays)
        
        err = 0.0
        n = 0
        for r in traced:
            for h in r.history:
                if h['surface_index'] == 23:
                    err += h['x']**2 + h['y']**2
                    n += 1
        return (err / max(n, 1)) + (0 if n > 0 else 1000)

    print(f'Initial error: {obj(x0):.6f}')
    
    res = minimize(obj, x0, method='Nelder-Mead', options={'maxiter': 1000, 'disp': True})
    print(f'Final error: {res.fun:.6f}')
    print(f'Optimized thicknesses: {res.x}')
    
    for i, idx in enumerate(var_indices):
        stack.surfaces[idx].thickness = res.x[i]
        
    # Check aberrations/spot size
    rays = get_rays([0,0,0], 0.1, wavelengths)
    traced = sys_opt.trace_rays(rays)
    det_x, det_y = [], []
    for r in traced:
        for h in r.history:
            if h['surface_index'] == 23:
                det_x.append(h['x'])
                det_y.append(h['y'])
    if det_x:
        rms = np.sqrt(np.mean(np.array(det_x)**2 + np.array(det_y)**2))
        print(f'RMS spot radius on detector (1.225um): {rms:.4f} mm')
        
    # Save
    orig_stack = SurfaceStack.load_from_json(json_path)['stack']
    for i in range(len(stack.surfaces)):
        stack.surfaces[i].semi_diameter = orig_stack.surfaces[i].semi_diameter
        stack.surfaces[i].aperture_type = orig_stack.surfaces[i].aperture_type
        stack.surfaces[i].aperture_x_half = orig_stack.surfaces[i].aperture_x_half
        stack.surfaces[i].aperture_y_half = orig_stack.surfaces[i].aperture_y_half
        
    stack.save_to_json('spectrometer_raman_50mm_100mm_manual_optimized.json', wavelengths=load_res['wavelengths'])
    
if __name__ == '__main__':
    main()
