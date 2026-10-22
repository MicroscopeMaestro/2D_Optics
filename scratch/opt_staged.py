import json, math, numpy as np, sys, os
from scipy.optimize import minimize_scalar, minimize
sys.path.insert(0, os.path.abspath('.'))
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem, Ray

def get_rays(start_pos, NA, wavelengths):
    rays = []
    max_theta = np.arcsin(NA)
    for wl in wavelengths:
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
    primary_wl = wavelengths[1] if len(wavelengths) > 1 else wavelengths[0]
    stack.primary_wavelength = primary_wl
    
    orig_apertures = []
    for s in stack.surfaces:
        orig_apertures.append((s.semi_diameter, s.aperture_type))
        s.semi_diameter = 1000.0
        s.aperture_type = 'None'
        
    sys_opt = OpticalSystem(stack)
    
    # STAGE 1: T0 (idx 0) and T2 (idx 7) to focus on Entrance Slit (idx 8)
    def obj_stage1(x):
        stack.surfaces[0].thickness = max(0.1, x[0])
        stack.surfaces[7].thickness = max(0.1, x[1])
        stack._update_accumulated_positions()
        rays = get_rays([0,0,0], 0.05, [primary_wl])
        traced = sys_opt.trace_rays(rays)
        err = 0.0; n = 0
        for r in traced:
            for h in r.history:
                if h['surface_index'] == 8:
                    err += h['x']**2 + h['y']**2
                    n += 1
        return (err / max(n, 1)) + (0 if n > 0 else 1000.0)

    res1 = minimize(obj_stage1, [22.0, 91.0], method='Nelder-Mead')
    stack.surfaces[0].thickness = max(0.1, res1.x[0])
    stack.surfaces[7].thickness = max(0.1, res1.x[1])
    
    # STAGE 2: S9 (idx 9) and T4 (idx 18) to focus on DMD (idx 20)
    # T3 (idx 12) is fixed to 5.0
    stack.surfaces[12].thickness = 5.0
    def obj_stage2(x):
        stack.surfaces[9].thickness = max(0.1, x[0])
        stack.surfaces[18].thickness = max(0.1, x[1])
        stack._update_accumulated_positions()
        rays = get_rays([0,0,0], 0.05, wavelengths)
        traced = sys_opt.trace_rays(rays)
        total_var = 0.0; n = 0; vig = 0
        for wl in wavelengths:
            pts_x, pts_y = [], []
            for r in traced:
                if abs(r.wavelength - wl) < 1e-5:
                    if r.vignetted: vig += 1
                    else:
                        for h in r.history:
                            if h['surface_index'] == 20:
                                pts_x.append(h['x'])
                                pts_y.append(h['y'])
            if pts_x:
                total_var += np.var(pts_x) + np.var(pts_y)
                n += 1
        return (total_var / max(n, 1)) + (vig * 1000.0)

    res2 = minimize(obj_stage2, [43.0, 43.0], method='Nelder-Mead')
    stack.surfaces[9].thickness = max(0.1, res2.x[0])
    stack.surfaces[18].thickness = max(0.1, res2.x[1])
    
    # STAGE 3: T5 (idx 22) to focus on Detector (idx 23)
    def obj_stage3(x):
        stack.surfaces[22].thickness = max(0.1, x[0])
        stack._update_accumulated_positions()
        rays = get_rays([0,0,0], 0.05, wavelengths)
        traced = sys_opt.trace_rays(rays)
        err = 0.0; n = 0; vig = 0
        for r in traced:
            if r.vignetted: vig += 1
            else:
                for h in r.history:
                    if h['surface_index'] == 23:
                        err += h['x']**2 + h['y']**2
                        n += 1
        return (err / max(n, 1)) + (vig * 1000.0)

    res3 = minimize(obj_stage3, [8.0], method='Nelder-Mead')
    stack.surfaces[22].thickness = max(0.1, res3.x[0])
    
    # Restore apertures
    for i in range(len(stack.surfaces)):
        stack.surfaces[i].semi_diameter = orig_apertures[i][0]
        stack.surfaces[i].aperture_type = orig_apertures[i][1]
        
    stack.save_to_json('C:/Users/Juan/Desktop/spectrometer_raman_50mm_100mm_manual_optimized.json', wavelengths=wavelengths)
    print('Done.')

if __name__ == '__main__':
    main()
