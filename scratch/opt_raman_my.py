
import json
import math
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath('.'))
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem, Ray
from optics_app.engine.optimizer import OpticalOptimizer
from optics_app.analysis.spot import SpotDiagram

def custom_generate_rays(self, pupil_radius: float, wavelengths: list, num_rays: int):
    # Overriding to simulate 0.22 NA from fiber object
    NA = 0.22
    max_theta = np.arcsin(NA)
    
    rays = []
    n_rings = max(1, int(np.sqrt(num_rays / np.pi)) + 1)
    for wl in wavelengths:
        # Central chief ray
        rays.append(Ray(pos=np.array([0.0, 0.0, 0.0], dtype=float), dir=np.array([0.0, 0.0, 1.0], dtype=float), wavelength=wl))
        for ring in range(1, n_rings + 1):
            r_norm = ring / n_rings
            theta_cone = r_norm * max_theta
            
            n_pts = max(6, int(6 * ring))
            for p in range(n_pts):
                phi = 2.0 * np.pi * p / n_pts
                dx = np.sin(theta_cone) * np.cos(phi)
                dy = np.sin(theta_cone) * np.sin(phi)
                dz = np.cos(theta_cone)
                rays.append(Ray(pos=np.array([0.0, 0.0, 0.0], dtype=float), dir=np.array([dx, dy, dz], dtype=float), wavelength=wl))
    return rays


def main():
    json_path = 'spectrometer_raman_50mm_100mm_manual.json'
    load_res = SurfaceStack.load_from_json(json_path)
    stack = load_res['stack']
    wavelengths = load_res['wavelengths']
    
    stack.primary_wavelength = wavelengths[1] if len(wavelengths) > 1 else wavelengths[0]
    stack._update_accumulated_positions()
    
    # Temporarily remove small apertures to allow gradient descent
    orig_apertures = []
    for i, s in enumerate(stack.surfaces):
        orig_apertures.append((s.semi_diameter, s.aperture_type))
        s.semi_diameter = 25.4
        if 'Slit' in s.name or 'Aperture' in s.name or 'DMD' in s.name:
             s.aperture_type = 'None'
             
    sys_opt = OpticalSystem(stack)
    opt = OpticalOptimizer(sys_opt, wavelengths, max_iter=200, target='Spot Size (Focus)')
    opt._generate_rays = custom_generate_rays.__get__(opt, OpticalOptimizer)
    
    print('Initial parameters:')
    for s in stack.surfaces:
        if getattr(s, 'thickness_variable', False):
            print(f'  {s.name}: {s.thickness:.4f}')
            
    res = opt.optimize()
    print(f'Success: {res.success}')
    print(f'Initial Merit: {res.initial_rms:.6f}')
    print(f'Final Merit: {res.final_rms:.6f}')
    
    # Restore apertures
    for i, s in enumerate(res.optimized_stack.surfaces):
        s.semi_diameter = orig_apertures[i][0]
        s.aperture_type = orig_apertures[i][1]
        
    out_path = 'spectrometer_raman_50mm_100mm_manual_optimized.json'
    res.optimized_stack.save_to_json(out_path, wavelengths=wavelengths)
    
    print('Optimized parameters:')
    for s in res.optimized_stack.surfaces:
        if getattr(s, 'thickness_variable', False):
            print(f'  {s.name}: {s.thickness:.4f}')

if __name__ == '__main__':
    main()
