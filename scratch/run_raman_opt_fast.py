import json
import os
import sys
import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.abspath('.'))

from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem, Ray
from optics_app.engine.optimizer import OpticalOptimizer

def custom_generate_rays(self, pupil_radius: float, wavelengths: list, num_rays: int):
    NA = 0.22
    max_theta = np.arcsin(NA)
    
    rays = []
    # Force exactly 2 rings (19 rays total per wavelength)
    n_rings = 2
    for wl in wavelengths:
        # Central chief ray
        rays.append(Ray(pos=np.array([0.0, 0.0, 0.0], dtype=float), dir=np.array([0.0, 0.0, 1.0], dtype=float), wavelength=wl))
        for ring in range(1, n_rings + 1):
            r_norm = ring / n_rings
            theta_cone = r_norm * max_theta
            
            n_pts = 6 * ring
            for p in range(n_pts):
                phi = 2.0 * np.pi * p / n_pts
                
                dx = np.sin(theta_cone) * np.cos(phi)
                dy = np.sin(theta_cone) * np.sin(phi)
                dz = np.cos(theta_cone)
                
                rays.append(Ray(pos=np.array([0.0, 0.0, 0.0], dtype=float), dir=np.array([dx, dy, dz], dtype=float), wavelength=wl))
    return rays

def main():
    json_path = "fiber_spectrometer_thorlabs_full.json"
    load_res = SurfaceStack.load_from_json(json_path)
    stack = load_res["stack"]
    wavelengths = load_res["wavelengths"]
    
    # CRITICAL: set primary_wavelength so lde.py folds the coordinate system correctly for the laser!
    stack.primary_wavelength = wavelengths[0]
    stack._update_accumulated_positions()
    
    sys_opt = OpticalSystem(stack)
    
    # Increase the semi-diameters slightly to avoid vignetting during optimization search
    for i in range(1, len(stack.surfaces)):
        stack.surfaces[i].semi_diameter = 25.4
    
    # ONLY OPTIMIZE FOR THE CENTRAL WAVELENGTH TO SPEED IT UP
    fast_wavelengths = [wavelengths[0]]
    opt = OpticalOptimizer(sys_opt, fast_wavelengths, max_iter=100, target="Spot Size (Focus)")
    
    # Override ray generation
    opt._generate_rays = custom_generate_rays.__get__(opt, OpticalOptimizer)
    
    print("Running fast optimization with custom point-source diverging rays (1 wavelength, 19 rays)...")
    res = opt.optimize()
    print(f"Optimization finished! Success: {res.success}")
    print(f"Initial Merit (RMS): {res.initial_rms:.6f}")
    print(f"Final Merit (RMS): {res.final_rms:.6f}")
    print(f"Iterations: {res.iterations}")
    
    optimized_path = "fiber_spectrometer_thorlabs_optimized.json"
    # Restore realistic semi-diameters
    for i in range(1, len(res.optimized_stack.surfaces)):
        if "Condenser" in res.optimized_stack.surfaces[i].name:
            res.optimized_stack.surfaces[i].semi_diameter = 12.5
        elif "L" in res.optimized_stack.surfaces[i].name:
            res.optimized_stack.surfaces[i].semi_diameter = 12.7
            
    res.optimized_stack.save_to_json(optimized_path, wavelengths=wavelengths)  # save with all 3 wavelengths
    print(f"Saved optimized system to {optimized_path}")
    
    print("\nNew Thicknesses:")
    for s in res.optimized_stack.surfaces:
        if getattr(s, 'thickness_variable', False):
            print(f"  {s.name}: {s.thickness:.4f} mm")

if __name__ == "__main__":
    main()
