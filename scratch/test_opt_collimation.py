import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optics_app.engine.tracer import OpticalSystem, SurfaceStack
from optics_app.engine.optimizer import OpticalOptimizer


import json

def main():
    json_path = os.path.join(os.path.dirname(__file__), '..', 'fiber_spectrometer_portable.json')
    stack_data = None
    with open(json_path, 'r', encoding='utf-8') as f:
        stack_data = json.load(f)
    
    surfaces = []
    from optics_app.engine.surface import Surface
    if isinstance(stack_data, dict) and "surfaces" in stack_data:
        for s_dict in stack_data["surfaces"]:
            s = Surface()
            for k, v in s_dict.items():
                if hasattr(s, k):
                    setattr(s, k, v)
            surfaces.append(s)
    
    stack = SurfaceStack()
    stack.surfaces = surfaces
    system = OpticalSystem(stack)
    
    # 0 = Fiber, 1 = L1 Front, 2 = L1 Back, 3 = Grating
    # We want to optimize distance from Fiber (0) to L1 Front (1)
    # So thickness of Fiber is variable
    stack.surfaces[0].thickness_variable = True
    
    # Target surface: evaluate Collimation at Grating (Surface 3)
    target_surface_index = 3
    
    # Pre-seed the thickness to 19.0 (focal length of AC127-019)
    stack.surfaces[0].thickness = 19.0
    stack._update_accumulated_positions()
    
    opt = OpticalOptimizer(
        system=system,
        wavelengths=[0.5876],
        method="Nelder-Mead",
        num_rays=100,
        entrance_pupil_radius=6.35, # L1 semi-diameter
        max_iter=100,
        target="Angular Spread (Collimation)",
        target_surface_index=target_surface_index
    )
    
    # Print initial angular spread
    rays = opt._generate_rays(6.35, [0.5876], 100, z_dist=19.0)
    traced = system.trace_rays(rays)
    
    dx, dy, dz = [], [], []
    valid = 0
    for r in traced:
        if not r.vignetted and len(r.history) > 0:
            for h in r.history:
                if h["surface_index"] == target_surface_index:
                    if "dir_x" in h:
                        dx.append(h["dir_x"])
                        dy.append(h["dir_y"])
                        dz.append(h["dir_z"])
                    valid += 1
                    break
    
    print(f"Initial valid rays at Grating: {valid}")
    if valid > 0:
        var_dx = np.var(dx)
        var_dy = np.var(dy)
        var_dz = np.var(dz)
        rms = np.sqrt(var_dx + var_dy + var_dz) * 1e4
        print(f"Initial Collimation RMS: {rms:.4f}")
        
    # Run optimization
    variable_indices = {"thickness": [0], "radius": []}
    res = opt.optimize(variable_indices=variable_indices)
    
    print(f"\nOptimization Success: {res.success}")
    print(f"Initial Metric: {res.initial_rms:.4f}")
    print(f"Final Metric: {res.final_rms:.4f}")
    print(f"Optimized Thickness (Fiber to L1): {res.optimized_stack.surfaces[0].thickness:.4f}")

if __name__ == "__main__":
    main()
