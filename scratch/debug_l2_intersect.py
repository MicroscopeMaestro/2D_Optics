import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import numpy as np
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem, Ray

def debug_trace():
    with open('fiber_spectrometer_portable.json', 'r') as f:
        data = json.load(f)
    
    surfaces = []
    for s_dict in data["surfaces"]:
        s = Surface()
        for k, v in s_dict.items():
            if hasattr(s, k):
                setattr(s, k, v)
        surfaces.append(s)
        
    stack = SurfaceStack()
    stack.surfaces = surfaces
    stack.primary_wavelength = 0.85
    stack._update_accumulated_positions()
    
    system = OpticalSystem(stack)
    
    # Trace a single on-axis ray and a marginal ray
    rays = [
        Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.85),
        Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.1, 0.994987]), wavelength=0.85),
        Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, -0.1, 0.994987]), wavelength=0.85),
    ]
    
    for idx, r in enumerate(rays):
        print(f"\n--- Tracing Ray {idx} ---")
        traced = system.trace_ray(r)
        print(f"Ray {idx} vignetted: {traced.vignetted}, TIR: {traced.tir}")
        for h in traced.history:
            s_name = stack.surfaces[h['surface_index']].name if h['surface_index'] < len(stack.surfaces) else 'unknown'
            print(f"  Surf {h['surface_index']} ({s_name}): hit_local=({h['x']:.3f}, {h['y']:.3f}, {h['z_local']:.3f}), hit_global=({h['x_global']:.3f}, {h['y_global']:.3f}, {h['z_global']:.3f})")

    # Now let's test OpticalOptimizer on this system
    print("\n=== Testing OpticalOptimizer ===")
    from optics_app.engine.optimizer import OpticalOptimizer
    opt = OpticalOptimizer(
        system=system,
        wavelengths=[0.785, 0.85, 0.9],
        method="Nelder-Mead",
        max_iter=50,
        target="RMS Spot Radius",
        num_rays=30
    )
    res = opt.optimize()
    print("Optimization finished:", res.success)
    print("Initial metric:", res.initial_metric, "Final metric:", res.final_metric)

if __name__ == "__main__":
    debug_trace()
