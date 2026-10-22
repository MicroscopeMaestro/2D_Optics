import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath('.'))
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.optimizer import OpticalOptimizer
from optics_app.engine.source import SourceConfig, SourceType

def main():
    json_path = "fiber_spectrometer_thorlabs_optimized.json"
    print(f"Loading {json_path}...")
    
    data = SurfaceStack.load_from_json(json_path)
    stack = data["stack"]
    system = OpticalSystem(stack)
    
    # Configure the fiber source (NA ~ 0.22 -> 12.7 deg half angle)
    source = SourceConfig()
    source.source_type = SourceType.FIBER
    source.beam_radius = 0.0125
    source.divergence_angle = 12.7
    
    # Freeing variables:
    # 0: Fiber Object (thickness)
    # 3: L1 Back (thickness) - space before grating
    # 4: Diffraction_Grating (thickness) - space after grating
    # 7: L2 Back (thickness) - space to detector
    
    var_indices = {
        "thickness": [0, 3, 4, 7],
        "radius": [],
        "tilt_x": [4, 8]  # Grating and Detector tilt
    }
    
    # Current values
    current_x = []
    for idx in var_indices["thickness"]:
        current_x.append(stack.surfaces[idx].thickness)
    for idx in var_indices["tilt_x"]:
        current_x.append(stack.surfaces[idx].tilt_x)
        
    print(f"Initial variables: {current_x}")
    
    # Set bounds
    bounds = []
    for idx in var_indices["thickness"]:
        bounds.append((5.0, 150.0))
    for idx in var_indices["tilt_x"]:
        if idx == 4: # Grating tilt
            bounds.append((25.0, 80.0)) # Force grating to fold at least 50 degrees! (25 x 2)
        else:
            bounds.append((-90.0, 90.0))
            
    # Modify initial guess for grating to be at least 25 degrees so Nelder-Mead starts safely
    current_x[4] = 30.0 # Force grating initial tilt_x to 30.0 deg (60 deg fold)
    stack.surfaces[4].tilt_x = 30.0
    
    # Push the lenses apart initially so we don't start with a massive penalty
    current_x[1] = 40.0 # L1 Back thickness
    current_x[2] = 40.0 # Grating thickness
    
    stack.surfaces[3].thickness = 40.0
    stack.surfaces[4].thickness = 40.0
    stack._update_accumulated_positions()
    
    opt = OpticalOptimizer(system, wavelengths=[0.450, 0.546, 0.650], target="Spot Size", source_config=source, max_iter=250)
    opt.method = "Nelder-Mead"
    
    print("Starting optimization with spatial constraints enabled...")
    initial_rms = opt.merit_function(current_x, var_indices)
    print(f"Manual initial RMS: {initial_rms}")
    
    res = opt.optimize(var_indices, bounds=bounds)
    
    print(f"Optimization finished in {res.iterations} iterations.")
    print(f"Final Merit Score: {res.final_rms}")
    
    print("\nFinal Parameters:")
    for key, indices in var_indices.items():
        for idx in indices:
            val = getattr(res.optimized_stack.surfaces[idx], key)
            print(f"Surface {idx} ({res.optimized_stack.surfaces[idx].name}) {key} = {val:.4f}")
            
    out_path = "fiber_spectrometer_thorlabs_spatial_optimized.json"
    res.optimized_stack.save_to_json(out_path, wavelengths=data["wavelengths"])
    print(f"\nSaved structurally sound design to {out_path}")

if __name__ == "__main__":
    main()
