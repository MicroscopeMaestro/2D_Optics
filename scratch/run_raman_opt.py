import json
import os
import sys

# Add path so we can import optics_app
sys.path.insert(0, os.path.abspath('.'))

from optics_app.engine.lde import SurfaceStack
from optics_app.engine.optimizer import OpticalOptimizer

def main():
    json_path = "fiber_spectrometer_thorlabs_full.json"
    
    # Load the system
    load_res = SurfaceStack.load_from_json(json_path)
    stack = load_res["stack"]
    wavelengths = load_res["wavelengths"]
    
    print(f"Loaded stack with {len(stack.surfaces)} surfaces. Wavelengths: {wavelengths}")
    
    # Initialize Optimizer
    # We want to optimize the thickness of the variable surfaces to minimize the spot size on the detector
    opt = OpticalOptimizer(stack, wavelengths)
    
    # Run optimization using scipy minimize
    print("Running optimization...")
    res = opt.optimize()
    print(f"Optimization finished! Success: {res['success']}")
    print(f"Initial Merit: {res['initial_merit']:.6f}")
    print(f"Final Merit: {res['final_merit']:.6f}")
    print(f"Iterations: {res['iterations']}")
    
    # Save the optimized system
    optimized_path = "fiber_spectrometer_thorlabs_optimized.json"
    stack.save_to_json(optimized_path, wavelengths=wavelengths)
    print(f"Saved optimized system to {optimized_path}")
    
    # Let's print out the new thicknesses
    print("\nNew Thicknesses:")
    for s in stack.surfaces:
        if getattr(s, 'thickness_variable', False):
            print(f"  {s.name}: {s.thickness:.4f} mm")

if __name__ == "__main__":
    main()
