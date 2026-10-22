import json
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.abspath('.'))

from optics_app.engine.lde import SurfaceStack
from optics_app.engine.optimizer import OpticalOptimizer

def main():
    json_path = "fiber_spectrometer_thorlabs_full.json"
    load_res = SurfaceStack.load_from_json(json_path)
    stack = load_res["stack"]
    wavelengths = load_res["wavelengths"]
    print(f"Loaded stack with {len(stack.surfaces)} surfaces. Wavelengths: {wavelengths}")
    
    # Check variable variables
    vars = []
    for s in stack.surfaces:
        if getattr(s, 'thickness_variable', False):
            vars.append(s.name)
    print(f"Variables: {vars}")
    
    class DummySystem:
        def __init__(self, s):
            self.stack = s
            self.wavelengths = wavelengths
            self.pupil_radius = s.surfaces[0].semi_diameter
            self.primary_wavelength = wavelengths[0]
            
    system = DummySystem(stack)
    try:
        opt = OpticalOptimizer(system, wavelengths, num_rays=15)
        opt.max_iter = 50
        print("Running optimization...")
        res = opt.optimize()
        print(f"Optimization finished! Success: {res.success}")
        print(f"Initial Merit: {res.initial_rms:.6f}")
        print(f"Final Merit: {res.final_rms:.6f}")
        
        optimized_path = "fiber_spectrometer_thorlabs_optimized.json"
        res.optimized_stack.save_to_json(optimized_path, wavelengths=wavelengths)
        print(f"Saved optimized system to {optimized_path}")
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
