import os
import sys

sys.path.insert(0, os.path.abspath('.'))
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.optimizer import OpticalOptimizer

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_final_opt.py <json_file>")
        return
        
    json_path = sys.argv[1]
    load_res = SurfaceStack.load_from_json(json_path)
    stack = load_res["stack"]
    wavelengths = load_res.get("wavelengths", [0.785, 0.850, 0.900])
    print(f"Loaded {json_path}. Wavelengths: {wavelengths}")
    
    class DummySystem:
        def __init__(self, s):
            self.stack = s
            self.wavelengths = wavelengths
            self.pupil_radius = s.surfaces[0].semi_diameter
            self.primary_wavelength = wavelengths[0]
            
    system = DummySystem(stack)
    
    opt = OpticalOptimizer(system, wavelengths, num_rays=30)
    opt.method = 'Nelder-Mead'
    opt.max_iter = 500
    
    print("Running aggressive optimization...")
    res = opt.optimize()
    print(f"Optimization finished! Success: {res.success}")
    print(f"Initial Merit: {res.initial_rms:.6f}")
    print(f"Final Merit: {res.final_rms:.6f}")
    
    out_path = json_path.replace(".json", "_final.json")
    res.optimized_stack.save_to_json(out_path, wavelengths=wavelengths)
    print(f"Saved optimized system to {out_path}")

if __name__ == "__main__":
    main()
