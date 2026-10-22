import os
import sys

sys.path.insert(0, os.path.abspath('.'))
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.optimizer import OpticalOptimizer
from optics_app.engine.source import SourceConfig, SourceType

def main():
    json_path = "fiber_spectrometer_portable.json"
    data = SurfaceStack.load_from_json(json_path)
    stack = data["stack"]
    system = OpticalSystem(stack)
    
    source = SourceConfig()
    source.source_type = SourceType.FIBER
    source.beam_radius = 0.0125
    source.divergence_angle = 12.7
    
    var_indices = {
        "thickness": [0, 4, 7, 8, 13], 
        "radius": [], 
        "tilt_x": [14]
    }
    
    # Pre-seed the thicknesses close to their theoretical Thorlabs focal lengths (f=19mm)
    stack.surfaces[0].thickness = 19.0
    stack.surfaces[4].thickness = 25.0 # Free space (Grating to L2)
    stack.surfaces[7].thickness = 19.0
    stack.surfaces[8].thickness = 19.0
    stack.surfaces[13].thickness = 19.0
    
    stack._update_accumulated_positions()
    
    opt = OpticalOptimizer(system, wavelengths=[0.450, 0.546, 0.650], target="Spot Size", source_config=source, max_iter=200)
    opt.method = "Nelder-Mead"
    
    initial_rms = opt.merit_function(
        [stack.surfaces[0].thickness, stack.surfaces[4].thickness, stack.surfaces[7].thickness, stack.surfaces[8].thickness, stack.surfaces[13].thickness, stack.surfaces[14].tilt_x], 
        var_indices
    )
    print(f"Initial RMS of portable layout: {initial_rms}")
    
    print("Starting Nelder-Mead robust focal optimization on micro-spectrometer geometry...")
    
    # Tighter bounds because it's a micro-spectrometer (f=19mm)
    bounds = [
        (10.0, 30.0),  # Fiber to L1
        (15.0, 40.0),  # Grating to L2
        (10.0, 30.0),  # L2 to DMD
        (10.0, 100.0), # DMD to Condenser (Unlock to 100mm for 10x demagnification)
        (1.0, 30.0),   # Condenser to Detector
        (-45.0, 45.0)  # Detector tilt
    ]
    res = opt.optimize(var_indices, bounds=bounds)
    print(f"Final RMS: {res.final_rms}")
    
    out_path = "fiber_spectrometer_portable_optimized.json"
    res.optimized_stack.save_to_json(out_path, wavelengths=data["wavelengths"])
    print(f"Saved optimized portable layout to {out_path}")

if __name__ == "__main__":
    main()
