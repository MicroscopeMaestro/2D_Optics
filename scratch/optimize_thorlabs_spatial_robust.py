import os
import sys

sys.path.insert(0, os.path.abspath('.'))
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.optimizer import OpticalOptimizer
from optics_app.engine.source import SourceConfig, SourceType

def main():
    json_path = "fiber_spectrometer_thorlabs_2inch_optimized.json"
    data = SurfaceStack.load_from_json(json_path)
    stack = data["stack"]
    system = OpticalSystem(stack)
    
    source = SourceConfig()
    source.source_type = SourceType.FIBER
    source.beam_radius = 0.0125
    source.divergence_angle = 12.7
    
    # Set generous folds to prevent physical collisions
    stack.surfaces[4].tilt_x = 45.0 # 90 degree fold
    stack.surfaces[8].tilt_x = 45.0 # 90 degree fold
    
    var_indices = {
        "thickness": [4, 7, 8, 13], 
        "radius": [], 
        "tilt_x": [14]
    }
    
    stack._update_accumulated_positions()
    
    # Use Nelder-Mead, which is immune to flat gradient stalling!
    opt = OpticalOptimizer(system, wavelengths=[0.450, 0.546, 0.650], target="Spot Size", source_config=source, max_iter=100)
    opt.method = "Nelder-Mead"
    
    initial_rms = opt.merit_function(
        [stack.surfaces[4].thickness, stack.surfaces[7].thickness, stack.surfaces[8].thickness, stack.surfaces[13].thickness, stack.surfaces[14].tilt_x], 
        var_indices
    )
    print(f"Initial RMS with structural folds: {initial_rms}")
    
    print("Starting Nelder-Mead robust focal optimization...")
    # Bound the thicknesses to >60 so the 2-inch (50.8mm) lenses clear each other!
    bounds = [
        (60.0, 120.0), # Grating to L2
        (60.0, 120.0), # L2 to DMD
        (60.0, 120.0), # DMD to Condenser
        (15.0, 120.0), # Condenser to Detector
        (-45.0, 45.0)  # Detector tilt
    ]
    res = opt.optimize(var_indices, bounds=bounds)
    print(f"Final RMS: {res.final_rms}")
    
    out_path = "fiber_spectrometer_thorlabs_2inch_spatial_optimized.json"
    res.optimized_stack.save_to_json(out_path, wavelengths=data["wavelengths"])
    print(f"Saved optimized spatial layout to {out_path}")

if __name__ == "__main__":
    main()
