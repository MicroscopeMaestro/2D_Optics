import os
import sys

sys.path.insert(0, os.path.abspath('.'))
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.optimizer import OpticalOptimizer
from optics_app.engine.source import SourceConfig, SourceType

def main():
    json_path = "fiber_spectrometer_thorlabs_compact.json"
    data = SurfaceStack.load_from_json(json_path)
    stack = data["stack"]
    system = OpticalSystem(stack)
    
    source = SourceConfig()
    source.source_type = SourceType.FIBER
    source.beam_radius = 0.0125
    source.divergence_angle = 12.7
    
    # We only optimize the exact focal spacing.
    # The AC127-025-A (L1) has f=25mm, so Fiber to L1 should be optimized to ~25mm.
    # L2 is AC127-050-A (f=50mm), so Grating to L2 is free space, L2 to DMD should be ~50mm.
    # Condenser is AC127-025-A relay, so DMD to Condenser ~25mm, Condenser to Detector ~25mm.
    
    var_indices = {
        "thickness": [0, 4, 7, 8, 13], 
        "radius": [], 
        "tilt_x": [14]
    }
    
    # Pre-seed the thicknesses close to their theoretical Thorlabs focal lengths to help Nelder-Mead
    stack.surfaces[0].thickness = 25.0
    stack.surfaces[4].thickness = 30.0 # Free space (Grating to L2)
    stack.surfaces[7].thickness = 50.0
    stack.surfaces[8].thickness = 25.0
    stack.surfaces[13].thickness = 25.0
    
    stack._update_accumulated_positions()
    
    opt = OpticalOptimizer(system, wavelengths=[0.450, 0.546, 0.650], target="Spot Size", source_config=source, max_iter=200)
    opt.method = "Nelder-Mead"
    
    initial_rms = opt.merit_function(
        [stack.surfaces[0].thickness, stack.surfaces[4].thickness, stack.surfaces[7].thickness, stack.surfaces[8].thickness, stack.surfaces[13].thickness, stack.surfaces[14].tilt_x], 
        var_indices
    )
    print(f"Initial RMS of Thorlabs AC127 layout: {initial_rms}")
    
    print("Starting Nelder-Mead robust focal optimization on compact geometry...")
    
    # Set tighter bounds that still ensure the small lenses clear each other
    bounds = [
        (15.0, 40.0),  # Fiber to L1
        (20.0, 60.0),  # Grating to L2
        (30.0, 70.0),  # L2 to DMD
        (20.0, 60.0),  # DMD to Condenser
        (15.0, 40.0),  # Condenser to Detector
        (-45.0, 45.0)  # Detector tilt
    ]
    res = opt.optimize(var_indices, bounds=bounds)
    print(f"Final RMS: {res.final_rms}")
    
    out_path = "fiber_spectrometer_thorlabs_compact_optimized.json"
    res.optimized_stack.save_to_json(out_path, wavelengths=data["wavelengths"])
    print(f"Saved optimized compact layout to {out_path}")

if __name__ == "__main__":
    main()
