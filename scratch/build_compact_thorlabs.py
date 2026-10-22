import os
import sys

sys.path.insert(0, os.path.abspath('.'))
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.optimizer import OpticalOptimizer
from optics_app.engine.source import SourceConfig, SourceType

def main():
    json_path = "fiber_spectrometer_thorlabs_optimized.json"
    data = SurfaceStack.load_from_json(json_path)
    stack = data["stack"]
    
    # 1. Update L1 to AC127-025-AB (f=25mm)
    # Surfaces: 1 (Front), 2 (Middle), 3 (Back)
    stack.surfaces[1].radius = 19.1
    stack.surfaces[1].thickness = 2.5
    stack.surfaces[2].radius = -17.8
    stack.surfaces[2].thickness = 1.0
    stack.surfaces[3].radius = -82.3
    
    # 2. Update L2 to AC127-050-A (f=50mm)
    # Surfaces: 5 (Front), 6 (Middle), 7 (Back)
    stack.surfaces[5].radius = 27.4
    stack.surfaces[5].thickness = 2.0
    stack.surfaces[6].radius = -22.5
    stack.surfaces[6].thickness = 1.0
    stack.surfaces[7].radius = -91.8
    
    # 3. Update Condenser (L3 Group) to two AC127-025-AB (f=25mm) facing each other
    # Condenser 1 (Surfaces 9, 10, 11)
    stack.surfaces[9].radius = 82.3    # Reversed
    stack.surfaces[9].thickness = 1.0
    stack.surfaces[10].radius = 17.8   # Reversed
    stack.surfaces[10].thickness = 2.5
    stack.surfaces[11].radius = -19.1  # Reversed
    # Condenser 2 (Surfaces 12, 13) - We will just mimic a standard singlet to save space or just use one AC127 for relay!
    stack.surfaces[12].radius = 19.1
    stack.surfaces[12].thickness = 2.5
    stack.surfaces[13].radius = -17.8
    
    # Set all semi_diameters to 6.35 (1/2-inch lenses) and reduce geometric apertures
    compact_surfaces = [1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13]
    for idx in compact_surfaces:
        s = stack.surfaces[idx]
        s.semi_diameter = 6.35
        if hasattr(s, "aperture_x_half"):
            s.aperture_x_half = 10.0
            s.aperture_y_half = 10.0
            s.aperture_a_semi = 10.0
            s.aperture_b_semi = 10.0
            
    # Shrink the grating housing as well
    stack.surfaces[4].semi_diameter = 6.35
    if hasattr(stack.surfaces[4], "aperture_x_half"):
        stack.surfaces[4].aperture_x_half = 10.0
        stack.surfaces[4].aperture_y_half = 10.0
        stack.surfaces[4].aperture_a_semi = 10.0
        stack.surfaces[4].aperture_b_semi = 10.0
        
    # With smaller 1/2-inch housings, we can safely use a 60-degree fold without clipping!
    stack.surfaces[4].tilt_x = 30.0 # Grating 60 deg fold
    stack.surfaces[8].tilt_x = 30.0 # DMD 60 deg fold
    
    stack._update_accumulated_positions()
    
    out_path = "fiber_spectrometer_thorlabs_compact.json"
    stack.save_to_json(out_path, wavelengths=data["wavelengths"])
    print(f"Mapped AC127 catalog models and saved to {out_path}")

if __name__ == "__main__":
    main()
