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
    
    # Target L2 and Condenser surfaces
    # L2: 5, 6, 7
    # Condenser (L3): 9, 10, 11, 12, 13
    large_surfaces = [5, 6, 7, 9, 10, 11, 12, 13]
    
    for idx in large_surfaces:
        s = stack.surfaces[idx]
        s.semi_diameter = 25.4 # 2-inch diameter lens!
        
        # Open up the geometric apertures as well so they don't clip artificially
        if hasattr(s, "aperture_x_half"):
            s.aperture_x_half = 30.0
            s.aperture_y_half = 30.0
            s.aperture_a_semi = 30.0
            s.aperture_b_semi = 30.0
            
    # Set the wide spatial folds to ensure no mechanical housing collisions
    stack.surfaces[4].tilt_x = 45.0 # Grating 90 deg fold
    stack.surfaces[8].tilt_x = 45.0 # DMD 90 deg fold
    
    out_path = "fiber_spectrometer_thorlabs_2inch_optimized.json"
    stack.save_to_json(out_path, wavelengths=data["wavelengths"])
    print(f"Upgraded lenses to 2-inch diameter and saved as {out_path}")
    
if __name__ == "__main__":
    main()
