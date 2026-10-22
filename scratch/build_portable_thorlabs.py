import os
import sys

sys.path.insert(0, os.path.abspath('.'))
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem

def main():
    json_path = "fiber_spectrometer_thorlabs_compact.json"
    data = SurfaceStack.load_from_json(json_path)
    stack = data["stack"]
    
    # AC127-019-A
    # R1: 12.9, R2: -11.0, R3: -59.3
    
    # 1. Update L1 to AC127-019-A (f=19mm)
    stack.surfaces[1].radius = 12.9
    stack.surfaces[1].thickness = 3.2
    stack.surfaces[2].radius = -11.0
    stack.surfaces[2].thickness = 1.5
    stack.surfaces[3].radius = -59.3
    
    # 2. Update L2 to AC127-019-A (f=19mm)
    stack.surfaces[5].radius = 12.9
    stack.surfaces[5].thickness = 3.2
    stack.surfaces[6].radius = -11.0
    stack.surfaces[6].thickness = 1.5
    stack.surfaces[7].radius = -59.3
    
    # 3. Update Condenser (L3 Group) to ACL12708U Aspheric Condenser
    # We will use surface 9 for the convex front (R=4.16 for f=8mm N-BK7 equiv),
    # surface 10 for the flat back, and nullify 11, 12, 13.
    stack.surfaces[9].radius = 4.16
    stack.surfaces[9].thickness = 4.0
    stack.surfaces[9].glass = "N-BK7"
    
    stack.surfaces[10].radius = 0.0
    stack.surfaces[10].thickness = 0.0
    stack.surfaces[10].glass = "AIR"
    
    stack.surfaces[11].radius = 0.0
    stack.surfaces[11].thickness = 0.0
    stack.surfaces[11].glass = "AIR"
    
    stack.surfaces[12].radius = 0.0
    stack.surfaces[12].thickness = 0.0
    stack.surfaces[12].glass = "AIR"
    
    stack.surfaces[13].radius = 0.0
    stack.surfaces[13].thickness = 0.0
    stack.surfaces[13].glass = "AIR"
    
    # Halve the DMD Area
    stack.surfaces[8].aperture_x_half = 5.0
    stack.surfaces[8].aperture_y_half = 5.0
    stack.surfaces[8].aperture_a_semi = 5.0
    stack.surfaces[8].aperture_b_semi = 5.0
    
    # Restrict Detector Area to 1mm^2 (Rectangular 1x1mm)
    stack.surfaces[14].aperture_type = "Rectangular"
    stack.surfaces[14].aperture_x_half = 0.5
    stack.surfaces[14].aperture_y_half = 0.5
    stack.surfaces[14].semi_diameter = 0.5
    
    stack._update_accumulated_positions()
    
    out_path = "fiber_spectrometer_portable.json"
    stack.save_to_json(out_path, wavelengths=data["wavelengths"])
    print(f"Mapped AC127-019-A portable models and saved to {out_path}")

if __name__ == "__main__":
    main()
