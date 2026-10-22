import os
import sys

sys.path.insert(0, os.path.abspath('.'))
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.cad_export import export_system_to_step

def main():
    json_path = "fiber_spectrometer_portable_optimized.json"
    data = SurfaceStack.load_from_json(json_path)
    
    out_path = "fiber_spectrometer_portable.step"
    export_system_to_step(data["stack"], out_path)
    print(f"Exported CAD model to {out_path}")

if __name__ == "__main__":
    main()
