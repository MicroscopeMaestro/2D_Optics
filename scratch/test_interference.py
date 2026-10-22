import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath('.'))
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.tracer import OpticalSystem, Ray
from optics_app.engine.optimizer import OpticalOptimizer
from optics_app.engine.source import SourceConfig, SourceType

def main():
    stack = SurfaceStack()
    
    # 0. Source
    stack.add_surface(Surface("Source", thickness=10.0, semi_diameter=2.0))
    # 1. Lens
    stack.add_surface(Surface("Lens", thickness=2.0, semi_diameter=12.7, glass="N-BK7"))
    # 2. Mirror folded 180 degrees backwards
    stack.add_surface(Surface("Mirror", thickness=5.0, semi_diameter=12.7, glass="MIRROR", tilt_x=180.0))
    # 3. Detector (should hit the lens on its way back!)
    stack.add_surface(Surface("Detector", surface_type=SurfaceType.DETECTOR, thickness=10.0))
    
    stack._update_accumulated_positions()
    
    system = OpticalSystem(stack)
    source = SourceConfig()
    source.source_type = SourceType.FIBER
    source.beam_radius = 0.1
    source.divergence_angle = 1.0
    
    opt = OpticalOptimizer(system, wavelengths=[0.5876], target="Spot Size", source_config=source)
    merit = opt.merit_function([10.0, 2.0, 5.0, 10.0], {"thickness": [0, 1, 2, 3], "radius": []})
    
    print(f"Merit Score: {merit}")
    if merit >= 5000:
        print("Success! Interference detected.")
    else:
        print("Failed to detect interference.")

if __name__ == "__main__":
    main()
