import os
import sys
import numpy as np

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
    
    # Pre-seed the thicknesses for 10x demagnification
    stack.surfaces[0].thickness = 19.0
    stack.surfaces[4].thickness = 19.0 
    stack.surfaces[7].thickness = 19.0
    stack.surfaces[8].thickness = 88.0
    stack.surfaces[13].thickness = 8.8
    
    stack._update_accumulated_positions()
    
    opt = OpticalOptimizer(system, wavelengths=[0.450, 0.546, 0.650], target="Spot Size", source_config=source, max_iter=1)
    rays = opt._generate_rays(6.35, opt.wavelengths, 100)
    for r in rays:
        r.pos[2] = -0.01
    
    traced_rays = system.trace_rays(rays)
    
    print("Debug Ray 0:")
    for h in traced_rays[0].history:
        print(" ", h["surface_name"])
    if traced_rays[0].vignetted:
        print(" Ray 0 vignetted!")
    
    for i, s in enumerate(stack.surfaces):
        print(f"Surface {i} ({s.name}):")
        alive = sum(1 for r in traced_rays if len(r.history) > i and r.history[i] is not None)
        print(f"  Alive rays: {alive}")
        if alive == 0 and i == 0:
            for r in traced_rays:
                if len(r.history) > 0 and r.history[0] is None:
                    # Not hit!
                    print("  Ray missed surface entirely!")
                    break
        if alive == 0:
            print(f"  -> ALL RAYS BLOCKED AT {s.name}!")
            break

if __name__ == "__main__":
    main()
