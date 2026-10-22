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
    
    stack.surfaces[0].thickness = 25.0
    stack.surfaces[4].thickness = 30.0
    stack.surfaces[7].thickness = 50.0
    stack.surfaces[8].thickness = 25.0
    stack.surfaces[13].thickness = 25.0
    stack._update_accumulated_positions()
    
    opt = OpticalOptimizer(system, wavelengths=[0.450, 0.546, 0.650], target="Spot Size", source_config=source, max_iter=1)
    
    print("Generating rays...")
    rays = opt._generate_rays(opt.entrance_pupil_radius, opt.wavelengths, opt.num_rays)
    
    print(f"Tracing {len(rays)} rays...")
    traced = system.trace_rays(rays)
    print("Done tracing!")

if __name__ == "__main__":
    main()
