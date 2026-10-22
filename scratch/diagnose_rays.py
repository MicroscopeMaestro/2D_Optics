import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath('.'))
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.optimizer import OpticalOptimizer
from optics_app.engine.source import SourceConfig, SourceType

def main():
    json_path = "fiber_spectrometer_thorlabs_optimized.json"
    data = SurfaceStack.load_from_json(json_path)
    stack = data["stack"]
    system = OpticalSystem(stack)
    
    source = SourceConfig()
    source.source_type = SourceType.FIBER
    source.beam_radius = 0.0125
    source.divergence_angle = 12.7
    
    # Just set the folds to physically separate the components
    stack.surfaces[4].tilt_x = 30.0 # Grating (60 deg fold)
    stack.surfaces[8].tilt_x = 45.0 # DMD (90 deg fold)
    stack.surfaces[14].tilt_x = 0.0 # Detector
    
    stack._update_accumulated_positions()
    
    opt = OpticalOptimizer(system, wavelengths=[0.450, 0.546, 0.650], target="Spot Size", source_config=source, max_iter=1)
    
    # We pass empty var indices, just to evaluate the merit function with spatial constraints
    initial_rms = opt.merit_function([], {"thickness": [], "tilt_x": []})
    print(f"Calculated physically clear initial RMS: {initial_rms}")
    
    rays = opt._generate_rays(opt.entrance_pupil_radius, opt.wavelengths, opt.num_rays)
    traced = system.trace_rays(rays)
    vignetted_count = 0
    detector_hits = 0
    total_penalty = 0.0
    from optics_app.engine.tracer import check_segment_surface_interference
    for ray in traced:
        if ray.vignetted or ray.interference:
            vignetted_count += 1
            # Check where interference happened
            if ray.interference:
                for i in range(len(ray.history) - 1):
                    h1 = ray.history[i]
                    h2 = ray.history[i+1]
                    seg_start_idx = h1["surface_index"]
                    seg_end_idx = h2["surface_index"]
                    
                    p1 = np.array([h1["x_global"], h1["y_global"], h1["z_global"]])
                    p2 = np.array([h2["x_global"], h2["y_global"], h2["z_global"]])
                    
                    for s_idx, surf in enumerate(stack.surfaces):
                        if s_idx == seg_start_idx or s_idx == seg_end_idx:
                            continue
                        if check_segment_surface_interference(p1, p2, surf):
                            print(f"Ray seg ({seg_start_idx}->{seg_end_idx}) hits surface {s_idx} ({surf.name})")
        elif len(ray.history) > 0 and ray.history[-1]["surface_index"] == len(stack.surfaces) - 1:
            detector_hits += 1
            
    print(f"Total rays: {len(traced)}")
    print(f"Vignetted/Interference: {vignetted_count}")
    print(f"Detector hits: {detector_hits}")

if __name__ == "__main__":
    main()
