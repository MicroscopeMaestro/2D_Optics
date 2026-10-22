import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath('.'))
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.source import SourceConfig, SourceType
from optics_app.analysis.spot import generate_spot_diagram_from_system

def evaluate_spectrometer(json_path: str, wavelengths: list, num_rays: int = 500):
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return
        
    print(f"\n==================================================")
    print(f"Evaluating Model: {json_path}")
    print(f"==================================================")
    
    res = SurfaceStack.load_from_json(json_path)
    stack = res["stack"]
    
    system = OpticalSystem(stack)
    
    # Configure high-density source
    semi_d = stack.surfaces[0].semi_diameter if len(stack.surfaces) > 0 else 0.0125
    source_config = SourceConfig()
    source_config.source_type = SourceType.FIBER
    source_config.beam_radius = semi_d
    source_config.divergence_angle = 13.0
    
    results = {}
    for wl in wavelengths:
        sd = generate_spot_diagram_from_system(system, num_rays=num_rays, wavelength_um=wl, target_surface_index=-1, source_config=source_config)
        
        valid_rays = len(sd.x)
        if valid_rays == 0:
            print(f"Wavelength {wl}nm: 0 valid rays reached the detector!")
            results[wl] = {"rms": float('inf'), "geo": float('inf'), "trans": 0.0, "x_span": 0.0, "y_span": 0.0}
            continue
            
        x_pts = np.array(sd.x)
        y_pts = np.array(sd.y)
        
        centroid_x = np.mean(x_pts)
        centroid_y = np.mean(y_pts)
        
        r_sq = (x_pts - centroid_x)**2 + (y_pts - centroid_y)**2
        rms_radius = np.sqrt(np.mean(r_sq))
        geo_radius = np.max(np.sqrt(r_sq))
        
        x_span = np.max(x_pts) - np.min(x_pts)
        y_span = np.max(y_pts) - np.min(y_pts)
        
        transmission = (valid_rays / num_rays) * 100.0
        
        results[wl] = {
            "rms": rms_radius,
            "geo": geo_radius,
            "trans": transmission,
            "x_span": x_span,
            "y_span": y_span
        }
        
        print(f"Wavelength: {wl} nm")
        print(f"  Transmission: {transmission:.1f}% ({valid_rays}/{num_rays} rays)")
        print(f"  RMS Radius:   {rms_radius*1000:.3f} um")
        print(f"  GEO Radius:   {geo_radius*1000:.3f} um")
        print(f"  X Span:       {x_span*1000:.3f} um")
        print(f"  Y Span:       {y_span*1000:.3f} um")
        
    return results

if __name__ == "__main__":
    wavelengths = [0.785, 0.850, 0.900]
    evaluate_spectrometer("fiber_spectrometer_thorlabs_optimized.json", wavelengths)
    evaluate_spectrometer("fiber_spectrometer_mirrors_optimized.json", wavelengths)
    evaluate_spectrometer("hybrid_spectrometer_optimized.json", wavelengths)
