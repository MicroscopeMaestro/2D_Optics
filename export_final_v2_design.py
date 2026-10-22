import os
import sys
import json
import numpy as np

# Ensure root is in path to import v2 modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optics_app.engine.lde import SurfaceStack
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.cad.exporter import STEPExporter, CADExporter

def create_final_design():
    # 1. Load the base JSON
    base_file = 'dmd4500_nir_fingerprint_0p8mm_spectrometer.json'
    with open(base_file, 'r') as f:
        data = json.load(f)
        
    # We keep up to the DMD
    # Index 7 is DMD
    surfaces = data['surfaces'][:8]
    
    # 2. Update DMD thickness (distance to Field Lens) = 15.0 mm
    surfaces[7]['thickness'] = 15.0
    
    # 3. Append the new optimized Pupil-Relay Lenses
    
    surfaces.append({
        "name": "Field_AC254-030-C_Front", "surface_type": "Standard",
        "radius": 19.9, "thickness": 5.0, "glass": "N-BAF10", "semi_diameter": 12.7, "is_active": True
    })
    
    surfaces.append({
        "name": "Field_AC254-030-C_Mid", "surface_type": "Standard",
        "radius": -15.3, "thickness": 2.0, "glass": "N-SF6", "semi_diameter": 12.7, "is_active": True
    })
    
    # Distance to Cond 1 = 20.0 mm
    surfaces.append({
        "name": "Field_AC254-030-C_Back", "surface_type": "Standard",
        "radius": -60.0, "thickness": 20.0, "glass": "AIR", "semi_diameter": 12.7, "is_active": True
    })
    
    # Condenser 1: AC127-019-C (f=19mm)
    surfaces.append({
        "name": "Cond1_AC127-019-C_Front", "surface_type": "Standard",
        "radius": 11.6, "thickness": 4.0, "glass": "N-BAF10", "semi_diameter": 6.35, "is_active": True
    })
    
    surfaces.append({
        "name": "Cond1_AC127-019-C_Mid", "surface_type": "Standard",
        "radius": -10.0, "thickness": 1.5, "glass": "N-SF6", "semi_diameter": 6.35, "is_active": True
    })
    
    # Distance Cond 1 to Cond 2 = 1.0 mm
    surfaces.append({
        "name": "Cond1_AC127-019-C_Back", "surface_type": "Standard",
        "radius": -40.0, "thickness": 1.0, "glass": "AIR", "semi_diameter": 6.35, "is_active": True
    })
    
    # Condenser 2: AC127-019-C (Reversed)
    surfaces.append({
        "name": "Cond2_AC127-019-C_Front", "surface_type": "Standard",
        "radius": 40.0, "thickness": 1.5, "glass": "N-SF6", "semi_diameter": 6.35, "is_active": True
    })
    
    surfaces.append({
        "name": "Cond2_AC127-019-C_Mid", "surface_type": "Standard",
        "radius": 10.0, "thickness": 4.0, "glass": "N-BAF10", "semi_diameter": 6.35, "is_active": True
    })
    
    # Distance Cond 2 to Detector = 10.0 mm
    surfaces.append({
        "name": "Cond2_AC127-019-C_Back", "surface_type": "Standard",
        "radius": -11.6, "thickness": 10.0, "glass": "AIR", "semi_diameter": 6.35, "is_active": True
    })
    
    # Detector
    surfaces.append({
        "name": "Detector_0.8mm", "surface_type": "Detector",
        "radius": 0.0, "thickness": 0.0, "glass": "AIR", "semi_diameter": 0.4, "is_active": True
    })
    
    data['surfaces'] = surfaces
    
    # Save the new optimized JSON file
    out_json = 'C:/Users/Juan/Desktop/Spectrometer_V2_Optimized.json'
    with open(out_json, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"1. Saved final optimized JSON to: {out_json}")
    
    # 4. Generate the CAD .STEP Export (using optics_app exporter)
    try:
        stack = SurfaceStack.from_dict(data)
        
        # We want to export lenses and the AIR gaps between them.
        exporter = STEPExporter(export_air_volumes=True)
        # Using the base cad exporter interface
        out_step = 'C:/Users/Juan/Desktop/Spectrometer_V2_Optimized.step'
        exporter.export(stack, out_step)
        print(f"2. Saved final 3D CAD .STEP assembly to: {out_step}")
    except Exception as e:
        print(f"Warning: Could not export STEP file directly (FreeCAD/build123d dependencies may be missing): {e}")

if __name__ == '__main__':
    create_final_design()
