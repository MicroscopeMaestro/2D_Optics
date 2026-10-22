import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from verify_layout import verify_spectrometer_layout

def update_and_verify(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    surfaces = data["surfaces"]
    # Surface 8 (DMD) thickness -> distance to relay lens = 10.0 mm
    surfaces[8]["thickness"] = 10.0
    # Surface 11 (Relay Back) thickness -> distance to condenser = 5.0 mm
    surfaces[11]["thickness"] = 5.0
    # Surface 13 (Condenser Back) thickness -> distance to detector = 5.0 mm
    surfaces[13]["thickness"] = 5.0
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
        
    print(f"Updated {json_path}")
    passed, report = verify_spectrometer_layout(json_path, verbose=True)
    return passed, report

if __name__ == '__main__':
    p1, r1 = update_and_verify('hybrid_raman_1064.json')
    p2, r2 = update_and_verify('hybrid_raman_1064_full_range.json')
    print(f"\nResults: hybrid_raman_1064.json: {p1}, hybrid_raman_1064_full_range.json: {p2}")
