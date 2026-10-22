import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from verify_layout import verify_spectrometer_layout

def update_and_verify(json_path, t8=20.0, t11=5.0, t13=5.2):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    surfaces = data["surfaces"]
    surfaces[8]["thickness"] = float(t8)
    surfaces[11]["thickness"] = float(t11)
    surfaces[13]["thickness"] = float(t13)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
        
    passed, report = verify_spectrometer_layout(json_path, verbose=False)
    print(f"File {json_path}: status={report['overall_status']}, vig={report['vignetting_pct']}%, dmd_span={report['dmd_dispersion_width_mm']:.3f}mm, det_diam={report['detector_max_spot_diameter_mm']:.4f}mm")
    return passed, report

if __name__ == '__main__':
    update_and_verify('hybrid_raman_1064.json', 20.0, 5.0, 5.2)
    update_and_verify('hybrid_raman_1064_full_range.json', 20.0, 5.0, 5.2)
