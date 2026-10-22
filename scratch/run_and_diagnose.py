import sys
import os
import json
import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verify_layout import verify_spectrometer_layout, RAMAN_WAVELENGTHS_UM, generate_fiber_phase_space_rays
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.lde import SurfaceStack

def diagnose(layout_path="hybrid_raman_1064_full_range.json"):
    passed, report = verify_spectrometer_layout(layout_path, verbose=True)
    print("\n--- DIAGNOSTIC METRICS ---")
    metrics = report.get("metrics", {})
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:35s}: {v:.6f}")
        else:
            print(f"  {k:35s}: {v}")
            
    print("\n--- ASSERTION SUMMARY ---")
    for a in report.get("assertions", []):
        st = "PASS" if a["passed"] else "FAIL"
        print(f"  [{st}] {a['name']}")
        print(f"         Expected: {a['expected']}")
        print(f"         Actual:   {a['actual']}")
        if a.get("detail"):
            print(f"         Detail:   {a['detail']}")

if __name__ == "__main__":
    diagnose()
