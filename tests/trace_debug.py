import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray

def trace_single(json_file):
    print("=" * 60)
    print("TRACING:", json_file)
    print("=" * 60)
    opt = OpticalSystem.from_json(json_file)
    wls = [1.1113, 1.2000, 1.3000, 1.38919, 1.5000, 1.6133, 1.66708]

    for wl in wls:
        r = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl)
        tr = opt.trace_rays([r])[0]
        print(f"\nWavelength {wl:.4f} um: vignetted={tr.vignetted}, len={len(tr.history)}")
        for h in tr.history:
            s = opt.surfaces[h['surface_index']]
            print(f"  Surf {h['surface_index']:2d} ({s.name:<25}): pos=({h['x']:8.3f}, {h['y']:8.3f}, {h['z']:8.3f}) dir=({h['dir_x']:7.4f}, {h['dir_y']:7.4f}, {h['dir_z']:7.4f})")

if __name__ == "__main__":
    trace_single("hybrid_raman_spectrometer.json")
