"""
Inspect ray trace history surface by surface to find vignetting and beam propagation behavior.
"""
import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scratch.optimize_raman_1064 import build_system, WAVELENGTHS
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray

def trace_debug():
    params = [64.57, 15.0, 12.0, 15.0, 36.5, 12.0, 75.0, 10.0, 4.85]
    stack = build_system(params)
    stack._update_accumulated_positions()
    opt = OpticalSystem(stack)

    # Launch chief rays
    print("=== CHIEF RAYS ===")
    for wl in WAVELENGTHS:
        r = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl)
        tr = opt.trace_rays([r])[0]
        print(f"\nWL = {wl:.4f} um (Vignetted={tr.vignetted}, TIR={tr.tir}):")
        for h in tr.history:
            s_idx = h['surface_index']
            s_name = stack.surfaces[s_idx].name
            sd = stack.surfaces[s_idx].semi_diameter
            rad = math.sqrt(h['x']**2 + h['y']**2)
            print(f"  Surf {s_idx:2d} ({s_name:25s}): x={h['x']:8.3f}, y={h['y']:8.3f}, r={rad:8.3f} / CA={sd:5.1f}")

trace_debug()
