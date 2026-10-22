import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scratch.trace_research_stack import build_research_stack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import RAMAN_WAVELENGTHS_UM, generate_fiber_phase_space_rays

stack = build_research_stack()
opt = OpticalSystem(stack)

rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)
traced = opt.trace_rays(rays)

print(f"Total traced rays: {len(traced)}")
for r_idx, r in enumerate(traced):
    h14 = next((h for h in r.history if h["surface_index"] == 14), None)
    if h14:
        rad = np.sqrt(h14["x"]**2 + h14["y"]**2)
        if rad > 1.0:
            h0 = r.history[0]
            h8 = next((h for h in r.history if h["surface_index"] == 8), None)
            h12 = next((h for h in r.history if h["surface_index"] == 12), None)
            h13 = next((h for h in r.history if h["surface_index"] == 13), None)
            print(f"Ray {r_idx:03d} | Wl={r.wavelength:.4f} um | pos0=({h0['x']:.3f}, {h0['y']:.3f}) dir0=({h0['dir_x']:.3f}, {h0['dir_y']:.3f}) | DMD_x={h8['x'] if h8 else 'N/A':+.3f} | Cond12_x={h12['x'] if h12 else 'N/A':+.3f} | Cond13_x={h13['x'] if h13 else 'N/A':+.3f} | Det_x={h14['x']:+8.3f}, Det_r={rad:8.3f} mm")
