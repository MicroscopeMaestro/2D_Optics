"""
Detailed trace diagnostics through collector and relay optics.
"""
import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scratch.optimize_collector import build_full_system, WAVELENGTHS
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray

def test_collector_detailed():
    # Let's inspect where rays vignette or miss apertures
    d_fiber = 64.57
    d_coll_grating = 15.0
    grating_tilt = 12.0
    d_grating_focus = 15.0
    d_focus_dmd = 36.5
    dmd_tilt = 12.0
    d_dmd_relay = 75.0
    d_relay_cond = 10.0
    d_cond_det = 5.0

    stack = build_full_system(d_fiber, d_coll_grating, grating_tilt, d_grating_focus, d_focus_dmd, dmd_tilt, d_dmd_relay, d_relay_cond, d_cond_det)
    stack._update_accumulated_positions()
    opt = OpticalSystem(stack)

    print("Checking where rays get vignetted:")
    for wl in WAVELENGTHS:
        # Chief ray and marginal rays
        rays = [
            Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl),
            Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.12, 0.0, math.sqrt(1-0.12**2)]), wavelength=wl),
            Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([-0.12, 0.0, math.sqrt(1-0.12**2)]), wavelength=wl),
            Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.12, math.sqrt(1-0.12**2)]), wavelength=wl),
            Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, -0.12, math.sqrt(1-0.12**2)]), wavelength=wl)
        ]
        traced = opt.trace_rays(rays)
        for idx, r in enumerate(traced):
            if r.vignetted:
                last_h = r.history[-1]
                s_idx = last_h['surface_index']
                s_name = stack.surfaces[s_idx].name
                print(f"  WL {wl:.4f} um, Ray {idx}: Vignetted at Surface {s_idx} ({s_name}): x={last_h['x']:.3f}, y={last_h['y']:.3f}, semi_diam={stack.surfaces[s_idx].semi_diameter}")
            else:
                last_h = r.history[-1]
                det_r = math.sqrt(last_h['x']**2 + last_h['y']**2)
                print(f"  WL {wl:.4f} um, Ray {idx}: Reached Detector (Surf 14): x={last_h['x']:.4f}, y={last_h['y']:.4f}, r={det_r:.4f} mm")

test_collector_detailed()
