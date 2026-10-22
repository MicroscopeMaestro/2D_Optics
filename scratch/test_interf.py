"""
Find exactly which surface triggered interference.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem, check_segment_surface_interference
from scratch.run_full_spectrometer_suite import build_portable_dmd_spectrometer

def test():
    stack = build_portable_dmd_spectrometer()
    stack.primary_wavelength = 0.850
    stack._update_accumulated_positions()
    
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([0.0, 0.0, 25.7])
    
    print("Testing segment [0, 0, 0] -> [0, 0, 25.7]:")
    for k, s in enumerate(stack.surfaces):
        if k != 0 and k != 1:
            interf = check_segment_surface_interference(p1, p2, s)
            p1_loc = s.global_R.T @ (p1 - s.global_origin)
            p2_loc = s.global_R.T @ (p2 - s.global_origin)
            print(f"  Surf {k:02d} ({s.name:<20}): Interf={interf} | z1={p1_loc[2]:7.2f}, z2={p2_loc[2]:7.2f}")
            if interf:
                t = p1_loc[2] / (p1_loc[2] - p2_loc[2])
                x_int = p1_loc[0] + t * (p2_loc[0] - p1_loc[0])
                y_int = p1_loc[1] + t * (p2_loc[1] - p1_loc[1])
                print(f"     -> Intersection local: ({x_int:.3f}, {y_int:.3f}), r={np.sqrt(x_int**2+y_int**2):.3f} mm vs sd={s.semi_diameter} mm")

if __name__ == "__main__":
    test()
