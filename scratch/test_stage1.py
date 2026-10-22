"""
Analyze and optimize each stage of the 15-surface Hybrid Raman Spectrometer.
"""
import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.analysis.spot import SpotDiagram

def test_stage1_collimation():
    print("=== STAGE 1: COLLIMATION (AC254-050-C) ===")
    # Let's find the exact fiber-to-lens distance that produces perfectly collimated beam at 1.3892 um.
    for d_fiber in np.linspace(40.0, 52.0, 25):
        stack = SurfaceStack()
        stack.primary_wavelength = 1.3892
        stack.add_surface(Surface(name="Fiber", surface_type=SurfaceType.OBJECT, thickness=d_fiber, glass="AIR"))
        stack.add_surface(Surface(name="L1_F", radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
        stack.add_surface(Surface(name="L1_M", radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
        stack.add_surface(Surface(name="L1_B", radius=-115.0, thickness=100.0, glass="AIR", semi_diameter=12.7))
        stack._update_accumulated_positions()

        # Launch marginal ray with NA=0.12 (theta = 6.892 deg)
        theta = math.asin(0.12)
        r1 = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.3892)
        r2 = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([math.sin(theta), 0.0, math.cos(theta)]), wavelength=1.3892)
        
        opt = OpticalSystem(stack)
        tr = opt.trace_rays([r1, r2])
        
        # Check angle of r2 after L1_B (Surface 3)
        # In history, surface 3 is L1_B
        h = tr[1].history[3] # exit of L1_B
        # Ray direction after surface 3
        # In tracer, tr[1].dir is the direction after last surface
        vx, vy, vz = tr[1].dir
        div_angle_mrad = math.atan2(math.sqrt(vx**2 + vy**2), vz) * 1000.0
        # Beam radius at 100mm vs at 0mm
        pos_at_lens = tr[1].history[3]['x']
        pos_at_100mm = tr[1].history[-1]['x']
        slope = (pos_at_100mm - pos_at_lens) / 100.0
        print(f"d_fiber = {d_fiber:.2f} mm -> ray exit slope = {slope:.6f}, div angle = {div_angle_mrad:.3f} mrad, beam radius = {pos_at_lens:.2f} mm")

test_stage1_collimation()
