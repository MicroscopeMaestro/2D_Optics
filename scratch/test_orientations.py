"""
Solve exact thicknesses for the 15-surface Thorlabs setup.
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

WAVELENGTHS = [1.1113, 1.2000, 1.3000, 1.3892, 1.5000, 1.6000, 1.6671]

# Let's test AC254-050-C collimation
# If fiber -> AC254-050-C (f=50mm)
# Front R1 = 34.0, T1=6.0, BK7; R2=-27.8, T2=2.0, SF5; R3=-115.0, T3=d3, AIR.
# Let's see: if we send a collimated beam into back R3=-115, R2=-27.8, R1=34, where does it focus?
# Or if we reverse the doublet (flint first): R1=-115.0, T1=2.0, SF5; R2=-27.8, T2=6.0, BK7; R3=-34.0?
# Standard Thorlabs orientation: more curved side faces collimated beam, less curved/flint faces focal point.
# If fiber is a point source, the less curved/flint side should face the fiber, OR the curved side faces the fiber?
# Let's check both orientations:

print("--- Orientation A: Crown front (R=34), Flint rear (R=-115) ---")
for d in np.linspace(40, 70, 31):
    s = SurfaceStack()
    s.primary_wavelength = 1.3892
    s.add_surface(Surface(thickness=d, glass="AIR"))
    s.add_surface(Surface(radius=34.0, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    s.add_surface(Surface(radius=-27.8, thickness=2.0, glass="SF5", semi_diameter=12.7))
    s.add_surface(Surface(radius=-115.0, thickness=100.0, glass="AIR", semi_diameter=12.7))
    s._update_accumulated_positions()
    r = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([math.sin(math.asin(0.12)), 0.0, math.cos(math.asin(0.12))]), wavelength=1.3892)
    tr = OpticalSystem(s).trace_rays([r])[0]
    if not tr.vignetted:
        vx, vy, vz = tr.dir
        print(f"d={d:.1f}: exit vx={vx:.6f}, angle={math.degrees(math.atan2(vx, vz)):.4f} deg")

print("\n--- Orientation B: Flint front (R=115), Crown rear (R=-34) ---")
for d in np.linspace(40, 70, 31):
    s = SurfaceStack()
    s.primary_wavelength = 1.3892
    s.add_surface(Surface(thickness=d, glass="AIR"))
    s.add_surface(Surface(radius=115.0, thickness=2.0, glass="SF5", semi_diameter=12.7))
    s.add_surface(Surface(radius=27.8, thickness=6.0, glass="N-BK7", semi_diameter=12.7))
    s.add_surface(Surface(radius=-34.0, thickness=100.0, glass="AIR", semi_diameter=12.7))
    s._update_accumulated_positions()
    r = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([math.sin(math.asin(0.12)), 0.0, math.cos(math.asin(0.12))]), wavelength=1.3892)
    tr = OpticalSystem(s).trace_rays([r])[0]
    if not tr.vignetted:
        vx, vy, vz = tr.dir
        print(f"d={d:.1f}: exit vx={vx:.6f}, angle={math.degrees(math.atan2(vx, vz)):.4f} deg")
