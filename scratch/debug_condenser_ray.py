import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scratch.trace_research_stack import build_research_stack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray

stack = build_research_stack()
opt = OpticalSystem(stack)

print("Tracing single chief ray at 1.38919 um:")
cr = Ray(pos=[0,0,0], dir=[0,0,1], wavelength=1.38919)
tr = opt.trace_rays([cr])[0]
for h in tr.history:
    print(f"Surface {h['surface_index']:02d}: pos=({h['x']:+7.3f}, {h['y']:+7.3f}, {h['z']:+7.3f}) dir=({h['dir_x']:+7.3f}, {h['dir_y']:+7.3f}, {h['dir_z']:+7.3f})")

print("\nTracing marginal ray (+NA) at 1.38919 um:")
max_theta = np.arcsin(0.12)
mr = Ray(pos=[0,0,0], dir=[np.sin(max_theta), 0, np.cos(max_theta)], wavelength=1.38919)
tr = opt.trace_rays([mr])[0]
for h in tr.history:
    print(f"Surface {h['surface_index']:02d}: pos=({h['x']:+7.3f}, {h['y']:+7.3f}, {h['z']:+7.3f}) dir=({h['dir_x']:+7.3f}, {h['dir_y']:+7.3f}, {h['dir_z']:+7.3f})")

print("\nTracing band edge chief ray at 1.11130 um:")
cr11 = Ray(pos=[0,0,0], dir=[0,0,1], wavelength=1.11130)
tr11 = opt.trace_rays([cr11])[0]
for h in tr11.history:
    print(f"Surface {h['surface_index']:02d}: pos=({h['x']:+7.3f}, {h['y']:+7.3f}, {h['z']:+7.3f}) dir=({h['dir_x']:+7.3f}, {h['dir_y']:+7.3f}, {h['dir_z']:+7.3f})")
