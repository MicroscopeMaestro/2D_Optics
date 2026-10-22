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
from verify_layout import RAMAN_WAVELENGTHS_UM, generate_fiber_phase_space_rays

# Let's inspect the current layout
stack_orig = SurfaceStack.load_from_json('hybrid_raman_1064.json')['stack']
print("Loaded hybrid_raman_1064.json with surfaces:")
for i, s in enumerate(stack_orig.surfaces):
    print(f"[{i:02d}] {s.name:<28} R={s.radius:<8.2f} T={s.thickness:<8.2f} Glass={s.glass:<8} SD={s.semi_diameter:<6.2f} TiltY={s.tilt_y:<6.2f}")

