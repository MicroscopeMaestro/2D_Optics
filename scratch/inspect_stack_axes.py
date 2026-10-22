import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import numpy as np
from optics_app.engine.surface import Surface
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem, Ray

with open('fiber_spectrometer_portable.json', 'r') as f:
    data = json.load(f)

surfaces = []
for s_dict in data["surfaces"]:
    s = Surface()
    for k, v in s_dict.items():
        if hasattr(s, k):
            setattr(s, k, v)
    surfaces.append(s)

stack = SurfaceStack()
stack.surfaces = surfaces
stack.primary_wavelength = 0.85
stack._update_accumulated_positions()

print("Surface Origins & Rotations:")
for idx, s in enumerate(stack.surfaces):
    print(f"Surf {idx:2d} ({s.name:20s}): origin = [{s.global_origin[0]:.2f}, {s.global_origin[1]:.2f}, {s.global_origin[2]:.2f}], Z-axis = [{s.global_R[0,2]:.3f}, {s.global_R[1,2]:.3f}, {s.global_R[2,2]:.3f}]")
