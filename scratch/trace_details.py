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

system = OpticalSystem(stack)

for wl in [0.785, 0.85, 0.90]:
    print(f"\n================ Wavelength {wl} um ================")
    r0 = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl)
    traced = system.trace_ray(r0)
    print(f"Traced ray: vignetted={traced.vignetted}, TIR={traced.tir}")
    for h in traced.history:
        s_name = stack.surfaces[h['surface_index']].name
        print(f"  Surf {h['surface_index']:2d} ({s_name:18s}): local hit=({h['x_local']:.3f}, {h['y_local']:.3f}, {h['z_local']:.3f}), global hit=({h['x_global']:.3f}, {h['y_global']:.3f}, {h['z_global']:.3f}), dir_out_global=({h['dir_x']:.3f}, {h['dir_y']:.3f}, {h['dir_z']:.3f})")
