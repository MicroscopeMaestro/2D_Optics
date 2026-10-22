import sys
import os
import json
import math

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
from optics_app.engine.tracer import OpticalSystem, Ray

system = OpticalSystem.from_json('spectrometer_yz_40mm_100mm_slit.json')

r = Ray(pos=(0, 0, 0), dir=(0, 0.05, 1.0), wavelength=1.087)
traced_ray = system.trace_ray(r)

for i, state in enumerate(traced_ray.history):
    surf_idx = state['surface_index']
    y = state['y_global']
    z = state['z_global']
    dy = state['dir_y']
    dz = state['dir_z']
    if surf_idx is not None and surf_idx < len(system.surfaces):
        name = system.surfaces[surf_idx].name
    else:
        name = "End"
    print(f"Seg {i}: to {name} at Z={z:.2f}, y={y:.5f}, dy={dy:.5f}, angle_deg={math.degrees(math.atan2(dy, dz)):.3f}")
