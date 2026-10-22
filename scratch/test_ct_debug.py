import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scratch.design_thorlabs_fiber_spectrometers import build_thorlabs_czerny_turner_spectrometer
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray

stack = build_thorlabs_czerny_turner_spectrometer()
stack.primary_wavelength = 0.785
stack._update_accumulated_positions()

opt_sys = OpticalSystem(stack)
r = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.785)
tr = opt_sys.trace_ray(r)
print(f"Traced surfaces: {len(tr.history)} / {len(stack.surfaces)}")
print(f"Vignetted: {tr.vignetted}, TIR: {tr.tir}")
for idx, h in enumerate(tr.history):
    s_idx = h['surface_index']
    print(f"  Step {idx} - Surf {s_idx} ({stack.surfaces[s_idx].name}): pos=({h['x']:.2f}, {h['y']:.2f}, {h['z']:.2f}) dir=({h['dir_x']:.2f}, {h['dir_y']:.2f}, {h['dir_z']:.2f})")
