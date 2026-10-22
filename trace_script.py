import json
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
import numpy as np

stack = SurfaceStack.load_from_json('fiber_spectrometer_setup.json')['stack']
stack.primary_wavelength = 0.7
sys = OpticalSystem(stack)

ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.7)
traced = sys.trace_ray(ray)

for i, h in enumerate(traced.history):
    print(f'Hit {i}:')
    print(f'  Global pos: {h.get("x_global", 0):.4f}, {h.get("y_global", 0):.4f}, {h.get("z_global", 0):.4f}')
    print(f'  Global dir: {h.get("dir_x", 0):.4f}, {h.get("dir_y", 0):.4f}, {h.get("dir_z", 0):.4f}')
    if i == 5:
        surf5 = stack.surfaces[5]
        print(f'  L2 Origin: {surf5.global_origin}')
        local_p = surf5.global_R.T @ (np.array([h.get("x_global", 0), h.get("y_global", 0), h.get("z_global", 0)]) - surf5.global_origin)
        print(f'  Local pos on L2: {local_p}')
