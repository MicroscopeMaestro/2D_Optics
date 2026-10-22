import sys, os
sys.path.insert(0, os.path.abspath('.'))
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
import numpy as np

stack = SurfaceStack.load_from_json('fiber_spectrometer_mirrors.json')['stack']
sys_opt = OpticalSystem(stack)
ray = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0.1, 0.995]), wavelength=0.785)
res = sys_opt.trace_ray(ray)
print('Vignetted:', res.vignetted)
for h in res.history:
    print(f"Surf {h['surface_index']}: x_local={h.get('x_local', 0):.4f}, y_local={h.get('y_local', 0):.4f}, vignetted={h.get('vignetted', False)}")
