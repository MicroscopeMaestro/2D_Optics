import sys, os
sys.path.insert(0, os.path.abspath('.'))

from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
import numpy as np

res = SurfaceStack.load_from_json(r'C:\Users\Juan\Desktop\spectrometer_raman_50mm_100mm_manual_optimized.json')
stack = res['stack']
wls = res.get('wavelengths', [1.087, 1.225, 1.35])

system = OpticalSystem(stack)
d0 = stack.surfaces[0].thickness
r1 = stack.surfaces[1].semi_diameter

py = 0.3
target_y = py * r1
length = np.sqrt(target_y**2 + d0**2)
r = Ray(pos=(0.0, 0.0, 0.0), dir=(0.0, target_y/length, d0/length), wavelength=wls[0])
tr = system.trace_ray(r)
print(f"Traced ray steps: {len(tr.history)}")
for h in tr.history:
    print(f"  surf {h.get('surface_idx')}: {h.get('surface_name')} | valid={h.get('valid')} | err={h.get('error')}")
