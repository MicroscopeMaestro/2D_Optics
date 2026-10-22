import sys
sys.path.insert(0, ".")
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
import numpy as np

stack_dict = SurfaceStack.load_from_json(r'C:\Users\Juan\Desktop\spectrometer_raman_50mm_100mm_manual.json')
stack = stack_dict['stack']
sys_opt = OpticalSystem(stack)
ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.064)
res = sys_opt.trace_ray(ray)
for i, h in enumerate(res.history):
    s = stack.surfaces[i]
    print(f"S{i:02d} {s.name:20s}: global=({h['x_global']:8.2f}, {h['y_global']:8.2f}, {h['z_global']:8.2f}) dir=({h['dir_x']:6.3f}, {h['dir_y']:6.3f}, {h['dir_z']:6.3f}) vig={h['vignetted']}")
