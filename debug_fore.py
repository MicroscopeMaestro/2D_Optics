
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath('')))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray
with open('C:/Users/Juan/Desktop/Spectrometer_V2_Full_System.json', 'r') as f: data = json.load(f)
stack = SurfaceStack()
for sd in data['surfaces'][:8]:
    s = Surface(name=sd.get('name', ''))
    for k,v in sd.items():
        if hasattr(s, k): setattr(s, k, v)
    stack.add_surface(s)
sys_opt = OpticalSystem(stack)
r = Ray(pos=np.array([0.,0.5,1e-3]), dir=np.array([0., 0.05, 1.0]), wavelength=1.225)
tr = sys_opt.trace_ray(r)
print('Vignetted:', tr.vignetted)
for h in tr.history:
    idx = h['surface_index']
    name = sys_opt.surfaces[idx].name
    y = h['y_local']
    print(str(idx) + ' ' + name + ': y=' + str(y))

