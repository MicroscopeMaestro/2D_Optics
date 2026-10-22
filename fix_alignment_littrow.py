
import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('spectrometer_refractive_1064.json', 'r') as f:
    data = json.load(f)

for s in data['surfaces']:
    if 'Grating' in s['name']:
        s['glass'] = 'MIRROR'
        s['tilt_x'] = -45.0

stack = SurfaceStack()
for sd in data['surfaces'][:5]:
    s = Surface(name=sd.get('name', ''))
    for k,v in sd.items():
        if k == 'surface_type':
            if 'grating' in v.lower(): s.surface_type = SurfaceType.GRATING
            elif 'object' in v.lower(): s.surface_type = SurfaceType.OBJECT
            else: s.surface_type = SurfaceType.STANDARD
        elif hasattr(s, k): setattr(s, k, v)
    stack.add_surface(s)

sys_opt = OpticalSystem(stack)
sys_opt._update_accumulated_positions()
r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=1.225)
tr = sys_opt.trace_ray(r)

print('Direction after reflective grating:', tr.dir)

