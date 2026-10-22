
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

for wl in [1.087, 1.225, 1.350]:
    r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=wl)
    tr = sys_opt.trace_ray(r)
    print(f'WL {wl}: dir = {tr.dir}')
    print(f'Angle from -Z = {np.degrees(np.arccos(tr.dir[2]))}')
    print(f'Angle in YZ plane = {np.degrees(np.arctan2(tr.dir[1], tr.dir[2]))}')

