
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath('')))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('spectrometer_relay_redesign/spectrometer_nir_900_1700nm.json', 'r') as f: data = json.load(f)

stack = SurfaceStack()
for sd in data['surfaces']:
    s = Surface(name=sd.get('name', ''))
    for k,v in sd.items():
        if k == 'surface_type': s.surface_type = SurfaceType.GRATING if v.lower() == 'grating' else (SurfaceType.STOP if v.lower() == 'stop' else (SurfaceType.OBJECT if v.lower() == 'object' else SurfaceType.STANDARD))
        elif hasattr(s, k): setattr(s, k, v)
    if 'Aspheric' in s.name:
        s.is_aspheric = True
        s.conic = sd.get('conic', 0.0)
    stack.add_surface(s)
sys_opt = OpticalSystem(stack)

wls = [0.900, 1.100, 1.300, 1.500, 1.700]
print('--- Chief Ray Pass ---')
for wl in wls:
    r_chief = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=wl)
    tr = sys_opt.trace_ray(r_chief)
    if tr.vignetted:
        print(f'{wl} Vignetted at:', sys_opt.surfaces[tr.history[-1]['surface_index']].name)
    else:
        print(f'{wl} passed. Final y_local:', tr.history[-1]['y_local'])

