
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath('')))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('C:/Users/Juan/Desktop/Spectrometer_V2_Full_System.json', 'r') as f: data = json.load(f)

stack = SurfaceStack()
for sd in data['surfaces']:
    s = Surface(name=sd.get('name', ''))
    for k,v in sd.items():
        if k == 'surface_type': s.surface_type = SurfaceType.GRATING if v.lower() == 'grating' else (SurfaceType.ASPHERIC if v.lower() == 'aspheric' else SurfaceType.STANDARD)
        elif hasattr(s, k): setattr(s, k, v)
    s.aperture_type = 'Circular'
    s.semi_diameter = float('inf')
    stack.add_surface(s)
sys_opt = OpticalSystem(stack)

r = Ray(pos=np.array([0.,0.,0.]), dir=np.array([0.,0.,1.]), wavelength=1.225)
tr = sys_opt.trace_ray(r)
print('Vignetted:', tr.vignetted)
for h in tr.history:
    idx = h['surface_index']
    name = sys_opt.surfaces[idx].name
    y = h['y_local']
    print(str(idx) + ' ' + name + ': y=' + str(y))

