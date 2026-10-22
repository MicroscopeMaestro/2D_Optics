
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath('')))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem

with open('C:/Users/Juan/Desktop/Spectrometer_V2_Complete_NoCollide.json', 'r') as f: data = json.load(f)

stack = SurfaceStack()
for sd in data['surfaces']:
    s = Surface(name=sd.get('name', ''))
    for k,v in sd.items():
        if k == 'surface_type': s.surface_type = SurfaceType.GRATING if v.lower() == 'grating' else (SurfaceType.ASPHERIC if v.lower() == 'aspheric' else SurfaceType.STANDARD)
        elif hasattr(s, k): setattr(s, k, v)
    stack.add_surface(s)
sys_opt = OpticalSystem(stack)

for i, s in enumerate(sys_opt.surfaces):
    pos = s.global_origin
    print(f'{i:2d} {s.name:25s} Z={pos[2]:6.2f} Y={pos[1]:6.2f}')

