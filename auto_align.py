
import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('spectrometer_refractive_1064.json', 'r') as f:
    data = json.load(f)

for s in data['surfaces']:
    if 'DMD' in s['name']:
        s['glass'] = 'MIRROR'

surfaces = data['surfaces']

# Insert CB1
cb1 = {'name': 'CB_Grating', 'surface_type': 'Coordinate Break', 'is_coord_break': True, 'tilt_x': 43.1230625}
for i, s in enumerate(surfaces):
    if 'Grating' in s['name']:
        cb1['thickness'] = s['thickness']
        s['thickness'] = 0.0
        surfaces.insert(i+1, cb1)
        break

def get_dir_after_surface(surfaces_list, stop_name):
    stack = SurfaceStack()
    for sd in surfaces_list:
        s = Surface(name=sd.get('name', ''))
        for k,v in sd.items():
            if k == 'surface_type':
                if v.lower() == 'grating': s.surface_type = SurfaceType.GRATING
                elif v.lower() == 'coordinate break': 
                    s.surface_type = SurfaceType.COORDINATE_BREAK
                    s.is_coord_break = True
                else: s.surface_type = SurfaceType.STANDARD
            elif hasattr(s, k): setattr(s, k, v)
        stack.add_surface(s)
        if s.name == stop_name:
            break
            
    sys_opt = OpticalSystem(stack)
    sys_opt._update_accumulated_positions()
    r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=1.225)
    tr = sys_opt.trace_ray(r)
    return tr.dir

dir2 = get_dir_after_surface(surfaces, 'DMD4500_Micromirrors')
print('Dir after DMD:', dir2)

# But wait, a coordinate break tilt_x rotates the frame.
# If we look at the global direction after DMD, how do we align the next lens?
# The frame before DMD is tilted by +43.1230625.
# Let's find the required CB2 tilt!
# We can just run trace up to DMD, look at the ray's LOCAL direction in the DMD's frame, and rotate by that!

