
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath('')))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('mirror_spectrometer/all_mirror_spectrometer.json', 'r') as f: data = json.load(f)
for i, s in enumerate(data['surfaces']):
    if 'M1' in s['name']: m1_idx = i
    if 'Grating' in s['name']: grat_idx = i
    if 'M2' in s['name']: m2_idx = i
    if 'DMD' in s['name']: dmd_idx = i
    if 'M3' in s['name']: m3_idx = i
    if 'M4' in s['name']: m4_idx = i

data['surfaces'][grat_idx]['grating_primary_wavelength_um'] = 1.225
data['surfaces'][m1_idx]['tilt_x'] = 20.0
data['surfaces'][grat_idx]['tilt_x'] = -28.1

data['surfaces'][m2_idx]['name'] = 'M2_CM254-025-P01_Focusing'
data['surfaces'][m2_idx]['radius'] = -50.8
data['surfaces'][m2_idx]['thickness'] = 25.4

data['surfaces'][m3_idx]['name'] = 'M3_CM254-200-P01_Relay_Collector'
data['surfaces'][m3_idx]['radius'] = -400.0
data['surfaces'][m3_idx]['thickness'] = 219.0
data['surfaces'][dmd_idx]['thickness'] = 200.0

data['surfaces'][dmd_idx]['semi_diameter'] = 1000.0
data['surfaces'][-1]['semi_diameter'] = 1000.0
data['surfaces'][m3_idx]['semi_diameter'] = 1000.0
data['surfaces'][m4_idx]['semi_diameter'] = 1000.0

stack = SurfaceStack()
for sd in data['surfaces']:
    s = Surface(name=sd.get('name', ''))
    for k,v in sd.items():
        if k == 'surface_type': s.surface_type = SurfaceType.GRATING if v.lower() == 'grating' else (SurfaceType.STOP if v.lower() == 'stop' else (SurfaceType.OBJECT if v.lower() == 'object' else SurfaceType.STANDARD))
        elif hasattr(s, k): setattr(s, k, v)
    stack.add_surface(s)
sys_opt = OpticalSystem(stack)

import v2.engine.tracer as tracer
orig_intersect = tracer.intersect_sphere
def mock_intersect(*args, **kwargs):
    res = orig_intersect(*args, **kwargs)
    print('Intersect:', res)
    return res
tracer.intersect_sphere = mock_intersect

r_chief = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=1.087)
tr = sys_opt.trace_ray(r_chief)

