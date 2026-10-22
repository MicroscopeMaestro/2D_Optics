
import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

sys_opt = OpticalSystem.from_json('spectrometer_refractive_1064.json')
r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=1.225)
tr = sys_opt.trace_ray(r)

print('Vignetted:', tr.vignetted)
for h in tr.history:
    s = sys_opt.surfaces[h['surface_index']]
    print(s.name)

# Let's manually check the intersection with Focusing_S1
focus_s1 = sys_opt.surfaces[6]  # Focusing_S1
print('Focusing_S1 global origin:', focus_s1.global_origin)
print('Focusing_S1 global R:', focus_s1.global_R)
print('Ray after Grating: pos =', tr.pos, 'dir =', tr.dir)

