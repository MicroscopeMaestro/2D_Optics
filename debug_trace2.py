
import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

sys_opt = OpticalSystem.from_json('spectrometer_refractive_1064.json')
r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=1.225)
tr = sys_opt.trace_ray(r)

f_s1 = sys_opt.surfaces[6]
print('Focusing_S1 R:', f_s1.global_R)
print('Focusing_S1 origin:', f_s1.global_origin)
print('CB R:', sys_opt.surfaces[5].global_R)

# Ray entering Focusing S1:
# Which history item is it?
print(tr.history[6])


