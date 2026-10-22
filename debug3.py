import json, numpy as np
import sys, os
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray
sys_opt = OpticalSystem.from_json('teamwork_yz_spectrometer/spectrometer_yz_1064.json')
sys_opt._update_accumulated_positions()
r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=1.087)
tr = sys_opt.trace_ray(r)
for i, h in enumerate(tr.history):
    print(i, h.get('name'), h.get('vignetted', False), np.round(h.get('z_global', 0), 2))
