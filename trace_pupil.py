
import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

sys_opt = OpticalSystem.from_json('spectrometer_yz_40mm_100mm_final.json')
sys_opt._update_accumulated_positions()

# Ray 1: 1225 nm (center)
r1 = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., 0.0, 1.0]), wavelength=1.225)
tr1 = sys_opt.trace_ray(r1)

# Ray 2: 1064 nm (edge)
r2 = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., 0.0, 1.0]), wavelength=1.064)
tr2 = sys_opt.trace_ray(r2)

print('1225 nm at Det:', tr1.history[-1]['y_local'])
print('1064 nm at Det:', tr2.history[-1]['y_local'])

