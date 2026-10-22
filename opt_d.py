
import sys, os, json, numpy as np
from scipy.optimize import minimize_scalar
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

data = json.load(open('spectrometer_yz_retrofocus.json'))
data['surfaces'][20]['thickness'] = 50.0

def opt_d(d):
    data['surfaces'][17]['thickness'] = d
    json.dump(data, open('t_temp.json', 'w'))
    sys_opt = OpticalSystem.from_json('t_temp.json')
    sys_opt._update_accumulated_positions()
    
    r0 = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., 0.0, 1.0]), wavelength=1.225)
    r1 = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., 0.01, 0.99995]), wavelength=1.225)
    tr0 = sys_opt.trace_ray(r0)
    tr1 = sys_opt.trace_ray(r1)
    if len(tr1.history) <= 21: return 1e6
    return np.abs(tr1.history[21]['y_local'] - tr0.history[21]['y_local'])

res = minimize_scalar(opt_d, bounds=(10.0, 80.0), method='bounded')
print('Optimal d =', res.x)
data['surfaces'][17]['thickness'] = res.x
json.dump(data, open('spectrometer_yz_retrofocus.json', 'w'), indent=4)

