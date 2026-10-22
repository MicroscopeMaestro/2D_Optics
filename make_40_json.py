
import sys, os, json, numpy as np
from scipy.optimize import minimize_scalar
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

data = json.load(open('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json'))

data['surfaces'][15]['name'] = 'Focus_AC254-040-C_S1'
data['surfaces'][15]['radius'] = 36.73634132
data['surfaces'][15]['thickness'] = 7.0
data['surfaces'][15]['glass'] = 'N-BAF10'

data['surfaces'][16]['name'] = 'Focus_AC254-040-C_S2'
data['surfaces'][16]['radius'] = -17.95865465
data['surfaces'][16]['thickness'] = 2.0
data['surfaces'][16]['glass'] = 'F2'

data['surfaces'][17]['name'] = 'Focus_AC254-040-C_S3'
data['surfaces'][17]['radius'] = -259.24872865

def opt_focus(t):
    data['surfaces'][17]['thickness'] = t
    json.dump(data, open('t_temp_40.json', 'w'))
    sys_opt = OpticalSystem.from_json('t_temp_40.json')
    sys_opt._update_accumulated_positions()
    
    r0 = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., 0.0, 1.0]), wavelength=1.225)
    r1 = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., 0.01, 0.99995]), wavelength=1.225)
    tr0 = sys_opt.trace_ray(r0)
    tr1 = sys_opt.trace_ray(r1)
    if len(tr1.history) <= 18: return 1e6
    
    return np.abs(tr1.history[18]['y_local'] - tr0.history[18]['y_local'])

res = minimize_scalar(opt_focus, bounds=(20.0, 50.0), method='bounded')
print('Optimal t_focus =', res.x)
data['surfaces'][17]['thickness'] = res.x
json.dump(data, open('spectrometer_yz_40mm_focus.json', 'w'), indent=4)

