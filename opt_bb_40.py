
import sys, os, json, numpy as np
from scipy.optimize import minimize_scalar
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

data = json.load(open('t_broadband.json'))
# Restore 40mm Focusing Lens
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
    json.dump(data, open('t_broadband_40.json', 'w'))
    sys_opt = OpticalSystem.from_json('t_broadband_40.json')
    sys_opt._update_accumulated_positions()
    r0 = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., 0.0, 1.0]), wavelength=1.35)
    tr0 = sys_opt.trace_ray(r0)
    r1 = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., 0.01, 0.99995]), wavelength=1.35)
    tr1 = sys_opt.trace_ray(r1)
    if len(tr1.history) <= 18: return 1e6
    return np.abs(tr1.history[18]['y_local'] - tr0.history[18]['y_local'])

res_focus = minimize_scalar(opt_focus, bounds=(30.0, 45.0), method='bounded')
data['surfaces'][17]['thickness'] = res_focus.x
json.dump(data, open('t_broadband_40.json', 'w'))

sys_opt = OpticalSystem.from_json('t_broadband_40.json')
sys_opt._update_accumulated_positions()

def get_y(wl):
    tr = sys_opt.trace_ray(Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., 0.0, 1.0]), wavelength=wl))
    return tr.history[18]['y_local']

print('1000 nm on DMD:', get_y(1.0))
print('1350 nm on DMD:', get_y(1.35))
print('1700 nm on DMD:', get_y(1.7))
print('DMD Spread with 40mm + 300 l/mm:', np.abs(get_y(1.7) - get_y(1.0)), 'mm')

