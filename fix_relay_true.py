
import sys, os, json, numpy as np
from scipy.optimize import minimize_scalar
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

data = json.load(open('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json'))
data['surfaces'][6]['semi_diameter'] = 5.0
data['surfaces'][-1]['semi_diameter'] = 5.0
relay_indices = [i for i, s in enumerate(data['surfaces']) if 'Relay' in s.get('name', '')]
dmd_idx = next(i for i, s in enumerate(data['surfaces']) if 'DMD' in s.get('name',''))
cb_idx = dmd_idx + 1

def collimate_relay(t):
    data['surfaces'][cb_idx]['thickness'] = t
    json.dump(data, open('t_relay.json', 'w'))
    sys_opt = OpticalSystem.from_json('t_relay.json')
    sys_opt._update_accumulated_positions()
    r1_ray = Ray(pos=sys_opt.surfaces[6].global_origin, dir=np.array([0., 0.01, 0.99995]), wavelength=1.225)
    tr = sys_opt.trace_ray(r1_ray)
    if tr.vignetted: return 1e6
    return np.abs(tr.history[relay_indices[2]]['dir_y'])

res_r = minimize_scalar(collimate_relay, bounds=(70.0, 90.0), method='bounded')
print('t_dmd_relay =', res_r.x)
data['surfaces'][cb_idx]['thickness'] = res_r.x

cond_idx = next(i for i, s in enumerate(data['surfaces']) if 'Cond' in s.get('name','') and 'S1' in s.get('name',''))
det_idx = len(data['surfaces']) - 1

def focus_det(t):
    data['surfaces'][cond_idx+1]['thickness'] = t
    json.dump(data, open('t_relay.json', 'w'))
    sys_opt = OpticalSystem.from_json('t_relay.json')
    sys_opt._update_accumulated_positions()
    r1_ray = Ray(pos=sys_opt.surfaces[6].global_origin, dir=np.array([0., 0.01, 0.99995]), wavelength=1.225)
    tr = sys_opt.trace_ray(r1_ray)
    if tr.vignetted: return 1e6
    return np.abs(tr.history[det_idx]['y_local'])

res_d = minimize_scalar(focus_det, bounds=(5.0, 15.0), method='bounded')
print('t_cond_det =', res_d.x)
data['surfaces'][cond_idx+1]['thickness'] = res_d.x

data['surfaces'][6]['semi_diameter'] = 0.1
data['surfaces'][-1]['semi_diameter'] = 0.4
with open('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json', 'w') as f:
    json.dump(data, f, indent=4)

