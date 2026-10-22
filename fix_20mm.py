
import sys, os, json, numpy as np
from scipy.optimize import minimize_scalar
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

data = json.load(open('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json'))

# Change Collector
data['surfaces'][1]['name'] = 'Collect_AL2520-C_S1'
data['surfaces'][1]['radius'] = 10.0
data['surfaces'][1]['thickness'] = 8.0 # thick
data['surfaces'][1]['conic'] = -1.0

data['surfaces'][2]['name'] = 'Collect_AL2520-C_S2'
data['surfaces'][2]['radius'] = 0.0

# Optimize Sample distance
def col_sample(t):
    data['surfaces'][0]['thickness'] = t
    json.dump(data, open('t20.json', 'w'))
    sys_opt = OpticalSystem.from_json('t20.json')
    sys_opt._update_accumulated_positions()
    r = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., 0.01, 0.99995]), wavelength=1.225)
    tr = sys_opt.trace_ray(r)
    return np.abs(tr.history[2]['dir_y'])
res_s = minimize_scalar(col_sample, bounds=(10.0, 30.0), method='bounded')
print('t_sample =', res_s.x)
data['surfaces'][0]['thickness'] = res_s.x

# Change Condenser
cond_idx = next(i for i, s in enumerate(data['surfaces']) if 'Cond' in s.get('name','') and 'S1' in s.get('name',''))
data['surfaces'][cond_idx]['name'] = 'Condenser_AL2520-C_S1'
data['surfaces'][cond_idx]['radius'] = 10.0
data['surfaces'][cond_idx]['thickness'] = 8.0
data['surfaces'][cond_idx]['conic'] = -1.0

data['surfaces'][cond_idx+1]['name'] = 'Condenser_AL2520-C_S2'
data['surfaces'][cond_idx+1]['radius'] = 0.0

# Optimize Detector distance
det_idx = len(data['surfaces']) - 1
data['surfaces'][-1]['semi_diameter'] = 5.0 # widen so optimization doesn't fail
data['surfaces'][6]['semi_diameter'] = 5.0

def foc_det(t):
    data['surfaces'][cond_idx+1]['thickness'] = t
    json.dump(data, open('t20.json', 'w'))
    sys_opt = OpticalSystem.from_json('t20.json')
    sys_opt._update_accumulated_positions()
    r1_ray = Ray(pos=sys_opt.surfaces[6].global_origin, dir=np.array([0., 0.01, 0.99995]), wavelength=1.225)
    tr = sys_opt.trace_ray(r1_ray)
    if tr.vignetted: return 1e6
    return np.abs(tr.history[det_idx]['y_local'])

res_d = minimize_scalar(foc_det, bounds=(10.0, 30.0), method='bounded')
print('t_det =', res_d.x)
data['surfaces'][cond_idx+1]['thickness'] = res_d.x

data['surfaces'][-1]['semi_diameter'] = 0.4
data['surfaces'][6]['semi_diameter'] = 0.1

with open('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json', 'w') as f:
    json.dump(data, f, indent=4)

