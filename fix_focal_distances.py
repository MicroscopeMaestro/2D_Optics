
import sys, os, json, numpy as np
from scipy.optimize import minimize_scalar
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

data = json.load(open('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json'))
def collimate(t):
    data['surfaces'][6]['thickness'] = t
    json.dump(data, open('t1.json', 'w'))
    sys_opt = OpticalSystem.from_json('t1.json')
    sys_opt._update_accumulated_positions()
    r1_ray = Ray(pos=sys_opt.surfaces[6].global_origin, dir=np.array([0., 0.01, 0.99995]), wavelength=1.225)
    tr = sys_opt.trace_ray(r1_ray)
    return np.abs(tr.history[9]['dir_y'])

res_col = minimize_scalar(collimate, bounds=(20.0, 30.0), method='bounded')
print('t_slit_collimator =', res_col.x)
data['surfaces'][6]['thickness'] = res_col.x

focus_indices = [i for i, s in enumerate(data['surfaces']) if 'Focusing' in s.get('name', '')]
r1, r2, r3 = 34.5, -23.1, -135.2
t1, t2 = 12.0, 2.0
data['surfaces'][focus_indices[0]]['radius'] = r1
data['surfaces'][focus_indices[0]]['thickness'] = t1
data['surfaces'][focus_indices[0]]['glass'] = 'N-BAF10'
data['surfaces'][focus_indices[1]]['radius'] = r2
data['surfaces'][focus_indices[1]]['thickness'] = t2
data['surfaces'][focus_indices[1]]['glass'] = 'N-SF6HT'
data['surfaces'][focus_indices[2]]['radius'] = r3
for i in range(3): data['surfaces'][focus_indices[i]]['name'] = f'Focusing_AC254-050-C_S{i+1}'
dmd_idx = next(i for i, s in enumerate(data['surfaces']) if 'DMD' in s.get('name',''))

def focus_dmd(t):
    data['surfaces'][focus_indices[2]]['thickness'] = t
    json.dump(data, open('t1.json', 'w'))
    sys_opt = OpticalSystem.from_json('t1.json')
    sys_opt._update_accumulated_positions()
    r1_ray = Ray(pos=sys_opt.surfaces[6].global_origin, dir=np.array([0., 0.01, 0.99995]), wavelength=1.225)
    tr = sys_opt.trace_ray(r1_ray)
    return np.abs(tr.history[dmd_idx]['y_local'])

res_foc = minimize_scalar(focus_dmd, bounds=(30.0, 60.0), method='bounded')
print('t_focus =', res_foc.x)
data['surfaces'][focus_indices[2]]['thickness'] = res_foc.x

with open('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json', 'w') as f: json.dump(data, f, indent=4)

