
import sys, os, json, numpy as np
from scipy.optimize import minimize_scalar
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

data = json.load(open('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json'))
data['surfaces'][6]['semi_diameter'] = 5.0
data['surfaces'][-1]['semi_diameter'] = 5.0

foc_s3_idx = next(i for i, s in enumerate(data['surfaces']) if 'Focusing' in s.get('name','') and 'S3' in s.get('name',''))
dmd_idx = next(i for i, s in enumerate(data['surfaces']) if 'DMD' in s.get('name',''))

def focus_dmd(t):
    data['surfaces'][foc_s3_idx]['thickness'] = t
    json.dump(data, open('t_temp.json', 'w'))
    sys_opt = OpticalSystem.from_json('t_temp.json')
    sys_opt._update_accumulated_positions()
    r0 = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., 0.0, 1.0]), wavelength=1.225)
    r1 = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., 0.01, 0.99995]), wavelength=1.225)
    tr0 = sys_opt.trace_ray(r0)
    tr1 = sys_opt.trace_ray(r1)
    if len(tr1.history) <= dmd_idx: return 1e6
    return np.abs(tr1.history[dmd_idx]['y_local'] - tr0.history[dmd_idx]['y_local'])

res_foc = minimize_scalar(focus_dmd, bounds=(40.0, 60.0), method='bounded')
print('t_focus =', res_foc.x)
data['surfaces'][foc_s3_idx]['thickness'] = res_foc.x

data['surfaces'][6]['semi_diameter'] = 0.1
data['surfaces'][-1]['semi_diameter'] = 0.4
with open('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json', 'w') as f:
    json.dump(data, f, indent=4)

