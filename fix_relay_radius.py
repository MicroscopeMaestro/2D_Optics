
import sys, os, json, numpy as np
from scipy.optimize import minimize_scalar
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json', 'r') as f: data = json.load(f)
data['surfaces'][6]['semi_diameter'] = 5.0
focus_indices = [i for i, s in enumerate(data['surfaces']) if 'Focusing' in s.get('name', '')]
focus_s3 = focus_indices[-1]
dmd_idx = next(i for i, s in enumerate(data['surfaces']) if 'DMD' in s.get('name',''))
cb_idx = dmd_idx + 1
relay_indices = [i for i, s in enumerate(data['surfaces']) if 'Relay' in s.get('name', '')]
relay_s1 = relay_indices[0]
relay_s2 = relay_indices[1]
relay_s3 = relay_indices[2]

data['surfaces'][cb_idx]['thickness'] = 71.0 # BFL of a 75mm doublet is usually ~71mm!

def set_radii(r1):
    # Standard achromatic doublet shape factor
    data['surfaces'][relay_s1]['radius'] = r1
    data['surfaces'][relay_s2]['radius'] = -r1 * 0.8  # typical
    data['surfaces'][relay_s3]['radius'] = -r1 * 2.5
    
    with open('temp_seq.json', 'w') as f: json.dump(data, f)
    sys_opt = OpticalSystem.from_json('temp_seq.json')
    sys_opt._update_accumulated_positions()
    
    r1_ray = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0.01, 0.99995]), wavelength=1.225)
    r2_ray = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., -0.01, 0.99995]), wavelength=1.225)
    tr1 = sys_opt.trace_ray(r1_ray)
    tr2 = sys_opt.trace_ray(r2_ray)
    if tr1.vignetted or tr2.vignetted or len(tr1.history) <= relay_s3: return 1e6
    return np.abs(tr1.history[relay_s3]['dir_y'] - tr2.history[relay_s3]['dir_y'])

res = minimize_scalar(set_radii, bounds=(20.0, 300.0), method='bounded')
print('Optimal r1 =', res.x)
set_radii(res.x)

# Now optimize t_cond_det
cond_idx = next(i for i, s in enumerate(data['surfaces']) if 'Cond' in s.get('name','') and 'S1' in s.get('name',''))
det_idx = len(data['surfaces']) - 1
data['surfaces'][relay_s3]['thickness'] = 10.0

def focus_det(t):
    data['surfaces'][cond_idx+1]['thickness'] = t
    with open('temp_seq.json', 'w') as f: json.dump(data, f)
    sys_opt = OpticalSystem.from_json('temp_seq.json')
    sys_opt._update_accumulated_positions()
    r1_ray = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0.01, 0.99995]), wavelength=1.225)
    r2_ray = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., -0.01, 0.99995]), wavelength=1.225)
    tr1 = sys_opt.trace_ray(r1_ray)
    tr2 = sys_opt.trace_ray(r2_ray)
    if tr1.vignetted or tr2.vignetted or len(tr1.history) <= det_idx: return 1e6
    return np.abs(tr1.history[det_idx]['y_local'] - tr2.history[det_idx]['y_local'])

res_det = minimize_scalar(focus_det, bounds=(5.0, 15.0), method='bounded')
print('t_cond_det =', res_det.x)
data['surfaces'][cond_idx+1]['thickness'] = res_det.x
data['surfaces'][6]['semi_diameter'] = 0.1

with open('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json', 'w') as f:
    json.dump(data, f, indent=4)

