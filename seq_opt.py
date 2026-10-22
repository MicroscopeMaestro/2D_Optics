
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
relay_s3 = relay_indices[-1]
cond_idx = next(i for i, s in enumerate(data['surfaces']) if 'Cond' in s.get('name','') and 'S1' in s.get('name',''))
det_idx = len(data['surfaces']) - 1

wl = 1.225

def collimate_relay(t):
    data['surfaces'][cb_idx]['thickness'] = t
    with open('temp_seq.json', 'w') as f: json.dump(data, f)
    sys_opt = OpticalSystem.from_json('temp_seq.json')
    sys_opt._update_accumulated_positions()
    r1 = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0.01, 0.99995]), wavelength=wl)
    r2 = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., -0.01, 0.99995]), wavelength=wl)
    tr1 = sys_opt.trace_ray(r1)
    tr2 = sys_opt.trace_ray(r2)
    if tr1.vignetted or tr2.vignetted or len(tr1.history) <= relay_s3: return 1e6
    return np.abs(tr1.history[relay_s3]['dir_y'] - tr2.history[relay_s3]['dir_y'])

res_relay = minimize_scalar(collimate_relay, bounds=(10.0, 150.0), method='bounded')
print('t_dmd_relay =', res_relay.x)

