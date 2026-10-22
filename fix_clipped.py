
import sys, os, json, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('spectrometer_clipped_0.16x.json', 'r') as f:
    data = json.load(f)

dmd_idx = next(i for i, s in enumerate(data['surfaces']) if 'DMD' in s.get('name',''))
cb_idx = dmd_idx + 1
cond_idx = next(i for i, s in enumerate(data['surfaces']) if 'Cond' in s.get('name','') and 'S1' in s.get('name',''))
relay_s3 = cond_idx - 1
det_idx = len(data['surfaces']) - 1

def test_system(x):
    t_dmd_relay, t_relay_cond, t_cond_det = x
    data['surfaces'][cb_idx]['thickness'] = t_dmd_relay
    data['surfaces'][relay_s3]['thickness'] = t_relay_cond
    data['surfaces'][cond_idx+1]['thickness'] = t_cond_det
    
    with open('temp_clipped2.json', 'w') as f: json.dump(data, f)
    sys_opt = OpticalSystem.from_json('temp_clipped2.json')
    sys_opt._update_accumulated_positions()
    
    r1 = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0.01, 0.99995]), wavelength=1.225)
    r2 = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., -0.01, 0.99995]), wavelength=1.225)
    tr1 = sys_opt.trace_ray(r1)
    tr2 = sys_opt.trace_ray(r2)
    
    if tr1.vignetted or tr2.vignetted or len(tr1.history) <= det_idx:
        return 1e6
        
    det_diff = np.abs(tr1.history[det_idx]['y_local'] - tr2.history[det_idx]['y_local'])
    # We want collimated light between relay and condenser!
    h1_after_relay = tr1.history[relay_s3]
    h2_after_relay = tr2.history[relay_s3]
    collimation_err = np.abs(h1_after_relay['dir_y'] - h2_after_relay['dir_y'])
    
    return det_diff + collimation_err*10

res = minimize(test_system, [88.0, 10.0, 11.0], method='Nelder-Mead')
print(f'Optimized! t_dmd_relay = {res.x[0]:.3f}, t_cond_det = {res.x[2]:.3f}')

data['surfaces'][cb_idx]['thickness'] = res.x[0]
data['surfaces'][relay_s3]['thickness'] = res.x[1]
data['surfaces'][cond_idx+1]['thickness'] = res.x[2]

with open('spectrometer_clipped_0.16x.json', 'w') as f:
    json.dump(data, f, indent=4)

