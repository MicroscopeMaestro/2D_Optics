
import sys, os, json, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('teamwork_yz_spectrometer/spectrometer_yz_1064_tilt5.json', 'r') as f:
    data = json.load(f)

# 1. Change Focusing to 30mm (AC254-030-C)
collimator_surfs = [s for s in data['surfaces'] if 'Collimator' in s.get('name', '')]
focus_indices = [i for i, s in enumerate(data['surfaces']) if 'Focusing' in s.get('name', '')]
for i, f_idx in enumerate(focus_indices):
    data['surfaces'][f_idx]['radius'] = collimator_surfs[i]['radius']
    data['surfaces'][f_idx]['thickness'] = collimator_surfs[i]['thickness']
    data['surfaces'][f_idx]['glass'] = collimator_surfs[i]['glass']
    data['surfaces'][f_idx]['name'] = f'Focusing_AC254-030-C_S{i+1}'

# 2. Change Relay to 100mm (AC254-100-C)
relay_indices = [i for i, s in enumerate(data['surfaces']) if 'Relay' in s.get('name', '')]
for i, r_idx in enumerate(relay_indices):
    orig_r = data['surfaces'][r_idx]['radius']
    if orig_r != 0: data['surfaces'][r_idx]['radius'] = orig_r * 2.0
    data['surfaces'][r_idx]['semi_diameter'] = 12.7 # 1-inch
    data['surfaces'][r_idx]['name'] = f'Relay_AC254-100-C_S{i+1}'

# 3. Optimize Focal Distances
focus_s3 = focus_indices[-1]
dmd_idx = next(i for i, s in enumerate(data['surfaces']) if 'DMD' in s.get('name',''))
cb_idx = dmd_idx + 1
relay_s3 = relay_indices[-1]
cond_idx = next(i for i, s in enumerate(data['surfaces']) if 'Cond' in s.get('name','') and 'S1' in s.get('name',''))
det_idx = len(data['surfaces']) - 1

def test_system(x):
    t_focus, t_dmd_relay, t_relay_cond, t_cond_det = x
    data['surfaces'][focus_s3]['thickness'] = t_focus
    data['surfaces'][cb_idx]['thickness'] = t_dmd_relay
    data['surfaces'][relay_s3]['thickness'] = t_relay_cond
    data['surfaces'][cond_idx+1]['thickness'] = t_cond_det
    
    with open('temp_1inch.json', 'w') as f: json.dump(data, f)
    sys_opt = OpticalSystem.from_json('temp_1inch.json')
    sys_opt._update_accumulated_positions()
    
    # Trace PARAXIAL rays to avoid clipping on the 1-inch lens!
    r1 = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0.001, 0.9999995]), wavelength=1.225)
    r2 = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., -0.001, 0.9999995]), wavelength=1.225)
    tr1 = sys_opt.trace_ray(r1)
    tr2 = sys_opt.trace_ray(r2)
    
    if tr1.vignetted or tr2.vignetted or len(tr1.history) <= det_idx:
        return 1e6
        
    dmd_diff = np.abs(tr1.history[dmd_idx]['y_local'] - tr2.history[dmd_idx]['y_local'])
    det_diff = np.abs(tr1.history[det_idx]['y_local'] - tr2.history[det_idx]['y_local'])
    h1_after_relay = tr1.history[relay_s3]
    h2_after_relay = tr2.history[relay_s3]
    collimation_err = np.abs(h1_after_relay['dir_y'] - h2_after_relay['dir_y'])
    
    return dmd_diff*5 + det_diff*5 + collimation_err*10

res = minimize(test_system, [28.0, 95.0, 10.0, 11.0], method='Nelder-Mead', bounds=[(20,40), (80,120), (5,30), (5,20)])
print(f'Optimized! {res.x}')

data['surfaces'][focus_s3]['thickness'] = res.x[0]
data['surfaces'][cb_idx]['thickness'] = res.x[1]
data['surfaces'][relay_s3]['thickness'] = res.x[2]
data['surfaces'][cond_idx+1]['thickness'] = res.x[3]

with open('spectrometer_yz_1inch_100mm.json', 'w') as f:
    json.dump(data, f, indent=4)

