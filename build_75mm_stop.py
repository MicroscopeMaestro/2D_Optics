
import sys, os, json, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json', 'r') as f:
    data = json.load(f)

# 1. Set Aperture Stop to 4 mm diameter (2.0 radius)
stop_idx = next(i for i, s in enumerate(data['surfaces']) if 'Stop' in s.get('name',''))
data['surfaces'][stop_idx]['semi_diameter'] = 2.0

# 2. Change Relay to 75mm (AC254-075-C)
relay_indices = [i for i, s in enumerate(data['surfaces']) if 'Relay' in s.get('name', '')]
# We'll just take the 30mm Focusing radii and multiply by 2.5
focus_indices = [i for i, s in enumerate(data['surfaces']) if 'Focusing' in s.get('name', '')]
for i, r_idx in enumerate(relay_indices):
    f_idx = focus_indices[i]
    r_focus = data['surfaces'][f_idx]['radius']
    if r_focus != 0:
        data['surfaces'][r_idx]['radius'] = r_focus * 2.5
    data['surfaces'][r_idx]['semi_diameter'] = 12.7
    data['surfaces'][r_idx]['name'] = f'Relay_AC254-075-C_S{i+1}'

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
    
    with open('temp_75mm.json', 'w') as f: json.dump(data, f)
    sys_opt = OpticalSystem.from_json('temp_75mm.json')
    sys_opt._update_accumulated_positions()
    
    # Trace PARAXIAL rays
    r1 = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0.001, 0.9999995]), wavelength=1.225)
    r2 = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., -0.001, 0.9999995]), wavelength=1.225)
    tr1 = sys_opt.trace_ray(r1)
    tr2 = sys_opt.trace_ray(r2)
    
    if tr1.vignetted or tr2.vignetted or len(tr1.history) <= det_idx: return 1e6
        
    dmd_diff = np.abs(tr1.history[dmd_idx]['y_local'] - tr2.history[dmd_idx]['y_local'])
    det_diff = np.abs(tr1.history[det_idx]['y_local'] - tr2.history[det_idx]['y_local'])
    h1_after_relay = tr1.history[relay_s3]
    h2_after_relay = tr2.history[relay_s3]
    collimation_err = np.abs(h1_after_relay['dir_y'] - h2_after_relay['dir_y'])
    
    return dmd_diff*5 + det_diff*5 + collimation_err*10

res = minimize(test_system, [30.0, 75.0, 10.0, 11.0], method='Nelder-Mead', bounds=[(20,40), (60,95), (5,30), (5,20)])
print(f'Optimized! {res.x}')

data['surfaces'][focus_s3]['thickness'] = res.x[0]
data['surfaces'][cb_idx]['thickness'] = res.x[1]
data['surfaces'][relay_s3]['thickness'] = res.x[2]
data['surfaces'][cond_idx+1]['thickness'] = res.x[3]

with open('spectrometer_yz_75mm_stop4.json', 'w') as f:
    json.dump(data, f, indent=4)

