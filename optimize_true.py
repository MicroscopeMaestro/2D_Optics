
import sys, os, json, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json', 'r') as f:
    data = json.load(f)

# Widen slit during optimization so rays from Z=0 don't clip there
data['surfaces'][6]['semi_diameter'] = 5.0

focus_indices = [i for i, s in enumerate(data['surfaces']) if 'Focusing' in s.get('name', '')]
focus_s3 = focus_indices[-1]
dmd_idx = next(i for i, s in enumerate(data['surfaces']) if 'DMD' in s.get('name',''))
cb_idx = dmd_idx + 1
relay_indices = [i for i, s in enumerate(data['surfaces']) if 'Relay' in s.get('name', '')]
relay_s3 = relay_indices[-1]
cond_idx = next(i for i, s in enumerate(data['surfaces']) if 'Cond' in s.get('name','') and 'S1' in s.get('name',''))
det_idx = len(data['surfaces']) - 1

def test_system(x):
    t_focus, t_dmd_relay, t_relay_cond, t_cond_det = x
    
    data['surfaces'][focus_s3]['thickness'] = t_focus
    data['surfaces'][cb_idx]['thickness'] = t_dmd_relay
    data['surfaces'][relay_s3]['thickness'] = t_relay_cond
    data['surfaces'][cond_idx+1]['thickness'] = t_cond_det
    
    with open('temp_opt_true.json', 'w') as f: json.dump(data, f)
    sys_opt = OpticalSystem.from_json('temp_opt_true.json')
    sys_opt._update_accumulated_positions()
    
    wls = [1.087, 1.225, 1.350]
    total_error = 0
    
    for wl in wls:
        r1 = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0.01, 0.99995]), wavelength=wl)
        r2 = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., -0.01, 0.99995]), wavelength=wl)
        tr1 = sys_opt.trace_ray(r1)
        tr2 = sys_opt.trace_ray(r2)
        
        if tr1.vignetted or tr2.vignetted or len(tr1.history) <= det_idx:
            return 1e6
            
        # Spot size on DMD should be zero
        dmd_y1 = tr1.history[dmd_idx]['y_local']
        dmd_y2 = tr2.history[dmd_idx]['y_local']
        dmd_spot = np.abs(dmd_y1 - dmd_y2)
        
        # Spot size on Detector should be zero
        det_y1 = tr1.history[det_idx]['y_local']
        det_y2 = tr2.history[det_idx]['y_local']
        det_spot = np.abs(det_y1 - det_y2)
        
        # Target: perfectly focused on DMD and Detector
        total_error += dmd_spot * 10.0 + det_spot * 10.0
        
    return total_error

# Good initial guess based on theoretical focal lengths
x0 = [30.1, 75.0, 10.0, 8.7]
bounds = [(29.5, 31.0), (70.0, 85.0), (5.0, 20.0), (6.0, 12.0)]

res = minimize(test_system, x0, method='Nelder-Mead', bounds=bounds, options={'maxiter': 500, 'xatol': 1e-4})
print(f'Optimized! {res.x} Error={res.fun}')

# Re-tighten slit
data['surfaces'][6]['semi_diameter'] = 0.1

data['surfaces'][focus_s3]['thickness'] = res.x[0]
data['surfaces'][cb_idx]['thickness'] = res.x[1]
data['surfaces'][relay_s3]['thickness'] = res.x[2]
data['surfaces'][cond_idx+1]['thickness'] = res.x[3]

with open('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json', 'w') as f:
    json.dump(data, f, indent=4)

