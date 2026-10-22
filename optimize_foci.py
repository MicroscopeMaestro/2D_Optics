
import sys, os, json, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('teamwork_yz_spectrometer/spectrometer_yz_1064.json', 'r') as f:
    data = json.load(f)

# Find indices
focus_idx = next(i for i, s in enumerate(data['surfaces']) if 'Focusing' in s.get('name','') and 'S3' in s.get('name',''))
dmd_idx = next(i for i, s in enumerate(data['surfaces']) if 'DMD' in s.get('name',''))
relay_idx = next(i for i, s in enumerate(data['surfaces']) if 'Relay' in s.get('name','') and 'S3' in s.get('name',''))
cond_idx = next(i for i, s in enumerate(data['surfaces']) if 'Cond' in s.get('name','') and 'S1' in s.get('name',''))
det_idx = len(data['surfaces']) - 1

def test_system(x):
    t_focus, t_relay, t_cond = x
    data['surfaces'][focus_idx]['thickness'] = t_focus
    data['surfaces'][relay_idx]['thickness'] = t_relay
    data['surfaces'][cond_idx+1]['thickness'] = t_cond # cond_S2 to detector
    
    with open('temp_opt.json', 'w') as f:
        json.dump(data, f)
        
    sys_opt = OpticalSystem.from_json('temp_opt.json')
    sys_opt._update_accumulated_positions()
    
    r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0.01, 0.99995]), wavelength=1.225)
    tr = sys_opt.trace_ray(r)
    
    if tr.vignetted or len(tr.history) <= det_idx:
        return 1e6
        
    h_dmd = tr.history[dmd_idx]
    dmd_error = np.abs(h_dmd['x_local']) + np.abs(h_dmd['y_local'] - 0.0436) # using centroid from earlier
    
    h_det = tr.history[det_idx]
    det_error = np.abs(h_det['x_local']) + np.abs(h_det['y_local'])
    
    # We just want them to be focused! So we trace two rays and minimize the distance between them at the plane!
    r2 = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., -0.01, 0.99995]), wavelength=1.225)
    tr2 = sys_opt.trace_ray(r2)
    
    if tr2.vignetted: return 1e6
    
    dmd_diff = np.abs(tr.history[dmd_idx]['y_local'] - tr2.history[dmd_idx]['y_local'])
    det_diff = np.abs(tr.history[det_idx]['y_local'] - tr2.history[det_idx]['y_local'])
    
    return dmd_diff + det_diff

res = minimize(test_system, [43.5, 100.0, 11.4], method='Nelder-Mead', bounds=[(20, 80), (10, 200), (5, 20)])
print(f'Optimized! t_focus = {res.x[0]:.3f}, t_relay = {res.x[1]:.3f}, t_cond = {res.x[2]:.3f}')

# Save optimized
data['surfaces'][focus_idx]['thickness'] = res.x[0]
data['surfaces'][relay_idx]['thickness'] = res.x[1]
data['surfaces'][cond_idx+1]['thickness'] = res.x[2]

with open('teamwork_yz_spectrometer/spectrometer_yz_1064_opt.json', 'w') as f:
    json.dump(data, f, indent=4)

