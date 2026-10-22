
import sys, os, json, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

data = json.load(open('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json'))
data['surfaces'][6]['semi_diameter'] = 5.0
data['surfaces'][-1]['semi_diameter'] = 5.0

cb_idx = next(i for i, s in enumerate(data['surfaces']) if 'DMD' in s.get('name','')) + 1
cond_idx = next(i for i, s in enumerate(data['surfaces']) if 'Cond' in s.get('name','') and 'S1' in s.get('name',''))
det_idx = len(data['surfaces']) - 1

# Reverse Relay to correct orientation (R1 facing DMD)
relay_indices = [i for i, s in enumerate(data['surfaces']) if 'Relay' in s.get('name', '')]
data['surfaces'][relay_indices[0]]['radius'] = 26.4
data['surfaces'][relay_indices[0]]['thickness'] = 7.6
data['surfaces'][relay_indices[0]]['glass'] = 'N-BAF10'
data['surfaces'][relay_indices[1]]['radius'] = -29.4
data['surfaces'][relay_indices[1]]['thickness'] = 1.8
data['surfaces'][relay_indices[1]]['glass'] = 'N-SF6HT'
data['surfaces'][relay_indices[2]]['radius'] = 84.9
for i in range(3): data['surfaces'][relay_indices[i]]['name'] = f'Relay_AC254-075-C_S{i+1}'

def spot_size(x):
    t_dmd_relay, t_cond_det = x
    data['surfaces'][cb_idx]['thickness'] = t_dmd_relay
    data['surfaces'][cond_idx+1]['thickness'] = t_cond_det
    json.dump(data, open('t_opt.json', 'w'))
    sys_opt = OpticalSystem.from_json('t_opt.json')
    sys_opt._update_accumulated_positions()
    
    r1 = Ray(pos=sys_opt.surfaces[6].global_origin, dir=np.array([0., 0.01, 0.99995]), wavelength=1.225)
    r2 = Ray(pos=sys_opt.surfaces[6].global_origin, dir=np.array([0., -0.01, 0.99995]), wavelength=1.225)
    tr1 = sys_opt.trace_ray(r1)
    tr2 = sys_opt.trace_ray(r2)
    
    if tr1.vignetted or tr2.vignetted or len(tr1.history) <= det_idx: return 1e6
    return np.abs(tr1.history[det_idx]['y_local'] - tr2.history[det_idx]['y_local'])

res = minimize(spot_size, [71.0, 7.3], method='Nelder-Mead', bounds=[(50.0, 90.0), (5.0, 15.0)])
print('Optimum:', res.x)

data['surfaces'][cb_idx]['thickness'] = res.x[0]
data['surfaces'][cond_idx+1]['thickness'] = res.x[1]
data['surfaces'][6]['semi_diameter'] = 0.1
data['surfaces'][-1]['semi_diameter'] = 0.4
with open('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json', 'w') as f: json.dump(data, f, indent=4)

