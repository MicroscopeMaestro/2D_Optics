
import sys, os, json, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('teamwork_yz_spectrometer/spectrometer_yz_1064_tilt5.json', 'r') as f:
    data = json.load(f)

# 1. Change Focusing Lens to AC254-030-C (R1=19.4, R2=-14.2, R3=-64.3, t1=7.0 N-BAF10, t2=2.0 N-SF6HT)
# Wait, the Collimator is already an AC254-030-C. Let's copy its parameters!
collimator_surfs = [s for s in data['surfaces'] if 'Collimator' in s.get('name', '')]
focus_indices = [i for i, s in enumerate(data['surfaces']) if 'Focusing' in s.get('name', '')]
for i, f_idx in enumerate(focus_indices):
    data['surfaces'][f_idx]['radius'] = collimator_surfs[i]['radius']
    data['surfaces'][f_idx]['thickness'] = collimator_surfs[i]['thickness']
    data['surfaces'][f_idx]['glass'] = collimator_surfs[i]['glass']

# 2. Change Relay Lens to AC508-100-C (2-inch, f=100mm)
# R1=63.3, R2=-46.8, R3=-177.3, t1=15.0 N-BAF10, t2=4.0 N-SF6HT, D=50.8
relay_indices = [i for i, s in enumerate(data['surfaces']) if 'Relay' in s.get('name', '')]
r_radii = [63.3, -46.8, -177.3]
r_thick = [15.0, 4.0, 100.0]
r_glass = ['N-BAF10', 'N-SF6HT', 'AIR']
for i, r_idx in enumerate(relay_indices):
    data['surfaces'][r_idx]['radius'] = r_radii[i]
    data['surfaces'][r_idx]['thickness'] = r_thick[i]
    data['surfaces'][r_idx]['glass'] = r_glass[i]
    data['surfaces'][r_idx]['semi_diameter'] = 25.4 # 2-inch lens
    data['surfaces'][r_idx]['name'] = f'Relay_AC508-100-C_S{i+1}'

# 3. Optimize Focal Distances
focus_s3 = focus_indices[-1]
dmd_idx = next(i for i, s in enumerate(data['surfaces']) if 'DMD' in s.get('name',''))
relay_s3 = relay_indices[-1]
cond_idx = next(i for i, s in enumerate(data['surfaces']) if 'Cond' in s.get('name','') and 'S1' in s.get('name',''))
det_idx = len(data['surfaces']) - 1

def test_system(x):
    t_focus, t_relay, t_cond = x
    data['surfaces'][focus_s3]['thickness'] = t_focus
    data['surfaces'][relay_s3]['thickness'] = t_relay
    data['surfaces'][cond_idx+1]['thickness'] = t_cond
    
    with open('temp_clipped.json', 'w') as f: json.dump(data, f)
    sys_opt = OpticalSystem.from_json('temp_clipped.json')
    sys_opt._update_accumulated_positions()
    
    r1 = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0.01, 0.99995]), wavelength=1.225)
    r2 = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., -0.01, 0.99995]), wavelength=1.225)
    tr1 = sys_opt.trace_ray(r1)
    tr2 = sys_opt.trace_ray(r2)
    
    if tr1.vignetted or tr2.vignetted or len(tr1.history) <= det_idx:
        return 1e6
        
    dmd_diff = np.abs(tr1.history[dmd_idx]['y_local'] - tr2.history[dmd_idx]['y_local'])
    det_diff = np.abs(tr1.history[det_idx]['y_local'] - tr2.history[det_idx]['y_local'])
    
    return dmd_diff*10 + det_diff

res = minimize(test_system, [23.0, 10.0, 11.0], method='Nelder-Mead', bounds=[(15, 35), (5, 50), (5, 20)])
print(f'Optimized! t_focus = {res.x[0]:.3f}, t_relay = {res.x[1]:.3f}, t_cond = {res.x[2]:.3f}')

data['surfaces'][focus_s3]['thickness'] = res.x[0]
data['surfaces'][relay_s3]['thickness'] = res.x[1]
data['surfaces'][cond_idx+1]['thickness'] = res.x[2]
# Fix the distance between DMD and Relay! We didn't optimize it, it's tied to CB_After_DMD thickness
cb_idx = dmd_idx + 1
data['surfaces'][cb_idx]['thickness'] = 89.0 # Approximate for 100mm lens back focal length

with open('spectrometer_clipped_0.16x.json', 'w') as f:
    json.dump(data, f, indent=4)

