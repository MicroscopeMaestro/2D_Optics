
import sys, os, json, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json', 'r') as f:
    data = json.load(f)

# Change Collimator to 1/2 inch
collimator_indices = [i for i, s in enumerate(data['surfaces']) if 'Collimator' in s.get('name', '')]
for i, c_idx in enumerate(collimator_indices):
    data['surfaces'][c_idx]['semi_diameter'] = 6.35
    data['surfaces'][c_idx]['name'] = f'Collimator_AC127-030-C_S{i+1}'

# Save a copy before optimization
with open('temp_goal.json', 'w') as f: json.dump(data, f)

# Optimize distances
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
    
    with open('temp_opt_goal.json', 'w') as f: json.dump(data, f)
    sys_opt = OpticalSystem.from_json('temp_opt_goal.json')
    sys_opt._update_accumulated_positions()
    
    slit_pos = sys_opt.surfaces[6].global_origin
    slit_R = sys_opt.surfaces[6].global_R
    
    r1 = Ray(pos=slit_pos, dir=slit_R @ np.array([0., 0.001, 0.9999995]), wavelength=1.225)
    r2 = Ray(pos=slit_pos, dir=slit_R @ np.array([0., -0.001, 0.9999995]), wavelength=1.225)
    
    # manual trace
    hx1, hx2 = [], []
    r_curr1 = Ray(pos=r1.pos, dir=r1.dir, wavelength=1.225)
    r_curr2 = Ray(pos=r2.pos, dir=r2.dir, wavelength=1.225)
    
    for i in range(7, len(sys_opt.surfaces)):
        surf = sys_opt.surfaces[i]
        
        # Ray 1
        pos_loc = surf.global_R.T @ (r_curr1.pos - surf.global_origin)
        dir_loc = surf.global_R.T @ r_curr1.dir
        if dir_loc[2] == 0: return 1e6
        t = -pos_loc[2]/dir_loc[2]
        if t < 0: return 1e6
        r_curr1.pos = surf.global_origin + surf.global_R @ (pos_loc + t * dir_loc)
        hx1.append(r_curr1.pos.copy())
        
        # Ray 2
        pos_loc = surf.global_R.T @ (r_curr2.pos - surf.global_origin)
        dir_loc = surf.global_R.T @ r_curr2.dir
        if dir_loc[2] == 0: return 1e6
        t = -pos_loc[2]/dir_loc[2]
        if t < 0: return 1e6
        r_curr2.pos = surf.global_origin + surf.global_R @ (pos_loc + t * dir_loc)
        hx2.append(r_curr2.pos.copy())
        
    # measure spot size
    p1 = sys_opt.surfaces[det_idx].global_R.T @ (r_curr1.pos - sys_opt.surfaces[det_idx].global_origin)
    p2 = sys_opt.surfaces[det_idx].global_R.T @ (r_curr2.pos - sys_opt.surfaces[det_idx].global_origin)
    det_diff = np.abs(p1[1] - p2[1])
    
    # measure dmd spot size
    dmd_p1 = sys_opt.surfaces[dmd_idx].global_R.T @ (hx1[dmd_idx-7] - sys_opt.surfaces[dmd_idx].global_origin)
    dmd_p2 = sys_opt.surfaces[dmd_idx].global_R.T @ (hx2[dmd_idx-7] - sys_opt.surfaces[dmd_idx].global_origin)
    dmd_diff = np.abs(dmd_p1[1] - dmd_p2[1])
    
    return dmd_diff*5 + det_diff*5

res = minimize(test_system, [30.12, 75.0, 10.6, 8.67], method='Nelder-Mead', bounds=[(20,40), (60,90), (5,30), (5,20)])
print(f'Optimized! {res.x}')

data['surfaces'][focus_s3]['thickness'] = res.x[0]
data['surfaces'][cb_idx]['thickness'] = res.x[1]
data['surfaces'][relay_s3]['thickness'] = res.x[2]
data['surfaces'][cond_idx+1]['thickness'] = res.x[3]

with open('spectrometer_yz_goal_final.json', 'w') as f:
    json.dump(data, f, indent=4)

