
import sys, os, json, numpy as np
from scipy.optimize import minimize_scalar
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

data = json.load(open('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json'))

def opt_relay(t):
    data['surfaces'][19]['thickness'] = t
    json.dump(data, open('t_temp.json', 'w'))
    sys_opt = OpticalSystem.from_json('t_temp.json')
    sys_opt._update_accumulated_positions()
    
    r1 = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., 0.01, 0.99995]), wavelength=1.225)
    tr1 = sys_opt.trace_ray(r1)
    
    if len(tr1.history) <= 23: return 1e6
    
    # local dir at Condenser S1
    global_dir = np.array([tr1.history[23]['dir_x'], tr1.history[23]['dir_y'], tr1.history[23]['dir_z']])
    local_dir = sys_opt.surfaces[23].global_R.T @ global_dir
    
    return np.abs(local_dir[1])

res = minimize_scalar(opt_relay, bounds=(70.0, 100.0), method='bounded')
print('t_dmd_relay =', res.x)
data['surfaces'][19]['thickness'] = res.x
json.dump(data, open('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json', 'w'), indent=4)

