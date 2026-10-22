
import sys, os, json, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

data = json.load(open('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json'))
data['surfaces'][6]['semi_diameter'] = 5.0
data['surfaces'][-1]['semi_diameter'] = 5.0

def opt(x):
    t_relay, t_det = x
    data['surfaces'][19]['thickness'] = t_relay
    data['surfaces'][24]['thickness'] = t_det
    json.dump(data, open('t_temp.json', 'w'))
    try:
        sys_opt = OpticalSystem.from_json('t_temp.json')
        sys_opt._update_accumulated_positions()
        r0 = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., 0.0, 1.0]), wavelength=1.225)
        r1 = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., 0.01, 0.99995]), wavelength=1.225)
        tr0 = sys_opt.trace_ray(r0)
        tr1 = sys_opt.trace_ray(r1)
        if len(tr1.history) <= 25: return 1e6
        
        # We want perfectly focused spot at detector
        spot_error = np.abs(tr1.history[-1]['y_local'] - tr0.history[-1]['y_local'])
        
        # We want the ray to be completely collimated at Condenser S1
        # global_dir = np.array([tr1.history[23]['dir_x'], tr1.history[23]['dir_y'], tr1.history[23]['dir_z']])
        # local_dir = sys_opt.surfaces[23].global_R.T @ global_dir
        # coll_error = np.abs(local_dir[1])
        # Actually, let's just trace a ray from the edge of the object to see if the chief ray is collimated?
        # A simpler way: we know it's a 100mm lens. The BFL is exactly 100 - 3.6/1.5 = 97.6! 
        return spot_error
    except:
        return 1e6

res = minimize(opt, [96.4, 11.45], method='Nelder-Mead')
print('Optimal t_relay, t_det:', res.x)
data['surfaces'][19]['thickness'] = res.x[0]
data['surfaces'][24]['thickness'] = res.x[1]
data['surfaces'][6]['semi_diameter'] = 0.1
data['surfaces'][-1]['semi_diameter'] = 0.4
json.dump(data, open('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json', 'w'), indent=4)

