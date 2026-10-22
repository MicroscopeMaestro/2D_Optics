
import sys, os, json, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

data = json.load(open('spectrometer_yz_40mm_100mm_final.json'))

def trace_max_radius(x):
    t_dmd, t_bet, t_det = x
    if t_dmd < 100 or t_bet < 5 or t_det < 1: return 1e6
    
    data['surfaces'][19]['thickness'] = t_dmd
    data['surfaces'][22]['thickness'] = t_bet
    data['surfaces'][24]['thickness'] = t_det
    
    json.dump(data, open('t_temp_40_goal.json', 'w'))
    try:
        sys_opt = OpticalSystem.from_json('t_temp_40_goal.json')
        sys_opt._update_accumulated_positions()
        
        max_r = 0
        for wl in [1.064, 1.225, 1.350]:
            r0 = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., 0.0, 1.0]), wavelength=wl)
            tr0 = sys_opt.trace_ray(r0)
            r1 = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., 0.01, 0.99995]), wavelength=wl)
            tr1 = sys_opt.trace_ray(r1)
            r2 = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., -0.01, 0.99995]), wavelength=wl)
            tr2 = sys_opt.trace_ray(r2)
            
            if len(tr1.history) <= 25 or len(tr2.history) <= 25: return 1e6
            
            max_r = max(max_r, abs(tr0.history[-1]['y_local']), abs(tr1.history[-1]['y_local']), abs(tr2.history[-1]['y_local']))
        return max_r
    except:
        return 1e6

res = minimize(trace_max_radius, [150.0, 5.0, 5.0], method='Nelder-Mead', bounds=[(100, 200), (5, 50), (1, 20)])
print('Optimal [t_dmd, t_bet, t_det]:', res.x)

data['surfaces'][19]['thickness'] = res.x[0]
data['surfaces'][22]['thickness'] = res.x[1]
data['surfaces'][24]['thickness'] = res.x[2]
json.dump(data, open('spectrometer_yz_40mm_100mm_optimized.json', 'w'), indent=4)

