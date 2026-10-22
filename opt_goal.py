
import sys, os, json, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

data = json.load(open('spectrometer_yz_40mm_100mm_final.json'))

def trace_max_radius(x):
    t_dmd, t_bet, t_det = x
    if t_dmd < 10 or t_bet < 5 or t_det < 1: return 1e6
    
    data['surfaces'][19]['thickness'] = t_dmd
    data['surfaces'][22]['thickness'] = t_bet
    data['surfaces'][24]['thickness'] = t_det
    
    json.dump(data, open('t_temp_goal.json', 'w'))
    try:
        sys_opt = OpticalSystem.from_json('t_temp_goal.json')
        sys_opt._update_accumulated_positions()
        
        max_r = 0
        for wl in [1.064, 1.225, 1.350]:
            # Chief ray
            r0 = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., 0.0, 1.0]), wavelength=wl)
            tr0 = sys_opt.trace_ray(r0)
            if len(tr0.history) <= 25: return 1e6
            
            # Marginal rays (top and bottom of NA)
            r1 = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., 0.01, 0.99995]), wavelength=wl)
            tr1 = sys_opt.trace_ray(r1)
            
            r2 = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., -0.01, 0.99995]), wavelength=wl)
            tr2 = sys_opt.trace_ray(r2)
            
            if len(tr1.history) <= 25 or len(tr2.history) <= 25: return 1e6
            
            y0 = tr0.history[-1]['y_local']
            y1 = tr1.history[-1]['y_local']
            y2 = tr2.history[-1]['y_local']
            
            # Find the absolute maximum deviation from the center of the detector
            max_r = max(max_r, abs(y0), abs(y1), abs(y2))
            
        return max_r
    except Exception as e:
        return 1e6

res = minimize(trace_max_radius, [95.0, 20.0, 8.13], method='Nelder-Mead', options={'maxiter': 2000})
print('Optimal [t_dmd, t_bet, t_det]:', res.x)
print('Minimum Max Radius on Detector:', res.fun)

