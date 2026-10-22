
import sys, os, json, numpy as np
from scipy.optimize import minimize, minimize_scalar
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

# Load the base optimized model
data = json.load(open('spectrometer_yz_40mm_100mm_optimized.json'))

# Restore 50mm Focusing Lens
data['surfaces'][15]['name'] = 'Focus_AC254-050-C_S1'
data['surfaces'][15]['radius'] = 34.5
data['surfaces'][15]['thickness'] = 7.0
data['surfaces'][15]['glass'] = 'N-BAF10'

data['surfaces'][16]['name'] = 'Focus_AC254-050-C_S2'
data['surfaces'][16]['radius'] = -23.1
data['surfaces'][16]['thickness'] = 2.0
data['surfaces'][16]['glass'] = 'N-SF6HT'

data['surfaces'][17]['name'] = 'Focus_AC254-050-C_S3'
data['surfaces'][17]['radius'] = -135.2

# 1. Optimize Focus to DMD distance
def opt_focus(t):
    data['surfaces'][17]['thickness'] = t
    json.dump(data, open('t_temp_50_goal.json', 'w'))
    sys_opt = OpticalSystem.from_json('t_temp_50_goal.json')
    sys_opt._update_accumulated_positions()
    
    r0 = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., 0.0, 1.0]), wavelength=1.225)
    r1 = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., 0.01, 0.99995]), wavelength=1.225)
    tr0 = sys_opt.trace_ray(r0)
    tr1 = sys_opt.trace_ray(r1)
    if len(tr1.history) <= 18: return 1e6
    return np.abs(tr1.history[18]['y_local'] - tr0.history[18]['y_local'])

res_foc = minimize_scalar(opt_focus, bounds=(35.0, 55.0), method='bounded')
print('Optimal t_focus =', res_foc.x)
data['surfaces'][17]['thickness'] = res_foc.x

# 2. Optimize Relay distances for the Pupil placement
def trace_max_radius(x):
    t_dmd, t_bet, t_det = x
    if t_dmd < 10 or t_bet < 5 or t_det < 1: return 1e6
    
    data['surfaces'][19]['thickness'] = t_dmd
    data['surfaces'][22]['thickness'] = t_bet
    data['surfaces'][24]['thickness'] = t_det
    
    json.dump(data, open('t_temp_50_goal.json', 'w'))
    try:
        sys_opt = OpticalSystem.from_json('t_temp_50_goal.json')
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

res_relay = minimize(trace_max_radius, [150.0, 5.0, 5.0], method='Nelder-Mead', options={'maxiter': 2000})
print('Optimal [t_dmd, t_bet, t_det]:', res_relay.x)
print('Minimum Max Radius on Detector:', res_relay.fun)

data['surfaces'][19]['thickness'] = res_relay.x[0]
data['surfaces'][22]['thickness'] = res_relay.x[1]
data['surfaces'][24]['thickness'] = res_relay.x[2]
json.dump(data, open('spectrometer_yz_50mm_100mm_optimized.json', 'w'), indent=4)

