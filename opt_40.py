
import sys, os, json, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

data = {
    'surfaces': [
        {'name':'OBJ', 'thickness':1e6, 'radius':0.0, 'glass':'AIR', 'surface_type':'Standard'},
        {'name':'S1', 'radius': 25.0, 'thickness': 7.0, 'glass':'N-BAF10', 'surface_type':'Standard'},
        {'name':'S2', 'radius': -20.0, 'thickness': 2.0, 'glass':'F2', 'surface_type':'Standard'},
        {'name':'S3', 'radius': -75.0, 'thickness': 30.0, 'glass':'AIR', 'surface_type':'Standard'},
        {'name':'IMA', 'radius':0.0, 'thickness':0.0, 'glass':'AIR', 'surface_type':'Detector'}
    ],
    'wavelengths': [1.225], 'primary_wavelength': 1.225
}

def opt(x):
    r1, r2, r3, t = x
    if abs(r1)<10 or abs(r2)<10 or abs(r3)<10: return 1e6
    data['surfaces'][1]['radius'] = r1
    data['surfaces'][2]['radius'] = r2
    data['surfaces'][3]['radius'] = r3
    data['surfaces'][3]['thickness'] = t
    json.dump(data, open('t_40_temp.json', 'w'))
    try:
        sys_opt = OpticalSystem.from_json('t_40_temp.json')
        sys_opt._update_accumulated_positions()
        
        # Marginal ray for spherical aberration
        r1 = Ray(pos=np.array([0., 5.0, -100.]), dir=np.array([0., 0., 1.]), wavelength=1.225)
        tr1 = sys_opt.trace_ray(r1)
        y1 = tr1.history[4]['y_local']
        
        # Paraxial ray for EFL
        r_p = Ray(pos=np.array([0., 0.1, -100.]), dir=np.array([0., 0., 1.]), wavelength=1.225)
        tr_p = sys_opt.trace_ray(r_p)
        y_p = tr_p.history[4]['y_local']
        m = tr_p.history[4]['dir_y'] / tr_p.history[4]['dir_z']
        efl = -tr_p.history[3]['y_local'] / m
        
        return (efl - 40.0)**2 + (y1*10)**2 + (y_p*10)**2
    except:
        return 1e6

res = minimize(opt, [25.0, -20.0, -75.0, 32.0], method='Nelder-Mead', options={'maxiter': 2000})
print('Optimal 40mm:', res.x)

