
import sys, os, json, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

data = {
    'surfaces': [
        {'name':'OBJ', 'thickness':1e6, 'radius':0.0, 'glass':'AIR', 'surface_type':'Standard'},
        {'name':'S1', 'radius': 50.0, 'thickness': 4.0, 'glass':'N-BAF10', 'surface_type':'Standard'},
        {'name':'S2', 'radius': -40.0, 'thickness': 2.0, 'glass':'F2', 'surface_type':'Standard'},
        {'name':'S3', 'radius': -150.0, 'thickness': 90.0, 'glass':'AIR', 'surface_type':'Standard'},
        {'name':'IMA', 'radius':0.0, 'thickness':0.0, 'glass':'AIR', 'surface_type':'Detector'}
    ],
    'wavelengths': [1.225], 'primary_wavelength': 1.225
}

def opt(x):
    r1, r2, r3, t = x
    data['surfaces'][1]['radius'] = r1
    data['surfaces'][2]['radius'] = r2
    data['surfaces'][3]['radius'] = r3
    data['surfaces'][3]['thickness'] = t
    json.dump(data, open('t_temp_achromat.json', 'w'))
    try:
        sys_opt = OpticalSystem.from_json('t_temp_achromat.json')
        sys_opt._update_accumulated_positions()
        
        # Marginal ray
        r1 = Ray(pos=np.array([0., 10.0, -100.]), dir=np.array([0., 0., 1.]), wavelength=1.225)
        tr1 = sys_opt.trace_ray(r1)
        y1 = tr1.history[4]['y_local']
        
        # Paraxial ray for EFL
        r_p = Ray(pos=np.array([0., 0.1, -100.]), dir=np.array([0., 0., 1.]), wavelength=1.225)
        tr_p = sys_opt.trace_ray(r_p)
        y_p = tr_p.history[4]['y_local']
        m = tr_p.history[4]['dir_y'] / tr_p.history[4]['dir_z']
        efl = -tr_p.history[3]['y_local'] / m
        
        # Cost: efl must be exactly 100.0, spherical aberration must be 0, defocus (y_p) must be 0
        return (efl - 100.0)**2 + (y1*100)**2 + (y_p*100)**2
    except:
        return 1e6

res = minimize(opt, [51.5, -44.2, -148.9, 90.0], method='Nelder-Mead', options={'maxiter':2000})
print('Optimal Forward:', res.x)

r1, r2, r3, t = res.x
data['surfaces'][1]['radius'] = r1
data['surfaces'][2]['radius'] = r2
data['surfaces'][3]['radius'] = r3
data['surfaces'][3]['thickness'] = t
json.dump(data, open('t_100_achromat_forward.json', 'w'))

