
import sys, os, json, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

data = {
    'surfaces': [
        {'name':'OBJ', 'thickness':1e6, 'radius':0.0, 'glass':'AIR', 'surface_type':'Standard'},
        {'name':'S1', 'radius': 50.0, 'thickness': 4.0, 'glass':'N-BAF10', 'surface_type':'Standard'},
        {'name':'S2', 'radius': -40.0, 'thickness': 2.0, 'glass':'N-SF6HT', 'surface_type':'Standard'},
        {'name':'S3', 'radius': -150.0, 'thickness': 90.0, 'glass':'AIR', 'surface_type':'Standard'},
        {'name':'IMA', 'radius':0.0, 'thickness':0.0, 'glass':'AIR', 'surface_type':'Detector'}
    ],
    'wavelengths': [1.225], 'primary_wavelength': 1.225
}

def obj(x):
    r1, r2, r3, t = x
    if abs(r1)<10 or abs(r2)<10 or abs(r3)<10: return 1e6
    data['surfaces'][1]['radius'] = r1
    data['surfaces'][2]['radius'] = r2
    data['surfaces'][3]['radius'] = r3
    data['surfaces'][3]['thickness'] = t
    json.dump(data, open('t_100_v3.json', 'w'))
    try:
        sys_opt = OpticalSystem.from_json('t_100_v3.json')
        sys_opt._update_accumulated_positions()
        # Find BFL (t)
        r0 = Ray(pos=np.array([0., 2.0, -100.]), dir=np.array([0., 0., 1.]), wavelength=1.225)
        tr0 = sys_opt.trace_ray(r0)
        y0 = tr0.history[4]['y_local']
        
        # Spherical aberration
        r1 = Ray(pos=np.array([0., 10.0, -100.]), dir=np.array([0., 0., 1.]), wavelength=1.225)
        tr1 = sys_opt.trace_ray(r1)
        y1 = tr1.history[4]['y_local']
        
        # Calculate EFL
        m = tr0.history[4]['dir_y'] / tr0.history[4]['dir_z']
        efl = -tr0.history[3]['y_local'] / m
        
        return (efl - 100.0)**2 + (y0*10)**2 + (y1*100)**2
    except:
        return 1e6

res = minimize(obj, [50.0, -30.0, -100.0, 95.0], method='Nelder-Mead')
print(res.x)
data['surfaces'][1]['radius'] = res.x[0]
data['surfaces'][2]['radius'] = res.x[1]
data['surfaces'][3]['radius'] = res.x[2]
data['surfaces'][3]['thickness'] = res.x[3]
json.dump(data, open('t_100_final.json', 'w'))

