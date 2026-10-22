
import sys, os, json, numpy as np
from scipy.optimize import minimize_scalar
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

data = {
    'surfaces': [
        {'name':'OBJ', 'thickness':1e6, 'radius':0.0, 'glass':'AIR', 'surface_type':'Standard'},
        {'name':'S1', 'radius': 113.6, 'thickness': 4.0, 'glass':'N-BAF10', 'surface_type':'Standard'},
        {'name':'S2', 'radius': -82.6, 'thickness': 2.0, 'glass':'N-SF6HT', 'surface_type':'Standard'},
        {'name':'S3', 'radius': -444.4, 'thickness': 30.0, 'glass':'AIR', 'surface_type':'Standard'},
        {'name':'IMA', 'radius':0.0, 'thickness':0.0, 'glass':'AIR', 'surface_type':'Detector'}
    ],
    'wavelengths': [1.225], 'primary_wavelength': 1.225
}

def opt(scale):
    data['surfaces'][1]['radius'] = 51.5 * scale
    data['surfaces'][2]['radius'] = -44.2 * scale
    data['surfaces'][3]['radius'] = -270.4 * scale
    json.dump(data, open('t_temp_scale.json', 'w'))
    try:
        sys_opt = OpticalSystem.from_json('t_temp_scale.json')
        sys_opt._update_accumulated_positions()
        
        r_p = Ray(pos=np.array([0., 0.1, -100.]), dir=np.array([0., 0., 1.]), wavelength=1.225)
        tr_p = sys_opt.trace_ray(r_p)
        m = tr_p.history[4]['dir_y'] / tr_p.history[4]['dir_z']
        efl = -tr_p.history[3]['y_local'] / m
        return np.abs(efl - 200.0)
    except:
        return 1e6

res = minimize_scalar(opt, bounds=(2.0, 5.0), method='bounded')
print('Optimal scale:', res.x)
print('R1:', 51.5 * res.x)
print('R2:', -44.2 * res.x)
print('R3:', -270.4 * res.x)

