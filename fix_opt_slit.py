
import sys, os, json, numpy as np
from scipy.optimize import minimize_scalar
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

data = json.load(open('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json'))
data['surfaces'][6]['semi_diameter'] = 5.0
data['surfaces'][-1]['semi_diameter'] = 5.0

# 1. Optimize Collimator
def collimate(t):
    data['surfaces'][6]['thickness'] = t
    json.dump(data, open('t_temp.json', 'w'))
    sys_opt = OpticalSystem.from_json('t_temp.json')
    sys_opt._update_accumulated_positions()
    
    # Ray originating exactly at the SLIT (index 6)
    r0 = Ray(pos=sys_opt.surfaces[6].global_origin, dir=np.array([0., 0.0, 1.0]), wavelength=1.225)
    r1 = Ray(pos=sys_opt.surfaces[6].global_origin, dir=np.array([0., 0.01, 0.99995]), wavelength=1.225)
    
    # To trace from surface 6, we can just trace from z=0 and override pos?
    # Actually, OpticalSystem.trace_ray automatically starts from surface 0.
    # We can just manually trace surface by surface, or set the first 6 surfaces to AIR/thickness=0!
    pass

# Easiest way to trace from Slit:
def get_col_dir(t):
    data['surfaces'][6]['thickness'] = t
    json.dump(data, open('t_temp.json', 'w'))
    sys_opt = OpticalSystem.from_json('t_temp.json')
    sys_opt._update_accumulated_positions()
    
    # We trace a ray that happens to pass through the center of the slit at an angle.
    # A ray from the object that crosses y=0 at the slit.
    # To do this, we can just manually push a ray into surface 7!
    r = Ray(pos=sys_opt.surfaces[6].global_origin, dir=np.array([0., 0.01, 0.99995]), wavelength=1.225)
    
    # manual propagation from surface 7 onwards
    tr = sys_opt.trace_ray(r) # wait, trace_ray traces from 0 and checks intersections. It will intersect surface 0!
    return tr

# Let's just create a sub-system from Slit to Aperture Stop!
sub_surfaces = data['surfaces'][6:13] # Slit to Aperture Stop
sub_data = {
    'surfaces': sub_surfaces,
    'wavelengths': [1.225], 'primary_wavelength': 1.225
}

def collimate_sub(t):
    sub_surfaces[0]['thickness'] = t
    json.dump(sub_data, open('t_sub.json', 'w'))
    sys_opt = OpticalSystem.from_json('t_sub.json')
    sys_opt._update_accumulated_positions()
    
    r0 = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., 0.0, 1.0]), wavelength=1.225)
    r1 = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., 0.01, 0.99995]), wavelength=1.225)
    tr0 = sys_opt.trace_ray(r0)
    tr1 = sys_opt.trace_ray(r1)
    
    if tr1.vignetted: return 1e6
    return np.abs(tr1.history[6]['dir_y'] - tr0.history[6]['dir_y'])

res_col = minimize_scalar(collimate_sub, bounds=(20.0, 30.0), method='bounded')
print('t_slit_collimator =', res_col.x)
data['surfaces'][6]['thickness'] = res_col.x

