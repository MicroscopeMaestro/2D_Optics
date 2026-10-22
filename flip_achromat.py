
import sys, os, json, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

data = json.load(open('C:/Users/Juan/.gemini/antigravity/brain/49fd158d-b0e1-421c-b15b-b237b2e3ad7f/spectrometer_yz_100mm_achromat.json'))

# Flip it back to FORWARD!
data['surfaces'][20]['name'] = 'Relay_AC254-100-C_S1'
data['surfaces'][20]['radius'] = 51.5
data['surfaces'][20]['thickness'] = 4.0
data['surfaces'][20]['glass'] = 'N-BAF10'

data['surfaces'][21]['name'] = 'Relay_AC254-100-C_S2'
data['surfaces'][21]['radius'] = -44.2
data['surfaces'][21]['thickness'] = 2.0
data['surfaces'][21]['glass'] = 'N-SF6HT'

data['surfaces'][22]['name'] = 'Relay_AC254-100-C_S3'
data['surfaces'][22]['radius'] = -270.4
data['surfaces'][22]['thickness'] = 20.0
data['surfaces'][22]['glass'] = 'AIR'

def opt_both(x):
    t_relay, t_det = x
    data['surfaces'][19]['thickness'] = t_relay
    data['surfaces'][24]['thickness'] = t_det
    json.dump(data, open('t_temp_flip.json', 'w'))
    try:
        sys_opt = OpticalSystem.from_json('t_temp_flip.json')
        sys_opt._update_accumulated_positions()
        
        r0 = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., 0.0, 1.0]), wavelength=1.225)
        r1 = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., 0.01, 0.99995]), wavelength=1.225)
        tr0 = sys_opt.trace_ray(r0)
        tr1 = sys_opt.trace_ray(r1)
        if len(tr1.history) <= 25: return 1e6
        
        err = np.abs(tr1.history[-1]['y_local'] - tr0.history[-1]['y_local'])
        return err + np.abs(t_relay - 95.0)*1e-4
    except Exception as e:
        return 1e6

res = minimize(opt_both, [95.0, 10.0], method='Nelder-Mead')
print('Optimal t_relay, t_det:', res.x)
data['surfaces'][19]['thickness'] = res.x[0]
data['surfaces'][24]['thickness'] = res.x[1]
json.dump(data, open('spectrometer_yz_100mm_achromat_flipped.json', 'w'), indent=4)

# Measure spherical aberration!
sys_opt = OpticalSystem.from_json('spectrometer_yz_100mm_achromat_flipped.json')
sys_opt._update_accumulated_positions()
r0 = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., 0.0, 1.0]), wavelength=1.225)
tr0 = sys_opt.trace_ray(r0)
r1 = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., 0.01, 0.99995]), wavelength=1.225)
tr1 = sys_opt.trace_ray(r1)
print('Error flipped:', np.abs(tr1.history[-1]['y_local'] - tr0.history[-1]['y_local']))

