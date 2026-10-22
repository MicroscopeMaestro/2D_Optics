
import sys, os, json, numpy as np
from scipy.optimize import minimize_scalar
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

data = json.load(open('spectrometer_yz_50mm_100mm_optimized.json'))
data['wavelengths'] = [1.0, 1.35, 1.7]
data['primary_wavelength'] = 1.35
data['system_config']['wavelengths'] = [
    {'wavelength_um': 1.0, 'weight': 1.0, 'is_primary': False, 'color': '#0000FF'},
    {'wavelength_um': 1.35, 'weight': 1.0, 'is_primary': True, 'color': '#00FF00'},
    {'wavelength_um': 1.7, 'weight': 1.0, 'is_primary': False, 'color': '#FF0000'}
]
data['surfaces'][13]['grating_lines_per_mm'] = 300.0
data['surfaces'][14]['tilt_x'] = 27.415814682537192

# Enlarge parts
for s in data['surfaces']:
    if 'Detector' not in s['surface_type']:
        s['semi_diameter'] = 25.4

def opt_focus(t):
    data['surfaces'][17]['thickness'] = t
    json.dump(data, open('t_broadband.json', 'w'))
    sys_opt = OpticalSystem.from_json('t_broadband.json')
    sys_opt._update_accumulated_positions()
    r0 = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., 0.0, 1.0]), wavelength=1.35)
    tr0 = sys_opt.trace_ray(r0)
    r1 = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., 0.01, 0.99995]), wavelength=1.35)
    tr1 = sys_opt.trace_ray(r1)
    if len(tr1.history) <= 18: return 1e6
    return np.abs(tr1.history[18]['y_local'] - tr0.history[18]['y_local'])

res_focus = minimize_scalar(opt_focus, bounds=(35.0, 55.0), method='bounded')
data['surfaces'][17]['thickness'] = res_focus.x
json.dump(data, open('t_broadband.json', 'w'))

sys_opt = OpticalSystem.from_json('t_broadband.json')
sys_opt._update_accumulated_positions()

def get_y(wl):
    tr = sys_opt.trace_ray(Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., 0.0, 1.0]), wavelength=wl))
    return tr.history[18]['y_local']

print('1000 nm on DMD:', get_y(1.0))
print('1350 nm on DMD:', get_y(1.35))
print('1700 nm on DMD:', get_y(1.7))
print('DMD Spread with 50mm + 300 l/mm:', np.abs(get_y(1.7) - get_y(1.0)), 'mm')

