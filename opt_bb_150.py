
import sys, os, json, numpy as np
from scipy.optimize import minimize_scalar
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

data = json.load(open('spectrometer_yz_50mm_100mm_optimized.json'))
data['wavelengths'] = [1.0, 1.35, 1.7]
data['primary_wavelength'] = 1.35
data['surfaces'][13]['grating_lines_per_mm'] = 150.0

wl = 1.35
g_dir = np.array([0, np.sin(np.radians(-45)), np.cos(np.radians(-45))])
s_out_y = g_dir[1] + 1 * wl * (150.0 / 1000.0)
s_out_z = np.sqrt(1 - s_out_y**2)
angle = np.degrees(np.arctan2(s_out_y, s_out_z))
data['surfaces'][14]['tilt_x'] = angle - (-45)
print('Tilt X for 150 l/mm:', data['surfaces'][14]['tilt_x'])

for s in data['surfaces']: s['semi_diameter'] = 25.4

def opt_focus(t):
    data['surfaces'][17]['thickness'] = t
    json.dump(data, open('t_broadband_150.json', 'w'))
    sys_opt = OpticalSystem.from_json('t_broadband_150.json')
    sys_opt._update_accumulated_positions()
    r0 = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., 0.0, 1.0]), wavelength=1.35)
    tr0 = sys_opt.trace_ray(r0)
    r1 = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., 0.01, 0.99995]), wavelength=1.35)
    tr1 = sys_opt.trace_ray(r1)
    if len(tr1.history) <= 18: return 1e6
    return np.abs(tr1.history[18]['y_local'] - tr0.history[18]['y_local'])

res_focus = minimize_scalar(opt_focus, bounds=(35.0, 55.0), method='bounded')
data['surfaces'][17]['thickness'] = res_focus.x
json.dump(data, open('t_broadband_150.json', 'w'))

sys_opt = OpticalSystem.from_json('t_broadband_150.json')
sys_opt._update_accumulated_positions()
def get_y(wl):
    tr = sys_opt.trace_ray(Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., 0.0, 1.0]), wavelength=wl))
    return tr.history[18]['y_local']

print('1000 nm on DMD:', get_y(1.0))
print('1350 nm on DMD:', get_y(1.35))
print('1700 nm on DMD:', get_y(1.7))
print('DMD Spread with 50mm + 150 l/mm:', np.abs(get_y(1.7) - get_y(1.0)), 'mm')

