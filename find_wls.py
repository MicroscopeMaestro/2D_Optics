
import sys, os, json, numpy as np
from scipy.optimize import minimize_scalar
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

sys_opt = OpticalSystem.from_json('spectrometer_yz_40mm_100mm_final.json')
sys_opt._update_accumulated_positions()

def get_y_dmd(wl):
    r = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., 0.0, 1.0]), wavelength=wl)
    tr = sys_opt.trace_ray(r)
    return tr.history[18]['y_local']

print('1225 nm y_local =', get_y_dmd(1.225))
print('1064 nm y_local =', get_y_dmd(1.064))
print('1350 nm y_local =', get_y_dmd(1.350))

wl_top = minimize_scalar(lambda w: np.abs(get_y_dmd(w) - 2.5), bounds=(1.2, 1.6), method='bounded').x
wl_bot = minimize_scalar(lambda w: np.abs(get_y_dmd(w) - (-2.5)), bounds=(0.8, 1.2), method='bounded').x

print('Wavelength for +2.5mm:', wl_top)
print('Wavelength for -2.5mm:', wl_bot)

