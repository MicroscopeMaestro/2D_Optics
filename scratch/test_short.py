import sys
import os
import json
from scipy.optimize import minimize_scalar

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
from optics_app.engine.tracer import OpticalSystem, Ray
import tempfile

with open('spectrometer_yz_40mm_100mm_slit.json') as f:
    config = json.load(f)

# Use original optics
# Focus
config['surfaces'][15]['radius'] = 34.5
config['surfaces'][15]['thickness'] = 12.0
config['surfaces'][15]['glass'] = 'N-BAK4'
config['surfaces'][16]['radius'] = -25.2
config['surfaces'][16]['thickness'] = 2.0
config['surfaces'][16]['glass'] = 'N-SF6HT'
config['surfaces'][17]['radius'] = -135.2
config['surfaces'][17]['glass'] = 'AIR'
config['surfaces'][17]['thickness'] = 43.80

config['surfaces'][18]['thickness'] = 97.03
config['surfaces'][22]['thickness'] = 5.0

def evaluate(t):
    config['surfaces'][24]['thickness'] = t
    with tempfile.NamedTemporaryFile('w', delete=False) as tf:
        json.dump(config, tf)
        tmp = tf.name
    sys_temp = OpticalSystem.from_json(tmp)
    r = Ray(pos=(0, 0.005, 0), dir=(0, 0, 1.0), wavelength=1.225)
    traced = sys_temp.trace_ray(r)
    y_marg = traced.history[-1]['y_global']
    r2 = Ray(pos=(0, 0, 0), dir=(0, 0, 1.0), wavelength=1.225)
    traced2 = sys_temp.trace_ray(r2)
    y_chief = traced2.history[-1]['y_global']
    os.remove(tmp)
    return abs(y_marg - y_chief)

res = minimize_scalar(evaluate, bounds=(5, 30), method='bounded')
config['surfaces'][24]['thickness'] = res.x

with tempfile.NamedTemporaryFile('w', delete=False) as tf:
    json.dump(config, tf)
    tmp = tf.name

sys_temp = OpticalSystem.from_json(tmp)
y_chiefs = []
for wl in [1.087, 1.225, 1.35]:
    r = Ray(pos=(0,0,0), dir=(0, 0, 1.0), wavelength=wl)
    y_chiefs.append(sys_temp.trace_ray(r).history[-1]['y_global'])

spread = max(y_chiefs) - min(y_chiefs)
print(f"Optimal Detector Dist: {res.x:.3f} mm")
print(f"Spread: {spread:.3f} mm")
os.remove(tmp)
