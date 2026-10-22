import sys
import os
import json
from scipy.optimize import minimize_scalar

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
from optics_app.engine.tracer import OpticalSystem, Ray
import tempfile

with open('spectrometer_yz_40mm_100mm_slit.json') as f:
    config = json.load(f)

def evaluate(t):
    config['surfaces'][0]['thickness'] = t
    with tempfile.NamedTemporaryFile('w', delete=False) as tf:
        json.dump(config, tf)
        tmp = tf.name
    sys_temp = OpticalSystem.from_json(tmp)
    r = Ray(pos=(0,0,0), dir=(0, 0.05, 1.0), wavelength=1.087)
    traced_ray = sys_temp.trace_ray(r)
    dy_out = traced_ray.history[2]['dir_y']
    os.remove(tmp)
    return abs(dy_out)

res = minimize_scalar(evaluate, bounds=(5, 15), method='bounded')
print("Optimal Object Thickness:", res.x)
