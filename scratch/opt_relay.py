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
    relay_config = {
        "surfaces": [
            {"name": "DMD", "surface_type": "Object", "thickness": t, "glass": "AIR", "radius": 0.0, "conic": 0.0, "asphere_coeffs": [],
             'is_active': True, 'is_coord_break': False, 'decenter_x': 0, 'decenter_y': 0, 'decenter_z': 0,
             'tilt_x': 0, 'tilt_y': 0, 'tilt_z': 0, 'aperture_type': 'Circular', 'aperture_x_half': 100},
            config['surfaces'][20],
            config['surfaces'][21],
            config['surfaces'][22]
        ]
    }
    with tempfile.NamedTemporaryFile('w', delete=False) as tf:
        json.dump(relay_config, tf)
        tmp = tf.name
    sys_mini = OpticalSystem.from_json(tmp)
    r = Ray(pos=(0,0,0), dir=(0, 0.05, 1.0), wavelength=1.087)
    traced = sys_mini.trace_ray(r)
    dy = traced.history[-1]['dir_y']
    os.remove(tmp)
    return abs(dy)

res = minimize_scalar(evaluate, bounds=(70, 80), method='bounded')
print("Optimal CB_After_DMD Thickness:", res.x)
