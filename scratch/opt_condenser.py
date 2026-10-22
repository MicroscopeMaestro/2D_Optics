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
    condenser_config = {
        "surfaces": [
            {"name": "Collimated_Source", "surface_type": "Object", "thickness": 10.0, "glass": "AIR", "radius": 0.0, "conic": 0.0, "asphere_coeffs": [],
             'is_active': True, 'is_coord_break': False, 'decenter_x': 0, 'decenter_y': 0, 'decenter_z': 0,
             'tilt_x': 0, 'tilt_y': 0, 'tilt_z': 0, 'aperture_type': 'Circular', 'aperture_x_half': 100},
            config['surfaces'][23],
            config['surfaces'][24],
            config['surfaces'][25] # S25 (IMA)
        ]
    }
    condenser_config['surfaces'][2]['thickness'] = t
    with tempfile.NamedTemporaryFile('w', delete=False) as tf:
        json.dump(condenser_config, tf)
        tmp = tf.name
    sys_mini = OpticalSystem.from_json(tmp)
    r = Ray(pos=(0,2.0,0), dir=(0, 0, 1.0), wavelength=1.087)
    traced = sys_mini.trace_ray(r)
    y_out = traced.history[-1]['y_global']
    os.remove(tmp)
    return abs(y_out)

res = minimize_scalar(evaluate, bounds=(5, 12), method='bounded')
print("Optimal Condenser Back Focal Distance:", res.x)
