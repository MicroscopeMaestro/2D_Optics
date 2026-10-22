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
    config['surfaces'][17]['thickness'] = t
    with tempfile.NamedTemporaryFile('w', delete=False) as tf:
        json.dump(config, tf)
        tmp = tf.name
    sys_temp = OpticalSystem.from_json(tmp)
    
    # Trace a parallel beam from before the focusing lens
    # The focusing lens is at index 15. We'll set a parallel beam entering it.
    focus_config = {
        "surfaces": [
            {"name": "Collimated_Source", "surface_type": "Object", "thickness": 10.0, "glass": "AIR", "radius": 0.0, "conic": 0.0, "asphere_coeffs": [],
             'is_active': True, 'is_coord_break': False, 'decenter_x': 0, 'decenter_y': 0, 'decenter_z': 0,
             'tilt_x': 0, 'tilt_y': 0, 'tilt_z': 0, 'aperture_type': 'Circular', 'aperture_x_half': 100},
            config['surfaces'][15],
            config['surfaces'][16],
            config['surfaces'][17],
            {"name": "Focus", "surface_type": "Standard", "thickness": 0.0, "glass": "AIR", "radius": 0.0, "conic": 0.0, "asphere_coeffs": [],
             'is_active': True, 'is_coord_break': False, 'decenter_x': 0, 'decenter_y': 0, 'decenter_z': 0,
             'tilt_x': 0, 'tilt_y': 0, 'tilt_z': 0}
        ]
    }
    with tempfile.NamedTemporaryFile('w', delete=False) as tf2:
        json.dump(focus_config, tf2)
        tmp2 = tf2.name
        
    sys_mini = OpticalSystem.from_json(tmp2)
    r = Ray(pos=(0, 2.0, 0), dir=(0, 0.0, 1.0), wavelength=1.087)
    traced = sys_mini.trace_ray(r)
    y_out = traced.history[-1]['y_global']
    os.remove(tmp)
    os.remove(tmp2)
    return abs(y_out)

res = minimize_scalar(evaluate, bounds=(30, 60), method='bounded')
print("Optimal Focusing Distance:", res.x)
