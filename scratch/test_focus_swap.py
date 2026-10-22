import sys
import os
import json

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
from optics_app.engine.tracer import OpticalSystem, Ray
import tempfile
from scipy.optimize import minimize_scalar

with open('spectrometer_yz_40mm_100mm_slit.json') as f:
    config = json.load(f)

# Crown (thick) faces collimated light
config['surfaces'][15]['radius'] = 20.3
config['surfaces'][15]['thickness'] = 7.0
config['surfaces'][15]['glass'] = 'N-BAF10'

# Interface
config['surfaces'][16]['radius'] = -16.4
config['surfaces'][16]['thickness'] = 2.0
config['surfaces'][16]['glass'] = 'N-SF6HT'

# Flint (thin) faces focus
config['surfaces'][17]['radius'] = -63.7
config['surfaces'][17]['glass'] = 'AIR'
# thickness to be optimized

# Reset Relay and Condenser to standard infinite conjugate
config['surfaces'][18]['thickness'] = 74.76
config['surfaces'][22]['thickness'] = 28.0
config['surfaces'][24]['thickness'] = 10.13

def eval_focus(t):
    config['surfaces'][17]['thickness'] = t
    with tempfile.NamedTemporaryFile('w', delete=False) as tf2:
        json.dump(config, tf2)
        tmp2 = tf2.name
    sys_temp = OpticalSystem.from_json(tmp2)
    
    focus_config = {
        "surfaces": [
            {"name": "Collimated", "surface_type": "Object", "thickness": 10.0, "glass": "AIR", "radius": 0.0, "conic": 0.0, "asphere_coeffs": [], 'is_active': True, 'is_coord_break': False, 'tilt_x':0, 'tilt_y':0, 'tilt_z':0, 'decenter_x':0, 'decenter_y':0, 'decenter_z':0, 'aperture_type': 'Circular', 'aperture_x_half': 100},
            config['surfaces'][15],
            config['surfaces'][16],
            config['surfaces'][17],
            {"name": "Focus", "surface_type": "Standard", "thickness": 0.0, "glass": "AIR"}
        ]
    }
    with tempfile.NamedTemporaryFile('w', delete=False) as tf3:
        json.dump(focus_config, tf3)
        tmp3 = tf3.name
    sys_mini = OpticalSystem.from_json(tmp3)
    r = Ray(pos=(0, 2.0, 0), dir=(0, 0.0, 1.0), wavelength=1.087)
    traced = sys_mini.trace_ray(r)
    y_out = traced.history[-1]['y_global']
    os.remove(tmp2)
    os.remove(tmp3)
    return abs(y_out)

res_focus = minimize_scalar(eval_focus, bounds=(15, 30), method='bounded')
config['surfaces'][17]['thickness'] = res_focus.x
print(f"Optimal AC254-030-C Focusing Distance: {res_focus.x:.2f} mm")

with tempfile.NamedTemporaryFile('w', delete=False) as tf:
    json.dump(config, tf)
    tmp = tf.name

sys_final = OpticalSystem.from_json(tmp)
y_chiefs = []
for wl in [1.087, 1.225, 1.35]:
    r = Ray(pos=(0,0,0), dir=(0, 0, 1.0), wavelength=wl)
    traced = sys_final.trace_ray(r)
    y_chiefs.append(traced.history[-1]['y_global'])
    dmd_y = traced.history[18]['y_global']
    print(f"WL {wl}: DMD y={dmd_y:.2f}, IMA y={y_chiefs[-1]:.2f}")

spread = max(y_chiefs) - min(y_chiefs)
print(f"Final Detector Footprint: {spread:.3f} mm")

os.remove(tmp)
