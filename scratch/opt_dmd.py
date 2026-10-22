import sys
import os
import json
from scipy.optimize import minimize_scalar

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
from optics_app.engine.tracer import OpticalSystem, Ray
import tempfile

with open('spectrometer_yz_40mm_100mm_slit.json') as f:
    config = json.load(f)

for i, s in enumerate(config['surfaces']):
    if 'DMD' in s['name'] and 'CB' not in s['name']:
        dmd_idx = i
cb_idx = dmd_idx + 1

def evaluate(t):
    config['surfaces'][cb_idx]['thickness'] = t
    with tempfile.NamedTemporaryFile('w', delete=False) as tf:
        json.dump(config, tf)
        tmp = tf.name
    sys_temp = OpticalSystem.from_json(tmp)
    
    # Trace a ray starting exactly after the DMD
    # We will launch it from the origin of CB_After_DMD!
    # To do this safely, let's just make the Slit aperture huge so it doesn't vignette
    sys_temp.surfaces[6].aperture_y_half = 1000
    
    r = Ray(pos=(0,0.01,0), dir=(0, 0, 1.0), wavelength=1.087)
    traced_ray = sys_temp.trace_ray(r)
    
    dy_out = 1000
    for state in traced_ray.history:
        if state['surface_index'] == 22:
            dy_out = state['dir_y']
            break
    os.remove(tmp)
    return abs(dy_out)

res = minimize_scalar(evaluate, bounds=(90, 120), method='bounded')
print("Optimal CB_After_DMD Thickness:", res.x)
