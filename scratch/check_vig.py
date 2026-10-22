import sys
import os
import json

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
from optics_app.engine.tracer import OpticalSystem, Ray
import tempfile

with open('spectrometer_yz_40mm_100mm_slit.json') as f:
    config = json.load(f)

for s in config['surfaces']:
    if 'aperture_y_half' in s:
        s['aperture_y_half'] = 1000
    if 'aperture_x_half' in s:
        s['aperture_x_half'] = 1000

for x0 in [
    [75, 28, 10],
    [97, 20, 8.5]
]:
    dmd_to_relay, relay_to_condenser, condenser_to_det = x0
    config['surfaces'][18]['thickness'] = dmd_to_relay
    config['surfaces'][22]['thickness'] = relay_to_condenser
    config['surfaces'][24]['thickness'] = condenser_to_det
    
    with tempfile.NamedTemporaryFile('w', delete=False) as tf:
        json.dump(config, tf)
        tmp = tf.name
    
    sys_temp = OpticalSystem.from_json(tmp)
    r = Ray(pos=(0, 0.005, 0), dir=(0, 0, 1.0), wavelength=1.225)
    traced = sys_temp.trace_ray(r)
    print("x0", x0, "Len:", len(traced.history))
    if len(traced.history) < 26:
        print("Vignetted at", sys_temp.surfaces[traced.history[-1]['surface_index']].name)
    os.remove(tmp)
