import sys
import os
import json

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
from optics_app.engine.tracer import OpticalSystem, Ray
import tempfile

with open('spectrometer_yz_40mm_100mm_slit.json') as f:
    config = json.load(f)

# Set optimal focusing distance to DMD
config['surfaces'][17]['thickness'] = 43.80

with tempfile.NamedTemporaryFile('w', delete=False) as tf:
    json.dump(config, tf)
    tmp = tf.name

sys_temp = OpticalSystem.from_json(tmp)

print("Tracing wavelengths...")
for wl in [1.087, 1.225, 1.35]:
    r = Ray(pos=(0,0,0), dir=(0, 0, 1.0), wavelength=wl)
    traced = sys_temp.trace_ray(r)
    dmd_y = next((s['y_global'] for s in traced.history if s['surface_index']==18), None)
    ima_y = traced.history[-1]['y_global']
    print(f"WL={wl}: DMD y={dmd_y:.3f}, IMA y={ima_y:.3f}")

os.remove(tmp)
