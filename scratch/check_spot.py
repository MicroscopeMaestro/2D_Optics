import sys
import os
import json

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
from optics_app.engine.tracer import OpticalSystem, Ray
import tempfile

with open('spectrometer_yz_40mm_100mm_slit.json') as f:
    config = json.load(f)

config['surfaces'][17]['thickness'] = 43.80

with tempfile.NamedTemporaryFile('w', delete=False) as tf:
    json.dump(config, tf)
    tmp = tf.name

sys_temp = OpticalSystem.from_json(tmp)

y_vals = []
# Trace a fan of rays from the object to see the spot size at Z=215
for dy in [-0.05, -0.025, 0, 0.025, 0.05]:
    r = Ray(pos=(0,0,0), dir=(0, dy, 1.0), wavelength=1.225)
    traced = sys_temp.trace_ray(r)
    seg = traced.history[24]
    z = 215
    dz = z - seg['z_global']
    y = seg['y_global'] + (seg['dir_y'] / seg['dir_z']) * dz
    y_vals.append(y)

spot_size = max(y_vals) - min(y_vals)
print(f"Single Wavelength Spot Size at Z=215: {spot_size:.3f} mm")
os.remove(tmp)
