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

rays = []
for wl in [1.087, 1.225, 1.35]:
    r = Ray(pos=(0,0,0), dir=(0, 0, 1.0), wavelength=wl)
    traced = sys_temp.trace_ray(r)
    rays.append(traced)

z_condenser = rays[0].history[24]['z_global']
print(f"Condenser S2 is at Z={z_condenser:.2f}")

for z in range(int(z_condenser), int(z_condenser + 40), 1):
    y_vals = []
    for r in rays:
        # Find segment after condenser
        seg = r.history[24]
        dz = z - seg['z_global']
        y = seg['y_global'] + (seg['dir_y'] / seg['dir_z']) * dz
        y_vals.append(y)
    spread = max(y_vals) - min(y_vals)
    print(f"Z={z}, Spread={spread:.3f}")

os.remove(tmp)
