import sys
import os
import json

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
from optics_app.engine.tracer import OpticalSystem, Ray
import tempfile

with open('spectrometer_yz_40mm_100mm_slit.json') as f:
    config = json.load(f)

# Widen all apertures after DMD
for s in config['surfaces'][19:]:
    if 'aperture_y_half' in s: s['aperture_y_half'] = 1000
    if 'aperture_x_half' in s: s['aperture_x_half'] = 1000

with tempfile.NamedTemporaryFile('w', delete=False) as tf:
    json.dump(config, tf)
    tmp = tf.name

sys_temp = OpticalSystem.from_json(tmp)

for wl in [1.137, 1.227, 1.316]:
    r = Ray(pos=(0,0,0), dir=(0,0,1), wavelength=wl)
    tr = sys_temp.trace_ray(r)
    s19_y = tr.history[19]['y_global'] if len(tr.history)>19 else None
    
    # Let's get local Y on Relay_AC254-100-C_S1 (surface 20)
    import numpy as np
    s20 = sys_temp.surfaces[20]
    if len(tr.history)>20:
        pt = np.array([tr.history[20]['x_global'], tr.history[20]['y_global'], tr.history[20]['z_global']])
        pt_local = np.dot(s20.global_R.T, pt - s20.global_origin)
        print(f"WL {wl:.3f}: Local Y on Relay S1 = {pt_local[1]:.3f} mm")
    else:
        print(f"WL {wl:.3f}: Missed Relay S1")

os.remove(tmp)
