import sys
import os
import json
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
from optics_app.engine.tracer import OpticalSystem, Ray
import tempfile

with open('spectrometer_yz_40mm_100mm_slit.json') as f:
    config = json.load(f)

# Restore AC254-050-C
config['surfaces'][15]['radius'] = 34.5
config['surfaces'][15]['thickness'] = 12.0
config['surfaces'][15]['glass'] = 'N-BAK4'
config['surfaces'][15]['name'] = 'Focusing_AC254-050-C_S1'
config['surfaces'][16]['radius'] = -25.2
config['surfaces'][16]['thickness'] = 2.0
config['surfaces'][16]['glass'] = 'N-SF6HT'
config['surfaces'][16]['name'] = 'Focusing_AC254-050-C_S2'
config['surfaces'][17]['radius'] = -135.2
config['surfaces'][17]['glass'] = 'AIR'
config['surfaces'][17]['thickness'] = 43.80
config['surfaces'][17]['name'] = 'Focusing_AC254-050-C_S3'

# We'll see what the original JSON had for distances
# config['surfaces'][18]['thickness'] = 74.76 
# Let's put it to 100 as the user expects? No, if it's 100, the focus is wrong. Let's stick to 74.76 first, or check what happens at 100.
# For now, let's just find the reflection angle off the DMD!

with tempfile.NamedTemporaryFile('w', delete=False) as tf:
    json.dump(config, tf)
    tmp = tf.name

sys_temp = OpticalSystem.from_json(tmp)

for wl in [1.137, 1.227, 1.316]:
    r = Ray(pos=(0,0,0), dir=(0,0,1), wavelength=wl)
    tr = sys_temp.trace_ray(r)
    if len(tr.history) > 18:
        # Before reflection
        s_dmd = sys_temp.surfaces[18]
        # dir before DMD? The segment ending at 18 has the direction before reflection!
        seg_in = tr.segments[17]
        # Wait, segments are not reliably indexed like that. Let's use history.
        h_dmd = tr.history[18]
        # The direction in history is AFTER interaction!
        dir_out = np.array([h_dmd['dir_x'], h_dmd['dir_y'], h_dmd['dir_z']])
        angle_out = np.degrees(np.arctan2(dir_out[1], dir_out[2]))
        print(f"WL: {wl}, DMD out angle: {angle_out:.2f} deg, Y global: {h_dmd['y_global']:.2f}")

os.remove(tmp)
