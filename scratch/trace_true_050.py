import sys
import os
import json
import numpy as np
from scipy.optimize import minimize_scalar

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
from optics_app.engine.tracer import OpticalSystem, Ray
import tempfile

with open('spectrometer_yz_40mm_100mm_slit.json') as f:
    config = json.load(f)

# TRUE AC254-050-C
config['surfaces'][15]['radius'] = 22.9
config['surfaces'][15]['thickness'] = 9.0
config['surfaces'][15]['glass'] = 'N-LAK22'
config['surfaces'][15]['name'] = 'Focusing_AC254-050-C_S1'
config['surfaces'][16]['radius'] = -25.9
config['surfaces'][16]['thickness'] = 1.8
config['surfaces'][16]['glass'] = 'N-SF6'
config['surfaces'][16]['name'] = 'Focusing_AC254-050-C_S2'
config['surfaces'][17]['radius'] = 194.5  # Wait, if R3=194.5 in standard Thorlabs, it is convex to the right. 
# But for focusing, the collimated light should hit the steeper surface. 
# R1=22.9 is steeper than R3=194.5. 
# Wait, if R3=194.5, it is convex. But the standard AC series has a nearly FLAT back surface.
# In Thorlabs, R1 is usually convex (positive), R2 is concave interface, R3 is negative (convex on the back) or positive (concave). 
# If R3=194.5, the center of curvature is to the right. So the surface curves to the left. It is a CONCAVE surface from the outside.
# Let's just use it as given:
config['surfaces'][17]['glass'] = 'AIR'
config['surfaces'][17]['thickness'] = 41.2
config['surfaces'][17]['name'] = 'Focusing_AC254-050-C_S3'

# DMD aperture
config['surfaces'][18]['semi_diameter'] = 1000 # Let's see if it fits on 8.66 later

with tempfile.NamedTemporaryFile('w', delete=False) as tf:
    json.dump(config, tf)
    tmp = tf.name

sys_temp = OpticalSystem.from_json(tmp)
r1 = sys_temp.trace_ray(Ray(pos=(0, 0, 0), dir=(0, 0, 1), wavelength=1.227))
print('0-deg Focus Y at 41.2 mm:', r1.history[18]['y_local'])

ys = []
for wl in [1.137, 1.227, 1.316]:
    tr = sys_temp.trace_ray(Ray(pos=(0,0,0), dir=(0,0,1), wavelength=wl))
    print(wl, 'Y_local DMD:', tr.history[18]['y_local'])
    ys.append(tr.history[18]['y_local'])
print('DMD spread:', max(ys)-min(ys))

os.remove(tmp)
