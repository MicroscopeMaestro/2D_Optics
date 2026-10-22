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

# Use AC254-050-C
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

# We want the CB_After_DMD to align with the 1.227 um ray!
with tempfile.NamedTemporaryFile('w', delete=False) as tf:
    json.dump(config, tf)
    tmp = tf.name

sys_temp = OpticalSystem.from_json(tmp)
r = sys_temp.trace_ray(Ray(pos=(0,0,0), dir=(0,0,1), wavelength=1.227))
# Find the angle of the ray in the DMD's local coordinate system!
# Wait, the CB_After_DMD is applied AFTER the DMD reflection.
# The DMD reflection mirrors the ray. In the global frame, the DMD is tilted by 14 deg.
# The global angle of the ray after DMD is:
dir_global = np.array([r.history[18]['dir_x'], r.history[18]['dir_y'], r.history[18]['dir_z']])
angle_global = np.degrees(np.arctan2(dir_global[1], dir_global[2]))

# The CB_After_DMD starts with the coordinate system of the DMD *after* reflection.
# If a ray hits a 14-deg mirror parallel to Z, it reflects at 28 deg globally.
# But this ray hits the DMD at some angle. 
# We can just optimize tilt_x of CB_After_DMD so that the ray hits the center of the Relay Lens!

def eval_cb(tilt):
    config['surfaces'][19]['tilt_x'] = tilt
    with tempfile.NamedTemporaryFile('w', delete=False) as tf2:
        json.dump(config, tf2)
        tmp2 = tf2.name
    sys_temp2 = OpticalSystem.from_json(tmp2)
    # Trace center wavelength
    r2 = Ray(pos=(0,0,0), dir=(0,0,1), wavelength=1.227)
    tr2 = sys_temp2.trace_ray(r2)
    if len(tr2.history) > 20:
        # local Y at Relay S1
        pt = np.array([tr2.history[20]['x_global'], tr2.history[20]['y_global'], tr2.history[20]['z_global']])
        s20 = sys_temp2.surfaces[20]
        local_pt = np.dot(s20.global_R.T, pt - s20.global_origin)
        os.remove(tmp2)
        return abs(local_pt[1])
    else:
        os.remove(tmp2)
        return 1000

res_tilt = minimize_scalar(eval_cb, bounds=(-30, 30), method='bounded')
config['surfaces'][19]['tilt_x'] = res_tilt.x
print(f"Optimal CB_After_DMD tilt_x: {res_tilt.x:.2f} deg")

# Now check if 1.137 and 1.316 make it through!
with tempfile.NamedTemporaryFile('w', delete=False) as tf3:
    json.dump(config, tf3)
    tmp3 = tf3.name

sys_temp3 = OpticalSystem.from_json(tmp3)
ys = []
for wl in [1.137, 1.227, 1.316]:
    tr = sys_temp3.trace_ray(Ray(pos=(0,0,0), dir=(0,0,1), wavelength=wl))
    if len(tr.history) < 26:
        print(f"WL {wl} vignetted at {sys_temp3.surfaces[tr.history[-1]['surface_index']].name}")
    else:
        ys.append(tr.history[-1]['y_global'])
        print(f"WL {wl} reached detector at {ys[-1]:.3f}")

if len(ys) == 3:
    print(f"Detector Spread: {max(ys)-min(ys):.3f} mm")

os.remove(tmp3)
