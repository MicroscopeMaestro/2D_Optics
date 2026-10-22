
import sys, os, json, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('teamwork_yz_spectrometer/spectrometer_yz_1064_opt.json', 'r') as f:
    data = json.load(f)

# Tilt the DMD board by 5 degrees!
# The DMD surface is index 16. It currently has tilt_x = 12.0
# We want it to be 17.0 (12 for mirror, 5 for board)
data['surfaces'][16]['tilt_x'] = 17.0

# The reflection angle will change!
# Currently CB_After_DMD has tilt_x = 55.661 (relative to DMD?)
# Wait, let's see what tilt_x it originally had:
cb_idx = 17
orig_cb_tilt = data['surfaces'][cb_idx]['tilt_x']

# If the DMD tilts by 5 degrees, the normal tilts by 5 degrees.
# The reflected beam will tilt by 10 degrees!
# Since the DMD frame is already tilted 5 degrees extra, we need to tilt the CB by an extra 5 degrees to match the 10 degree beam tilt.
# Let's verify by tracing a ray!
data['surfaces'][cb_idx]['tilt_x'] = orig_cb_tilt + 10.0

with open('temp_tilt.json', 'w') as f:
    json.dump(data, f)

sys_opt = OpticalSystem.from_json('temp_tilt.json')
sys_opt._update_accumulated_positions()

r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=1.225)
tr = sys_opt.trace_ray(r)
# Check angle of ray vs angle of CB Z-axis
h_cb = tr.history[cb_idx]
ray_angle = np.arcsin(h_cb['dir_y']) * 180 / np.pi
cb_z_axis = sys_opt.surfaces[cb_idx].global_R @ np.array([0,0,1])
cb_angle = np.arcsin(cb_z_axis[1]) * 180 / np.pi
print(f'Ray angle: {ray_angle:.3f}, CB angle: {cb_angle:.3f}')

