
import sys, os, json, numpy as np
from scipy.optimize import minimize_scalar
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

data = json.load(open('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json'))

cb_idx = next(i for i, s in enumerate(data['surfaces']) if 'CB_After_Grating' in s.get('name',''))
data['surfaces'][cb_idx]['tilt_x'] = 0.0
json.dump(data, open('t_temp.json', 'w'))
sys_opt = OpticalSystem.from_json('t_temp.json')
sys_opt._update_accumulated_positions()

# Trace chief ray
r0 = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., 0.0, 1.0]), wavelength=1.225)
tr0 = sys_opt.trace_ray(r0)

# Get the direction of the diffracted chief ray in global coordinates
dir_y = tr0.history[cb_idx-1]['dir_y']
dir_z = tr0.history[cb_idx-1]['dir_z']

# Angle of the ray relative to global +Z
ray_angle = np.degrees(np.arctan2(dir_y, dir_z))

# The mirror reflected the coordinate system (flipped Z). 
# So the incoming Z axis of the next surface is currently aligned to -Z globally?
# Let's just find the angle that makes the ray's local dir = [0, 0, 1]
def obj_angle(angle):
    data['surfaces'][cb_idx]['tilt_x'] = angle
    json.dump(data, open('t_temp.json', 'w'))
    sys_opt = OpticalSystem.from_json('t_temp.json')
    sys_opt._update_accumulated_positions()
    r = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., 0.0, 1.0]), wavelength=1.225)
    tr = sys_opt.trace_ray(r)
    # Target: at Focusing S1 (which is index cb_idx+1), the local dir_y should be 0 and dir_z should be 1
    # Actually, we can just check the y_local coordinate at Focus S1 and Focus S3! They should be exactly the same (meaning it's parallel to the local Z axis!)
    y1 = tr.history[cb_idx+1]['y_local']
    y3 = tr.history[cb_idx+3]['y_local']
    return np.abs(y1 - y3) + np.abs(y1) # Also want it to be centered!

res_angle = minimize_scalar(obj_angle, bounds=(35.0, 55.0), method='bounded')
print('Optimal CB_After_Grating tilt_x =', res_angle.x)
data['surfaces'][cb_idx]['tilt_x'] = res_angle.x

json.dump(data, open('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json', 'w'), indent=4)

