
import sys, os, json, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('teamwork_yz_spectrometer/spectrometer_yz_1064_opt.json', 'r') as f:
    data = json.load(f)

data['surfaces'][16]['tilt_x'] = 17.0

def test_tilt(x):
    data['surfaces'][17]['tilt_x'] = x[0]
    with open('temp_tilt.json', 'w') as f:
        json.dump(data, f)
    sys_opt = OpticalSystem.from_json('temp_tilt.json')
    sys_opt._update_accumulated_positions()
    
    r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=1.225)
    tr = sys_opt.trace_ray(r)
    if tr.vignetted or len(tr.history) <= 17: return 1e6
    h = tr.history[17]
    dir_y = h['dir_y']
    cb_z = sys_opt.surfaces[17].global_R @ np.array([0,0,1])
    return np.abs(dir_y - cb_z[1])

res = minimize(test_tilt, [65.0], method='Nelder-Mead')
print(f'Optimal CB tilt_x: {res.x[0]:.3f}')
data['surfaces'][17]['tilt_x'] = res.x[0]

# Now let's check distance!
with open('temp_tilt.json', 'w') as f:
    json.dump(data, f)
sys_opt = OpticalSystem.from_json('temp_tilt.json')
sys_opt._update_accumulated_positions()
focus_c = sys_opt.surfaces[15].global_origin
relay_c = sys_opt.surfaces[18].global_origin
dist = np.linalg.norm(focus_c - relay_c)
print(f'New distance between lenses: {dist:.2f} mm')

with open('teamwork_yz_spectrometer/spectrometer_yz_1064_tilt5.json', 'w') as f:
    json.dump(data, f, indent=4)

