
import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('teamwork_yz_spectrometer/spectrometer_yz_1064_opt.json', 'r') as f:
    data = json.load(f)

data['surfaces'][16]['tilt_x'] = 17.0

for t in [17.0, -17.0, 34.0, -34.0, 10.0, -10.0, 68.0, 146.0]:
    data['surfaces'][17]['tilt_x'] = t
    with open('temp_tilt.json', 'w') as f: json.dump(data, f)
    sys_opt = OpticalSystem.from_json('temp_tilt.json')
    sys_opt._update_accumulated_positions()
    
    cb_z = sys_opt.surfaces[17].global_R @ np.array([0,0,1])
    cb_angle = np.arcsin(cb_z[1]) * 180 / np.pi
    
    r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0.0, 1.0]), wavelength=1.225)
    tr = sys_opt.trace_ray(r)
    
    if len(tr.history) > 17:
        dir_y = tr.history[17]['dir_y']
        ray_angle = np.arcsin(dir_y) * 180 / np.pi
        err = np.abs(ray_angle - cb_angle)
        print(f'Tilt {t}: err = {err:.3f} (Ray={ray_angle:.3f}, CB={cb_angle:.3f})')

