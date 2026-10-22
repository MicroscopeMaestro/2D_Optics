
import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

sys_opt = OpticalSystem.from_json('spectrometer_yz_40mm_100mm_optimized.json')
sys_opt._update_accumulated_positions()

# Clearance check
focus_center = sys_opt.surfaces[15].global_origin

clearances = []
for wl in [1.064, 1.225, 1.350]:
    r = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., 0.0, 1.0]), wavelength=wl)
    tr = sys_opt.trace_ray(r)
    
    p_dmd = sys_opt.surfaces[18].global_origin + sys_opt.surfaces[18].global_R @ np.array([tr.history[18]['x_local'], tr.history[18]['y_local'], tr.history[18]['z_local']])
    p_relay = sys_opt.surfaces[20].global_origin + sys_opt.surfaces[20].global_R @ np.array([tr.history[20]['x_local'], tr.history[20]['y_local'], tr.history[20]['z_local']])
    
    v_ray = p_relay - p_dmd
    len_ray = np.linalg.norm(v_ray)
    dir_ray = v_ray / len_ray
    v_focus = focus_center - p_dmd
    t = np.dot(v_focus, dir_ray)
    if 0 < t < len_ray:
        closest_point = p_dmd + t * dir_ray
        distance = np.linalg.norm(closest_point - focus_center)
        clearances.append(distance)

if clearances:
    min_clear = min(clearances)
    print(f'Minimum clearance from returning beam center to 40mm Focusing Lens optical axis: {min_clear:.2f} mm')
    print(f'Assuming standard 1-inch (25.4mm) housing (Radius = 15.0 mm):')
    if min_clear > 15.0:
        print(f'  [PASS] Beam safely clears the 40mm lens housing by {min_clear - 15.0:.2f} mm!')
    else:
        print(f'  [FAIL] Beam CRASHES into the 40mm lens housing! (Needs 15.0mm, only has {min_clear:.2f}mm)')
else:
    print('No valid ray path found for clearance check.')

