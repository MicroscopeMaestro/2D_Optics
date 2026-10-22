
import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

data = json.load(open('spectrometer_yz_40mm_100mm_optimized.json'))
sys_opt = OpticalSystem.from_json('spectrometer_yz_40mm_100mm_optimized.json')
sys_opt._update_accumulated_positions()

# 1. Check spots on Detector
print('--- OPTICAL PERFORMANCE (Detector at Z={:.2f}) ---'.format(sys_opt.surfaces[24].global_origin[2]))
max_radius = 0
for wl in [1.064, 1.225, 1.350]:
    r0 = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., 0.0, 1.0]), wavelength=wl)
    tr0 = sys_opt.trace_ray(r0)
    
    r_top = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., 0.01, 0.99995]), wavelength=wl)
    tr_top = sys_opt.trace_ray(r_top)
    
    r_bot = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., -0.01, 0.99995]), wavelength=wl)
    tr_bot = sys_opt.trace_ray(r_bot)
    
    c_y = tr0.history[-1]['y_local']
    top_y = tr_top.history[-1]['y_local']
    bot_y = tr_bot.history[-1]['y_local']
    
    spot_radius = max(abs(top_y - c_y), abs(bot_y - c_y))
    max_radius = max(max_radius, abs(top_y), abs(bot_y), abs(c_y))
    
    print(f'Wavelength {wl} nm:')
    print(f'  Chief Ray Center: y = {c_y:.4f} mm')
    print(f'  Spot Radius: {spot_radius:.4f} mm')
    print(f'  Max absolute deviation from axis: {max(abs(top_y), abs(bot_y)):.4f} mm')

print(f'\nTOTAL MAX RADIUS ACROSS ALL RAYS: {max_radius:.4f} mm (Must be < 0.400 mm)')

# 2. Check Clearance at Focusing Lens
print('\n--- MECHANICAL CLEARANCE ---')
# The incoming beam hits the Grating, then the Focusing Lens, then the DMD.
# The reflected beam from the DMD heads towards the Relay Lens.
# As it travels from DMD to Relay, it passes by the physical housing of the Focusing Lens.
# The Focusing Lens is at index 15, 16, 17.
# We need to find the shortest distance between the returning chief rays (from DMD to Relay) 
# and the center of the Focusing Lens.

focus_center = sys_opt.surfaces[15].global_origin

clearances = []
for wl in [1.064, 1.225, 1.350]:
    r = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., 0.0, 1.0]), wavelength=wl)
    tr = sys_opt.trace_ray(r)
    
    # Ray segment from DMD (index 18) to Relay (index 19)
    p_dmd = sys_opt.surfaces[18].global_origin + sys_opt.surfaces[18].global_R @ np.array([tr.history[18]['x_local'], tr.history[18]['y_local'], tr.history[18]['z_local']])
    
    p_relay = sys_opt.surfaces[19].global_origin + sys_opt.surfaces[19].global_R @ np.array([tr.history[19]['x_local'], tr.history[19]['y_local'], tr.history[19]['z_local']])
    
    # Vector from DMD to Relay
    v_ray = p_relay - p_dmd
    len_ray = np.linalg.norm(v_ray)
    dir_ray = v_ray / len_ray
    
    # Vector from DMD to Focus Lens Center
    v_focus = focus_center - p_dmd
    
    # Project v_focus onto the ray direction to find the closest point
    t = np.dot(v_focus, dir_ray)
    if 0 < t < len_ray:
        closest_point = p_dmd + t * dir_ray
        distance = np.linalg.norm(closest_point - focus_center)
        clearances.append(distance)

if clearances:
    min_clear = min(clearances)
    print(f'Minimum clearance from returning beam center to Focusing Lens optical axis: {min_clear:.2f} mm')
    print(f'Assuming standard 1-inch (25.4mm) housing (Radius = 15.0 mm):')
    if min_clear > 15.0:
        print(f'  [PASS] Beam safely clears the lens housing by {min_clear - 15.0:.2f} mm!')
    else:
        print(f'  [FAIL] Beam CRASHES into the lens housing! (Needs 15.0mm, only has {min_clear:.2f}mm)')
else:
    print('No valid ray path found for clearance check.')

# 3. Check Relay to Condenser thickness
t_bet = data['surfaces'][22]['thickness']
print(f'\nRelay to Condenser Distance: {t_bet:.2f} mm')
if t_bet >= 1.0:
    print('  [PASS] Physically possible to space the lenses.')
else:
    print('  [FAIL] Lenses physically intersect!')

