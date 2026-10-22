
import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

sys_opt = OpticalSystem.from_json('spectrometer_yz_40mm_100mm_final.json')
sys_opt._update_accumulated_positions()

def trace_chief(wl):
    r = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., 0.0, 1.0]), wavelength=wl)
    tr = sys_opt.trace_ray(r)
    return tr

tr1 = trace_chief(1.064)
tr2 = trace_chief(1.225)
tr3 = trace_chief(1.350)

# We want to find Z (distance from Condenser S2) where all y's are 0
p2_start = sys_opt.surfaces[23].global_origin + sys_opt.surfaces[23].global_R @ np.array([tr2.history[23]['x_local'], tr2.history[23]['y_local'], tr2.history[23]['z_local']])
d2 = sys_opt.surfaces[23].global_R @ np.array([tr2.history[23]['dir_x'], tr2.history[23]['dir_y'], tr2.history[23]['dir_z']])

# z where ray2 hits y=0:
z2_cross = -p2_start[1] / d2[1] * d2[2] if d2[1] != 0 else 0
print('1225 nm crosses y=0 at z=', z2_cross)

p1_start = sys_opt.surfaces[23].global_origin + sys_opt.surfaces[23].global_R @ np.array([tr1.history[23]['x_local'], tr1.history[23]['y_local'], tr1.history[23]['z_local']])
d1 = sys_opt.surfaces[23].global_R @ np.array([tr1.history[23]['dir_x'], tr1.history[23]['dir_y'], tr1.history[23]['dir_z']])
z1_cross = -p1_start[1] / d1[1] * d1[2]
print('1064 nm crosses y=0 at z=', z1_cross)

