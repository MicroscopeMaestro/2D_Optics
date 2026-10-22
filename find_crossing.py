
import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray
import matplotlib.pyplot as plt

sys_opt = OpticalSystem.from_json('spectrometer_yz_40mm_100mm_final.json')
sys_opt._update_accumulated_positions()

r1 = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., 0.0, 1.0]), wavelength=1.225)
tr1 = sys_opt.trace_ray(r1)

r2 = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([0., 0.0, 1.0]), wavelength=1.064)
tr2 = sys_opt.trace_ray(r2)

h1, y1 = [], []
h2, y2 = [], []

# After Condenser (index 23 is S2 of Condenser)
p1_start = sys_opt.surfaces[23].global_origin + sys_opt.surfaces[23].global_R @ np.array([tr1.history[23]['x_local'], tr1.history[23]['y_local'], tr1.history[23]['z_local']])
d1 = sys_opt.surfaces[23].global_R @ np.array([tr1.history[23]['dir_x'], tr1.history[23]['dir_y'], tr1.history[23]['dir_z']])

p2_start = sys_opt.surfaces[23].global_origin + sys_opt.surfaces[23].global_R @ np.array([tr2.history[23]['x_local'], tr2.history[23]['y_local'], tr2.history[23]['z_local']])
d2 = sys_opt.surfaces[23].global_R @ np.array([tr2.history[23]['dir_x'], tr2.history[23]['dir_y'], tr2.history[23]['dir_z']])

for z in np.linspace(0, 30, 100):
    p1 = p1_start + d1 * z
    p2 = p2_start + d2 * z
    h1.append(z)
    y1.append(p1[1])
    h2.append(z)
    y2.append(p2[1])

plt.plot(h1, y1, label='1225')
plt.plot(h2, y2, label='1064')
plt.legend()
plt.savefig('crossing.png')

# Find z where y1 == y2
diff = np.abs(np.array(y1) - np.array(y2))
idx = np.argmin(diff)
print('Crossing z distance from Condenser S2:', h1[idx])
print('y at crossing:', y1[idx])

