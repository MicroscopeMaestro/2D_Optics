
import sys, os, json, numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

sys_opt = OpticalSystem.from_json('spectrometer_refractive_1064.json')
r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=1.225)
tr = sys_opt.trace_ray(r)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for s in sys_opt.surfaces:
    p = s.global_origin
    ax.scatter(p[0], p[1], p[2], label=s.name, s=20)
    
pts = []
for h in tr.history:
    pts.append([h['x_global'], h['y_global'], h['z_global']])
pts = np.array(pts)
ax.plot(pts[:,0], pts[:,1], pts[:,2], color='red')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.savefig('layout_3d.png')

