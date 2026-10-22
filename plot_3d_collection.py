
import sys, os, json, numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

sys_opt = OpticalSystem.from_json('spectrometer_with_collection.json')
sys_opt._update_accumulated_positions()

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Draw elements
for s in sys_opt.surfaces:
    if s.is_coord_break: continue
    center = s.global_origin
    n = s.global_R @ np.array([0, 0, 1])
    d = s.semi_diameter
    # approximate circle
    theta = np.linspace(0, 2*np.pi, 20)
    u = s.global_R @ np.array([1, 0, 0])
    v = s.global_R @ np.array([0, 1, 0])
    cx = center[0] + d * (np.cos(theta)*u[0] + np.sin(theta)*v[0])
    cy = center[1] + d * (np.cos(theta)*u[1] + np.sin(theta)*v[1])
    cz = center[2] + d * (np.cos(theta)*u[2] + np.sin(theta)*v[2])
    
    color = 'k'
    if 'Slit' in s.name: color = 'magenta'
    elif 'DMD' in s.name: color = 'orange'
    elif 'Grating' in s.name: color = 'cyan'
    
    ax.plot(cx, cy, cz, color=color, alpha=0.3)
    ax.text(center[0], center[1], center[2], s.name, fontsize=6)

# Trace rays from Sample
colors = {1.087: 'b', 1.225: 'g', 1.350: 'r'}
for wl in [1.087, 1.225, 1.350]:
    for theta in [np.arcsin(0.2)]: # NA 0.2 from sample
        for phi in np.linspace(0, 2*np.pi, 6, endpoint=False):
            dx, dy, dz = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
            r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([dx, dy, dz]), wavelength=wl)
            tr = sys_opt.trace_ray(r)
            hx, hy, hz = [], [], []
            for h in tr.history:
                if 'x_global' in h:
                    hx.append(h['x_global'])
                    hy.append(h['y_global'])
                    hz.append(h['z_global'])
            ax.plot(hx, hy, hz, color=colors[wl], alpha=0.6)

ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
ax.set_title('Complete System: Collection Optics + Spectrometer')
plt.savefig('full_system_3d.png', dpi=300)

