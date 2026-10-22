
import sys, os, json, numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

sys_opt = OpticalSystem.from_json('spectrometer_yz_1inch_100mm.json')
sys_opt._update_accumulated_positions()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for s in sys_opt.surfaces:
    if s.is_coord_break: continue
    center = s.global_origin
    n = s.global_R @ np.array([0, 0, 1])
    d = s.semi_diameter
    if 'Slit' in s.name: d = 5.0
    if 'DMD' in s.name: d = 5.0
    theta = np.linspace(0, 2*np.pi, 20)
    u = s.global_R @ np.array([1, 0, 0])
    v = s.global_R @ np.array([0, 1, 0])
    cx = center[0] + d * (np.cos(theta)*u[0] + np.sin(theta)*v[0])
    cy = center[1] + d * (np.cos(theta)*u[1] + np.sin(theta)*v[1])
    cz = center[2] + d * (np.cos(theta)*u[2] + np.sin(theta)*v[2])
    color = 'k'
    if 'Grating' in s.name: color = 'cyan'
    elif 'DMD' in s.name: color = 'orange'
    ax.plot(cx, cy, cz, color=color, alpha=0.3)
    ax.text(center[0], center[1], center[2], s.name, fontsize=6)

colors = {1.087: 'b', 1.225: 'g', 1.350: 'r'}
for wl in [1.087, 1.225, 1.350]:
    for theta in [np.arcsin(0.04)]:
        for phi in [np.pi/2, -np.pi/2]:
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

ax.set_xlim(-50, 50)
ax.view_init(elev=0, azim=0)
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
ax.set_title('Strict 1-Inch Layout (30mm Focus, 100mm Relay, 16mm Cond)')
plt.savefig('C:/Users/Juan/.gemini/antigravity/brain/49fd158d-b0e1-421c-b15b-b237b2e3ad7f/layout_yz_1inch.png', dpi=300)

