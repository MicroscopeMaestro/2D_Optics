
import sys, os, json, numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

sys_opt = OpticalSystem.from_json('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json')
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

slit_pos = sys_opt.surfaces[6].global_origin
slit_R = sys_opt.surfaces[6].global_R
colors = {1.087: 'b', 1.225: 'g', 1.350: 'r'}
for wl in [1.087, 1.225, 1.350]:
    for theta in [np.arcsin(0.065)]:
        for phi in [np.pi/2, -np.pi/2]:
            dx, dy, dz = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
            dir_global = slit_R @ np.array([dx, dy, dz])
            current_ray = Ray(pos=slit_pos, dir=dir_global, wavelength=wl)
            hx, hy, hz = [slit_pos[0]], [slit_pos[1]], [slit_pos[2]]
            for i in range(7, len(sys_opt.surfaces)):
                surf = sys_opt.surfaces[i]
                R_mat = surf.global_R
                origin = surf.global_origin
                pos_local = R_mat.T @ (current_ray.pos - origin)
                dir_local = R_mat.T @ current_ray.dir
                hit_local = np.array([0,0,0])
                if hasattr(sys_opt, '_intersect_surface'):
                    h = sys_opt._intersect_surface(surf, pos_local, dir_local)
                    if h is not None: hit_local = h
                else:
                    t = -pos_local[2] / dir_local[2] if dir_local[2] != 0 else 0
                    hit_local = pos_local + t * dir_local
                hit_global = origin + R_mat @ hit_local
                hx.append(hit_global[0])
                hy.append(hit_global[1])
                hz.append(hit_global[2])
                if surf.glass.upper() == 'AIR' and not surf.is_coord_break:
                    current_ray.pos = hit_global
                else:
                    current_ray.pos = hit_global
            ax.plot(hx, hy, hz, color=colors[wl], alpha=0.6)

ax.set_xlim(-50, 50)
ax.view_init(elev=0, azim=0)
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
ax.set_title('Final Optimized Layout (1/2-in Collimator & Relay)')
plt.savefig('C:/Users/Juan/.gemini/antigravity/brain/49fd158d-b0e1-421c-b15b-b237b2e3ad7f/layout_yz_goal_final.png', dpi=300)

