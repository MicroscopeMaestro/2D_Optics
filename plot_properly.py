
import sys, os, json, numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

sys_opt = OpticalSystem.from_json('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json')
sys_opt._update_accumulated_positions()

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)

# Draw lenses (flattened to 2D for simplicity, just lines at Z, or rectangles)
for s in sys_opt.surfaces:
    if s.is_coord_break: continue
    center = s.global_origin
    n = s.global_R @ np.array([0, 0, 1])
    d = s.semi_diameter
    if 'Slit' in s.name: d = 5.0
    if 'DMD' in s.name: d = 5.0
    
    # 2D projection (Y vs Z)
    u = s.global_R @ np.array([1, 0, 0])
    v = s.global_R @ np.array([0, 1, 0])
    
    # Draw straight line for surface
    pts_z = [center[2] - d*v[2], center[2] + d*v[2]]
    pts_y = [center[1] - d*v[1], center[1] + d*v[1]]
    
    color = 'k'
    if 'Grating' in s.name: color = 'cyan'
    elif 'DMD' in s.name: color = 'orange'
    ax.plot(pts_z, pts_y, color=color, alpha=0.5)
    ax.text(center[2], center[1] + d*1.1, s.name, fontsize=6, rotation=90)

colors = {1.087: 'b', 1.225: 'g', 1.350: 'r'}

# Trace proper rays from the Object!
obj_pos = sys_opt.surfaces[0].global_origin
for wl in [1.225]: # Just 1225 for now to show perfect focus
    for theta in np.linspace(-0.01, 0.01, 5):
        # Ray from center of object
        dy = theta
        dz = np.sqrt(1 - dy**2)
        r = Ray(pos=obj_pos, dir=np.array([0., dy, dz]), wavelength=wl)
        tr = sys_opt.trace_ray(r)
        
        # Plot ray path
        hz, hy = [], []
        for i, h in enumerate(tr.history):
            if i < len(sys_opt.surfaces):
                s = sys_opt.surfaces[i]
                if s.is_coord_break: continue
                # Calculate global hit point
                pt = s.global_origin + s.global_R @ np.array([h['x_local'], h['y_local'], h['z_local']])
                hz.append(pt[2])
                hy.append(pt[1])
        ax.plot(hz, hy, color=colors[wl], alpha=0.6)

ax.set_aspect('equal')
ax.set_xlabel('Z (mm)')
ax.set_ylabel('Y (mm)')
ax.set_title('True Optical Ray Trace (YZ Plane)')
plt.savefig('C:/Users/Juan/.gemini/antigravity/brain/49fd158d-b0e1-421c-b15b-b237b2e3ad7f/layout_yz_goal_final.png', dpi=300)

