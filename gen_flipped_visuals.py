
import sys, os, json, numpy as np
import matplotlib.pyplot as plt
import trimesh
from scipy.spatial import ConvexHull
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

sys_opt = OpticalSystem.from_json('spectrometer_yz_100mm_achromat_flipped.json')
sys_opt._update_accumulated_positions()

# 2D Plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
colors = {1.225: 'g'}
obj_pos = sys_opt.surfaces[0].global_origin
for wl in [1.225]:
    for theta in np.linspace(-0.01, 0.01, 5):
        dy, dz = theta, np.sqrt(1 - theta**2)
        r = Ray(pos=obj_pos, dir=np.array([0., dy, dz]), wavelength=wl)
        tr = sys_opt.trace_ray(r)
        hz, hy = [], []
        for i, h in enumerate(tr.history):
            if i < len(sys_opt.surfaces):
                s = sys_opt.surfaces[i]
                if s.is_coord_break: continue
                pt = s.global_origin + s.global_R @ np.array([h['x_local'], h['y_local'], h['z_local']])
                hz.append(pt[2])
                hy.append(pt[1])
        ax.plot(hz, hy, color=colors[wl], alpha=0.6)
        
for s in sys_opt.surfaces:
    if s.is_coord_break: continue
    center = s.global_origin
    n = s.global_R @ np.array([0, 0, 1])
    d = s.semi_diameter
    if 'Slit' in s.name or 'DMD' in s.name: d = 5.0
    u, v = s.global_R @ np.array([1, 0, 0]), s.global_R @ np.array([0, 1, 0])
    pts_z = [center[2] - d*v[2], center[2] + d*v[2]]
    pts_y = [center[1] - d*v[1], center[1] + d*v[1]]
    ax.plot(pts_z, pts_y, color='k' if 'Grating' not in s.name else 'cyan', alpha=0.5)
ax.set_aspect('equal')
plt.savefig('C:/Users/Juan/.gemini/antigravity/brain/49fd158d-b0e1-421c-b15b-b237b2e3ad7f/layout_yz_100mm_achromat_flipped.png', dpi=300)

# 3D CAD
meshes = []
physical_indices = [i for i, s in enumerate(sys_opt.surfaces) if not s.is_coord_break]
for i in range(len(physical_indices)-1):
    idx1, idx2 = physical_indices[i], physical_indices[i+1]
    s1, s2 = sys_opt.surfaces[idx1], sys_opt.surfaces[idx2]
    if s1.glass.upper() not in ['AIR', 'MIRROR'] and s1.thickness > 0:
        p1, p2, d = s1.global_origin, s2.global_origin, s1.semi_diameter
        pts = []
        for theta in np.linspace(0, 2*np.pi, 32, endpoint=False):
            u, v = s1.global_R @ np.array([1, 0, 0]), s1.global_R @ np.array([0, 1, 0])
            c1 = 1.0/s1.radius if s1.radius != 0 else 0
            z1 = (c1*d**2)/(1+np.sqrt(1-(1+s1.conic)*c1**2*d**2)) if c1!=0 else 0
            pts.append(p1 + d*(np.cos(theta)*u + np.sin(theta)*v) + s1.global_R @ np.array([0,0,z1]))
            u2, v2 = s2.global_R @ np.array([1, 0, 0]), s2.global_R @ np.array([0, 1, 0])
            c2 = 1.0/s2.radius if s2.radius != 0 else 0
            z2 = (c2*d**2)/(1+np.sqrt(1-(1+s2.conic)*c2**2*d**2)) if c2!=0 else 0
            pts.append(p2 + d*(np.cos(theta)*u2 + np.sin(theta)*v2) + s2.global_R @ np.array([0,0,z2]))
        try:
            hull = ConvexHull(pts)
            mesh = trimesh.Trimesh(vertices=np.array(pts), faces=hull.simplices)
            mesh.visual.face_colors = [100, 200, 255, 150]
            meshes.append(mesh)
        except: pass

for s in sys_opt.surfaces: s.semi_diameter = 100.0
ray_histories = []
for angle in np.linspace(0, 2*np.pi, 16, endpoint=False):
    dx, dy = np.cos(angle)*0.01, np.sin(angle)*0.01
    r = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([dx, dy, np.sqrt(1-dx**2-dy**2)]), wavelength=1.225)
    ray_histories.append(sys_opt.trace_ray(r))

slit_idx = next(k for k, idx in enumerate(physical_indices) if 'Slit' in sys_opt.surfaces[idx].name)
for k in range(slit_idx, len(physical_indices)-1):
    idx1, idx2 = physical_indices[k], physical_indices[k+1]
    s1, s2 = sys_opt.surfaces[idx1], sys_opt.surfaces[idx2]
    if s1.glass.upper() not in ['AIR', 'MIRROR'] and s1.thickness > 0: continue
    pts = []
    for tr in ray_histories:
        if idx1 < len(tr.history) and idx2 < len(tr.history):
            h1, h2 = tr.history[idx1], tr.history[idx2]
            pts.append(s1.global_origin + s1.global_R @ np.array([h1['x_local'], h1['y_local'], h1['z_local']]))
            pts.append(s2.global_origin + s2.global_R @ np.array([h2['x_local'], h2['y_local'], h2['z_local']]))
    if len(pts) > 4:
        try:
            hull = ConvexHull(pts)
            mesh = trimesh.Trimesh(vertices=np.array(pts), faces=hull.simplices)
            mesh.visual.face_colors = [255, 100, 100, 80]
            meshes.append(mesh)
        except: pass
scene = trimesh.Scene(meshes)
scene.export('C:/Users/Juan/.gemini/antigravity/brain/49fd158d-b0e1-421c-b15b-b237b2e3ad7f/Final_Assembly_100mm_Achromat_Flipped.glb')

