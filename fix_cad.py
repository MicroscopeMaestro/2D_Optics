
import sys, os, json, numpy as np
import trimesh
from scipy.spatial import ConvexHull
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

sys_opt = OpticalSystem.from_json('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json')
# Widen everything to avoid false vignetting during envelope trace
for s in sys_opt.surfaces: s.semi_diameter = 100.0
sys_opt._update_accumulated_positions()

meshes = []
# 1. Glass
sys_opt_orig = OpticalSystem.from_json('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json')
sys_opt_orig._update_accumulated_positions()
for i in range(len(sys_opt_orig.surfaces)-1):
    s1 = sys_opt_orig.surfaces[i]
    s2 = sys_opt_orig.surfaces[i+1]
    if s1.is_coord_break: continue
    if s1.glass.upper() not in ['AIR', 'MIRROR'] and s1.thickness > 0:
        p1, p2, d = s1.global_origin, s2.global_origin, s1.semi_diameter
        pts = []
        for theta in np.linspace(0, 2*np.pi, 32, endpoint=False):
            u = s1.global_R @ np.array([1, 0, 0])
            v = s1.global_R @ np.array([0, 1, 0])
            z1 = 0
            if s1.radius != 0:
                c1 = 1.0 / s1.radius
                z1 = (c1 * d**2) / (1 + np.sqrt(1 - (1+s1.conic)*c1**2 * d**2))
            pts.append(p1 + d * (np.cos(theta)*u + np.sin(theta)*v) + s1.global_R @ np.array([0,0,z1]))
            z2 = 0
            if s2.radius != 0:
                c2 = 1.0 / s2.radius
                z2 = (c2 * d**2) / (1 + np.sqrt(1 - (1+s2.conic)*c2**2 * d**2))
            pts.append(p2 + d * (np.cos(theta)*u2 + np.sin(theta)*v2) if 'u2' in locals() else p2 + d * (np.cos(theta)*u + np.sin(theta)*v) + s2.global_R @ np.array([0,0,z2]))
        try:
            hull = ConvexHull(pts)
            mesh = trimesh.Trimesh(vertices=np.array(pts), faces=hull.simplices)
            mesh.visual.face_colors = [100, 200, 255, 150]
            meshes.append(mesh)
        except: pass

# 2. Air Elements
ray_histories = []
for angle in np.linspace(0, 2*np.pi, 16, endpoint=False):
    dx, dy = np.cos(angle)*0.01, np.sin(angle)*0.01
    dz = np.sqrt(1 - dx**2 - dy**2)
    r = Ray(pos=sys_opt.surfaces[0].global_origin, dir=np.array([dx, dy, dz]), wavelength=1.225)
    ray_histories.append(sys_opt.trace_ray(r))

for i in range(len(sys_opt_orig.surfaces)-1):
    s_curr, s_next = sys_opt_orig.surfaces[i], sys_opt_orig.surfaces[i+1]
    if s_curr.is_coord_break or s_curr.thickness == 0: continue
    
    pts = []
    for tr in ray_histories:
        if i < len(tr.history):
            h = tr.history[i]
            pts.append(s_curr.global_origin + s_curr.global_R @ np.array([h['x_local'], h['y_local'], h['z_local']]))
        if i+1 < len(tr.history):
            h = tr.history[i+1]
            pts.append(s_next.global_origin + s_next.global_R @ np.array([h['x_local'], h['y_local'], h['z_local']]))
    if len(pts) > 4:
        try:
            hull = ConvexHull(pts)
            mesh = trimesh.Trimesh(vertices=np.array(pts), faces=hull.simplices)
            mesh.visual.face_colors = [255, 100, 100, 80]
            meshes.append(mesh)
        except: pass

scene = trimesh.Scene(meshes)
scene.export('C:/Users/Juan/.gemini/antigravity/brain/49fd158d-b0e1-421c-b15b-b237b2e3ad7f/Final_Assembly_With_Air.glb')

