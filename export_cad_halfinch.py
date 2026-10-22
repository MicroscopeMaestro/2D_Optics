
import sys, os, json, numpy as np
import trimesh
from scipy.spatial import ConvexHull
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

sys_opt = OpticalSystem.from_json('spectrometer_yz_75mm_halfinch.json')
sys_opt._update_accumulated_positions()

meshes = []

for i in range(len(sys_opt.surfaces)-1):
    s1 = sys_opt.surfaces[i]
    s2 = sys_opt.surfaces[i+1]
    if s1.is_coord_break: continue
    if s1.glass.upper() not in ['AIR', 'MIRROR']:
        p1, p2, d = s1.global_origin, s2.global_origin, s1.semi_diameter
        pts = []
        for theta in np.linspace(0, 2*np.pi, 32, endpoint=False):
            u = s1.global_R @ np.array([1, 0, 0])
            v = s1.global_R @ np.array([0, 1, 0])
            pts.append(p1 + d * (np.cos(theta)*u + np.sin(theta)*v))
            u2 = s2.global_R @ np.array([1, 0, 0])
            v2 = s2.global_R @ np.array([0, 1, 0])
            pts.append(p2 + d * (np.cos(theta)*u2 + np.sin(theta)*v2))
        hull = ConvexHull(pts)
        mesh = trimesh.Trimesh(vertices=pts, faces=hull.simplices)
        mesh.visual.face_colors = [0, 0, 255, 100]
        meshes.append(mesh)

beam_hits = []
r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0, 0, 1]), wavelength=1.087)
tr_ref = sys_opt.trace_ray(r)
num_hist = len(tr_ref.history)
beam_hits_ordered = [[] for _ in range(num_hist)]

for theta in [np.arcsin(0.065)]:
    for phi in np.linspace(0, 2*np.pi, 16, endpoint=False):
        dx, dy, dz = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
        r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([dx, dy, dz]), wavelength=1.087)
        tr = sys_opt.trace_ray(r)
        for i, h in enumerate(tr.history):
            if 'x_global' in h:
                pt = np.array([h['x_global'], h['y_global'], h['z_global']])
                beam_hits_ordered[i].append(pt)

for i in range(num_hist - 1):
    pts1, pts2 = beam_hits_ordered[i], beam_hits_ordered[i+1]
    if len(pts1) < 4 or len(pts2) < 4: continue
    if np.linalg.norm(np.mean(pts1, axis=0) - np.mean(pts2, axis=0)) < 1e-3: continue
    pts = pts1 + pts2
    try:
        hull = ConvexHull(pts)
        mesh = trimesh.Trimesh(vertices=pts, faces=hull.simplices)
        mesh.visual.face_colors = [255, 0, 0, 50]
        meshes.append(mesh)
    except Exception: pass

scene = trimesh.Scene(meshes)
scene.export('C:/Users/Juan/.gemini/antigravity/brain/49fd158d-b0e1-421c-b15b-b237b2e3ad7f/Final_Assembly_12mm_Relay.glb')

