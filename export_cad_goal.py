
import sys, os, json, numpy as np
import trimesh
from scipy.spatial import ConvexHull
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

sys_opt = OpticalSystem.from_json('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json')
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

scene = trimesh.Scene(meshes)
scene.export('C:/Users/Juan/.gemini/antigravity/brain/49fd158d-b0e1-421c-b15b-b237b2e3ad7f/Final_Goal_Assembly.glb')

