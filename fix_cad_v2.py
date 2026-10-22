
import sys, os, json, numpy as np
import trimesh
from scipy.spatial import ConvexHull
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

sys_opt_orig = OpticalSystem.from_json('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json')
sys_opt_orig._update_accumulated_positions()

# Trace a ray envelope
ray_histories = []
for angle in np.linspace(0, 2*np.pi, 16, endpoint=False):
    dx, dy = np.cos(angle)*0.01, np.sin(angle)*0.01
    dz = np.sqrt(1 - dx**2 - dy**2)
    r = Ray(pos=sys_opt_orig.surfaces[0].global_origin, dir=np.array([dx, dy, dz]), wavelength=1.225)
    ray_histories.append(sys_opt_orig.trace_ray(r))

# Find all PHYSICAL surfaces (not Coord Breaks)
physical_indices = []
for i, s in enumerate(sys_opt_orig.surfaces):
    if not s.is_coord_break:
        physical_indices.append(i)

meshes = []

# 1. Glass Lenses
for i in range(len(physical_indices)-1):
    idx1, idx2 = physical_indices[i], physical_indices[i+1]
    s1, s2 = sys_opt_orig.surfaces[idx1], sys_opt_orig.surfaces[idx2]
    
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
            
            u2 = s2.global_R @ np.array([1, 0, 0])
            v2 = s2.global_R @ np.array([0, 1, 0])
            z2 = 0
            if s2.radius != 0:
                c2 = 1.0 / s2.radius
                z2 = (c2 * d**2) / (1 + np.sqrt(1 - (1+s2.conic)*c2**2 * d**2))
            pts.append(p2 + d * (np.cos(theta)*u2 + np.sin(theta)*v2) + s2.global_R @ np.array([0,0,z2]))
        try:
            hull = ConvexHull(pts)
            mesh = trimesh.Trimesh(vertices=np.array(pts), faces=hull.simplices)
            mesh.visual.face_colors = [100, 200, 255, 150]
            meshes.append(mesh)
        except: pass

# 2. Air Elements (Beam Volume)
# Start from Slit (which is the first physical surface after Collect S3)
slit_list_idx = 0
for k, idx in enumerate(physical_indices):
    if 'Slit' in sys_opt_orig.surfaces[idx].name:
        slit_list_idx = k
        break

for k in range(slit_list_idx, len(physical_indices)-1):
    idx1, idx2 = physical_indices[k], physical_indices[k+1]
    s1, s2 = sys_opt_orig.surfaces[idx1], sys_opt_orig.surfaces[idx2]
    
    # We only draw AIR between surfaces if it's not inside a glass lens
    if s1.glass.upper() not in ['AIR', 'MIRROR'] and s1.thickness > 0:
        continue # This is a solid lens, skip air drawing
        
    pts = []
    for tr in ray_histories:
        if idx1 < len(tr.history) and idx2 < len(tr.history):
            h1 = tr.history[idx1]
            pts.append(s1.global_origin + s1.global_R @ np.array([h1['x_local'], h1['y_local'], h1['z_local']]))
            h2 = tr.history[idx2]
            pts.append(s2.global_origin + s2.global_R @ np.array([h2['x_local'], h2['y_local'], h2['z_local']]))
            
    if len(pts) > 4:
        try:
            hull = ConvexHull(pts)
            mesh = trimesh.Trimesh(vertices=np.array(pts), faces=hull.simplices)
            mesh.visual.face_colors = [255, 100, 100, 80]
            meshes.append(mesh)
        except: pass

scene = trimesh.Scene(meshes)
scene.export('C:/Users/Juan/.gemini/antigravity/brain/49fd158d-b0e1-421c-b15b-b237b2e3ad7f/Final_Assembly_With_Air.glb')

