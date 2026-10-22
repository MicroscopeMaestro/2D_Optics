
import sys, os, json, numpy as np
import trimesh
from scipy.spatial import ConvexHull
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

sys_opt = OpticalSystem.from_json('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json')
sys_opt._update_accumulated_positions()

meshes = []

# 1. Generate Glass Lenses
for i in range(len(sys_opt.surfaces)-1):
    s1 = sys_opt.surfaces[i]
    s2 = sys_opt.surfaces[i+1]
    if s1.is_coord_break: continue
    if s1.glass.upper() not in ['AIR', 'MIRROR'] and s1.thickness > 0:
        p1, p2, d = s1.global_origin, s2.global_origin, s1.semi_diameter
        pts = []
        for theta in np.linspace(0, 2*np.pi, 32, endpoint=False):
            u = s1.global_R @ np.array([1, 0, 0])
            v = s1.global_R @ np.array([0, 1, 0])
            z_sag = 0
            if s1.radius != 0:
                c = 1.0 / s1.radius
                k = s1.conic
                z_sag = (c * d**2) / (1 + np.sqrt(1 - (1+k)*c**2 * d**2))
            pts.append(p1 + d * (np.cos(theta)*u + np.sin(theta)*v) + s1.global_R @ np.array([0,0,z_sag]))
            
            u2 = s2.global_R @ np.array([1, 0, 0])
            v2 = s2.global_R @ np.array([0, 1, 0])
            z_sag2 = 0
            if s2.radius != 0:
                c2 = 1.0 / s2.radius
                k2 = s2.conic
                z_sag2 = (c2 * d**2) / (1 + np.sqrt(1 - (1+k2)*c2**2 * d**2))
            pts.append(p2 + d * (np.cos(theta)*u2 + np.sin(theta)*v2) + s2.global_R @ np.array([0,0,z_sag2]))
        
        hull = ConvexHull(pts)
        mesh = trimesh.Trimesh(vertices=np.array(pts), faces=hull.simplices)
        mesh.visual.face_colors = [100, 200, 255, 150]
        meshes.append(mesh)

# 2. Generate Air Elements (Beam Volume)
# The aperture stop has semi_diameter = 2.0. We trace rays to hit the edge of it.
# The slit is at index 6. Stop is at index 10.
# The distance is ~ 30 + 41.5 = 71.5 mm. So theta ~ 2.0 / 71.5 = 0.028 radians.
slit_pos = sys_opt.surfaces[6].global_origin
slit_R = sys_opt.surfaces[6].global_R

ray_histories = []
N_rays = 16
for angle in np.linspace(0, 2*np.pi, N_rays, endpoint=False):
    # Dir pointing slightly out to hit the 2mm stop
    dx = np.cos(angle) * 0.028
    dy = np.sin(angle) * 0.028
    dz = np.sqrt(1 - dx**2 - dy**2)
    dir_global = slit_R @ np.array([dx, dy, dz])
    
    r = Ray(pos=slit_pos, dir=dir_global, wavelength=1.225)
    tr = sys_opt.trace_ray(r)
    ray_histories.append(tr)

# Build a segment of 'air' between every pair of surfaces (from Slit to Detector)
for i in range(6, len(sys_opt.surfaces)-1):
    s_curr = sys_opt.surfaces[i]
    s_next = sys_opt.surfaces[i+1]
    if s_curr.is_coord_break: continue
    if s_curr.thickness == 0: continue
    
    # Check if all rays made it to this segment
    valid = True
    for tr in ray_histories:
        if i+1 >= len(tr.history) or (tr.vignetted and tr.vignetted_surface <= i):
            valid = False
    if not valid: continue
    
    pts = []
    # Collect points at surface i
    for tr in ray_histories:
        h = tr.history[i]
        pt = s_curr.global_origin + s_curr.global_R @ np.array([h['x_local'], h['y_local'], h['z_local']])
        pts.append(pt)
    
    # Collect points at surface i+1
    for tr in ray_histories:
        h = tr.history[i+1]
        pt = s_next.global_origin + s_next.global_R @ np.array([h['x_local'], h['y_local'], h['z_local']])
        pts.append(pt)
        
    hull = ConvexHull(pts)
    mesh = trimesh.Trimesh(vertices=np.array(pts), faces=hull.simplices)
    mesh.visual.face_colors = [255, 100, 100, 80] # transparent red for beam
    meshes.append(mesh)

scene = trimesh.Scene(meshes)
scene.export('C:/Users/Juan/.gemini/antigravity/brain/49fd158d-b0e1-421c-b15b-b237b2e3ad7f/Final_Assembly_With_Air.glb')

