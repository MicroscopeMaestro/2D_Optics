
import sys, os, json, numpy as np
import trimesh
from scipy.spatial import ConvexHull
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

sys_opt = OpticalSystem.from_json('teamwork_yz_spectrometer/spectrometer_yz_1064.json')
sys_opt._update_accumulated_positions()

# We will store a list of meshes
meshes = []
colors = []

# 1. Create simple cylindrical representations for the lenses
for i in range(len(sys_opt.surfaces)-1):
    s1 = sys_opt.surfaces[i]
    s2 = sys_opt.surfaces[i+1]
    
    if s1.is_coord_break: continue
    
    # If this gap is glass, it's a lens element
    if s1.glass.upper() not in ['AIR', 'MIRROR']:
        # Create a cylinder from s1 origin to s2 origin
        p1 = s1.global_origin
        p2 = s2.global_origin
        d = s1.semi_diameter
        
        # Create points for circles at p1 and p2
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
        mesh.visual.face_colors = [0, 0, 255, 100] # Blue for glass
        meshes.append(mesh)
        mesh.export(f'cad_export/Lens_{s1.name}.stl')

# 2. Create the Air path (Beam volume)
# We will trace marginal rays and record their hits at each surface
# Then we will build a convex hull between surface N and N+1
beam_hits = {i: [] for i in range(len(sys_opt.surfaces))}

for theta in [np.arcsin(0.2)]: # NA 0.2 from sample
    for phi in np.linspace(0, 2*np.pi, 16, endpoint=False):
        dx, dy, dz = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
        r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([dx, dy, dz]), wavelength=1.087)
        tr = sys_opt.trace_ray(r)
        
        # Assign history points to surfaces
        s_idx = 0
        for h in tr.history:
            if 'x_global' in h:
                # Find the closest surface origin
                pt = np.array([h['x_global'], h['y_global'], h['z_global']])
                beam_hits[s_idx].append(pt)
                s_idx += 1
                
                # If we hit the coordinate break, skip incrementing the actual physical hit
                # Wait, tracer history includes coord breaks if they are traced.
                # Let's just collect all history points in order.

# Rebuild beam hits based on history index
history_pts = []
r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0, 0, 1]), wavelength=1.087)
tr_ref = sys_opt.trace_ray(r)
num_hist = len(tr_ref.history)
beam_hits_ordered = [[] for _ in range(num_hist)]

for phi in np.linspace(0, 2*np.pi, 16, endpoint=False):
    dx, dy, dz = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
    r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([dx, dy, dz]), wavelength=1.087)
    tr = sys_opt.trace_ray(r)
    for i, h in enumerate(tr.history):
        if 'x_global' in h:
            pt = np.array([h['x_global'], h['y_global'], h['z_global']])
            beam_hits_ordered[i].append(pt)

for i in range(num_hist - 1):
    pts1 = beam_hits_ordered[i]
    pts2 = beam_hits_ordered[i+1]
    if len(pts1) < 4 or len(pts2) < 4: continue
    
    # Check if the points are identical (e.g. dummy surface or coord break at same location)
    if np.linalg.norm(np.mean(pts1, axis=0) - np.mean(pts2, axis=0)) < 1e-3:
        continue
        
    pts = pts1 + pts2
    try:
        hull = ConvexHull(pts)
        mesh = trimesh.Trimesh(vertices=pts, faces=hull.simplices)
        mesh.visual.face_colors = [255, 0, 0, 50] # Red for Air beam
        meshes.append(mesh)
        mesh.export(f'cad_export/AirPath_Segment_{i}.stl')
    except Exception as e:
        print(f'Skipped segment {i} due to flat hull: {e}')

# Export combined assembly
scene = trimesh.Scene(meshes)
scene.export('cad_export/Full_Assembly.glb')
print('CAD Export Complete! Check the cad_export directory.')

