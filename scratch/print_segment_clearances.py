import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray
from scratch.test_dmd_tilts import test_config

sys_opt = test_config(dmd_tilt_y=-12.0, d_dmd_relay=50.0, d_relay_cond=20.0, d_cond_det=5.5, use_30mm_focus=True, f_relay=75.0)

components = {
    'Collimator': {'surfs': [1, 2, 3], 'radius': 12.7},
    'Grating': {'surfs': [6], 'radius': 12.7},
    'Focusing': {'surfs': [8, 9, 10], 'radius': 12.7},
    'DMD4500': {'surfs': [11], 'radius': 12.7},
    'Relay': {'surfs': [13, 14, 15], 'radius': 12.7},
    'Condenser': {'surfs': [16, 17], 'radius': 6.35},
}
for comp_name, comp_info in components.items():
    centers = [sys_opt.surfaces[idx].global_origin for idx in comp_info['surfs']]
    comp_info['center'] = np.mean(centers, axis=0)

r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=1.225)
tr = sys_opt.trace_ray(r)

for seg_idx in range(len(tr.history) - 1):
    h1 = tr.history[seg_idx]
    h2 = tr.history[seg_idx+1]
    p1 = np.array([h1['x_global'], h1['y_global'], h1['z_global']])
    p2 = np.array([h2['x_global'], h2['y_global'], h2['z_global']])
    s1_idx = h1['surface_index']
    s2_idx = h2['surface_index']
    s1_name = sys_opt.surfaces[s1_idx].name
    s2_name = sys_opt.surfaces[s2_idx].name
    
    seg_vec = p2 - p1
    seg_len = np.linalg.norm(seg_vec)
    if seg_len < 1e-6: continue
    seg_dir = seg_vec / seg_len
    
    for comp_name, comp_info in components.items():
        comp_surfs = set(comp_info['surfs'])
        if comp_name == 'Grating': comp_surfs.update([5, 7])
        elif comp_name == 'DMD4500': comp_surfs.update([12])
        if s1_idx in comp_surfs or s2_idx in comp_surfs: continue
        
        comp_center = comp_info['center']
        req_dist = comp_info['radius'] + 5.0
        
        v = comp_center - p1
        t = np.clip(np.dot(v, seg_dir), 0.0, seg_len)
        closest_pt = p1 + t * seg_dir
        dist = np.linalg.norm(comp_center - closest_pt)
        margin_clr = dist - req_dist
        
        print(f"Seg {s1_name:18s} -> {s2_name:18s} vs {comp_name:10s}: dist={dist:6.2f} mm, req={req_dist:5.2f} mm, margin={margin_clr:6.2f} mm (t/len={t:.2f}/{seg_len:.2f})")
