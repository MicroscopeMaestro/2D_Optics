
import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

sys_opt = OpticalSystem.from_json('spectrometer_refractive_1064.json')
sys_opt._update_accumulated_positions()

r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=1.225)
tr = sys_opt.trace_ray(r)

for i in range(len(tr.history) - 1):
    h1 = tr.history[i]
    h2 = tr.history[i+1]
    p1 = np.array([h1['x_global'], h1['y_global'], h1['z_global']])
    p2 = np.array([h2['x_global'], h2['y_global'], h2['z_global']])
    
    s_idx_1 = h1['surface_index']
    s_idx_2 = h2['surface_index']
    
    segment_vec = p2 - p1
    segment_len = np.linalg.norm(segment_vec)
    if segment_len < 1e-6: continue
    segment_dir = segment_vec / segment_len
    
    for j, s in enumerate(sys_opt.surfaces):
        if j == s_idx_1 or j == s_idx_2: continue
        if s.is_coord_break: continue
        if 'object' in str(s.surface_type).lower() or 'stop' in str(s.surface_type).lower(): continue
        
        center = s.global_origin
        v = center - p1
        t = np.dot(v, segment_dir)
        
        if 0 <= t <= segment_len:
            proj = p1 + t * segment_dir
            dist = np.linalg.norm(center - proj)
            if dist < s.semi_diameter + 5.0:
                print(f'\n[Segment {sys_opt.surfaces[s_idx_1].name} -> {sys_opt.surfaces[s_idx_2].name}] crosses near {s.name}')
                print(f'  Ray projection point: {proj}')
                print(f'  Surface center: {center}')
                print(f'  Distance: {dist:.2f} mm')

