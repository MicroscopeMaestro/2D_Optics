import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

sys_opt = OpticalSystem.from_json('spectrometer_refractive_1064.json')
sys_opt._update_accumulated_positions()

# Elements in spectrometer_refractive_1064.json:
# 0: Slit (0.00, 0.00, 0.00)
# 1-3: Collimator (0.00, 0.00, 30.53) - radius 12.7
# 4: Aperture Stop (0.00, 0.00, 64.80) - radius 5.0
# 5: CB_Before_Grating (0.00, 0.00, 89.80)
# 6: Grating_600lpmm (0.00, 0.00, 89.80) - radius 12.7
# 7: CB_After_Grating (0.00, 0.00, 89.80)
# 8-10: Focusing Lens (36.87, -0.00, 50.81) - radius 12.7
# 11: DMD4500 (71.46, -0.00, 14.24) - radius 12.7 (active 5.0)
# 12: CB_After_DMD (71.46, -0.00, 14.24)
# 13-15: Imaging_Relay ( -26.57, -63.52, 117.89) - radius 25.4
# 16-17: Imaging_Cond ( -366.11, -283.53, 476.93) - radius 12.7
# 18: Detector (-378.72, -291.71, 490.26) - radius 0.4

components = {
    'Collimator': {'surfs': [1, 2, 3], 'radius': 12.7, 'center': np.mean([sys_opt.surfaces[k].global_origin for k in [1,2,3]], axis=0)},
    'Grating': {'surfs': [6], 'radius': 12.7, 'center': sys_opt.surfaces[6].global_origin},
    'Focusing': {'surfs': [8, 9, 10], 'radius': 12.7, 'center': np.mean([sys_opt.surfaces[k].global_origin for k in [8,9,10]], axis=0)},
    'DMD4500': {'surfs': [11], 'radius': 12.7, 'center': sys_opt.surfaces[11].global_origin},
    'Relay': {'surfs': [13, 14, 15], 'radius': 25.4, 'center': np.mean([sys_opt.surfaces[k].global_origin for k in [13,14,15]], axis=0)},
    'Condenser': {'surfs': [16, 17], 'radius': 12.7, 'center': np.mean([sys_opt.surfaces[k].global_origin for k in [16,17]], axis=0)},
}

print("Physical component centers in spectrometer_refractive_1064.json:")
for name, c in components.items():
    print(f"  {name:12s}: center = {np.round(c['center'], 2)}, radius = {c['radius']} mm")

# Trace rays
wls = [1.087, 1.225, 1.350]
rays = []
for wl in wls:
    for theta in [0.001, np.arcsin(0.04)]:
        for phi in np.linspace(0, 2*np.pi, 4, endpoint=False):
            dx, dy, dz = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
            rays.append(Ray(pos=np.array([0., 0., 0.]), dir=np.array([dx, dy, dz]), wavelength=wl))

traced = [sys_opt.trace_ray(r) for r in rays]

# Check segment clearance to each unrelated component
for r_idx, tr in enumerate(traced[:3]):
    print(f"\n--- Ray {r_idx} (WL {tr.wavelength} um) ---")
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
            
            if margin_clr < 10.0:
                print(f"  Seg {s1_name:18s} -> {s2_name:18s} vs {comp_name:10s}: dist={dist:6.2f} mm, req={req_dist:5.2f} mm, margin={margin_clr:6.2f} mm")
