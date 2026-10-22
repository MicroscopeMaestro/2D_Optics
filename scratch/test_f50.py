import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray
from scratch.test_dmd_tilts import test_config

print("Testing f=50mm focusing lens with various relay distances...")
for dmd_tilt in [12.0, -12.0]:
    for d_dmd in [50.0, 75.0, 100.0]:
        for d_mid in [15.0, 25.0]:
            for d_det in np.linspace(4.5, 6.5, 5):
                sys_opt = test_config(dmd_tilt_y=dmd_tilt, d_dmd_relay=d_dmd, d_relay_cond=d_mid, d_cond_det=d_det, use_30mm_focus=False, f_relay=75.0)

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

                wls = [1.087, 1.225, 1.350]
                rays = []
                for wl in wls:
                    rays.append(Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=wl))
                    theta_max = np.arcsin(0.04)
                    for theta in [theta_max]:
                        for phi in [0, np.pi/2, np.pi, 3*np.pi/2]:
                            dx = np.sin(theta)*np.cos(phi)
                            dy = np.sin(theta)*np.sin(phi)
                            dz = np.cos(theta)
                            rays.append(Ray(pos=np.array([0., 0., 0.]), dir=np.array([dx, dy, dz]), wavelength=wl))

                traced = [sys_opt.trace_ray(r) for r in rays]
                vignetted_count = sum(1 for tr in traced if tr.vignetted)
                
                min_margin = 999.0
                min_center_dist = 999.0
                for tr in traced:
                    for seg_idx in range(len(tr.history) - 1):
                        h1 = tr.history[seg_idx]
                        h2 = tr.history[seg_idx+1]
                        p1 = np.array([h1['x_global'], h1['y_global'], h1['z_global']])
                        p2 = np.array([h2['x_global'], h2['y_global'], h2['z_global']])
                        s1_idx = h1['surface_index']
                        s2_idx = h2['surface_index']
                        
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
                            if dist < min_center_dist: min_center_dist = dist
                            if margin_clr < min_margin: min_margin = margin_clr

                hits_x = [tr.history[-1]['x_local'] for tr in traced if not tr.vignetted]
                hits_y = [tr.history[-1]['y_local'] for tr in traced if not tr.vignetted]
                if hits_x:
                    mx, my = np.mean(hits_x), np.mean(hits_y)
                    max_r = np.max(np.sqrt((np.array(hits_x)-mx)**2 + (np.array(hits_y)-my)**2))
                    r_origin = np.max(np.sqrt(np.array(hits_x)**2 + np.array(hits_y)**2))
                else:
                    max_r = 999.0
                    r_origin = 999.0
                    
                if vignetted_count == 0 and min_margin > 0 and max_r <= 0.40:
                    print(f"PERFECT MATCH: tilt={dmd_tilt:5.1f}, d_dmd={d_dmd:5.1f}, d_mid={d_mid:5.1f}, d_det={d_det:5.2f} -> Vig={vignetted_count:2d}/{len(rays)}, Max_R={max_r:.4f} mm, R_origin={r_origin:.4f} mm, MinMargin={min_margin:.2f} mm, MinCenterDist={min_center_dist:.2f} mm")
