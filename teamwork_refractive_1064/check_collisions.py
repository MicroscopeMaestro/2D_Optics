import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

def verify_collisions(json_path=None, verbose=True):
    if json_path is None:
        json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spectrometer_refractive_1064.json')
        
    sys_opt = OpticalSystem.from_json(json_path)
    sys_opt._update_accumulated_positions()
    
    # Map physical optical components and their mechanical semi-diameters
    components = {
        'Collimator': {'surfs': [1, 2, 3], 'radius': 12.7},
        'Grating': {'surfs': [6], 'radius': 12.7, 'cbs': [5, 7]},
        'Focusing': {'surfs': [8, 9, 10], 'radius': 12.7},
        'DMD4500': {'surfs': [11], 'radius': 12.7, 'cbs': [12]},
        'Relay': {'surfs': [13, 14, 15], 'radius': sys_opt.surfaces[13].semi_diameter},
        'Condenser': {'surfs': [16, 17], 'radius': 12.7},
    }
    
    for comp_name, comp_info in components.items():
        centers = [sys_opt.surfaces[idx].global_origin for idx in comp_info['surfs']]
        comp_info['center'] = np.mean(centers, axis=0)
        
    if verbose:
        print(f"=== 3D Collision & Clearance Verification ({os.path.basename(json_path)}) ===")
        print("Optical Component Center Positions and Physical Radii:")
        for name, c in components.items():
            print(f"  {name:12s}: center = [{c['center'][0]:6.2f}, {c['center'][1]:6.2f}, {c['center'][2]:6.2f}] mm, radius = {c['radius']:5.1f} mm")
            
    # Trace rays (chief and marginal rays across Raman spectrum)
    wls = [1.087, 1.225, 1.350]
    rays = []
    for wl in wls:
        # Chief ray
        rays.append(Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=wl))
        # Marginal rays across NA cone
        theta_max = np.arcsin(0.04)
        for theta in [theta_max / 2.0, theta_max]:
            for phi in np.linspace(0, 2*np.pi, 8, endpoint=False):
                dx = np.sin(theta) * np.cos(phi)
                dy = np.sin(theta) * np.sin(phi)
                dz = np.cos(theta)
                rays.append(Ray(pos=np.array([0., 0., 0.]), dir=np.array([dx, dy, dz]), wavelength=wl))

    traced = [sys_opt.trace_ray(r) for r in rays]
    
    min_clearance_to_center = 999.0
    min_margin_clearance = 999.0
    violations = []
    clearance_records = []
    
    for r_idx, tr in enumerate(traced):
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
            if seg_len < 1e-6:
                continue
            seg_dir = seg_vec / seg_len
            
            for comp_name, comp_info in components.items():
                comp_surfs = set(comp_info['surfs'])
                if 'cbs' in comp_info:
                    comp_surfs.update(comp_info['cbs'])
                    
                # Skip if ray segment connects to this component
                if s1_idx in comp_surfs or s2_idx in comp_surfs:
                    continue
                    
                comp_center = comp_info['center']
                req_dist = comp_info['radius'] + 5.0  # +5mm physical margin
                
                v = comp_center - p1
                t = np.clip(np.dot(v, seg_dir), 0.0, seg_len)
                closest_pt = p1 + t * seg_dir
                dist = float(np.linalg.norm(comp_center - closest_pt))
                margin_clr = dist - req_dist
                
                if dist < min_clearance_to_center:
                    min_clearance_to_center = dist
                if margin_clr < min_margin_clearance:
                    min_margin_clearance = margin_clr
                    
                clearance_records.append({
                    'ray_index': r_idx,
                    'wavelength': tr.wavelength,
                    'segment': f"{s1_name} -> {s2_name}",
                    'component': comp_name,
                    'distance': dist,
                    'required_distance': req_dist,
                    'margin_clearance': margin_clr
                })
                
                if margin_clr < 0:
                    violations.append({
                        'ray_index': r_idx,
                        'wavelength': tr.wavelength,
                        'segment': f"{s1_name} -> {s2_name}",
                        'component': comp_name,
                        'distance': dist,
                        'required_distance': req_dist,
                        'margin_clearance': margin_clr
                    })

    passed = (len(violations) == 0) and (min_margin_clearance > 0)
    
    if verbose:
        print(f"\n--- Clearance Summary across {len(rays)} Rays ---")
        print(f"  Minimum distance to any unrelated housing center: {min_clearance_to_center:.2f} mm (> 0 REQUIRED)")
        print(f"  Minimum margin clearance beyond (radius + 5mm):  {min_margin_clearance:.2f} mm (> 0 REQUIRED)")
        print(f"  Total Segments Checked: {len(clearance_records)}")
        print(f"  Violations Count:       {len(violations)}")
        print(f"  Collision Check Result: {'PASS' if passed else 'FAIL'}")
        
        if violations:
            print("\n  VIOLATIONS:")
            for v in violations[:10]:
                print(f"    Ray {v['ray_index']} ({v['wavelength']} um) {v['segment']} vs {v['component']}: dist={v['distance']:.2f} mm, req={v['required_distance']:.2f} mm, diff={v['margin_clearance']:.2f} mm")
                
    return passed, min_clearance_to_center, min_margin_clearance, len(violations)

if __name__ == '__main__':
    passed, min_dist, min_margin, n_viol = verify_collisions()
    assert passed, f"Collision check failed with {n_viol} violations (min margin: {min_margin:.2f} mm)!"
    sys.exit(0 if passed else 1)
