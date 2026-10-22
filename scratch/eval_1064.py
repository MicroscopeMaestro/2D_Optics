import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

def full_eval(json_path):
    sys_opt = OpticalSystem.from_json(json_path)
    sys_opt._update_accumulated_positions()
    
    print(f"=== Evaluating {json_path} ===")
    
    # 1. DMD Dispersion
    dmd_idx = next(i for i, s in enumerate(sys_opt.surfaces) if 'DMD' in s.name)
    det_idx = next(i for i, s in enumerate(sys_opt.surfaces) if s.surface_type == 'Detector' or 'Detector' in s.name)
    
    wls = [1.087, 1.225, 1.350]
    dmd_hits_y = []
    dmd_hits_x = []
    
    for wl in wls:
        r0 = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=wl)
        tr = sys_opt.trace_ray(r0)
        for h in tr.history:
            if h['surface_index'] == dmd_idx:
                dmd_hits_x.append(h['x_local'])
                dmd_hits_y.append(h['y_local'])
    
    span_y = max(dmd_hits_y) - min(dmd_hits_y)
    span_x = max(dmd_hits_x) - min(dmd_hits_x)
    print(f"DMD dispersion span: X={span_x:.3f} mm, Y={span_y:.3f} mm, Total Span={np.sqrt(span_x**2 + span_y**2):.3f} mm")
    
    # 2. Spot radius across bundle
    total_rays = 0
    vignetted_rays = 0
    det_x, det_y = [], []
    
    # Ray bundle: 3 wavelengths, NA up to 0.04 (or 0.12)
    theta_max = np.arcsin(0.04)
    for wl in wls:
        for theta in np.linspace(0.001, theta_max, 5):
            for phi in np.linspace(0, 2*np.pi, 8, endpoint=False):
                dx = np.sin(theta) * np.cos(phi)
                dy = np.sin(theta) * np.sin(phi)
                dz = np.cos(theta)
                r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([dx, dy, dz]), wavelength=wl)
                tr = sys_opt.trace_ray(r)
                total_rays += 1
                if tr.vignetted:
                    vignetted_rays += 1
                else:
                    h = tr.history[-1]
                    det_x.append(h['x_local'])
                    det_y.append(h['y_local'])
    
    print(f"Total rays: {total_rays}, Vignetted: {vignetted_rays}, Non-vignetted: {len(det_x)}")
    if det_x:
        mx, my = np.mean(det_x), np.mean(det_y)
        radii_from_centroid = np.sqrt((np.array(det_x) - mx)**2 + (np.array(det_y) - my)**2)
        radii_from_origin = np.sqrt(np.array(det_x)**2 + np.array(det_y)**2)
        print(f"Max spot radius from centroid: {np.max(radii_from_centroid):.4f} mm")
        print(f"Max spot radius from origin (0,0): {np.max(radii_from_origin):.4f} mm")
        print(f"RMS spot radius: {np.sqrt(np.mean(radii_from_centroid**2)):.4f} mm")
    
    # 3. Collision / clearance check
    # Check ray segments against surface physical centers
    chief_rays = [Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=wl) for wl in wls]
    traced_chief = [sys_opt.trace_ray(r) for r in chief_rays]
    
    min_clearance = 999.0
    collision_violations = []
    
    for tr in traced_chief:
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
            
            for j, s in enumerate(sys_opt.surfaces):
                if j == s1_idx or j == s2_idx: continue
                if s.is_coord_break: continue
                if str(s.surface_type).lower() in ('object', 'stop', 'detector'): continue
                # Same element surface check
                if s1_idx < len(sys_opt.surfaces) and s2_idx < len(sys_opt.surfaces):
                    name_s = s.name
                    name_s1 = sys_opt.surfaces[s1_idx].name
                    name_s2 = sys_opt.surfaces[s2_idx].name
                    # If same lens group, skip
                    if any(grp in name_s and (grp in name_s1 or grp in name_s2) for grp in ['Collimator', 'Focusing', 'Relay', 'Cond']):
                        continue
                
                center = s.global_origin
                v = center - p1
                t = np.dot(v, seg_dir)
                t_clamped = np.clip(t, 0.0, seg_len)
                proj = p1 + t_clamped * seg_dir
                dist = np.linalg.norm(center - proj)
                margin_req = s.semi_diameter + 5.0
                clearance = dist - margin_req
                if clearance < min_clearance:
                    min_clearance = clearance
                if clearance < 0:
                    collision_violations.append((name_s1, name_s2, s.name, dist, margin_req, clearance))
    
    print(f"Minimum clearance (dist - (semi_diameter+5mm)): {min_clearance:.2f} mm")
    if collision_violations:
        print("Collision violations:")
        for v in collision_violations:
            print(f"  Segment {v[0]} -> {v[1]} near {v[2]}: dist={v[3]:.2f} mm, required={v[4]:.2f} mm, margin diff={v[5]:.2f} mm")
    else:
        print("No collision violations! All ray segments clear optical housings with > 5mm margin.")

if __name__ == '__main__':
    full_eval('spectrometer_refractive_1064.json')
