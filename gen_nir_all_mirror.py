
import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

def build_layout(m1_tilt, grat_tilt, m3_tilt, m4_tilt):
    data = {
        'system_config': {'primary_wavelength_um': 1.75},
        'surfaces': [
            {'name': 'Input_Slit_30um', 'surface_type': 'Object', 'thickness': 0.0},
            {'name': 'Thorlabs_SM1D12_Stop', 'surface_type': 'Stop', 'thickness': 50.0, 'semi_diameter': 12.7},
            {'name': 'M1_CM254-050-P01_Collimator', 'surface_type': 'Standard', 'radius': -100.0, 'thickness': 0.0, 'tilt_x': m1_tilt, 'semi_diameter': 12.7},
            {'name': 'Grating_100lpmm', 'surface_type': 'Grating', 'thickness': 25.4, 'lines_per_um': 0.1, 'diffraction_order': 1, 'grating_primary_wavelength_um': 1.75, 'tilt_x': grat_tilt, 'semi_diameter': 12.7},
            {'name': 'M2_CM254-025-P01_Focusing', 'surface_type': 'Standard', 'radius': -50.8, 'thickness': 25.4, 'semi_diameter': 12.7},
            {'name': 'DMD4500_Micromirrors', 'surface_type': 'Standard', 'thickness': 200.0, 'tilt_x': -24.0, 'semi_diameter': 5.0},
            {'name': 'M3_CM254-200-P01_Relay_Collector', 'surface_type': 'Standard', 'radius': -400.0, 'thickness': 0.0, 'tilt_x': m3_tilt, 'semi_diameter': 12.7},
            {'name': 'M4_CM254-025-P01_Focusing', 'surface_type': 'Standard', 'radius': -50.8, 'thickness': 25.4, 'tilt_x': m4_tilt, 'semi_diameter': 12.7},
            {'name': 'Detector_0.8mm_IMA', 'surface_type': 'Detector', 'thickness': 0.0, 'semi_diameter': 0.4}
        ]
    }
    
    stack = SurfaceStack()
    for sd in data['surfaces']:
        s = Surface(name=sd.get('name', ''))
        for k,v in sd.items():
            if k == 'surface_type': s.surface_type = SurfaceType.GRATING if v.lower() == 'grating' else (SurfaceType.STOP if v.lower() == 'stop' else (SurfaceType.OBJECT if v.lower() == 'object' else (SurfaceType.DETECTOR if v.lower() == 'detector' else SurfaceType.STANDARD)))
            elif hasattr(s, k): setattr(s, k, v)
        if 'Aspheric' in s.name:
            s.is_aspheric = True
            s.conic = sd.get('conic', 0.0)
        stack.add_surface(s)
    sys_opt = OpticalSystem(stack)
    sys_opt._update_accumulated_positions()
    return sys_opt, data

def evaluate(sys_opt):
    r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=1.75)
    tr = sys_opt.trace_ray(r)
    if tr.vignetted: return False, 0, 999
    
    path_pts = [np.array([0., 0., 0.])]
    for i, h in enumerate(tr.history):
        s = sys_opt.surfaces[h['surface_index']]
        p_loc = np.array([h['x_local'], h['y_local'], 0.0])
        p_glob = s.global_origin + s.global_R @ p_loc
        path_pts.append(p_glob)
        
    min_dist = 999.0
    for i_surf, s in enumerate(sys_opt.surfaces):
        if s.surface_type == SurfaceType.OBJECT: continue
        for i_seg in range(len(path_pts)-1):
            if i_seg == i_surf - 1 or i_seg == i_surf:
                continue
            
            pA = path_pts[i_seg]
            pB = path_pts[i_seg+1]
            normal = s.global_R[:, 2]
            denom = np.dot(pB - pA, normal)
            if abs(denom) < 1e-6: continue
            t = np.dot(s.global_origin - pA, normal) / denom
            if 1e-4 < t < 1 - 1e-4:
                p_int = pA + t * (pB - pA)
                dist = np.linalg.norm(p_int - s.global_origin)
                if dist < s.semi_diameter + 15.0:
                    return False, 0, 999
                if dist < min_dist:
                    min_dist = dist
                    
    hits_x, hits_y = [], []
    for wl in [0.8, 1.75, 2.7]:
        for theta in np.linspace(0.001, np.arcsin(0.04), 3):
            for phi in np.linspace(0, 2*np.pi, 4, endpoint=False):
                dx, dy, dz = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
                r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([dx, dy, dz]), wavelength=wl)
                tr = sys_opt.trace_ray(r)
                if not tr.vignetted:
                    h = tr.history[-1]
                    hits_x.append(h['x_local'])
                    hits_y.append(h['y_local'])
    if len(hits_x) == 0: return False, 0, 999
    
    mx, my = np.mean(hits_x), np.mean(hits_y)
    r_sq = (np.array(hits_x) - mx)**2 + (np.array(hits_y) - my)**2
    max_r = np.max(np.sqrt(r_sq))
    
    if max_r > 0.4: return False, min_dist, max_r
    return True, min_dist, max_r

print('Searching NIR layouts...')
best_dist = 0
best_params = None
best_data = None

for m1 in [-20, -35, -45]:
    for grat in [15, 25, 45, -15, -45]:
        for m3 in [-15, -25, -35, -45, 15, 30]:
            for m4 in [-15, -30, 15, 30]:
                try:
                    sys_opt, data = build_layout(m1, grat, m3, m4)
                    ok, dist, rad = evaluate(sys_opt)
                    if ok:
                        if dist > best_dist:
                            best_dist = dist
                            best_params = (m1, grat, m3, m4)
                            best_data = data
                            print(f'Valid! {best_params}, clearance: {dist:.2f}, rad: {rad:.3f}')
                except: pass

if best_data:
    with open('spectrometer_v3_nir_raman/spectrometer_nir_800_2700_all_mirror.json', 'w') as f:
        json.dump(best_data, f, indent=4)
    print('Saved NIR layout!')
else:
    print('Failed to find NIR layout!')

