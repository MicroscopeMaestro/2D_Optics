
import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('spectrometer_relay_redesign/spectrometer_nir_900_1700nm.json', 'r') as f: data = json.load(f)

m1_idx = next(i for i,s in enumerate(data['surfaces']) if 'M1' in s['name'])
grat_idx = next(i for i,s in enumerate(data['surfaces']) if 'Grating' in s['name'])

def check_layout(m1_tilt, grat_tilt):
    data['surfaces'][m1_idx]['tilt_x'] = m1_tilt
    data['surfaces'][grat_idx]['tilt_x'] = grat_tilt
    
    stack = SurfaceStack()
    for sd in data['surfaces']:
        s = Surface(name=sd.get('name', ''))
        for k,v in sd.items():
            if k == 'surface_type': s.surface_type = SurfaceType.GRATING if v.lower() == 'grating' else (SurfaceType.STOP if v.lower() == 'stop' else (SurfaceType.OBJECT if v.lower() == 'object' else SurfaceType.STANDARD))
            elif hasattr(s, k): setattr(s, k, v)
        if 'Aspheric' in s.name:
            s.is_aspheric = True
            s.conic = sd.get('conic', 0.0)
        stack.add_surface(s)
    sys_opt = OpticalSystem(stack)
    sys_opt._update_accumulated_positions()
    
    r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=1.3)
    tr = sys_opt.trace_ray(r)
    if tr.vignetted: return False, 0
    
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
            if 'Foc' in s.name and 'Foc' in sys_opt.surfaces[tr.history[i_seg]['surface_index']].name: continue
            if 'Relay' in s.name and 'Relay' in sys_opt.surfaces[tr.history[i_seg]['surface_index']].name: continue
            if 'Cond' in s.name and 'Cond' in sys_opt.surfaces[tr.history[i_seg]['surface_index']].name: continue
            
            pA = path_pts[i_seg]
            pB = path_pts[i_seg+1]
            normal = s.global_R[:, 2]
            denom = np.dot(pB - pA, normal)
            if abs(denom) < 1e-6: continue
            t = np.dot(s.global_origin - pA, normal) / denom
            if 1e-4 < t < 1 - 1e-4:
                p_int = pA + t * (pB - pA)
                dist = np.linalg.norm(p_int - s.global_origin)
                if dist < min_dist:
                    min_dist = dist
                    
    return True, min_dist

for grat in np.arange(-90, 90, 2):
    try:
        ok, dist = check_layout(-45.0, grat)
        if ok:
            print(f'M1: -45, Grat: {grat}, Clearance: {dist:.2f}')
    except Exception as e: pass

