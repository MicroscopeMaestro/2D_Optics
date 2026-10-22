
import sys, os, json, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath('')))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('mirror_spectrometer/all_mirror_spectrometer.json', 'r') as f: data = json.load(f)

for i, s in enumerate(data['surfaces']):
    if 'M1' in s['name']: m1_idx = i
    if 'Grating' in s['name']: grat_idx = i
    if 'M2' in s['name']: m2_idx = i
    if 'DMD' in s['name']: dmd_idx = i
    if 'M3' in s['name']: m3_idx = i
    if 'M4' in s['name']: m4_idx = i

wls = [1.087, 1.150, 1.225, 1.290, 1.350]

def build_sys(t):
    # t = [grat_tilt, m2_tilt, m3_tilt, m4_tilt, m4_thick]
    stack = SurfaceStack()
    for sd in data['surfaces']:
        s = Surface(name=sd.get('name', ''))
        for k,v in sd.items():
            if k == 'surface_type': s.surface_type = SurfaceType.GRATING if v.lower() == 'grating' else (SurfaceType.STOP if v.lower() == 'stop' else (SurfaceType.OBJECT if v.lower() == 'object' else SurfaceType.STANDARD))
            elif hasattr(s, k): setattr(s, k, v)
        stack.add_surface(s)
    
    # Fix M1 tilt to a large angle to avoid the slit
    stack.surfaces[m1_idx].tilt_x = 20.0
    
    stack.surfaces[grat_idx].tilt_x = t[0]
    stack.surfaces[m2_idx].tilt_x = t[1]
    # DMD tilt is fixed to 12.0 degrees (ON state)
    stack.surfaces[m3_idx].tilt_x = t[2]
    stack.surfaces[m4_idx].tilt_x = t[3]
    stack.surfaces[m4_idx].thickness = t[4]
    return OpticalSystem(stack)

def merit(t):
    sys_opt = build_sys(t)
    
    p_slit = np.array([0., 0., 0.])
    p_grat = sys_opt.surfaces[grat_idx].global_origin
    p_m2 = sys_opt.surfaces[m2_idx].global_origin
    p_m3 = sys_opt.surfaces[m3_idx].global_origin
    p_m4 = sys_opt.surfaces[m4_idx].global_origin
    
    penalty = 0.0
    
    # Clearance from Grating-to-M2 ray to the slit
    dir_grat_m2 = p_m2 - p_grat
    dir_grat_m2 = dir_grat_m2 / np.linalg.norm(dir_grat_m2)
    # cross product of (p_grat - p_slit) and dir_grat_m2 gives distance
    dist_slit = np.linalg.norm(np.cross(p_grat - p_slit, dir_grat_m2))
    
    if dist_slit < 25.0: penalty += (25.0 - dist_slit) * 1000.0
    if np.linalg.norm(p_m2 - np.array([0., 0., 50.])) < 25.0: penalty += 1000.0
        
    theta_max = np.arcsin(0.06)
    
    r_hits = []
    cy_list = []
    
    for wl in wls:
        hits_x, hits_y = [], []
        r_chief = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=wl)
        tr_chief = sys_opt.trace_ray(r_chief)
        if tr_chief.vignetted: return 9999.0 + penalty
        cy = tr_chief.history[-1]['y_local']
        cy_list.append(cy)
        
        for theta in np.linspace(0.001, theta_max, 3):
            for phi in np.linspace(0, 2*np.pi, 4, endpoint=False):
                dx, dy, dz = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
                r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([dx, dy, dz]), wavelength=wl)
                tr = sys_opt.trace_ray(r)
                if not tr.vignetted:
                    h = tr.history[-1]
                    hits_x.append(h['x_local'])
                    hits_y.append(h['y_local'])
        
        if len(hits_x) == 0: return 9999.0 + penalty
        mx, my = np.mean(hits_x), np.mean(hits_y)
        r_sq = (np.array(hits_x) - mx)**2 + (np.array(hits_y) - my)**2
        r_hits.append(np.max(np.sqrt(r_sq)))
        
    rms_radii = np.mean(r_hits)
    spread_y = np.max(cy_list) - np.min(cy_list)
    
    return rms_radii + spread_y*0.01 + abs(np.mean(cy_list))*1.0 + penalty

# Start near the teamwork's solution but offset for the new M1 tilt
t0 = [30.0, -10.0, -8.0, 8.0, 18.8]
res = minimize(merit, t0, method='Nelder-Mead', options={'maxiter': 500})
print('Opt:', res.x, 'Merit:', res.fun)

t = res.x
data['surfaces'][m1_idx]['tilt_x'] = 20.0
data['surfaces'][grat_idx]['tilt_x'] = float(t[0])
data['surfaces'][m2_idx]['tilt_x'] = float(t[1])
data['surfaces'][m3_idx]['tilt_x'] = float(t[2])
data['surfaces'][m4_idx]['tilt_x'] = float(t[3])
data['surfaces'][m4_idx]['thickness'] = float(t[4])

with open('mirror_spectrometer/all_mirror_spectrometer.json', 'w') as f: json.dump(data, f, indent=4)

