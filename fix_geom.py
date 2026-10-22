
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

# Set M1 and Grating manually!
data['surfaces'][m1_idx]['tilt_x'] = 20.0
data['surfaces'][grat_idx]['tilt_x'] = 28.1

# We need to find M2 tilt that points to the DMD!
def build_sys(t):
    # t = [m2_tilt, dmd_tilt, m3_tilt, m4_tilt, m4_thick]
    stack = SurfaceStack()
    for sd in data['surfaces']:
        s = Surface(name=sd.get('name', ''))
        for k,v in sd.items():
            if k == 'surface_type': s.surface_type = SurfaceType.GRATING if v.lower() == 'grating' else (SurfaceType.STOP if v.lower() == 'stop' else (SurfaceType.OBJECT if v.lower() == 'object' else SurfaceType.STANDARD))
            elif hasattr(s, k): setattr(s, k, v)
        stack.add_surface(s)
    
    stack.surfaces[m2_idx].tilt_x = t[0]
    stack.surfaces[dmd_idx].tilt_x = t[1]
    stack.surfaces[m3_idx].tilt_x = t[2]
    stack.surfaces[m4_idx].tilt_x = t[3]
    stack.surfaces[m4_idx].thickness = t[4]
    return OpticalSystem(stack)

def merit(t):
    sys_opt = build_sys(t)
    r_chief = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=1.225)
    tr = sys_opt.trace_ray(r_chief)
    if tr.vignetted: return 9999.0
    
    p_grat = sys_opt.surfaces[grat_idx].global_origin
    p_m2 = sys_opt.surfaces[m2_idx].global_origin
    dir_grat_m2 = p_m2 - p_grat
    dir_grat_m2 = dir_grat_m2 / np.linalg.norm(dir_grat_m2)
    dist_slit = np.linalg.norm(np.cross(p_grat, dir_grat_m2))
    
    penalty = 0.0
    if dist_slit < 25.0: penalty += (25.0 - dist_slit) * 1000.0
    
    cy = tr.history[-1]['y_local']
    
    return abs(cy) + penalty

t0 = [30.0, 12.0, -8.0, 8.0, 18.8]
res = minimize(merit, t0, method='Nelder-Mead', options={'maxiter': 500})
print('Opt:', res.x, 'Merit:', res.fun)

