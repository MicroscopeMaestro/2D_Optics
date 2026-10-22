
import sys, os, json, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('spectrometer_refractive_1064.json', 'r') as f:
    data = json.load(f)

def build_system(t_cond, t_det):
    for s in data['surfaces']:
        if s['name'] == 'Imaging_Relay_S3': s['thickness'] = t_cond
        if s['name'] == 'Imaging_Cond_S2': s['thickness'] = t_det
        # Reset semi_diameter of detector to 5.0 to measure real spot size
        if s['name'] == 'Detector': s['semi_diameter'] = 5.0
        
    stack = SurfaceStack()
    for s_dict in data['surfaces']:
        s = Surface()
        for k,v in s_dict.items():
            if k == 'surface_type':
                if 'grating' in v.lower(): s.surface_type = SurfaceType.GRATING
                elif 'object' in v.lower(): s.surface_type = SurfaceType.OBJECT
                elif 'detector' in v.lower(): s.surface_type = SurfaceType.DETECTOR
                elif 'aspheric' in v.lower(): s.surface_type = SurfaceType.ASPHERIC
                elif 'stop' in v.lower(): s.surface_type = SurfaceType.STOP
                elif 'coordinate' in v.lower():
                    s.surface_type = SurfaceType.COORDINATE_BREAK
                    s.is_coord_break = True
                else: s.surface_type = SurfaceType.STANDARD
            elif hasattr(s, k): setattr(s, k, v)
        stack.add_surface(s)
    
    sys_opt = OpticalSystem(stack)
    sys_opt._update_accumulated_positions()
    return sys_opt

def obj(x):
    t_cond, t_det = x
    if t_cond < 0 or t_det < 0: return 1e6
    sys_opt = build_system(t_cond, t_det)
    
    hits_x, hits_y = [], []
    for wl in [1.225]:
        for theta in [np.arcsin(0.04)]:
            for phi in np.linspace(0, 2*np.pi, 4, endpoint=False):
                dx, dy, dz = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
                r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([dx, dy, dz]), wavelength=wl)
                tr = sys_opt.trace_ray(r)
                if not tr.vignetted:
                    h = tr.history[-1]
                    hits_x.append(h['x_local'])
                    hits_y.append(h['y_local'])
    if not hits_x: return 1e6
    mx, my = np.mean(hits_x), np.mean(hits_y)
    r_sq = (np.array(hits_x) - mx)**2 + (np.array(hits_y) - my)**2
    max_r = np.max(np.sqrt(r_sq))
    
    # We want max_r to be as small as possible
    return max_r

res = minimize(obj, [10.0, 16.0], method='Nelder-Mead', bounds=[(2.0, 500.0), (2.0, 100.0)])
print(f'Optimized t_cond: {res.x[0]:.3f}, t_det: {res.x[1]:.3f}')
print(f'Max spot radius: {res.fun:.3f} mm')

sys_opt = build_system(res.x[0], res.x[1])
# Put the detector constraint back
sys_opt.surfaces[-1].semi_diameter = 0.4
sys_opt.stack.save_to_json('spectrometer_refractive_1064.json', wavelengths=[1.087, 1.225, 1.350])

