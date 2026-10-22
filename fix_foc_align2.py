
import sys, os, json, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath('')))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('C:/Users/Juan/Desktop/Spectrometer_V2_Zemax.json', 'r') as f: data = json.load(f)

for i, s in enumerate(data['surfaces']):
    if 'Grating' in s.get('name', ''): grat_idx = i
    if 'Foc_S1' in s.get('name', ''): foc_idx = i
    if 'DMD' in s.get('name', ''): dmd_idx = i; break

def merit(t):
    stack = SurfaceStack()
    for sd in data['surfaces'][:dmd_idx+1]:
        s = Surface(name=sd.get('name', ''))
        for k,v in sd.items():
            if k == 'surface_type': s.surface_type = SurfaceType.GRATING if v.lower() == 'grating' else (SurfaceType.STOP if v.lower() == 'stop' else (SurfaceType.OBJECT if v.lower() == 'object' else SurfaceType.STANDARD))
            elif hasattr(s, k): setattr(s, k, v)
        stack.add_surface(s)
    
    stack.surfaces[grat_idx].tilt_x = t[0]
    stack.surfaces[foc_idx].decenter_y = t[1]
    stack.surfaces[foc_idx].tilt_x = t[2]
    stack.surfaces[foc_idx+1].decenter_y = t[1]
    stack.surfaces[foc_idx+1].tilt_x = t[2]
    stack.surfaces[foc_idx+2].decenter_y = t[1]
    stack.surfaces[foc_idx+2].tilt_x = t[2]
    stack.surfaces[foc_idx+2].thickness = t[3]
    sys_opt = OpticalSystem(stack)
    
    theta_max = np.arcsin(0.06)
    hits_x, hits_y = [], []
    for theta in np.linspace(0.001, theta_max, 5):
        for phi in np.linspace(0, 2*np.pi, 8, endpoint=False):
            dx, dy, dz = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
            r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([dx, dy, dz]), wavelength=1.225)
            tr = sys_opt.trace_ray(r)
            if not tr.vignetted:
                h = tr.history[-1]
                hits_x.append(h['x_local'])
                hits_y.append(h['y_local'])
    
    if len(hits_x) == 0: return 999.0
    r_sq = (np.array(hits_x) - np.mean(hits_x))**2 + (np.array(hits_y) - np.mean(hits_y))**2
    rms = np.sqrt(np.mean(r_sq))
    
    r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=1.225)
    tr = sys_opt.trace_ray(r)
    if tr.vignetted: return 999.0
    cy = tr.history[-1]['y_local']
    
    return rms + abs(cy)*10.0

t0 = [data['surfaces'][grat_idx].get('tilt_x', -10.0), 0.0, 0.0, 46.8]
res = minimize(merit, t0, method='Nelder-Mead', options={'maxiter': 500})
print('Opt:', res.x, 'Merit:', res.fun)

t = res.x
data['surfaces'][grat_idx]['tilt_x'] = float(t[0])
data['surfaces'][foc_idx]['decenter_y'] = float(t[1])
data['surfaces'][foc_idx]['tilt_x'] = float(t[2])
data['surfaces'][foc_idx+1]['decenter_y'] = float(t[1])
data['surfaces'][foc_idx+1]['tilt_x'] = float(t[2])
data['surfaces'][foc_idx+2]['decenter_y'] = float(t[1])
data['surfaces'][foc_idx+2]['tilt_x'] = float(t[2])
data['surfaces'][foc_idx+2]['thickness'] = float(t[3])

with open('C:/Users/Juan/Desktop/Spectrometer_V2_Zemax_FocOpt.json', 'w') as f: json.dump(data, f, indent=4)

