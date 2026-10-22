
import sys, os, json, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath('')))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('C:/Users/Juan/Desktop/Spectrometer_V2_Full_System.json', 'r') as f: data = json.load(f)

def build_sys(t):
    stack = SurfaceStack()
    for sd in data['surfaces'][:5]:
        s = Surface(name=sd.get('name', ''))
        for k,v in sd.items():
            if hasattr(s, k): setattr(s, k, v)
        if 'Object_Plane' in s.name: s.thickness = 100.0
        if 'Fore_L1_B' in s.name: s.thickness = t[0]
        s.aperture_type = 'Circular'
        s.semi_diameter = float('inf')
        stack.add_surface(s)
    sys_opt = OpticalSystem(stack)
    sys_opt.stack._update_accumulated_positions()
    return sys_opt

def merit(t):
    sys_opt = build_sys(t)
    rays = [
        Ray(pos=np.array([0.,0.5,1e-3]), dir=np.array([0., 0.05, 1.0]), wavelength=1.225),
        Ray(pos=np.array([0.,0.5,1e-3]), dir=np.array([0., 0.0, 1.0]), wavelength=1.225),
        Ray(pos=np.array([0.,0.5,1e-3]), dir=np.array([0., -0.05, 1.0]), wavelength=1.225)
    ]
    pts = []
    for r in rays:
        tr = sys_opt.trace_ray(r)
        if tr.vignetted: return 9999.0
        h = tr.history[-1]
        pts.append(h['y_local'])
    pts = np.array(pts)
    return np.std(pts)

res = minimize(merit, [100.0], method='Nelder-Mead')
print('Fore Opt V:', res.x, 'Spread:', res.fun)

sys_opt = build_sys(res.x)
pts = []
for r in [
    Ray(pos=np.array([0.,0.5,1e-3]), dir=np.array([0., 0.05, 1.0]), wavelength=1.225),
    Ray(pos=np.array([0.,0.5,1e-3]), dir=np.array([0., 0.0, 1.0]), wavelength=1.225),
    Ray(pos=np.array([0.,0.5,1e-3]), dir=np.array([0., -0.05, 1.0]), wavelength=1.225)
]:
    tr = sys_opt.trace_ray(r)
    pts.append(tr.history[-1]['y_local'])
print('Mean Y (Magnification):', np.mean(pts))

data['surfaces'][0]['thickness'] = 100.0
data['surfaces'][3]['thickness'] = float(res.x[0])
with open('C:/Users/Juan/Desktop/Spectrometer_V2_Full_System.json', 'w') as f: json.dump(data, f, indent=4)
print('Saved!')

