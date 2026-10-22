
import sys, os, json, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath('')))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('C:/Users/Juan/Desktop/Spectrometer_V2_Full_System.json', 'r') as f: data = json.load(f)

# Keep only Object + 1 Lens (3 surfaces) + Slit + ...
# Currently fore_surfaces has Object + L1 + L2 + Slit (8 surfaces).
# Let's remove L2!
# 0: Object, 1: L1_F, 2: L1_M, 3: L1_B, 4: Slit (was 7)
new_surfaces = []
new_surfaces.append(data['surfaces'][0])
new_surfaces.append(data['surfaces'][1])
new_surfaces.append(data['surfaces'][2])
new_surfaces.append(data['surfaces'][3])
new_surfaces.append(data['surfaces'][7]) # Slit
new_surfaces.extend(data['surfaces'][8:]) # Spec

data['surfaces'] = new_surfaces

def build_sys(t):
    stack = SurfaceStack()
    for sd in data['surfaces'][:5]:
        s = Surface(name=sd.get('name', ''))
        for k,v in sd.items():
            if hasattr(s, k): setattr(s, k, v)
        if 'Object_Plane' in s.name: s.thickness = t[0]
        if 'Fore_L1_B' in s.name: s.thickness = t[1]
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
    
    spread = np.std(pts)
    mag_err = abs(abs(np.mean(pts)) - 0.5)
    return spread * 10.0 + mag_err

res = minimize(merit, [95.0, 95.0], method='Nelder-Mead')
print('Fore Opt:', res.x, 'RMS:', res.fun)

data['surfaces'][0]['thickness'] = float(res.x[0])
data['surfaces'][3]['thickness'] = float(res.x[1])
with open('C:/Users/Juan/Desktop/Spectrometer_V2_Full_System.json', 'w') as f: json.dump(data, f, indent=4)
print('Saved!')

