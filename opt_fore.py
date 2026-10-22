
import sys, os, json, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath('')))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('C:/Users/Juan/Desktop/Spectrometer_V2_Full_System.json', 'r') as f: data = json.load(f)

# Fix lens orientations
# L1 (Reversed, to collimate light from Object)
data['surfaces'][1].update({'radius': 83.0, 'thickness': 2.0, 'glass': 'N-SF5'})
data['surfaces'][2].update({'radius': 22.3, 'thickness': 6.5, 'glass': 'N-BAK4'})
data['surfaces'][3].update({'radius': -33.3, 'thickness': 2.0, 'glass': 'AIR'})

# L2 (Normal, to focus collimated light to Slit)
data['surfaces'][4].update({'radius': 33.3, 'thickness': 6.5, 'glass': 'N-BAK4'})
data['surfaces'][5].update({'radius': -22.3, 'thickness': 2.0, 'glass': 'N-SF5'})
data['surfaces'][6].update({'radius': -83.0, 'thickness': 50.0, 'glass': 'AIR'})

def build_sys(t):
    stack = SurfaceStack()
    for sd in data['surfaces'][:8]:
        s = Surface(name=sd.get('name', ''))
        for k,v in sd.items():
            if hasattr(s, k): setattr(s, k, v)
        if 'Object_Plane' in s.name: s.thickness = t[0]
        if 'Fore_L2_B' in s.name: s.thickness = t[1]
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
    return spread * 100.0 + mag_err

res = minimize(merit, [43.5, 43.5], method='Nelder-Mead')
print('Fore Opt:', res.x, 'RMS:', res.fun)

data['surfaces'][0]['thickness'] = float(res.x[0])
data['surfaces'][6]['thickness'] = float(res.x[1])
with open('C:/Users/Juan/Desktop/Spectrometer_V2_Full_System.json', 'w') as f: json.dump(data, f, indent=4)
print('Saved!')

