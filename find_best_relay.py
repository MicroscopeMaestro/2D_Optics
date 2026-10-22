
import sys, os, json, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath('')))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('C:/Users/Juan/Desktop/Spectrometer_V2_Zemax.json', 'r') as f: data = json.load(f)

for i, s in enumerate(data['surfaces']):
    if 'DMD' in s.get('name', ''):
        dmd_idx = i
        break

def merit(t):
    stack = SurfaceStack()
    dummy = {'name': 'Dummy_DMD_Obj', 'surface_type': 'Object', 'thickness': t[0], 'glass': 'AIR'}
    stack.add_surface(Surface(**dummy))

    for sd in data['surfaces'][dmd_idx+1:]:
        s = Surface(name=sd.get('name', ''))
        for k,v in sd.items():
            if k == 'surface_type': s.surface_type = SurfaceType.DETECTOR if v.lower() == 'detector' else SurfaceType.STANDARD
            elif hasattr(s, k): setattr(s, k, v)
        s.semi_diameter = 15.0
        stack.add_surface(s)
    
    stack.surfaces[3].thickness = t[1] # L1_B to L2_F
    stack.surfaces[5].thickness = t[2] # L2_B to detector
    sys_opt = OpticalSystem(stack)

    theta_max = np.arcsin(0.06)
    det_hits_x = []
    det_hits_y = []

    for r_obj in [0.0, 1.25, 2.5]:
        for theta_obj in np.linspace(0, 2*np.pi, 4, endpoint=False):
            ox = r_obj * np.cos(theta_obj)
            oy = r_obj * np.sin(theta_obj)
            for theta in [0, theta_max]:
                for phi in np.linspace(0, 2*np.pi, 4, endpoint=False):
                    dx, dy, dz = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
                    r = Ray(pos=np.array([ox, oy, 0.]), dir=np.array([dx, dy, dz]), wavelength=1.225)
                    tr = sys_opt.trace_ray(r)
                    det_hits_x.append(tr.history[-1]['x_local'])
                    det_hits_y.append(tr.history[-1]['y_local'])

    r_hits = np.sqrt(np.array(det_hits_x)**2 + np.array(det_hits_y)**2)
    
    # We want max radius to be <= 0.4
    return np.max(r_hits)

# Initial guess: 4f distances roughly (DMD-L1, L1-L2, L2-Det)
res = minimize(merit, [100.0, 116.0, 10.0], method='Nelder-Mead')
print('Opt:', res.x, 'Max Radius:', res.fun)

