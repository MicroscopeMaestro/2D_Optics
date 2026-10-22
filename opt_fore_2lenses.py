
import sys, os, json, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath('')))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('C:/Users/Juan/Desktop/Spectrometer_V2_4f_1inch.json', 'r') as f: data = json.load(f)
base_surf = data['surfaces'][0]

fore = []
# Object
s = base_surf.copy(); s.update({'name': 'Object_Plane', 'surface_type': 'Object', 'thickness': 43.5, 'glass': 'AIR', 'semi_diameter': 12.7, 'radius': 0.0})
fore.append(s)

# L1 (AC254-050-C, Reversed)
s = base_surf.copy(); s.update({'name': 'Fore_L1_F', 'radius': 83.0, 'thickness': 2.0, 'glass': 'N-SF5', 'semi_diameter': 12.7, 'surface_type': 'Standard'})
fore.append(s)
s = base_surf.copy(); s.update({'name': 'Fore_L1_M', 'radius': 22.3, 'thickness': 6.5, 'glass': 'N-BK7', 'semi_diameter': 12.7, 'surface_type': 'Standard'})
fore.append(s)
s = base_surf.copy(); s.update({'name': 'Fore_L1_B', 'radius': -33.3, 'thickness': 10.0, 'glass': 'AIR', 'semi_diameter': 12.7, 'surface_type': 'Standard'})
fore.append(s)

# L2 (AC254-050-C, Normal)
s = base_surf.copy(); s.update({'name': 'Fore_L2_F', 'radius': 33.3, 'thickness': 6.5, 'glass': 'N-BK7', 'semi_diameter': 12.7, 'surface_type': 'Standard'})
fore.append(s)
s = base_surf.copy(); s.update({'name': 'Fore_L2_M', 'radius': -22.3, 'thickness': 2.0, 'glass': 'N-SF5', 'semi_diameter': 12.7, 'surface_type': 'Standard'})
fore.append(s)
s = base_surf.copy(); s.update({'name': 'Fore_L2_B', 'radius': -83.0, 'thickness': 43.5, 'glass': 'AIR', 'semi_diameter': 12.7, 'surface_type': 'Standard'})
fore.append(s)

# Slit (Rectangular)
s = base_surf.copy()
s.update({'name': 'Input_Slit', 'surface_type': 'Standard', 'thickness': 50.0, 'radius': 0.0, 'glass': 'AIR'})
s['aperture_type'] = 'Rectangular'
s['aperture_x_half'] = 0.015 # 30 um width
s['aperture_y_half'] = 0.5   # 1 mm height
fore.append(s)

def build_sys(t):
    stack = SurfaceStack()
    for sd in fore:
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
    s_det = sys_opt.surfaces[-1] # Slit plane
    rays = []
    theta = np.arcsin(0.05)
    for phi in [0, np.pi/2, np.pi, 3*np.pi/2]:
        dx, dy, dz = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
        rays.append(Ray(pos=np.array([0., 0.5, 0.]), dir=np.array([dx,dy,dz]), wavelength=1.225))
    pts = []
    for r in rays:
        tr = sys_opt.trace_ray(r)
        if tr.vignetted: return 9999.0
        loc = s_det.global_R.T @ (tr.pos - s_det.global_origin)
        pts.append([loc[0], loc[1]])
    pts = np.array(pts)
    spot = np.sqrt(np.mean(pts[:,0]**2 + (pts[:,1] + 0.5)**2)) # Magnification is -1, so 0.5 maps to -0.5
    return spot * 100.0 + (t[0]-43.5)**2 * 0.001

res = minimize(merit, [43.5, 43.5], method='Nelder-Mead')
print('Fore 2-lens Opt:', res.x, 'RMS:', res.fun)

fore[0]['thickness'] = float(res.x[0])
fore[6]['thickness'] = float(res.x[1])
data['surfaces'] = fore + data['surfaces'][1:]

with open('C:/Users/Juan/Desktop/Spectrometer_V2_Complete.json', 'w') as f: json.dump(data, f, indent=4)
print('Saved!')

