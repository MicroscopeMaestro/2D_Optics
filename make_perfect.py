
import sys, os, json, copy, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath('')))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('C:/Users/Juan/Desktop/Spectrometer_V2_2inch_Optimized.json', 'r') as f: data = json.load(f)

# Also fix the Field lens in the Spectrometer to use N-BK7 instead of N-BAK4, and B270 to N-BK7 just in case
for s in data['surfaces']:
    if 'BAK4' in s.get('glass', ''): s['glass'] = 'N-BK7'
    if 'B270' in s.get('glass', ''): s['glass'] = 'B270' # B270 is valid

# Now prepend the fore-optics
fore_surfaces = []
obj = copy.deepcopy(data['surfaces'][-1])
obj.update({'name': 'Object_Plane', 'surface_type': 'Object', 'thickness': 100.0, 'glass': 'AIR', 'semi_diameter': 12.7})
fore_surfaces.append(obj)

def create_surf(name, radius, thickness, glass):
    s = copy.deepcopy(data['surfaces'][-1])
    s.update({'name': name, 'radius': radius, 'thickness': thickness, 'glass': glass, 'surface_type': 'Standard', 'semi_diameter': 12.7, 'aperture_type': 'Circular'})
    return s

# Fore L1 (N-BK7 / N-SF5)
fore_surfaces.append(create_surf('Fore_L1_F', 33.3, 6.5, 'N-BK7'))
fore_surfaces.append(create_surf('Fore_L1_M', -22.3, 2.0, 'N-SF5'))
fore_surfaces.append(create_surf('Fore_L1_B', -83.0, 100.0, 'AIR'))

# Slit
slit = copy.deepcopy(data['surfaces'][0])
slit['surface_type'] = 'Standard'
slit['aperture_type'] = 'Rectangular'
slit['aperture_x_half'] = 0.015
slit['aperture_y_half'] = 0.5
slit['thickness'] = 50.0 # Distance to M1
fore_surfaces.append(slit)

data['surfaces'] = fore_surfaces + data['surfaces'][1:]

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
        pts.append(tr.history[-1]['y_local'])
    pts = np.array(pts)
    
    # We want perfect imaging (minimize spread)
    spread = np.std(pts)
    # We want magnification around -1 or +1
    mag_err = abs(abs(np.mean(pts)) - 0.5)
    return spread * 100.0 + mag_err

res = minimize(merit, [95.0, 95.0], method='Nelder-Mead')
print('Fore Opt:', res.x, 'RMS:', res.fun)

data['surfaces'][0]['thickness'] = float(res.x[0])
data['surfaces'][3]['thickness'] = float(res.x[1])
data['system_config']['aperture']['slit_width_um'] = 100.0
data['source_config']['beam_radius'] = 0.5
data['source_config']['input_aperture_shape'] = 'Circular'
data['source_config']['input_aperture_width'] = 1.0
data['source_config']['input_aperture_height'] = 1.0

with open('C:/Users/Juan/Desktop/Spectrometer_V2_Full_System.json', 'w') as f: json.dump(data, f, indent=4)
print('Saved!')

