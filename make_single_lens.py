
import sys, os, json, copy, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath('')))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('C:/Users/Juan/Desktop/Spectrometer_V2_2inch_Optimized.json', 'r') as f: data = json.load(f)

fore_surfaces = []
obj = copy.deepcopy(data['surfaces'][-1])
obj.update({'name': 'Object_Plane', 'surface_type': 'Object', 'thickness': 92.5, 'glass': 'AIR', 'semi_diameter': 12.7})
fore_surfaces.append(obj)

def create_surf(name, radius, thickness, glass):
    s = copy.deepcopy(data['surfaces'][-1])
    s.update({'name': name, 'radius': radius, 'thickness': thickness, 'glass': glass, 'surface_type': 'Standard', 'semi_diameter': 12.7, 'aperture_type': 'Circular'})
    return s

fore_surfaces.append(create_surf('Fore_L1_F', 33.3, 6.5, 'N-BAK4'))
fore_surfaces.append(create_surf('Fore_L1_M', -22.3, 2.0, 'N-SF5'))
fore_surfaces.append(create_surf('Fore_L1_B', -83.0, 96.0, 'AIR'))

slit = copy.deepcopy(data['surfaces'][0])
slit['surface_type'] = 'Standard'
slit['aperture_type'] = 'Rectangular'
slit['aperture_x_half'] = 0.015
slit['aperture_y_half'] = 0.5
fore_surfaces.append(slit)

data['surfaces'] = fore_surfaces + data['surfaces'][1:]

stack = SurfaceStack()
for sd in data['surfaces'][:5]:
    s = Surface(name=sd.get('name', ''))
    for k,v in sd.items():
        if hasattr(s, k): setattr(s, k, v)
    s.aperture_type = 'Circular'
    s.semi_diameter = float('inf')
    stack.add_surface(s)
sys_opt = OpticalSystem(stack)

rays = [
    Ray(pos=np.array([0.,0.5,1e-3]), dir=np.array([0., 0.05, 1.0]), wavelength=1.225),
    Ray(pos=np.array([0.,0.5,1e-3]), dir=np.array([0., 0.0, 1.0]), wavelength=1.225),
    Ray(pos=np.array([0.,0.5,1e-3]), dir=np.array([0., -0.05, 1.0]), wavelength=1.225)
]
for i, r in enumerate(rays):
    tr = sys_opt.trace_ray(r)
    h = tr.history[-1]
    print('Ray ' + str(i) + ' hit slit Y=' + str(h['y_local']))

with open('C:/Users/Juan/Desktop/Spectrometer_V2_Full_System.json', 'w') as f: json.dump(data, f, indent=4)
print('Saved with hardcoded 92.5 and 96.0 gaps!')

