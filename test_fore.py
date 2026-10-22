
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath('')))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('C:/Users/Juan/Desktop/Spectrometer_V2_Full_System.json', 'r') as f: data = json.load(f)

data['surfaces'][0]['thickness'] = 43.5
data['surfaces'][6]['thickness'] = 43.5

stack = SurfaceStack()
for sd in data['surfaces'][:8]:
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
    print('Ray ' + str(i) + ' hitting slit at Y = ' + str(h['y_local']))

with open('C:/Users/Juan/Desktop/Spectrometer_V2_Full_System.json', 'w') as f: json.dump(data, f, indent=4)
print('Saved!')

