
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

stack = SurfaceStack()
dummy = {'name': 'Dummy_DMD_Obj', 'surface_type': 'Object', 'thickness': 200.0, 'glass': 'AIR'}
stack.add_surface(Surface(**dummy))

# Use paraxial lenses for the relay to see if it even fits!
l1 = Surface(name='Paraxial_L1', surface_type=SurfaceType.STANDARD, radius=0, thickness=230.0, glass='AIR', semi_diameter=12.5)
# Actually, a paraxial lens isn't natively built into the tracer unless I use an ideal lens formula.
# I will just trace rays and manually calculate where they hit L1!

stack = SurfaceStack()
for sd in data['surfaces'][:dmd_idx+1]:
    s = Surface(name=sd.get('name', ''))
    for k,v in sd.items():
        if k == 'surface_type': s.surface_type = SurfaceType.GRATING if v.lower() == 'grating' else (SurfaceType.STOP if v.lower() == 'stop' else (SurfaceType.OBJECT if v.lower() == 'object' else SurfaceType.STANDARD))
        elif hasattr(s, k): setattr(s, k, v)
    stack.add_surface(s)

sys_opt = OpticalSystem(stack)

# Trace chief ray from top of the object
# The object is the slit? No, the Object_Plane!
r = Ray(pos=np.array([0., 2.5, 0.]), dir=np.array([0., 0., 1.]), wavelength=1.225)
tr = sys_opt.trace_ray(r)
print('DMD Y:', tr.history[-1]['y_local'])
dx, dy, dz = tr.dir
# Extrapolate 200 mm from DMD
t = 200.0 / dz
y_L1 = tr.history[-1]['y_local'] + dy * t
print('Y at L1 (200mm away):', y_L1)

# Extrapolate 100 mm from DMD
t = 100.0 / dz
y_L1_100 = tr.history[-1]['y_local'] + dy * t
print('Y at L1 (100mm away):', y_L1_100)


