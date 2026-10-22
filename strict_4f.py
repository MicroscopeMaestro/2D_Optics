
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath('')))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('C:/Users/Juan/Desktop/Spectrometer_V2_4f_1inch.json', 'r') as f: data = json.load(f)

data['surfaces'][6]['thickness'] = 75.0
data['surfaces'][11]['thickness'] = 16.0

def build_sys():
    stack = SurfaceStack()
    for sd in data['surfaces']:
        s = Surface(name=sd.get('name', ''))
        for k,v in sd.items():
            if k == 'surface_type': s.surface_type = SurfaceType.GRATING if v.lower() == 'grating' else (SurfaceType.ASPHERIC if v.lower() == 'aspheric' else SurfaceType.STANDARD)
            elif hasattr(s, k): setattr(s, k, v)
        s.aperture_type = 'Circular'
        s.semi_diameter = float('inf')
        stack.add_surface(s)
    sys_opt = OpticalSystem(stack)
    sys_opt.stack._update_accumulated_positions()
    return sys_opt

sys_opt = build_sys()
s_det = sys_opt.surfaces[-1]

r_chief = Ray(pos=np.array([0.,0.,0.]), dir=np.array([0.,0.,1.]), wavelength=1.225)
tr_chief = sys_opt.trace_ray(r_chief)
print('Chief:', tr_chief.history[-1]['y_local'])

rays = []
theta = np.arcsin(0.12)
for phi in [0, np.pi/2, np.pi, 3*np.pi/2]:
    dx, dy, dz = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
    rays.append(Ray(pos=np.array([0.,0.,0.]), dir=np.array([dx,dy,dz]), wavelength=1.225))

pts = []
for r in rays:
    tr2 = sys_opt.trace_ray(r)
    loc = s_det.global_R.T @ (tr2.pos - s_det.global_origin)
    pts.append([loc[0], loc[1]])
pts = np.array(pts)
print('RMS:', np.sqrt(np.mean(pts[:,0]**2 + pts[:,1]**2)))

