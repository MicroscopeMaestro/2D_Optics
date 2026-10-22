
import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('spectrometer_relay_redesign/Spectrometer_Hybrid_1064_Repaired.json', 'r') as f: data = json.load(f)
m1_idx = next(i for i,s in enumerate(data['surfaces']) if 'M1' in s['name'])
data['surfaces'][m1_idx]['tilt_x'] = -24.0

stack = SurfaceStack()
for sd in data['surfaces']:
    s = Surface(name=sd.get('name', ''))
    for k,v in sd.items():
        if k == 'surface_type': s.surface_type = SurfaceType.GRATING if v.lower() == 'grating' else (SurfaceType.STOP if v.lower() == 'stop' else (SurfaceType.OBJECT if v.lower() == 'object' else SurfaceType.STANDARD))
        elif hasattr(s, k): setattr(s, k, v)
    if 'Aspheric' in s.name:
        s.is_aspheric = True
        s.conic = sd.get('conic', 0.0)
    stack.add_surface(s)
sys_opt = OpticalSystem(stack)

wls = [1.087, 1.150, 1.225, 1.290, 1.350]
theta_max = np.arcsin(0.04)

for wl in wls:
    print(f'\nWavelength {wl}:')
    hits_x, hits_y = [], []
    for theta in np.linspace(0.001, theta_max, 5):
        for phi in np.linspace(0, 2*np.pi, 6, endpoint=False):
            dx, dy, dz = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
            r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([dx, dy, dz]), wavelength=wl)
            tr = sys_opt.trace_ray(r)
            if tr.vignetted:
                pass
            else:
                h = tr.history[-1]
                hits_x.append(h['x_local'])
                hits_y.append(h['y_local'])
    if len(hits_x) > 0:
        mx, my = np.mean(hits_x), np.mean(hits_y)
        r_sq = (np.array(hits_x) - mx)**2 + (np.array(hits_y) - my)**2
        print('  Max spot radius:', np.max(np.sqrt(r_sq)))
        print('  Centroid Y:', my)
    else:
        print('  ALL VIGNETTED!')

with open('spectrometer_relay_redesign/Spectrometer_Hybrid_1064_Repaired.json', 'w') as f: json.dump(data, f, indent=4)

