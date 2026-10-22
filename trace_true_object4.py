
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath('')))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('C:/Users/Juan/Desktop/Spectrometer_V2_Zemax.json', 'r') as f: data = json.load(f)

stack = SurfaceStack()
for sd in data['surfaces']:
    s = Surface(name=sd.get('name', ''))
    for k,v in sd.items():
        if k == 'surface_type': s.surface_type = SurfaceType.GRATING if v.lower() == 'grating' else (SurfaceType.STOP if v.lower() == 'stop' else (SurfaceType.OBJECT if v.lower() == 'object' else SurfaceType.STANDARD))
        elif hasattr(s, k): setattr(s, k, v)
    if 'Detector' in s.name:
        s.semi_diameter = 15.0 # OPEN THE DETECTOR APERTURE TO SEE WHERE THEY HIT!
    stack.add_surface(s)
sys_opt = OpticalSystem(stack)

theta_max = np.arcsin(0.06)
det_hits_x = []
det_hits_y = []

for oy in np.linspace(-0.5, 0.5, 5):
    for ox in np.linspace(-0.015, 0.015, 3):
        for theta in np.linspace(0.001, theta_max, 5):
            for phi in np.linspace(0, 2*np.pi, 8, endpoint=False):
                dx, dy, dz = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
                r = Ray(pos=np.array([ox, oy, 0.]), dir=np.array([dx, dy, dz]), wavelength=1.225)
                tr = sys_opt.trace_ray(r)
                if not tr.vignetted:
                    det_hits_x.append(tr.history[-1]['x_local'])
                    det_hits_y.append(tr.history[-1]['y_local'])

r_hits = np.sqrt(np.array(det_hits_x)**2 + np.array(det_hits_y)**2)
print(f'Max radius on detector: {np.max(r_hits):.4f} mm')
print(f'Mean radius on detector: {np.mean(r_hits):.4f} mm')

