
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
    stack.add_surface(s)
sys_opt = OpticalSystem(stack)

theta_max = np.arcsin(0.06)
det_hits_x = []
det_hits_y = []

passed_stop = 0
passed_detector = 0
vignetted_detector = 0

for oy in np.linspace(-0.5, 0.5, 5):
    for ox in np.linspace(-0.015, 0.015, 3):
        for theta in np.linspace(0.001, theta_max, 5):
            for phi in np.linspace(0, 2*np.pi, 8, endpoint=False):
                dx, dy, dz = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
                r = Ray(pos=np.array([ox, oy, 0.]), dir=np.array([dx, dy, dz]), wavelength=1.225)
                tr = sys_opt.trace_ray(r)
                
                # Check if it passed the stop
                hit_stop = False
                for h in tr.history:
                    if 'Stop' in sys_opt.surfaces[h['surface_index']].name:
                        hit_stop = True
                        break
                
                if hit_stop:
                    # It hit the stop. Did it get vignetted BEFORE or AT the stop?
                    idx_vig = tr.history[-1]['surface_index']
                    if tr.vignetted and 'Stop' in sys_opt.surfaces[idx_vig].name:
                        # Vignetted by the stop itself, ignore
                        pass
                    elif not tr.vignetted:
                        passed_stop += 1
                        passed_detector += 1
                        det_hits_x.append(tr.history[-1]['x_local'])
                        det_hits_y.append(tr.history[-1]['y_local'])
                    elif tr.vignetted and 'Detector' in sys_opt.surfaces[idx_vig].name:
                        passed_stop += 1
                        vignetted_detector += 1

print(f'Rays that made it through the Stop: {passed_stop}')
print(f'Rays that hit the Detector: {passed_detector}')
print(f'Rays that missed the Detector: {vignetted_detector}')

r_hits = np.sqrt(np.array(det_hits_x)**2 + np.array(det_hits_y)**2)
print(f'Max radius on detector: {np.max(r_hits):.4f} mm')

