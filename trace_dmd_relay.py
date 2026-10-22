
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath('')))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('C:/Users/Juan/Desktop/Spectrometer_V2_Zemax.json', 'r') as f: data = json.load(f)

# Find DMD index
dmd_idx = -1
for i, s in enumerate(data['surfaces']):
    if 'DMD' in s.get('name', ''):
        dmd_idx = i
        break

stack = SurfaceStack()
dummy = {'name': 'Dummy_DMD_Obj', 'surface_type': 'Object', 'thickness': 86.9, 'glass': 'AIR'}
stack.add_surface(Surface(**dummy))

for sd in data['surfaces'][dmd_idx+1:]:
    s = Surface(name=sd.get('name', ''))
    for k,v in sd.items():
        if k == 'surface_type': s.surface_type = SurfaceType.DETECTOR if v.lower() == 'detector' else SurfaceType.STANDARD
        elif hasattr(s, k): setattr(s, k, v)
    stack.add_surface(s)
sys_opt = OpticalSystem(stack)

# Note: The distance from DMD to Relay L1 is 86.9 mm. I put that in the dummy thickness!
# So the Dummy_DMD_Obj is exactly at the DMD location!

theta_max = np.arcsin(0.06)
vignetted_rays = []
passed_rays = []
det_hits_x = []
det_hits_y = []

# Source points on DMD up to 2.5 mm radius (5 mm diameter)
for r_obj in [0.0, 1.25, 2.5]:
    for theta_obj in np.linspace(0, 2*np.pi, 8, endpoint=False):
        ox = r_obj * np.cos(theta_obj)
        oy = r_obj * np.sin(theta_obj)
        
        # Ray angles
        for theta in [0, theta_max]:
            for phi in np.linspace(0, 2*np.pi, 8, endpoint=False):
                dx, dy, dz = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
                r = Ray(pos=np.array([ox, oy, 0.]), dir=np.array([dx, dy, dz]), wavelength=1.225)
                tr = sys_opt.trace_ray(r)
                if tr.vignetted:
                    vignetted_rays.append((ox, oy, theta, phi, sys_opt.surfaces[tr.history[-1]['surface_index']].name))
                else:
                    passed_rays.append((ox, oy))
                    det_hits_x.append(tr.history[-1]['x_local'])
                    det_hits_y.append(tr.history[-1]['y_local'])

print(f'Passed: {len(passed_rays)}')
print(f'Vignetted: {len(vignetted_rays)}')
if vignetted_rays:
    surfs = {}
    for r in vignetted_rays:
        surfs[r[4]] = surfs.get(r[4], 0) + 1
    print('Vignetted at:')
    for k, v in surfs.items():
        print(f'  {k}: {v}')
if det_hits_x:
    r_hits = np.sqrt(np.array(det_hits_x)**2 + np.array(det_hits_y)**2)
    print(f'Max radius on detector: {np.max(r_hits):.4f} mm')

