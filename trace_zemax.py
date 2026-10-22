
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
        if k == 'surface_type': s.surface_type = SurfaceType.GRATING if v.lower() == 'grating' else (SurfaceType.ASPHERIC if v.lower() == 'aspheric' else SurfaceType.STANDARD)
        elif hasattr(s, k): setattr(s, k, v)
    stack.add_surface(s)
sys_opt = OpticalSystem(stack)

r_chief = Ray(pos=np.array([0.,0.,0.]), dir=np.array([0.,0.,1.]), wavelength=1.225)
tr = sys_opt.trace_ray(r_chief)
print('Chief Ray Vignetted:', tr.vignetted)

theta = np.arcsin(0.12)
vignetted_rays = []
passed_rays = []
for oy in [-0.5, 0.0, 0.5]:
    for ox in [-0.015, 0.0, 0.015]:
        for phi in np.linspace(0, 2*np.pi, 8, endpoint=False):
            # Trace from slit to image
            dx, dy, dz = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
            r = Ray(pos=np.array([ox, oy, 0.]), dir=np.array([dx, dy, dz]), wavelength=1.225)
            # Wait, the ray needs to start at the SLIT to bypass fore-optics vignetting!
            # Let's just create a dummy system starting at the slit!
            pass

# Sub-system starting at Slit
stack_sub = SurfaceStack()
dummy = {'name': 'Dummy_Slit', 'surface_type': 'Object', 'thickness': 50.0, 'glass': 'AIR'}
stack_sub.add_surface(Surface(**dummy))
slit_idx = -1
for i, s in enumerate(data['surfaces']):
    if 'Input_Slit' in s['name']:
        slit_idx = i
        break
for sd in data['surfaces'][slit_idx+1:]:
    s = Surface(name=sd.get('name', ''))
    for k,v in sd.items():
        if k == 'surface_type': s.surface_type = SurfaceType.GRATING if v.lower() == 'grating' else (SurfaceType.ASPHERIC if v.lower() == 'aspheric' else SurfaceType.STANDARD)
        elif hasattr(s, k): setattr(s, k, v)
    stack_sub.add_surface(s)
sys_opt_sub = OpticalSystem(stack_sub)

for oy in [-0.5, 0.0, 0.5]:
    for ox in [-0.015, 0.0, 0.015]:
        for phi in np.linspace(0, 2*np.pi, 8, endpoint=False):
            dx, dy, dz = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
            r = Ray(pos=np.array([ox, oy, 0.]), dir=np.array([dx, dy, dz]), wavelength=1.225)
            tr = sys_opt_sub.trace_ray(r)
            if tr.vignetted:
                vignetted_rays.append((oy, phi, tr.history[-1]['surface_index'], sys_opt_sub.surfaces[tr.history[-1]['surface_index']].name))
            else:
                passed_rays.append((oy, phi))

print(f'Passed: {len(passed_rays)}')
print(f'Vignetted: {len(vignetted_rays)}')
if vignetted_rays:
    surfs = {}
    for r in vignetted_rays:
        surfs[r[3]] = surfs.get(r[3], 0) + 1
    print('Vignetted at:')
    for k, v in surfs.items():
        print(f'  {k}: {v}')

