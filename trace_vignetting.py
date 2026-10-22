
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath('')))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('C:/Users/Juan/Desktop/Spectrometer_V2_Complete.json', 'r') as f: data = json.load(f)

stack = SurfaceStack()
for sd in data['surfaces']:
    s = Surface(name=sd.get('name', ''))
    for k,v in sd.items():
        if k == 'surface_type': s.surface_type = SurfaceType.GRATING if v.lower() == 'grating' else (SurfaceType.ASPHERIC if v.lower() == 'aspheric' else SurfaceType.STANDARD)
        elif hasattr(s, k): setattr(s, k, v)
    stack.add_surface(s)
sys_opt = OpticalSystem(stack)

# The NA from the object is limited by the NA of the system.
# Actually, the user has an NA=0.12 in their other scripts.
# Let's trace a cone of rays from the object plane!
# The object plane is imaged 1:1 to the slit.
# The slit is 1 mm high (y = +/- 0.5) and 30 um wide (x = +/- 0.015).
# So the object is effectively bound to the same size!
# Let's trace from the top edge of the object (y=0.5, x=0) and the bottom edge (y=-0.5, x=0)
# and trace marginal rays at NA=0.12 (theta = arcsin(0.12) = 6.89 degrees)

theta = np.arcsin(0.12)
vignetted_rays = []
passed_rays = []

for oy in [-0.5, 0.0, 0.5]:
    for ox in [-0.015, 0.0, 0.015]:
        for phi in np.linspace(0, 2*np.pi, 8, endpoint=False):
            dx, dy, dz = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
            r = Ray(pos=np.array([ox, oy, 0.]), dir=np.array([dx, dy, dz]), wavelength=1.225)
            tr = sys_opt.trace_ray(r)
            if tr.vignetted:
                vignetted_rays.append((oy, phi, tr.history[-1]['surface_index'], sys_opt.surfaces[tr.history[-1]['surface_index']].name))
            else:
                passed_rays.append((oy, phi))

print(f'Passed: {len(passed_rays)}')
print(f'Vignetted: {len(vignetted_rays)}')
if vignetted_rays:
    # Group by surface
    surfs = {}
    for r in vignetted_rays:
        surfs[r[3]] = surfs.get(r[3], 0) + 1
    print('Vignetted at:')
    for k, v in surfs.items():
        print(f'  {k}: {v}')

