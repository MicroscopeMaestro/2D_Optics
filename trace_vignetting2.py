
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath('')))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('C:/Users/Juan/Desktop/Spectrometer_V2_Complete.json', 'r') as f: data = json.load(f)

# Keep only surfaces from Slit onwards to isolate the back-end!
stack = SurfaceStack()
# Add a dummy Object at the Slit location
dummy = {'name': 'Dummy_Slit', 'surface_type': 'Object', 'thickness': 50.0, 'glass': 'AIR'}
stack.add_surface(Surface(**dummy))

for sd in data['surfaces'][8:]:
    s = Surface(name=sd.get('name', ''))
    for k,v in sd.items():
        if k == 'surface_type': s.surface_type = SurfaceType.GRATING if v.lower() == 'grating' else (SurfaceType.ASPHERIC if v.lower() == 'aspheric' else SurfaceType.STANDARD)
        elif hasattr(s, k): setattr(s, k, v)
    stack.add_surface(s)
sys_opt = OpticalSystem(stack)

theta = np.arcsin(0.12)
vignetted_rays = []
passed_rays = []

# For a 5mm DMD area, the magnification from Slit to DMD is:
# Slit -> M1 (f=50) -> Foc (f=50) -> DMD.
# It's a 1:1 relay! So a 5mm area on the DMD corresponds to a 5mm area at the slit!
# But the slit is 1 mm high!
# If the user is worried about a 5 mm area, they probably want to see what happens for rays starting at y = 2.5 mm on the slit!
# Wait, if the slit is 1 mm high, there ARE NO rays at y = 2.5 mm!
# Ah! In the user's previous prompt, they said:
# 'We can use smaller area of DMD defined by the size of the slit so that the rays arrive to the single detector. For example, just have of the DMD area - 5 mm'
# Wait! They thought the area of the DMD was defined by the size of the slit!
# If the slit is 1 mm high, the image on the DMD is 1 mm high!
# If the image on the DMD is 1 mm high, then the 4f relay ONLY needs to image a 1 mm area, NOT 5 mm!
# Let's trace from the edges of a 1 mm slit (y = +/- 0.5 mm)!

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
    surfs = {}
    for r in vignetted_rays:
        surfs[r[3]] = surfs.get(r[3], 0) + 1
    print('Vignetted at:')
    for k, v in surfs.items():
        print(f'  {k}: {v}')

