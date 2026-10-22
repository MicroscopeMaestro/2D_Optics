
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath('')))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

with open('mirror_spectrometer/all_mirror_spectrometer.json', 'r') as f: data = json.load(f)
for i, s in enumerate(data['surfaces']):
    if 'M1' in s['name']: m1_idx = i
    if 'Grating' in s['name']: grat_idx = i
    if 'M2' in s['name']: m2_idx = i
    if 'DMD' in s['name']: dmd_idx = i
    if 'M3' in s['name']: m3_idx = i
    if 'M4' in s['name']: m4_idx = i

data['surfaces'][grat_idx]['grating_primary_wavelength_um'] = 1.225
data['surfaces'][m1_idx]['tilt_x'] = 20.0
data['surfaces'][grat_idx]['tilt_x'] = -28.1
data['surfaces'][m2_idx]['tilt_x'] = 15.0
data['surfaces'][dmd_idx]['tilt_x'] = 12.0
data['surfaces'][m3_idx]['tilt_x'] = -8.0
data['surfaces'][m4_idx]['tilt_x'] = 8.0
data['surfaces'][m4_idx]['thickness'] = 18.8

stack = SurfaceStack()
for sd in data['surfaces']:
    s = Surface(name=sd.get('name', ''))
    for k,v in sd.items():
        if k == 'surface_type': s.surface_type = SurfaceType.GRATING if v.lower() == 'grating' else (SurfaceType.STOP if v.lower() == 'stop' else (SurfaceType.OBJECT if v.lower() == 'object' else SurfaceType.STANDARD))
        elif hasattr(s, k): setattr(s, k, v)
    stack.add_surface(s)
sys_opt = OpticalSystem(stack)

wls = [1.087, 1.150, 1.225, 1.290, 1.350]
theta_max = np.arcsin(0.06)

for wl in wls:
    print(f'\nWavelength {wl}:')
    for theta in np.linspace(0.001, theta_max, 5):
        for phi in np.linspace(0, 2*np.pi, 6, endpoint=False):
            dx, dy, dz = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
            r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([dx, dy, dz]), wavelength=wl)
            tr = sys_opt.trace_ray(r)
            if tr.vignetted:
                last_surf = sys_opt.surfaces[tr.history[-1]['surface_index']].name
                if 'Stop' not in last_surf:
                    print(f'  UNEXPECTED VIGNETTE: {last_surf}')

