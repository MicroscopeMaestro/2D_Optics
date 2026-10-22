
import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

sys_opt = OpticalSystem.from_json('spectrometer_refractive_1064.json')

for wl in [1.087, 1.225, 1.350]:
    for theta in np.linspace(0.001, np.arcsin(0.04), 3):
        for phi in np.linspace(0, 2*np.pi, 4, endpoint=False):
            dx, dy, dz = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
            r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([dx, dy, dz]), wavelength=wl)
            tr = sys_opt.trace_ray(r)
            if tr.vignetted:
                h = tr.history[-1]
                s = sys_opt.surfaces[h['surface_index']]
                print(f'Vignetted at {s.name}')

