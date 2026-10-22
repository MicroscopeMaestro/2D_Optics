
import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

import gen_refractive_1064
sys_opt = gen_refractive_1064.build_opt(20.0, 20.0, 100.0, 90.0)

for wl in [1.087, 1.225, 1.350]:
    for theta in np.linspace(0.001, np.arcsin(0.04), 3):
        for phi in np.linspace(0, 2*np.pi, 4, endpoint=False):
            dx, dy, dz = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
            r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([dx, dy, dz]), wavelength=wl)
            tr = sys_opt.trace_ray(r)
            if tr.vignetted:
                s_idx = tr.history[-1]['surface_index']
                s = sys_opt.surfaces[s_idx]
                h = tr.history[-1]
                print('Vignetted wl', wl, 'at surface', s.name, 'local r', np.hypot(h.get('x_local', 0), h.get('y_local', 0)), 'max r', s.semi_diameter)

