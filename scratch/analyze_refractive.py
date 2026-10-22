import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

sys_opt = OpticalSystem.from_json('spectrometer_refractive_1064.json')
sys_opt._update_accumulated_positions()

wls = [1.087, 1.225, 1.350]
for wl in wls:
    r0 = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=wl)
    tr = sys_opt.trace_ray(r0)
    for h in tr.history:
        s = sys_opt.surfaces[h['surface_index']]
        if 'DMD' in s.name:
            print(f"WL {wl:.3f} Chief Ray at DMD: local=({h['x_local']:.3f}, {h['y_local']:.3f})")
        if 'Detector' in s.name:
            print(f"WL {wl:.3f} Chief Ray at Detector: local=({h['x_local']:.3f}, {h['y_local']:.3f})")
