
import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

sys_opt = OpticalSystem.from_json('teamwork_yz_spectrometer/spectrometer_yz_1064.json')
sys_opt._update_accumulated_positions()

for wl in [1.087, 1.225, 1.350]:
    r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=wl)
    tr = sys_opt.trace_ray(r)
    # The DMD is surface 16
    h_dmd = tr.history[16]
    y_dmd = h_dmd.get('y_local')
    print(f'Wavelength {wl} um -> Y position on DMD = {y_dmd:.4f} mm')

