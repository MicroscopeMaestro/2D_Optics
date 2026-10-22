import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray
from scratch.test_telecentric import build_test

sys_opt, _ = build_test(95.0, 50.0, 13.0, use_30mm_focus=False)
wls = [1.087, 1.225, 1.350]
for wl in wls:
    r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=wl)
    tr = sys_opt.trace_ray(r)
    h_det = tr.history[-1]
    xl, yl = h_det['x_local'], h_det['y_local']
    r_det = np.sqrt(xl**2 + yl**2)
    print(f"WL {wl}: Detector hit at ({xl:.4f}, {yl:.4f}), r={r_det:.4f} mm")
