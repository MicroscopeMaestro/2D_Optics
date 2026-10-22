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
    print(f"WL {wl}: vignetted={tr.vignetted}, history_len={len(tr.history)}")
    if tr.vignetted:
        last = tr.history[-1]
        print(f"  Vignetted at surface {last['surface_index']}: {sys_opt.surfaces[last['surface_index']].name}")
    else:
        h = tr.history[-1]
        print(f"  Hit detector at: ({h['x_local']:.4f}, {h['y_local']:.4f})")
