import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from scratch.superfast_optimizer import build_spectrometer, full_rays
from optics_app.engine.tracer import OpticalSystem
from verify_layout import RAMAN_WAVELENGTHS_UM

for pri_wl in [1.3620, 1.3650, 1.3700, 1.3750, 1.3800, 1.3850, 1.3892]:
    for dmd_r in [15.0, 18.0, 20.0, 22.0, 25.0]:
        for r_c in [3.0, 5.0, 7.0, 10.0]:
            for c_d in [4.8, 5.0, 5.2, 5.4, 5.6]:
                stack = build_spectrometer(
                    f_coll=47.5, c_g=20.0, g_f=20.0, f_dmd=44.0,
                    dmd_r=dmd_r, relay_name='AC254-075-C', orient='forward', r_c=r_c, c_d=c_d, pri_wl=pri_wl
                )
                opt = OpticalSystem(stack)
                traced = opt.trace_rays(full_rays)
                det_idx = len(stack.surfaces) - 1
                
                det_hits = [(h['x'], h['y']) for r in traced for h in r.history if h['surface_index'] == det_idx]
                if len(det_hits) == len(full_rays):
                    radii = [np.sqrt(x**2 + y**2) for x, y in det_hits]
                    max_r = max(radii)
                    rms_r = np.sqrt(np.mean(np.array(radii)**2))
                    if max_r < 2.0:
                        print(f"pri_wl={pri_wl:.4f}, dmd_r={dmd_r:.1f}, r_c={r_c:.1f}, c_d={c_d:.1f} -> max_r={max_r:.4f} mm (dia={max_r*2:.4f} mm), rms_r={rms_r:.4f} mm")
                        sys.stdout.flush()
