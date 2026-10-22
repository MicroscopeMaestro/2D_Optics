import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from scratch.test_spot_fast import evaluate, rays
from verify_layout import RAMAN_WAVELENGTHS_UM

print("Testing f_dmd=39.0mm with AC254-050-C relay and AC254-075-C relay:")
for relay_f in [50.0, 75.0]:
    for f_dmd in [38.0, 38.5, 39.0, 39.5, 40.0]:
        for dmd_r in [15.0, 20.0, 25.0]:
            for c_d in [4.5, 4.8, 5.0, 5.2]:
                hits, vig = evaluate(f_dmd=f_dmd, dmd_r=dmd_r, c_d=c_d, pri_wl=1.38919, relay_f=relay_f)
                if vig == 0 and hits:
                    rads = [np.sqrt(x**2 + y**2) for x, y, wl in hits]
                    max_r = max(rads)
                    if max_r <= 1.000:
                        max_per_wl = {}
                        for x, y, wl in hits:
                            r = np.sqrt(x**2 + y**2)
                            max_per_wl[wl] = max(max_per_wl.get(wl, 0.0), r)
                        wl_str = ", ".join([f"{wl}:{max_per_wl.get(wl, 0):.2f}" for wl in RAMAN_WAVELENGTHS_UM])
                        print(f"Relay f={relay_f:.0f}, f_dmd={f_dmd:.1f}, dmd_r={dmd_r:.0f}, cd={c_d:.1f} | max_r={max_r:.4f} mm | {wl_str}")
