import sys, os, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scratch.test_lens_replacement import build_spectrometer_with_l2, evaluate_spectrometer

best_rmax = 9999.0
best_cfg = None

for d12 in np.linspace(5.0, 15.0, 11):
    for bfl in np.linspace(8.0, 9.0, 11):
        stack = build_spectrometer_with_l2("AC254-040-B")
        stack.surfaces[12].thickness = d12
        stack.surfaces[14].thickness = bfl
        stack._update_accumulated_positions()
        res = evaluate_spectrometer(stack)
        if res["total_transmitted"] > 60 and res["r_max_det_um"] < best_rmax:
            best_rmax = res["r_max_det_um"]
            best_cfg = (d12, bfl, res)

print(f"Optimal Config: d12 = {best_cfg[0]:.2f} mm, BFL = {best_cfg[1]:.2f} mm -> r_max = {best_cfg[2]['r_max_det_um']:.1f} um, r_rms = {best_cfg[2]['r_rms_det_um']:.1f} um, trans = {best_cfg[2]['transmission_pct']:.1f}%, encircled_100% = {best_cfg[2]['encircled_100pct']}")
