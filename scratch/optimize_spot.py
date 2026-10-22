import sys
import os
import math
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scratch.solve_relay import create_spectrometer, CATALOG
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import RAMAN_WAVELENGTHS_UM, generate_fiber_phase_space_rays

rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM)

def cost_fn(params, relay_name, reverse_relay):
    d1, d2, d3 = params
    if d1 < 1.0 or d2 < 0.5 or d3 < 1.0:
        return 1e6
    stack = create_spectrometer(relay_name=relay_name, reverse_relay=reverse_relay, t_dmd_relay=d1, t_relay_cond=d2, t_cond_det=d3)
    opt = OpticalSystem(stack)
    traced = opt.trace_rays(rays)
    
    det_hits = [h for r in traced for h in r.history if h["surface_index"] == 14]
    if len(det_hits) < len(rays):
        return 1e5 + (len(rays) - len(det_hits)) * 1000.0
        
    rs = [np.sqrt(h["x"]**2 + h["y"]**2) for h in det_hits]
    max_r = max(rs)
    rms_r = np.sqrt(np.mean(np.array(rs)**2))
    return max_r + 0.1 * rms_r

results = []
print("Starting Scipy Optimization for all 1-inch doublets...")

for name in ["AC254-075-C", "AC254-050-C", "AC254-040-C", "AC254-030-C", "AC254-100-C"]:
    f_nom = CATALOG[name]["f_approx"]
    for rev in [False, True]:
        for d1_init in [f_nom - 5.0, f_nom, f_nom + 5.0, 15.0, 20.0]:
            for d2_init in [2.0, 5.0, 10.0]:
                for d3_init in [4.8, 5.2, 5.5]:
                    x0 = [d1_init, d2_init, d3_init]
                    res = minimize(cost_fn, x0, args=(name, rev), method='Nelder-Mead', options={'maxiter': 40, 'xatol': 0.1, 'fatol': 0.01})
                    d1_opt, d2_opt, d3_opt = res.x
                    val = res.fun
                    if val < 5.0:
                        stack = create_spectrometer(relay_name=name, reverse_relay=rev, t_dmd_relay=d1_opt, t_relay_cond=d2_opt, t_cond_det=d3_opt)
                        opt = OpticalSystem(stack)
                        traced = opt.trace_rays(rays)
                        det_hits = [h for r in traced for h in r.history if h["surface_index"] == 14]
                        vig_optics = [r for r in traced if r.vignetted and (not r.history or r.history[-1]["surface_index"] < 14)]
                        if det_hits:
                            rs = [np.sqrt(h["x"]**2 + h["y"]**2) for h in det_hits]
                            max_r = max(rs)
                            cx = np.mean([h["x"] for h in det_hits])
                            cy = np.mean([h["y"] for h in det_hits])
                            results.append((max_r, len(det_hits), len(vig_optics), name, rev, d1_opt, d2_opt, d3_opt, cx, cy))
                            if max_r <= 0.60 and len(vig_optics) == 0:
                                print(f"GREAT CANDIDATE: max_r: {max_r:.4f} mm | {name} (rev={rev}) | d1={d1_opt:.2f}, d2={d2_opt:.2f}, d3={d3_opt:.2f} | vigs={len(vig_optics)} | cx={cx:+.3f}")

results.sort(key=lambda x: x[0])
print("\n" + "="*80)
print(f"Top 10 Results across all optimizations:")
for r in results[:10]:
    print(f"Max R: {r[0]:.4f} mm (Diam: {2*r[0]:.4f} mm) | Hits: {r[1]}/{len(rays)} | Vig: {r[2]} | {r[3]} (rev={r[4]}) | d1={r[5]:.2f}, d2={r[6]:.2f}, d3={r[7]:.2f} | centroid=({r[8]:+.3f}, {r[9]:+.3f})")
