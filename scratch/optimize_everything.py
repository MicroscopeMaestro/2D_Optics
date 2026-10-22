"""
Optimize everything.
"""
import os
from scratch.optimize_hybrid_zmx import build_hybrid_spectrometer, generate_fiber_rays
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.optimizer import OpticalOptimizer

import numpy as np

stack = build_hybrid_spectrometer()
opt_sys = OpticalSystem(stack)
optimizer = OpticalOptimizer(opt_sys, wavelengths=0.785, method="Nelder-Mead", max_iter=200)

res = optimizer.optimize(variable_indices= {
    "thickness": [0, 3, 4, 7, 8, 11, 13],
    "tilt_x": [9, 12],
    "decenter_y": [9, 12]
})

rays = generate_fiber_rays([0.785, 0.808, 0.830, 0.850, 0.880])
traced_rays = opt_sys.trace_rays(rays)
det_hits_x, det_hits_y = [], []
for tr in traced_rays:
    for h in tr.history:
        if h["surface_index"] == 14:
            det_hits_x.append(h["x"])
            det_hits_y.append(h["y"])

max_r = max(np.max(np.abs(det_hits_x)), np.max(np.abs(det_hits_y)))
if max_r < 0.5:
    print("SUCCESS: MaxR=", max_r)
else:
    print("FAIL: MaxR=", max_r)
