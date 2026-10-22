import json
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
import numpy as np

with open("spectrometer_raman_50mm_100mm_manual.json", "r") as f:
    data = json.load(f)

stack = SurfaceStack.from_dict(data)
opt = OpticalSystem(stack)

wls = [1.087, 1.225, 1.350]

print("=== Check at Surface 15 (Grating) and Surface 21 (DMD) ===")
for wl in wls:
    ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl)
    res = opt.trace_ray(ray)
    
    # Find step for surface 15 (grating) and surface 21 (DMD)
    s15 = next((s for s in res.history if s['surface_index'] == 15), None)
    s21 = next((s for s in res.history if s['surface_index'] == 21), None)
    s25 = next((s for s in res.history if s['surface_index'] == 25), None)
    
    if s15:
        print(f"wl={wl:.3f} um: at Grating (S15) dir={s15['dir']}")
    if s21:
        print(f"wl={wl:.3f} um: at DMD (S21) y_local={s21['y_local']:.4f} mm, y_global={s21['y_global']:.4f} mm, pos={s21['x']}, {s21['y']}, {s21['z']}")
    if s25:
        print(f"wl={wl:.3f} um: at Det (S25) y_local={s25['y_local']:.4f} mm, y_global={s25['y_global']:.4f} mm")
