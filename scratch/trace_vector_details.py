import sys, os, math
import numpy as np

sys.path.insert(0, r"c:\Users\Juan\Documents\GitHub\2D_Optics")
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem, Ray

loaded = SurfaceStack.load_from_json("mirror_spectrometer/all_mirror_spectrometer.json")
stack = loaded["stack"]
opt = OpticalSystem(stack)

wls = [0.785, 0.808, 0.830, 0.850, 0.880]
for wl in wls:
    r = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl)
    tr = opt.trace_ray(r)
    print(f"\n=== Wavelength {wl*1000:.0f} nm (vignetted={tr.vignetted}) ===")
    for h in tr.history:
        s = stack.surfaces[h["surface_index"]]
        print(f"  S{h['surface_index']} ({s.name:<32}): local=({h['x_local']:7.3f}, {h['y_local']:7.3f})")
