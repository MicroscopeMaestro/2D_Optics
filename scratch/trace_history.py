import json
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
import numpy as np

with open("spectrometer_raman_50mm_100mm_manual.json", "r") as f:
    data = json.load(f)

stack = SurfaceStack.from_dict(data)
opt = OpticalSystem(stack)

wls = [1.087, 1.225, 1.350]

for wl in wls:
    print(f"\n==================== Wavelength {wl:.3f} um ====================")
    ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl)
    res = opt.trace_ray(ray)
    if hasattr(res, "history"):
        for step in res.history:
            print(step)
    elif hasattr(ray, "history"):
        for step in ray.history:
            print(step)
    else:
        print("Dir:", dir(res))
