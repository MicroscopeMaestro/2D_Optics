import json, math, numpy as np, sys, os
sys.path.insert(0, os.path.abspath('.'))
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem, Ray

json_path = 'C:/Users/Juan/Desktop/spectrometer_raman_50mm_100mm_manual_optimized.json'
load_res = SurfaceStack.load_from_json(json_path)
stack = load_res['stack']

sys_opt = OpticalSystem(stack)
for wl in [1.087, 1.225, 1.35]:
    ray = Ray(pos=np.array([0.,0.,0.]), dir=np.array([0.,0.,1.]), wavelength=wl)
    traced = sys_opt.trace_rays([ray])[0]
    print(f'WL {wl} Vignetted: {traced.vignetted}')
    if traced.history:
        print(f'  Last surf: {traced.history[-1]["surface_index"]}')
