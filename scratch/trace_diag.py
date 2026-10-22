import sys, os, json, numpy as np
sys.path.insert(0, '.')
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray

stack = SurfaceStack.load_from_json('dmd4500_nir_fingerprint_0p8mm_spectrometer.json')['stack']
sys = OpticalSystem(stack)

for wl in [1.1, 1.225, 1.35]:
    r = Ray(pos=[0,0,0], dir=[0,0,1], wavelength=wl)
    traced = sys.trace_ray(r)
    print('--- Wavelength', wl, '---')
    print('Hits:', len(traced.history), 'Vignetted:', traced.vignetted)
    for h in traced.history:
        s_idx = h["surface_index"]
        s = stack.surfaces[s_idx]
        x1 = h["x"]
        y1 = h["y"]
        vig_flag = h["vignetted"]
        rad = np.sqrt(x1**2 + y1**2)
        print(f'[Prog {s_idx:2d}] {s.name:<28}: x = {x1:7.3f}, y = {y1:7.3f}, radius = {rad:7.3f}, sd = {s.semi_diameter:5.1}, vig = {vig_flag}')
