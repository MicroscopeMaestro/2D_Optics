import json, math, numpy as np, sys, os
sys.path.insert(0, os.path.abspath('.'))
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem, Ray

json_path = 'C:/Users/Juan/Desktop/spectrometer_raman_50mm_100mm_manual_optimized.json'
load_res = SurfaceStack.load_from_json(json_path)
stack = load_res['stack']

def get_rays(start_pos, NA, wavelengths):
    rays = []
    max_theta = np.arcsin(NA)
    for wl in wavelengths:
        for ring in [1]:
            theta = ring * max_theta
            n_pts = 6
            for p in range(n_pts):
                phi = 2.0 * np.pi * p / n_pts
                dx = np.sin(theta) * np.cos(phi)
                dy = np.sin(theta) * np.sin(phi)
                dz = np.cos(theta)
                rays.append(Ray(pos=np.array(start_pos, dtype=float), dir=np.array([dx, dy, dz], dtype=float), wavelength=wl))
    return rays

sys_opt = OpticalSystem(stack)
rays = get_rays([0,0,0], 0.05, [1.087, 1.225, 1.35])
traced = sys_opt.trace_rays(rays)
for r in traced:
    if r.vignetted:
        print(f'WL {r.wavelength} ray vignetted at {r.history[-1]["surface_index"] if r.history else "None"}')
