import json, math, numpy as np, sys, os
sys.path.insert(0, os.path.abspath('.'))
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem, Ray

json_path = 'C:/Users/Juan/Desktop/spectrometer_raman_50mm_100mm_manual_optimized.json'
load_res = SurfaceStack.load_from_json(json_path)
stack = load_res['stack']
wavelengths = load_res['wavelengths']

sys_opt = OpticalSystem(stack)
rays = []
max_theta = np.arcsin(0.05)
for wl in wavelengths:
    for p in range(12):
        phi = 2.0 * np.pi * p / 12
        dx = np.sin(max_theta) * np.cos(phi)
        dy = np.sin(max_theta) * np.sin(phi)
        dz = np.cos(max_theta)
        rays.append(Ray(pos=np.array([0.,0.,0.]), dir=np.array([dx, dy, dz]), wavelength=wl))
    # and the chief ray
    rays.append(Ray(pos=np.array([0.,0.,0.]), dir=np.array([0., 0., 1.]), wavelength=wl))

traced = sys_opt.trace_rays(rays)
for wl in wavelengths:
    pts23 = []
    vig = 0
    for r in traced:
        if abs(r.wavelength - wl) > 1e-5: continue
        if r.vignetted: 
            vig += 1
            continue
        for h in r.history:
            if h['surface_index'] == 23:
                pts23.append((h['x'], h['y']))
    
    if not pts23:
        print(f'WL {wl}: ALL VIGNETTED ({vig})')
    else:
        rms = np.sqrt(np.mean(np.array([p[0]**2 + p[1]**2 for p in pts23])))
        print(f'WL {wl}: RMS Det = {rms*1000:.2f} um, Vig = {vig}')
