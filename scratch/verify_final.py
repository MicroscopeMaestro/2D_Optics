import json, math, numpy as np, sys, os
sys.path.insert(0, os.path.abspath('.'))
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem, Ray

json_path = 'C:/Users/Juan/Desktop/spectrometer_raman_50mm_100mm_manual_optimized.json'
load_res = SurfaceStack.load_from_json(json_path)
stack = load_res['stack']
wavelengths = load_res['wavelengths']

def trace_and_print(stack_obj, name):
    sys_opt = OpticalSystem(stack_obj)
    rays = []
    max_theta = np.arcsin(0.05)
    for p in range(12):
        phi = 2.0 * np.pi * p / 12
        dx = np.sin(max_theta) * np.cos(phi)
        dy = np.sin(max_theta) * np.sin(phi)
        dz = np.cos(max_theta)
        rays.append(Ray(pos=np.array([0.,0.,0.]), dir=np.array([dx, dy, dz]), wavelength=wavelengths[1])) # chief
    
    traced = sys_opt.trace_rays(rays)
    pts8 = []
    pts20 = []
    pts23 = []
    for r in traced:
        if r.vignetted: continue
        for h in r.history:
            if h['surface_index'] == 8: pts8.append((h['x'], h['y']))
            if h['surface_index'] == 20: pts20.append((h['x'], h['y']))
            if h['surface_index'] == 23: pts23.append((h['x'], h['y']))
    
    def rms(pts):
        if not pts: return -1
        return np.sqrt(np.mean(np.array([p[0]**2 + p[1]**2 for p in pts])))
    
    print(f'{name} RMS 8 (Slit): {rms(pts8)*1000:.2f} um')
    print(f'{name} RMS 20 (DMD): {rms(pts20)*1000:.2f} um')
    print(f'{name} RMS 23 (Det): {rms(pts23)*1000:.2f} um')

trace_and_print(stack, 'Final')
