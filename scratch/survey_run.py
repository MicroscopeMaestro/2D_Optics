import sys, os, time, json, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.analysis.spot import SpotDiagram
from optics_app.analysis.rayfan import RayFanAnalysis

json_file = 'dmd4500_nir_fingerprint_0p8mm_spectrometer.json'
loaded = SurfaceStack.load_from_json(json_file)
stack = loaded['stack']
wls = loaded['wavelengths']
primary_wl = loaded.get('primary_wavelength', 1.225)
sys = OpticalSystem(stack)

print('FILE:', json_file)
print('TOTAL SURFACESs', len(stack.surfaces))
print('WAVELENGTHS:', wls, 'Primary:', primary_wl)

for i, s in enumerate(stack.surfaces):
    print(f'[{i:2d}] {s.name:<28} type={s.surface_type:<10} R={s.radius:8.2f} t={s.thickness:7.2f} glass={s.glass:<8} sd={s.semi_diameter:5.1f} tx={s.tilt_x:6.2f} ty={s.tilt_y:6.2f}')

print('\n--- CHIEF RAYS ---')
for wl in wls:
    r = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl)
    t = sys.trace_ray(r)
    dmd_h = [h for h in t.history if h['surface_index'] == 7]
    det_h = [h for h in t.history if h['surface_index'] == 16]
    dmd_pos = dmd_h[0]['y'] if dmd_h else None
    det_x = det_h[0]['x'] if det_h else None
    det_y = det_h[0]['y'] if det_h else None
    print(f'WL ;{wl:.4f} um: Hits={len(t.history)}, VNG={t.vignetted}, DMD Y={dmd_pos}, Det=({det_x}, {det_y})')

print('\n--- PERFORMANCE BENCHMARKS ---')
t0 = time.perf_counter()
for _ in range(1000):
    sys.trace_ray(Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.225))
t1 = time.perf_counter()
print(f'Single ray time: {(t1-t0)/1000*1e6:.2f} us')

t0 = time.perf_counter()
for _ in range(100):
    rays = [Ray(pos=np.array([0.0,0.0,0.0]), dir=np.array([0.0, np.sin(py*np.radians(6.8)), np.cos(py*np.radians(6.8))]), wavelength=wl) for wl in wls for py in np.linspace(-1,1,7)]
    sys.trace_rays(rays)
t1 = time.perf_counter()
fast_ms = (t1-t0)/100*1e3
print(f'Fast fan (21 rays): {fast_ms:.3f} ms ({1000.0/fast_ms:.1f} fps)')

t0 = time.perf_counter()
for _ in range(50):
    rays = [Ray(pos=np.array([0.0,0.0,0.0]), dir=np.array([0.0, np.sin(py*np.radians(6.8)), np.cos(py*np.radians(6.8))]), wavelength=wl) for wl in wls for py in np.linspace(-1,1,50)]
    sys.trace_rays(rays)
t1 = time.perf_counter()
dense_ms = (t1-t0)/50*1e3
print(f'Dense fan (150 rays): {dense_ms:.3f} ms ({1000.0/dense_ms:.1f} fps)')
