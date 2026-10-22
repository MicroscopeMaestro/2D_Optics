import sys, os
sys.path.insert(0, os.path.abspath('.'))

from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray

res = SurfaceStack.load_from_json(r'C:\Users\Juan\Desktop\spectrometer_raman_50mm_100mm_manual_optimized.json')
stack = res['stack']
wls = res.get('wavelengths', [1.064])

sys_opt = OpticalSystem(stack)
ray = Ray(pos=(0.0, 0.0, 0.0), dir=(0.0, 0.0, 1.0), wavelength=wls[0])
traced = sys_opt.trace_ray(ray)

print("Traced ray history steps:", len(getattr(traced, 'history', [])))
for h in getattr(traced, 'history', []):
    s_idx = h.get('surface_idx', -1)
    s_name = h.get('surface_name', '')
    pos = (h.get('x', 0.0), h.get('y', 0.0), h.get('z', 0.0))
    d = (h.get('kx', 0.0), h.get('ky', 0.0), h.get('kz', 0.0))
    valid = h.get('valid', True)
    err = h.get('error', '')
    z_glob = h.get('z_global', 0.0)
    print(f"[{s_idx:02d}] {s_name:<25} | valid={valid:<5} | z_glob={z_glob:6.2f} | pos=({pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f}) | dir=({d[0]:6.3f}, {d[1]:6.3f}, {d[2]:6.3f}) {err}")
