import sys, os
sys.path.insert(0, os.path.abspath('.'))

from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray

res = SurfaceStack.load_from_json(r'C:\Users\Juan\Desktop\spectrometer_raman_50mm_100mm_manual_optimized.json')
stack = res['stack']
wls = res.get('wavelengths', [1.087, 1.225, 1.35])

print(f"Loaded stack with {len(stack.surfaces)} surfaces.")
s15 = stack.surfaces[15]
s16 = stack.surfaces[16]
print(f"S15: {s15.name} | t={s15.thickness} | is_air_space={s15.is_air_space}")
print(f"S16: {s16.name} | t={s16.thickness} | is_air_space={s16.is_air_space}")

sys_opt = OpticalSystem(stack)
ray = Ray(pos=(0.0, 0.0, 0.0), dir=(0.0, 0.0, 1.0), wavelength=wls[0])
tr = sys_opt.trace_ray(ray)

print(f"Total steps traced: {len(tr.history)}")
for i, h in enumerate(tr.history):
    s_idx = h['surface_index']
    s = stack.surfaces[s_idx]
    pos = (h['x'], h['y'], h['z'])
    d = h['dir']
    vig = h['vignetted']
    tir = h['tir']
    print(f"[{s_idx:02d}] {s.name:<25} | vig={vig} tir={tir} | pos=({pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f}) | dir=({d[0]:6.3f}, {d[1]:6.3f}, {d[2]:6.3f})")
