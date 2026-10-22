import sys, os
sys.path.insert(0, os.path.abspath('.'))

from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import trace_system, trace_system_batch

res = SurfaceStack.load_from_json(r'C:\Users\Juan\Desktop\spectrometer_raman_50mm_100mm_manual_optimized.json')
stack = res['stack']
wls = res.get('wavelengths', [1.064])
print("Surfaces count:", len(stack.surfaces))
print("Wavelengths:", wls)

# Trace sequentially and report status at each surface
try:
    rays = trace_system(stack, wavelengths=wls, num_rays=5)
    print(f"Traced {len(rays)} rays successfully.")
    for idx, r in enumerate(rays[:2]):
        print(f"Ray {idx}: length of history={len(getattr(r, 'history', []))}")
        for h in getattr(r, 'history', []):
            print(f"  surf {h.get('surface_idx')}: name={h.get('surface_name')} valid={h.get('valid')} z_glob={h.get('z_global', 0.0):.2f}")
except Exception as e:
    import traceback
    traceback.print_exc()
