import sys, os
sys.path.insert(0, os.path.abspath('.'))
from optics_app.engine.lde import SurfaceStack

res = SurfaceStack.load_from_json(r'C:\Users\Juan\Desktop\spectrometer_raman_50mm_100mm_manual_optimized.json')
stack = res['stack']
for i in [14, 15, 16, 17, 19, 20, 21]:
    s = stack.surfaces[i]
    print(f"S{i:02d}: {s.name:<24} | glass: {s.glass:<7} | type: {s.surface_type:<10} | t: {s.thickness}")
