import sys, os
sys.path.insert(0, os.path.abspath('.'))

from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray

res = SurfaceStack.load_from_json(r'C:\Users\Juan\Desktop\spectrometer_raman_50mm_100mm_manual_optimized.json')
stack = res['stack']
wls = res.get('wavelengths', [1.087, 1.225, 1.35])

for idx in [15, 16, 20]:
    s = stack.surfaces[idx]
    print(f"Surface {idx} ({s.name}):")
    print(f"  global_origin: {s.global_origin}")
    print(f"  thickness: {s.thickness}")
    print(f"  tilt_x: {getattr(s, 'tilt_x', 0.0)}")
