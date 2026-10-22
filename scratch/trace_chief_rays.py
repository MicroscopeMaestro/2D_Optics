import numpy as np
import sys
sys.path.insert(0, ".")
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import RAMAN_WAVELENGTHS_UM

loaded = SurfaceStack.load_from_json("hybrid_raman_1064_full_range.json")
stack = loaded["stack"]
opt_sys = OpticalSystem(stack)

print("Primary wavelength:", getattr(stack, 'primary_wavelength', None))
print("\nTracing chief rays at each Raman wavelength:")
for wl in RAMAN_WAVELENGTHS_UM:
    r = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl)
    traced = opt_sys.trace_ray(r)
    print(f"\nWavelength {wl*1000:.2f} nm (vignetted={traced.vignetted}):")
    for h in traced.history:
        s_idx = h["surface_index"]
        s_name = stack.surfaces[s_idx].name
        print(f"  Surf {s_idx:2d} ({s_name:22s}): hit=({h['x']:7.3f}, {h['y']:7.3f}, {h['z']:7.3f})")
