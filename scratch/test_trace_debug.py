import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray

loaded = SurfaceStack.load_from_json('hybrid_raman_spectrometer.json')
stack = loaded['stack'] if isinstance(loaded, dict) else loaded
system = OpticalSystem(stack)

print("--- SURFACES IN STACK ---")
for i, s in enumerate(stack.surfaces):
    ap_type = getattr(s, 'aperture_type', None)
    ap_r_outer = getattr(s, 'aperture_r_outer', None)
    print(f"{i}: {s.name} type={s.surface_type} semi={s.semi_diameter} glass={s.glass} t={s.thickness} ap_type={ap_type} ap_r_outer={ap_r_outer}")

print("\n--- TRACING CHIEF RAYS ---")
for wl in [1.1113, 1.2, 1.3, 1.38919, 1.5, 1.6133, 1.66708]:
    r = Ray(pos=(0.0, 0.0, 0.0), dir=(0.0, 0.0, 1.0), wavelength=wl)
    tr = system.trace_ray(r)
    print(f"\nWavelength {wl} um: vignetted={tr.vignetted}, len(history)={len(tr.history)}/{len(stack.surfaces)}")
    for h in tr.history:
        s_idx = h['surface_index']
        s_name = stack.surfaces[s_idx].name
        print(f"  surf {s_idx} ({s_name}): local_x={h['x']:.4f}, local_y={h['y']:.4f}")
