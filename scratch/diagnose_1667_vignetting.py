import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from verify_layout import generate_fiber_phase_space_rays

stack = SurfaceStack.load_from_json("hybrid_raman_1064.json")["stack"]
opt = OpticalSystem(stack)

rays_standard = generate_fiber_phase_space_rays(0.0125, 0.12, [1.66708], dense_mode=False)
traced_std = opt.trace_rays(rays_standard)
lost_std = [r for r in traced_std if r.vignetted]
print(f"Standard sampling: Total={len(traced_std)}, Lost={len(lost_std)}")

rays_dense = generate_fiber_phase_space_rays(0.0125, 0.12, [1.66708], dense_mode=True)
traced_dense = opt.trace_rays(rays_dense)
lost_dense = [r for r in traced_dense if r.vignetted]
print(f"Dense sampling: Total={len(traced_dense)}, Lost={len(lost_dense)}")

for i, r in enumerate(lost_dense):
    print(f"\nLost ray {i+1}: initial pos={r.pos}, dir={r.dir}")
    print(f"History length: {len(r.history)}")
    for step in r.history:
        s_idx = step["surface_index"]
        s_name = stack.surfaces[s_idx].name
        s_semi = stack.surfaces[s_idx].semi_diameter
        r_hit = np.sqrt(step["x"]**2 + step["y"]**2)
        print(f"  Surf {s_idx} ({s_name}): hit=({step['x']:.3f}, {step['y']:.3f}, {step['z']:.3f}), r={r_hit:.3f} mm, CA semi={s_semi:.3f} mm")
