import os
import sys
import numpy as np

sys.path.insert(0, ".")
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import RAMAN_WAVELENGTHS_UM, generate_fiber_phase_space_rays

loaded = SurfaceStack.load_from_json("hybrid_raman_1064_full_range.json")
stack = loaded["stack"]
opt_sys = OpticalSystem(stack)

print("=== SURFACE STACK ===")
for i, s in enumerate(stack.surfaces):
    st = s.surface_type.name if hasattr(s.surface_type, "name") else str(s.surface_type)
    print(f"[{i:2d}] {s.name:25s} type={st:10s} R={s.radius:8.2f} T={s.thickness:8.2f} Glass={s.glass:8s} SD={s.semi_diameter:6.2f} tilt_y={s.tilt_y:6.2f}")

rays = generate_fiber_phase_space_rays(fiber_radius_mm=0.0125, numerical_aperture=0.12, wavelengths=RAMAN_WAVELENGTHS_UM, dense_mode=False)
print(f"\nGenerated {len(rays)} rays.")

traced = opt_sys.trace_rays(rays)

print("\n=== PER-SURFACE INTERCEPT & VIGNETTING ANALYSIS ===")
for i, s in enumerate(stack.surfaces):
    hits = []
    reached = 0
    for r in traced:
        for h in r.history:
            if h["surface_index"] == i:
                reached += 1
                hits.append((h["x"], h["y"], h["z"]))
    
    if hits:
        xs = [h[0] for h in hits]
        ys = [h[1] for h in hits]
        rs = [np.sqrt(x**2 + y**2) for x, y in zip(xs, ys)]
        print(f"Surf {i:2d} ({s.name:22s}): Reached={reached:3d}/{len(rays)} | X=[{min(xs):7.3f}, {max(xs):7.3f}] Y=[{min(ys):7.3f}, {max(ys):7.3f}] max_r={max(rs):6.3f} (SD={s.semi_diameter:5.2f})")
    else:
        print(f"Surf {i:2d} ({s.name:22s}): Reached=  0/{len(rays)}")

vignetted_rays = [r for r in traced if r.vignetted]
print(f"\nTotal vignetted rays: {len(vignetted_rays)} / {len(rays)} ({len(vignetted_rays)/len(rays)*100:.1f}%)")

# Let's inspect where rays get vignetted:
vig_reasons = {}
for r in traced:
    if r.vignetted:
        # see which surface caused vignetting
        last_h = r.history[-1] if r.history else None
        surf_idx = last_h["surface_index"] if last_h else -1
        vig_reasons[surf_idx] = vig_reasons.get(surf_idx, 0) + 1

print("\nVignetting occurred at surface indices:")
for s_idx, cnt in sorted(vig_reasons.items()):
    name = stack.surfaces[s_idx].name if 0 <= s_idx < len(stack.surfaces) else "Unknown"
    print(f"  Surface {s_idx} ({name}): {cnt} rays")
