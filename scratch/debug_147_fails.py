import sys, os, math
import numpy as np
from collections import Counter

sys.path.insert(0, r"c:\Users\Juan\Documents\GitHub\2D_Optics")
from scratch.test_zero_vignetting import test_complete_zero_vignetting

stack, max_r, num_vig = test_complete_zero_vignetting(f4=19.0)
from v2.engine.tracer import OpticalSystem, Ray

opt = OpticalSystem(stack)
wls = stack.wavelengths
theta_max = math.asin(0.10)
rays = []
for wl in wls:
    for x in [-0.25, 0.0, 0.25]:
        for y in [-0.015, 0.0, 0.015]:
            rays.append(Ray(pos=np.array([x, y, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl))
            for ring in [0.5, 1.0]:
                for p in range(8):
                    phi = 2 * math.pi * p / 8
                    kx = math.sin(ring * theta_max) * math.cos(phi)
                    ky = math.sin(ring * theta_max) * math.sin(phi)
                    kz = math.cos(ring * theta_max)
                    rays.append(Ray(pos=np.array([x, y, 0.0]), dir=np.array([kx, ky, kz]), wavelength=wl))

traced = [opt.trace_ray(r) for r in rays]
fails = []
for r_i, tr in enumerate(traced):
    if tr.vignetted:
        # Check which surface failed
        for idx, h in enumerate(tr.history):
            s = stack.surfaces[h["surface_index"]]
            if not s.check_aperture(h["x_local"], h["y_local"]):
                fails.append((h["surface_index"], s.name, h["x_local"], h["y_local"], s.semi_diameter, s.aperture_type, getattr(s, "aperture_r_outer", 0)))
                break
        else:
            last = tr.history[-1] if tr.history else None
            fails.append((-1, "None", 0, 0, 0, "", 0))

print("\nExact Failure Points:")
c = Counter([f[0] for f in fails])
for s_idx, cnt in c.items():
    s_name = stack.surfaces[s_idx].name if s_idx >= 0 else "Unknown"
    print(f"  Surface {s_idx} ({s_name}): {cnt} rays")

print("\nSample fail hits:")
for f in fails[:15]:
    print(f"  S{f[0]} ({f[1]}): hit=({f[2]:.4f}, {f[3]:.4f}), semi={f[4]}, ap_type={f[5]}, ap_r={f[6]}")
