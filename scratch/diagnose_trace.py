import sys, os
import numpy as np

sys.path.insert(0, r"c:\Users\Juan\Documents\GitHub\2D_Optics")
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem, Ray

loaded = SurfaceStack.load_from_json("mirror_spectrometer/all_mirror_spectrometer.json")
stack = loaded["stack"]
opt = OpticalSystem(stack)

theta_max = np.arcsin(0.10)
r = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, np.sin(theta_max), np.cos(theta_max)]), wavelength=0.830)
tr = opt.trace_ray(r)
print(f"Vignetted: {tr.vignetted}")
for idx, h in enumerate(tr.history):
    s_idx = h["surface_index"]
    s = stack.surfaces[s_idx]
    print(f"Hit {idx}: s_idx={s_idx} ({s.name}): local=({h['x_local']:.3f}, {h['y_local']:.3f}), semi={s.semi_diameter}, ap_type={s.aperture_type}, ap_r={s.aperture_r_outer}, ap_box=({s.aperture_x_half}, {s.aperture_y_half})")

# Let's also check all rays
wls = stack.wavelengths
rays = []
for wl in wls:
    for x in [-0.25, 0.0, 0.25]:
        for y in [-0.015, 0.0, 0.015]:
            rays.append(Ray(pos=np.array([x, y, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl))
            for ring in [0.5, 1.0]:
                for phi in np.linspace(0, 2*np.pi, 8, endpoint=False):
                    kx = np.sin(ring * theta_max) * np.cos(phi)
                    ky = np.sin(ring * theta_max) * np.sin(phi)
                    kz = np.cos(ring * theta_max)
                    rays.append(Ray(pos=np.array([x, y, 0.0]), dir=np.array([kx, ky, kz]), wavelength=wl))

print(f"\nTracing {len(rays)} rays...")
vig_reasons = []
for r_i, r in enumerate(rays):
    tr = opt.trace_ray(r)
    if tr.vignetted:
        last_h = tr.history[-1] if tr.history else None
        last_s = last_h["surface_index"] if last_h else -1
        # Test which surface failed check_aperture
        vig_reasons.append((r_i, len(tr.history), last_s, last_h))

print(f"Vignetted rays: {len(vig_reasons)} / {len(rays)}")
from collections import Counter
counts = Counter([v[1] for v in vig_reasons])
print("Failed after N surface intersections:", counts)
for v in vig_reasons[:10]:
    print(f"Ray {v[0]}: stopped at history len {v[1]}, last surface {v[2]}, hit={v[3]}")
