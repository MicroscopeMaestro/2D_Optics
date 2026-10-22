import json
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
import numpy as np

with open("spectrometer_raman_50mm_100mm_manual.json", "r") as f:
    data = json.load(f)

print("system_config:", data.get("system_config"))
print("wavelengths:", data.get("wavelengths"))
print("primary_wavelength:", data.get("primary_wavelength"))

stack = SurfaceStack.from_dict(data)
print(f"Loaded stack with {len(stack.surfaces)} surfaces.")

for i, s in enumerate(stack.surfaces):
    tilts = f"tx={s.tilt_x}, ty={s.tilt_y}, tz={s.tilt_z}" if (s.tilt_x or s.tilt_y or s.tilt_z) else ""
    decenters = f"dx={s.decenter_x}, dy={s.decenter_y}, dz={s.decenter_z}" if (s.decenter_x or s.decenter_y or s.decenter_z) else ""
    grat = f"lines={s.grating_lines_per_mm}, order={s.grating_order}" if s.surface_type == SurfaceType.GRATING else ""
    conic_asph = f"k={s.conic}, asph={s.asphere_coeffs}" if (s.conic or s.asphere_coeffs) else ""
    extra = " | ".join(x for x in [tilts, decenters, grat, conic_asph] if x)
    st_val = s.surface_type.value if hasattr(s.surface_type, 'value') else str(s.surface_type)
    print(f"{i:2d}: {s.name:<24} t={s.thickness:<6.2f} {s.glass:<10} {st_val:<12} {extra}")

opt = OpticalSystem(stack)
print("\n--- Ray Tracing Test ---")
wls = data.get("wavelengths", [1.064])
for wl in wls:
    ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl)
    res = opt.trace_ray(ray)
    print(f"wl={wl:.4f} um: vignetted={res.vignetted}, final_pos={res.pos}")
