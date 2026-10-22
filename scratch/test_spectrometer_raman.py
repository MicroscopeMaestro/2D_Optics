import sys
sys.path.insert(0, ".")
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
import numpy as np

stack_dict = SurfaceStack.load_from_json(r'C:\Users\Juan\Desktop\spectrometer_raman_50mm_100mm.json')
stack = stack_dict['stack']
print("Surfaces before:", len(stack.surfaces))
for i, s in enumerate(stack.surfaces):
    if "grating" in s.name.lower() or "cb_after" in s.name.lower():
        print(f"  {i}: {s.name}, thick={s.thickness}, tilt_x={s.tilt_x}, is_cb={s.is_coord_break}")

# Find grating and remove CB_After_Grating
grating_idx = None
cb_idx = None
for i, s in enumerate(stack.surfaces):
    if "grating" in s.name.lower() and s.surface_type == "Grating":
        grating_idx = i
    elif "cb_after" in s.name.lower():
        cb_idx = i

if grating_idx is not None and cb_idx is not None and cb_idx == grating_idx + 1:
    cb_surf = stack.surfaces[cb_idx]
    grating_surf = stack.surfaces[grating_idx]
    grating_surf.thickness = cb_surf.thickness
    grating_surf.grating_primary_wavelength_um = 1.064
    stack.remove_surface(cb_idx)

print("Surfaces after:", len(stack.surfaces))
sys_opt = OpticalSystem(stack)
ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.064)
res = sys_opt.trace_ray(ray)
for i, h in enumerate(res.history):
    s = stack.surfaces[i]
    if any(k in s.name.lower() for k in ["grating", "mirror", "dmd", "detector", "ima"]):
        print(f"S{i:02d} {s.name:20s}: global=({h['x_global']:8.2f}, {h['y_global']:8.2f}, {h['z_global']:8.2f}) dir=({h['dir_x']:6.3f}, {h['dir_y']:6.3f}, {h['dir_z']:6.3f}) vig={h['vignetted']}")
