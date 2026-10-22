import sys
sys.path.insert(0, ".")
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
import numpy as np

stack_dict = SurfaceStack.load_from_json(r'C:\Users\Juan\Desktop\spectrometer_raman_50mm_100mm_manual.json')
stack = stack_dict['stack']

# Remove dummy CB after grating and make grating single surface
cb_idx = None
grating_idx = None
for i, s in enumerate(stack.surfaces):
    if "grating" in s.name.lower() and s.surface_type == "Grating":
        grating_idx = i
    elif "cb_after" in s.name.lower():
        cb_idx = i
if cb_idx == grating_idx + 1:
    stack.surfaces[grating_idx].thickness = stack.surfaces[cb_idx].thickness
    stack.surfaces[grating_idx].grating_primary_wavelength_um = 1.064
    stack.remove_surface(cb_idx)

# Find DMD surface
dmd_surf = None
for s in stack.surfaces:
    if "dmd" in s.name.lower():
        dmd_surf = s
        break

print(f"DMD Surface: {dmd_surf.name}")
print(f"DMD Mechanical States: {list(dmd_surf.mechanical_states.keys())}")

# Trace with DMD in ON state
dmd_surf.mechanical_state = "ON"
sys_on = OpticalSystem(stack)
ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.064)
res_on = sys_on.trace_ray(ray)
last_on = res_on.history[-1]
pos_on = (last_on['x_global'], last_on['y_global'], last_on['z_global'])
print(f"DMD ON  -> Detector Hit: {np.round(pos_on, 2)}, Vignetted: {last_on['vignetted']}, Reached: {not last_on['vignetted']}")

# Trace with DMD in OFF state
dmd_surf.mechanical_state = "OFF"
sys_off = OpticalSystem(stack)
ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=1.064)
res_off = sys_off.trace_ray(ray)
last_off = res_off.history[-1]
pos_off = (last_off['x_global'], last_off['y_global'], last_off['z_global'])
print(f"DMD OFF -> Detector Hit: {np.round(pos_off, 2)}, Vignetted: {last_off['vignetted']}, Reached: {not last_off['vignetted']}")

for i, h in enumerate(res_off.history):
    s = stack.surfaces[i]
    if h['vignetted']:
        print(f"DMD OFF vignetted at S{i:02d} {s.name}")
        break
