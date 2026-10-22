import json
import sys

# Load JSON directly, convert to objects
sys.path.append('c:/Users/Juan/Documents/GitHub/2D_Optics')
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.optimizer import OpticalOptimizer

with open('thorlabs_spectrometer.json', 'r') as f:
    data = json.load(f)

stack = SurfaceStack()
for s in data["surfaces"]:
    surf = Surface()
    surf.name = s.get("name", "")
    surf.surface_type = getattr(SurfaceType, s.get("type", "STANDARD"))
    surf.radius = s.get("radius", 0.0)
    surf.thickness = s.get("thickness", 0.0)
    surf.glass_name = s.get("glass", "AIR")
    surf.semi_diameter = s.get("semi_diameter", 10.0)
    surf.radius_variable = s.get("radius_variable", False)
    surf.thickness_variable = s.get("thickness_variable", False)
    surf.tilt_x = s.get("tilt_x", 0.0)
    surf.tilt_x_variable = s.get("tilt_x_variable", False)
    if surf.surface_type == SurfaceType.GRATING:
        surf.grating_lines_per_mm = s.get("grating_lines_per_mm", 600.0)
        surf.grating_order = s.get("grating_order", 1)
    stack.add_surface(surf)

wls = data["wavelengths"]

class DummySystem:
    def __init__(self, stack, wls):
        self.stack = stack
        self.wavelengths = wls
sys_obj = DummySystem(stack, wls)
optimizer = OpticalOptimizer(sys_obj)

# Find variable indices
var_indices = []
for i, s in enumerate(stack.surfaces):
    if s.radius_variable or s.thickness_variable or s.tilt_x_variable:
        var_indices.append(i)

print(f"Variables to optimize: {var_indices}")

res = optimizer.optimize(var_indices)
print("Optimization finished!")
# Removed print

for i, s in enumerate(stack.surfaces):
    if i in var_indices:
        print(f"Surface {i} ({s.name}):")
        if s.thickness_variable:
            print(f"  Thickness = {s.thickness:.3f} mm")
        if s.tilt_x_variable:
            print(f"  Tilt X    = {s.tilt_x:.3f} deg")

# Save the optimized result back
out_data = {"surfaces": [], "wavelengths": wls}
for s in stack.surfaces:
    d = {
        "name": s.name,
        "type": s.surface_type.name,
        "radius": s.radius,
        "thickness": s.thickness,
        "glass": s.glass_name,
        "semi_diameter": s.semi_diameter,
        "radius_variable": s.radius_variable,
        "thickness_variable": s.thickness_variable,
        "tilt_x": s.tilt_x,
        "tilt_x_variable": s.tilt_x_variable
    }
    if s.surface_type == SurfaceType.GRATING:
        d["grating_lines_per_mm"] = s.grating_lines_per_mm
        d["grating_order"] = s.grating_order
    out_data["surfaces"].append(d)

with open('thorlabs_spectrometer_optimized.json', 'w') as f:
    json.dump(out_data, f, indent=4)
print("Saved thorlabs_spectrometer_optimized.json")
