import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part

doc = FreeCAD.openDocument('model1.FCStd')
dmd = next((obj for obj in doc.Objects if "22_CB_After_DMD_Hardware" in obj.Label), None)

vZ = FreeCAD.Vector(0, 0, 1)
z_vec = dmd.Placement.Rotation.multVec(vZ)
center = dmd.Placement.Base - z_vec

y_vec_global = dmd.Placement.Rotation.multVec(FreeCAD.Vector(0, 1, 0))

# We want: center.y + y_vec_global.y * y_local >= -85.5 (leave 0.5mm clearance)
y_local = (-85.5 - center.y) / y_vec_global.y
print(f"Max negative y_local to stay above Y=-85.5: {y_local:.3f}")

# And what about the positive y_local?
# The top casing goes up to some height, it's probably fine.
