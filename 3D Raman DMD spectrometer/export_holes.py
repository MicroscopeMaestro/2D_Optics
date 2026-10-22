import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part

doc = FreeCAD.openDocument('model1.FCStd')
dmd = next((obj for obj in doc.Objects if "22_CB_After_DMD_Hardware" in obj.Label), None)

vZ = FreeCAD.Vector(0, 0, 1)
z_vec = dmd.Placement.Rotation.multVec(vZ)
z_offset = 0.0
target_z = z_vec

compound = []
for dx, dy in [(-15, -15), (-15, 15), (15, -15), (15, 15)]:
    pos = dmd.Placement.Base + z_vec * z_offset + dmd.Placement.Rotation.multVec(FreeCAD.Vector(dx, dy, 0))
    cyl = Part.makeCylinder(1.65, 8.0, pos, target_z)
    compound.append(cyl)

res = compound[0]
for c in compound[1:]:
    res = res.fuse(c)

res.exportStep("holes_debug.step")
print("Exported holes_debug.step")
