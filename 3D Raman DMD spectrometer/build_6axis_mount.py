import sys
import os
import shutil
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD as App
import Part
import MeshPart

print("="*80)
print("BUILDING 6-AXIS CLEVER FLUSH MOUNT")
print("="*80)

# Base Block (55x55 mm, 22.5 mm total height)
base = Part.makeBox(55.0, 55.0, 22.5, App.Vector(-27.5, -27.5, 0))

# Detector Pocket (41x41 mm, 20 mm deep) -> Floor is 2.5 mm
pocket = Part.makeBox(41.0, 41.0, 20.0, App.Vector(-20.5, -20.5, 2.5))
base = base.cut(pocket)

# Optical Window (20 mm diameter)
window = Part.makeCylinder(10.0, 5.0, App.Vector(0,0,-1), App.Vector(0,0,1))
base = base.cut(window)

# Countersunk M4 holes to casing (30x30 spacing)
for dx in [-15.0, 15.0]:
    for dy in [-15.0, 15.0]:
        cyl = Part.makeCylinder(2.2, 5.0, App.Vector(dx, dy, -1.0), App.Vector(0,0,1))
        cone = Part.makeCone(2.2, 4.2, 2.5, App.Vector(dx, dy, 0), App.Vector(0,0,1))
        base = base.cut(cyl).cut(cone)

# 8x Radial M4 Set Screw Holes for X, Y, Z, Tip, Tilt
vX = App.Vector(1,0,0)
vY = App.Vector(0,1,0)

for z_height in [8.0, 17.0]:
    # Left/Right
    for dx in [-28.0, 20.0]:
        hole = Part.makeCylinder(1.65, 10.0, App.Vector(dx, 0, z_height), vX)
        base = base.cut(hole)
    # Top/Bottom
    for dy in [-28.0, 20.0]:
        hole = Part.makeCylinder(1.65, 10.0, App.Vector(0, dy, z_height), vY)
        base = base.cut(hole)

base.exportStep("PDAPC3_6Axis_Clever_Mount.step")

mesh_doc = App.newDocument()
m1 = mesh_doc.addObject("Mesh::Feature", "M1")
m1.Mesh = MeshPart.meshFromShape(base, 0.01, 0.1)
m1.Mesh.write("PDAPC3_6Axis_Clever_Mount.stl")

desktop = r"C:\Users\Juan\Desktop\Spectrometer_Print_Files"
for f in ["PDAPC3_6Axis_Clever_Mount.step", "PDAPC3_6Axis_Clever_Mount.stl"]:
    shutil.copy(f, os.path.join(desktop, f))

print("Successfully exported 6-Axis Clever Mount to Desktop!")
