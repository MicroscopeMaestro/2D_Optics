import sys
import math
import os
import shutil
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD as App
import Part
import MeshPart

print("="*80)
print("BUILDING ZERO-THICKNESS XY FLUSH MOUNT")
print("="*80)

base = Part.makeBox(55.0, 55.0, 17.5, App.Vector(-27.5, -27.5, 0))
pocket = Part.makeBox(41.0, 41.0, 16.0, App.Vector(-20.5, -20.5, 2.5))
base = base.cut(pocket)

window = Part.makeCylinder(10.0, 5.0, App.Vector(0,0,-1), App.Vector(0,0,1))
base = base.cut(window)

for dx in [-15.0, 15.0]:
    for dy in [-15.0, 15.0]:
        cyl = Part.makeCylinder(2.2, 5.0, App.Vector(dx, dy, -1.0), App.Vector(0,0,1))
        cone = Part.makeCone(2.2, 4.2, 2.5, App.Vector(dx, dy, 0), App.Vector(0,0,1))
        base = base.cut(cyl).cut(cone)

vX = App.Vector(1,0,0)
vY = App.Vector(0,1,0)
h_z = 10.0 
for dx in [-28.0, 20.0]:
    hole = Part.makeCylinder(1.65, 10.0, App.Vector(dx, 0, h_z), vX)
    base = base.cut(hole)
for dy in [-28.0, 20.0]:
    hole = Part.makeCylinder(1.65, 10.0, App.Vector(0, dy, h_z), vY)
    base = base.cut(hole)

for dx in [-23.5, 23.5]:
    for dy in [-23.5, 23.5]:
        hole = Part.makeCylinder(1.25, 10.0, App.Vector(dx, dy, 8.0), App.Vector(0,0,1))
        base = base.cut(hole)

lid = Part.makeBox(55.0, 55.0, 3.0, App.Vector(-27.5, -27.5, 0))
lid_cut = Part.makeBox(35.0, 35.0, 5.0, App.Vector(-17.5, -17.5, -1.0))
cable_slot = Part.makeBox(15.0, 20.0, 5.0, App.Vector(-7.5, 15.0, -1.0))
lid = lid.cut(lid_cut).cut(cable_slot)
for dx in [-23.5, 23.5]:
    for dy in [-23.5, 23.5]:
        hole = Part.makeCylinder(1.7, 5.0, App.Vector(dx, dy, -1.0), App.Vector(0,0,1))
        cb = Part.makeCylinder(3.0, 2.0, App.Vector(dx, dy, 1.0), App.Vector(0,0,1))
        lid = lid.cut(hole).cut(cb)

base.exportStep("PDAPC3_Clever_Flush_Mount.step")
lid.exportStep("PDAPC3_Clever_Flush_Lid.step")

mesh_doc = App.newDocument()
m1 = mesh_doc.addObject("Mesh::Feature", "M1")
m1.Mesh = MeshPart.meshFromShape(base, 0.01, 0.1)
m1.Mesh.write("PDAPC3_Clever_Flush_Mount.stl")

m2 = mesh_doc.addObject("Mesh::Feature", "M2")
m2.Mesh = MeshPart.meshFromShape(lid, 0.01, 0.1)
m2.Mesh.write("PDAPC3_Clever_Flush_Lid.stl")

desktop = r"C:\Users\Juan\Desktop\Spectrometer_Print_Files"
os.makedirs(desktop, exist_ok=True)
for f in ["PDAPC3_Clever_Flush_Mount.step", "PDAPC3_Clever_Flush_Lid.step", "PDAPC3_Clever_Flush_Mount.stl", "PDAPC3_Clever_Flush_Lid.stl"]:
    shutil.copy(f, os.path.join(desktop, f))

print("Successfully exported clever flush mount files to Desktop!")
