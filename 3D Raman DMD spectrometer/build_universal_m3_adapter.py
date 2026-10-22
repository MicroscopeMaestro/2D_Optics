import sys
import math
import os
import shutil
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD as App
import Part
import MeshPart

print("="*80)
print("BUILDING UNIVERSAL M3/M4 SLOTTED ADAPTER")
print("="*80)

adapter = Part.makeBox(45.0, 45.0, 4.0, App.Vector(-22.5, -22.5, 0))
window = Part.makeCylinder(9.0, 4.0, App.Vector(0,0,0), App.Vector(0,0,1))
adapter = adapter.cut(window)

for dx in [-15.0, 15.0]:
    for dy in [-15.0, 15.0]:
        cyl = Part.makeCylinder(2.2, 4.0, App.Vector(dx, dy, 0), App.Vector(0,0,1))
        cone = Part.makeCone(2.2, 4.2, 2.5, App.Vector(dx, dy, 1.5), App.Vector(0,0,1))
        adapter = adapter.cut(cyl).cut(cone)

def make_sliding_slot(is_x_axis, sign):
    if is_x_axis:
        channel = Part.makeBox(10.0, 5.8, 2.5, App.Vector(sign*11.0 if sign>0 else sign*21.0, -2.9, 0))
        slot = Part.makeBox(10.0, 3.2, 4.0, App.Vector(sign*11.0 if sign>0 else sign*21.0, -1.6, 0))
    else:
        channel = Part.makeBox(5.8, 10.0, 2.5, App.Vector(-2.9, sign*11.0 if sign>0 else sign*21.0, 0))
        slot = Part.makeBox(3.2, 10.0, 4.0, App.Vector(-1.6, sign*11.0 if sign>0 else sign*21.0, 0))
    return channel, slot

for is_x in [True, False]:
    for sign in [1, -1]:
        chan, slt = make_sliding_slot(is_x, sign)
        adapter = adapter.cut(chan).cut(slt)

for dx in [-10.0, 10.0]:
    for dy in [-10.0, 10.0]:
        cyl = Part.makeCylinder(1.25, 4.0, App.Vector(dx, dy, 0), App.Vector(0,0,1))
        adapter = adapter.cut(cyl)

adapter.exportStep("Universal_M3_M4_Adapter.step")

mesh_doc = App.newDocument()
m = mesh_doc.addObject("Mesh::Feature", "Adapter")
m.Mesh = MeshPart.meshFromShape(adapter, 0.01, 0.1)
m.Mesh.write("Universal_M3_M4_Adapter.stl")

desktop = r"C:\Users\Juan\Desktop\Spectrometer_Print_Files"
os.makedirs(desktop, exist_ok=True)
for f in ["Universal_M3_M4_Adapter.step", "Universal_M3_M4_Adapter.stl"]:
    shutil.copy(f, os.path.join(desktop, f))

print("Successfully exported Universal M3/M4 Adapter to Desktop!")
