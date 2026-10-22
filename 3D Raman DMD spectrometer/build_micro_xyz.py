import sys
import math
import os
import shutil
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD as App
import Part
import MeshPart

print("="*80)
print("BUILDING MICRO-XYZ TELESCOPING MOUNT")
print("="*80)

def make_hex_pocket(r_circ, depth, center):
    pts = []
    for i in range(7):
        ang = i * math.pi / 3.0
        pts.append(App.Vector(center.x + r_circ * math.cos(ang), center.y + r_circ * math.sin(ang), center.z))
    poly = Part.makePolygon(pts)
    return Part.Face(poly).extrude(App.Vector(0,0,depth))

s1 = Part.makeBox(50, 50, 2.5, App.Vector(-25, -25, 0))
s1 = s1.cut(Part.makeCylinder(10.0, 5, App.Vector(0,0,-1), App.Vector(0,0,1)))
for dx in [-15, 15]:
    for dy in [-15, 15]:
        cyl = Part.makeCylinder(2.2, 5, App.Vector(dx, dy, -1), App.Vector(0,0,1))
        cone = Part.makeCone(2.2, 4.2, 2.5, App.Vector(dx, dy, 0), App.Vector(0,0,1))
        s1 = s1.cut(cyl).cut(cone)
for dx in [-20, 20]:
    for dy in [-20, 20]:
        cyl = Part.makeCylinder(2.2, 5, App.Vector(dx, dy, -1), App.Vector(0,0,1))
        nut = make_hex_pocket(4.16, 2.0, App.Vector(dx, dy, 0.5))
        s1 = s1.cut(cyl).cut(nut)

s2_plate = Part.makeBox(50, 50, 2.5, App.Vector(-25, -25, 0))
s2_sleeve = Part.makeCylinder(13.0, 8.5, App.Vector(0,0,0), App.Vector(0,0,1))
s2 = s2_plate.fuse(s2_sleeve)
for dx in [-20, 20]:
    for dy in [-20, 20]:
        cyl = Part.makeCylinder(4.0, 5, App.Vector(dx, dy, -1), App.Vector(0,0,1))
        s2 = s2.cut(cyl)
s2 = s2.cut(Part.makeCylinder(9.1, 10, App.Vector(0,0,-1), App.Vector(0,0,1)))

lug = Part.makeBox(12, 6, 8.5, App.Vector(-6, 10, 0))
s2 = s2.fuse(lug)
slit = Part.makeBox(2, 20, 10, App.Vector(-1, 0, -1))
s2 = s2.cut(slit)
s2 = s2.cut(Part.makeCylinder(1.7, 20, App.Vector(-10, 14, 4), App.Vector(1,0,0)))

n3_poly = []
for i in range(7):
    ang = i * math.pi / 3.0
    n3_poly.append(App.Vector(0, 14 + 3.1*math.cos(ang), 4 + 3.1*math.sin(ang)))
n3_face = Part.Face(Part.makePolygon(n3_poly))
n3_pocket = n3_face.extrude(App.Vector(-4,0,0)).translate(App.Vector(-6,0,0))
s2 = s2.cut(n3_pocket)

s3_flange = Part.makeBox(45, 45, 10, App.Vector(-22.5, -22.5, 0))
pocket = Part.makeBox(37.5, 37.5, 9, App.Vector(-18.75, -18.75, 1.0))
s3 = s3_flange.cut(pocket)
swallow = Part.makeCylinder(17.0, 8.0, App.Vector(0,0,-1), App.Vector(0,0,1))
swallow_lug = Part.makeBox(14, 18, 8.0, App.Vector(-7, 0, -1))
s3 = s3.cut(swallow).cut(swallow_lug)
s3_barrel = Part.makeCylinder(8.9, 8.0, App.Vector(0,0,-7), App.Vector(0,0,1))
s3 = s3.fuse(s3_barrel)
s3 = s3.cut(Part.makeCylinder(7.5, 20, App.Vector(0,0,-10), App.Vector(0,0,1)))

for dx in [-19.5, 19.5]:
    for dy in [-19.5, 19.5]:
        s3 = s3.cut(Part.makeCylinder(1.25, 15, App.Vector(dx, dy, 0), App.Vector(0,0,1)))

s1.exportStep("Micro_XYZ_Stage1.step")
s2.exportStep("Micro_XYZ_Stage2.step")
s3.exportStep("Micro_XYZ_Stage3.step")

mesh_doc = App.newDocument()
for name, part in [("Stage1", s1), ("Stage2", s2), ("Stage3", s3)]:
    m = mesh_doc.addObject("Mesh::Feature", name)
    m.Mesh = MeshPart.meshFromShape(part, 0.01, 0.1)
    m.Mesh.write(f"Micro_XYZ_{name}.stl")

desktop = r"C:\Users\Juan\Desktop\Spectrometer_Print_Files"
os.makedirs(desktop, exist_ok=True)
for f in os.listdir("."):
    if f.startswith("Micro_XYZ_") and (f.endswith(".step") or f.endswith(".stl")):
        shutil.copy(f, os.path.join(desktop, f))

print("Successfully generated Micro-XYZ Mount!")
