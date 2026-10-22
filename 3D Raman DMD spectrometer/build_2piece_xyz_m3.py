import sys
import math
import os
import shutil
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD as App
import Part
import MeshPart

print("="*80)
print("BUILDING 2-PIECE COMPACT XYZ MOUNT (NON-AGGRESSIVE)")
print("="*80)

def make_hex_pocket(r_circ, depth, center):
    pts = []
    for i in range(7):
        ang = i * math.pi / 3.0
        pts.append(App.Vector(center.x + r_circ * math.cos(ang), center.y + r_circ * math.sin(ang), center.z))
    poly = Part.makePolygon(pts)
    return Part.Face(poly).extrude(App.Vector(0,0,depth))

# ---------------------------------------------------------
# PIECE 1: XY BASE
# ---------------------------------------------------------
p1_plate = Part.makeBox(45, 55, 2.5, App.Vector(-22.5, -27.5, 0))
# Shortened barrel (4.5 mm tall instead of 8 mm)
p1_barrel = Part.makeCylinder(10.5, 4.5, App.Vector(0,0,2.5), App.Vector(0,0,1))
p1 = p1_plate.fuse(p1_barrel)
p1 = p1.cut(Part.makeCylinder(6.0, 15, App.Vector(0,0,-1), App.Vector(0,0,1)))

for dx in [-15.0, 15.0]:
    for dy in [-15.0, 15.0]:
        cyl = Part.makeCylinder(3.0, 5, App.Vector(dx, dy, -1), App.Vector(0,0,1))
        p1 = p1.cut(cyl)

# ---------------------------------------------------------
# PIECE 2: Z CARRIER
# ---------------------------------------------------------
p2_flange = Part.makeBox(45, 45, 10, App.Vector(-22.5, -22.5, 0))
# Detector pocket (8mm deep, floor is 2mm)
pocket = Part.makeBox(37.5, 37.5, 8.5, App.Vector(-18.75, -18.75, 2.0))
p2 = p2_flange.cut(pocket)
p2 = p2.cut(Part.makeCylinder(7.5, 15, App.Vector(0,0,-1), App.Vector(0,0,1)))

# Shortened sleeve (4.5 mm long)
sleeve = Part.makeCylinder(13.5, 4.5, App.Vector(0,0,-4.5), App.Vector(0,0,1))
sleeve = sleeve.cut(Part.makeCylinder(10.6, 5.5, App.Vector(0,0,-5.0), App.Vector(0,0,1)))
p2 = p2.fuse(sleeve)

# Pinch Lug on the sleeve
lug = Part.makeBox(12, 6, 4.5, App.Vector(-6, 12, -4.5))
p2 = p2.fuse(lug)
slit = Part.makeBox(2, 20, 5.5, App.Vector(-1, 0, -5.0))
p2 = p2.cut(slit)
p2 = p2.cut(Part.makeCylinder(1.7, 20, App.Vector(-10, 15, -2.25), App.Vector(1,0,0)))

n3_poly = []
for i in range(7):
    ang = i * math.pi / 3.0
    n3_poly.append(App.Vector(0, 15 + 3.1*math.cos(ang), -2.25 + 3.1*math.sin(ang)))
n3_face = Part.Face(Part.makePolygon(n3_poly))
n3_pocket = n3_face.extrude(App.Vector(-4,0,0)).translate(App.Vector(-6,0,0))
p2 = p2.cut(n3_pocket)

# No clearance pockets needed! The shortened 4.5mm sleeve naturally suspends 
# the flange above the M3 screw heads on Piece 1.

for dx in [-19.5, 19.5]:
    for dy in [-19.5, 19.5]:
        p2 = p2.cut(Part.makeCylinder(1.25, 15, App.Vector(dx, dy, 0), App.Vector(0,0,1)))

# ---------------------------------------------------------
# PIECE 3: CLAMPING LID
# ---------------------------------------------------------
p3 = Part.makeBox(45, 45, 2.0, App.Vector(-22.5, -22.5, 0))
p3 = p3.cut(Part.makeCylinder(13.0, 5.0, App.Vector(0,0,-1), App.Vector(0,0,1)))
for dx in [-19.5, 19.5]:
    for dy in [-19.5, 19.5]:
        p3 = p3.cut(Part.makeCylinder(1.7, 5.0, App.Vector(dx, dy, -1), App.Vector(0,0,1)))


p1.exportStep("Micro_2Piece_XY_Base.step")
p2.exportStep("Micro_2Piece_Z_Carrier.step")
p3.exportStep("Micro_2Piece_Lid.step")

mesh_doc = App.newDocument()
for name, part in [("XY_Base", p1), ("Z_Carrier", p2), ("Lid", p3)]:
    m = mesh_doc.addObject("Mesh::Feature", name)
    m.Mesh = MeshPart.meshFromShape(part, 0.01, 0.1)
    m.Mesh.write(f"Micro_2Piece_{name}.stl")

desktop = r"C:\Users\Juan\Desktop\Spectrometer_Print_Files"
os.makedirs(desktop, exist_ok=True)
for f in os.listdir("."):
    if f.startswith("Micro_2Piece_") and (f.endswith(".step") or f.endswith(".stl")):
        shutil.copy(f, os.path.join(desktop, f))

print("Successfully generated Non-Aggressive Better 2-Piece Mount!")
