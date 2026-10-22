import sys
import math
import os
import shutil
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD as App
import Part
import MeshPart

print("="*80)
print("BUILDING MICRO-XYZ MOUNT (ALL M3 HARDWARE)")
print("="*80)

def make_hex_pocket(r_circ, depth, center):
    pts = []
    for i in range(7):
        ang = i * math.pi / 3.0
        pts.append(App.Vector(center.x + r_circ * math.cos(ang), center.y + r_circ * math.sin(ang), center.z))
    poly = Part.makePolygon(pts)
    return Part.Face(poly).extrude(App.Vector(0,0,depth))

# ---------------------------------------------------------
# STAGE 1: BASE PLATE (2.5mm thick)
# ---------------------------------------------------------
s1 = Part.makeBox(50, 50, 2.5, App.Vector(-25, -25, 0))
s1 = s1.cut(Part.makeCylinder(10.0, 5, App.Vector(0,0,-1), App.Vector(0,0,1)))

# M3 countersunk holes to casing (30x30 spacing)
for dx in [-15, 15]:
    for dy in [-15, 15]:
        cyl = Part.makeCylinder(1.7, 5, App.Vector(dx, dy, -1), App.Vector(0,0,1))
        # M3 Flat head cone
        cone = Part.makeCone(1.7, 3.2, 1.5, App.Vector(dx, dy, 1.0), App.Vector(0,0,1))
        s1 = s1.cut(cyl).cut(cone)

# M3 captive nuts for Stage 2 (40x40 spacing)
for dx in [-20, 20]:
    for dy in [-20, 20]:
        cyl = Part.makeCylinder(1.7, 5, App.Vector(dx, dy, -1), App.Vector(0,0,1))
        nut = make_hex_pocket(3.1, 2.0, App.Vector(dx, dy, 0.5))
        s1 = s1.cut(cyl).cut(nut)

# ---------------------------------------------------------
# STAGE 2: XY SLIDER + BARREL
# ---------------------------------------------------------
s2_plate = Part.makeBox(50, 50, 2.5, App.Vector(-25, -25, 0))
s2_barrel = Part.makeCylinder(10.5, 8.0, App.Vector(0,0,2.5), App.Vector(0,0,1))
s2 = s2_plate.fuse(s2_barrel)

# Oversized holes for M3 screws (allows XY sliding)
for dx in [-20, 20]:
    for dy in [-20, 20]:
        cyl = Part.makeCylinder(3.5, 5, App.Vector(dx, dy, -1), App.Vector(0,0,1))
        s2 = s2.cut(cyl)

s2 = s2.cut(Part.makeCylinder(8.0, 15, App.Vector(0,0,-1), App.Vector(0,0,1)))

# ---------------------------------------------------------
# STAGE 3: Z CARRIER + PINCH SLEEVE
# ---------------------------------------------------------
s3_flange = Part.makeBox(45, 45, 10, App.Vector(-22.5, -22.5, 0))
# Pocket on top (floor is 2.0 mm thick)
pocket = Part.makeBox(37.5, 37.5, 8, App.Vector(-18.75, -18.75, 2.0))
s3 = s3_flange.cut(pocket)
s3 = s3.cut(Part.makeCylinder(7.5, 15, App.Vector(0,0,-1), App.Vector(0,0,1)))

# M3 clearance holes for mounting the photodetector (30x30 spacing)
for dx in [-15, 15]:
    for dy in [-15, 15]:
        s3 = s3.cut(Part.makeCylinder(1.7, 15, App.Vector(dx, dy, -1), App.Vector(0,0,1)))

# Sleeve sticking DOWN from Z=0 to Z=-8
sleeve = Part.makeCylinder(13.5, 8.0, App.Vector(0,0,-8.0), App.Vector(0,0,1))
sleeve = sleeve.cut(Part.makeCylinder(10.6, 9.0, App.Vector(0,0,-8.5), App.Vector(0,0,1)))
s3 = s3.fuse(sleeve)

# Pinch Lug
lug = Part.makeBox(12, 6, 8, App.Vector(-6, 12, -8))
s3 = s3.fuse(lug)
slit = Part.makeBox(2, 20, 9, App.Vector(-1, 0, -8.5))
s3 = s3.cut(slit)
s3 = s3.cut(Part.makeCylinder(1.7, 20, App.Vector(-10, 15, -4), App.Vector(1,0,0)))

n3_poly = []
for i in range(7):
    ang = i * math.pi / 3.0
    n3_poly.append(App.Vector(0, 15 + 3.1*math.cos(ang), -4 + 3.1*math.sin(ang)))
n3_face = Part.Face(Part.makePolygon(n3_poly))
n3_pocket = n3_face.extrude(App.Vector(-4,0,0)).translate(App.Vector(-6,0,0))
s3 = s3.cut(n3_pocket)

# Corner M3 holes for lid
for dx in [-19.5, 19.5]:
    for dy in [-19.5, 19.5]:
        s3 = s3.cut(Part.makeCylinder(1.25, 15, App.Vector(dx, dy, 0), App.Vector(0,0,1)))

s1.exportStep("Micro_XYZ_Stage1_M3.step")
s2.exportStep("Micro_XYZ_Stage2_M3.step")
s3.exportStep("Micro_XYZ_Stage3_M3.step")

mesh_doc = App.newDocument()
for name, part in [("Stage1", s1), ("Stage2", s2), ("Stage3", s3)]:
    m = mesh_doc.addObject("Mesh::Feature", name)
    m.Mesh = MeshPart.meshFromShape(part, 0.01, 0.1)
    m.Mesh.write(f"Micro_XYZ_{name}_M3.stl")

desktop = r"C:\Users\Juan\Desktop\Spectrometer_Print_Files"
os.makedirs(desktop, exist_ok=True)
for f in os.listdir("."):
    if f.startswith("Micro_XYZ_") and "_M3" in f and (f.endswith(".step") or f.endswith(".stl")):
        shutil.copy(f, os.path.join(desktop, f))

print("Successfully generated ALL-M3 Micro-XYZ Mount!")
