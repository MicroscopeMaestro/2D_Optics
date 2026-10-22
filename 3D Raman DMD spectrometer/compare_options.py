import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part

shape = Part.Shape()
shape.read('support_top_half.step')

# Option 1: Orthogonal cut
box1 = Part.makeBox(25, 70, 28, FreeCAD.Vector(-2, -110, 94))
dmd1 = shape.common(box1)
rest1 = shape.cut(box1)

# Check face count and edge count
print("=== Option 1 (Orthogonal Cut) ===")
print(f"dmd1:  Faces={len(dmd1.Faces)}, Edges={len(dmd1.Edges)}, Solids={len(dmd1.Solids)}, Vol={dmd1.Volume:.1f}")
print(f"rest1: Faces={len(rest1.Faces)}, Edges={len(rest1.Edges)}, Solids={len(rest1.Solids)}, Vol={rest1.Volume:.1f}")
print(f"dmd1 isValid: {dmd1.isValid()}, rest1 isValid: {rest1.isValid()}")

# Option 2: Diagonal cut
pts = [
    FreeCAD.Vector(0, -110, 94),
    FreeCAD.Vector(0, -40, 94),
    FreeCAD.Vector(0, -40, 108),
    FreeCAD.Vector(0, -105, 126),
    FreeCAD.Vector(0, -110, 126),
    FreeCAD.Vector(0, -110, 94)
]
wire = Part.makePolygon(pts)
face = Part.Face(wire)
cutter2 = face.extrude(FreeCAD.Vector(25, 0, 0))
cutter2.translate(FreeCAD.Vector(-2, 0, 0))

dmd2 = shape.common(cutter2)
rest2 = shape.cut(cutter2)

print("\n=== Option 2 (Diagonal Cut) ===")
print(f"dmd2:  Faces={len(dmd2.Faces)}, Edges={len(dmd2.Edges)}, Solids={len(dmd2.Solids)}, Vol={dmd2.Volume:.1f}")
print(f"rest2: Faces={len(rest2.Faces)}, Edges={len(rest2.Edges)}, Solids={len(rest2.Solids)}, Vol={rest2.Volume:.1f}")
print(f"dmd2 isValid: {dmd2.isValid()}, rest2 isValid: {rest2.isValid()}")
