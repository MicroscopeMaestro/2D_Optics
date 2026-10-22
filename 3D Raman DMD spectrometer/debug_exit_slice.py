import sys
sys.path.append(r"C:\Program Files\FreeCAD 1.1\bin")
import FreeCAD
import Part

s = Part.Shape()
s.read("support_top_half.step")
comp = s.slice(FreeCAD.Vector(1, 0, 0), 0.5)
print("Number of compound sections:", len(comp))
for i, sec in enumerate(comp):
    print(f"Section {i}: type={sec.ShapeType}, {len(sec.Wires)} wires")
    for j, w in enumerate(sec.Wires):
        print(f"  Wire {j}: Length={w.Length:.2f}, Closed={w.isClosed()}, BB={w.BoundBox}")
