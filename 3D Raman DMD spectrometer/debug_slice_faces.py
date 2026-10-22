import sys
FREECAD_BIN = r"C:\Program Files\FreeCAD 1.1\bin"
if FREECAD_BIN not in sys.path:
    sys.path.append(FREECAD_BIN)
import FreeCAD
import Part

s = Part.Shape()
s.read("support_top_half.step")
comp = s.slice(FreeCAD.Vector(1, 0, 0), 0.5)

# Try making faces from the wires of the slice
faces = []
for sec in comp:
    for w in sec.Wires:
        if w.isClosed():
            try:
                faces.append(Part.Face(w))
            except:
                pass

print(f"Total closed wire loops in slice: {len(faces)}")
for i, f in enumerate(faces):
    print(f"Face {i}: Area={f.Area:.2f}, BB={f.BoundBox}")
