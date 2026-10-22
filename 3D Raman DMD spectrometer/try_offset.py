import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD
import Part

lens = Part.Shape()
lens.read('ACL25416U-Step.step')
lens = lens.Solids[0]
lens.transformShape(FreeCAD.Placement(FreeCAD.Vector(), FreeCAD.Rotation(FreeCAD.Vector(1, 0, 0), 90)).Matrix)

try:
    offset = lens.makeOffsetShape(0.2, 0.1, join=1)
    print("Offset successful! Volume:", offset.Volume)
except Exception as e:
    print("Offset failed:", e)
