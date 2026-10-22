import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD

doc = FreeCAD.open('model1.FCStd')
asm = doc.getObject('Optical_System_Assembly')
for obj in asm.Group:
    if "23_s23" in obj.Label.lower():
        print(f"Sensor Position: {obj.Placement.Base.x:.2f}, {obj.Placement.Base.y:.2f}, {obj.Placement.Base.z:.2f}")
