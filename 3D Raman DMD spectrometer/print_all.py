import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD

doc = FreeCAD.open('model1.FCStd')
optics = doc.getObject('Optical_System_Assembly').Group
for obj in optics:
    print(f"{obj.Label}: Base = {obj.Placement.Base}")
