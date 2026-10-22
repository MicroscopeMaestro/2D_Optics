import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD

doc = FreeCAD.openDocument('model1.FCStd')
optics = doc.getObject('Optical_System_Assembly')
for obj in optics.Group:
    print(f"Name: {obj.Name}, Label: {obj.Label}, Type: {obj.TypeId}")
