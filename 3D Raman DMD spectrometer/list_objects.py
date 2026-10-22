import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD

doc = FreeCAD.openDocument('model1.FCStd')
for obj in doc.Objects:
    print(obj.Label)
