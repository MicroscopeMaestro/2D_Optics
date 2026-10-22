import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part

doc = FreeCAD.openDocument('model1.FCStd')
for obj in doc.Objects:
    if hasattr(obj, 'Shape') and not obj.Shape.isNull():
        bb = obj.Shape.BoundBox
        print(f"{obj.Label:30} | Y:[{bb.YMin:6.1f}, {bb.YMax:6.1f}] Z:[{bb.ZMin:6.1f}, {bb.ZMax:6.1f}]")
