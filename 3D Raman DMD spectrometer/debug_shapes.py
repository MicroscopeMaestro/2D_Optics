import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD
doc = FreeCAD.openDocument('model1.FCStd')
for name in ['Part__Feature002', 'Part__Feature005', 'Part__Feature007']:
    obj = doc.getObject(name)
    bb = obj.Shape.BoundBox
    print(f"Name: {name}, Label: {obj.Label}")
    print(f"  Shape BB: Z={bb.ZMin}..{bb.ZMax}")
    print(f"  Placement: Base={obj.Placement.Base}")
    glob_shape = obj.Shape.copy()
    glob_shape.transformShape(obj.Placement.Matrix)
    gbb = glob_shape.BoundBox
    print(f"  Global BB: Z={gbb.ZMin}..{gbb.ZMax}")
