import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part

doc = FreeCAD.open('model1.FCStd')
optics = doc.getObject('Optical_System_Assembly').Group

poly_sensor_tr = Part.makePolygon([
    FreeCAD.Vector(0, -15, 233),
    FreeCAD.Vector(0, 0, 250),
    FreeCAD.Vector(0, 50, 250),
    FreeCAD.Vector(0, 50, 233),
    FreeCAD.Vector(0, -15, 233)
])
face_sensor_tr = Part.Face(poly_sensor_tr).extrude(FreeCAD.Vector(100, 0, 0)).translate(FreeCAD.Vector(-50, 0, 0))

for obj in optics:
    if hasattr(obj, 'Shape') and not obj.Shape.isNull():
        if face_sensor_tr.common(obj.Shape).Volume > 1e-4: print("WARNING: sensor_tr intersects", obj.Label)
        
print("Check complete.")
