import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part

doc = FreeCAD.open('model1.FCStd')
optics = doc.getObject('Optical_System_Assembly').Group

poly_bl_base = Part.makePolygon([
    FreeCAD.Vector(0, -18, -10),
    FreeCAD.Vector(0, -50, -10),
    FreeCAD.Vector(0, -50, 50),
    FreeCAD.Vector(0, -18, 50),
    FreeCAD.Vector(0, -18, -10)
])
face_bl_base = Part.Face(poly_bl_base).extrude(FreeCAD.Vector(100, 0, 0)).translate(FreeCAD.Vector(-50, 0, 0))

poly_br_base = Part.makePolygon([
    FreeCAD.Vector(0, 18, -10),
    FreeCAD.Vector(0, 50, -10),
    FreeCAD.Vector(0, 50, 160),
    FreeCAD.Vector(0, 18, 160),
    FreeCAD.Vector(0, 18, -10)
])
face_br_base = Part.Face(poly_br_base).extrude(FreeCAD.Vector(100, 0, 0)).translate(FreeCAD.Vector(-50, 0, 0))

for obj in optics:
    if hasattr(obj, 'Shape') and not obj.Shape.isNull():
        if face_bl_base.common(obj.Shape).Volume > 1e-4: print("WARNING: bl intersects", obj.Label)
        if face_br_base.common(obj.Shape).Volume > 1e-4: print("WARNING: br intersects", obj.Label)
        
print("Check complete.")
