import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part

doc = FreeCAD.open('model1.FCStd')
optics = doc.getObject('Optical_System_Assembly').Group

poly_dmd_bot = Part.makePolygon([
    FreeCAD.Vector(0, -90, 95),
    FreeCAD.Vector(0, -40, 50),
    FreeCAD.Vector(0, -100, 50),
    FreeCAD.Vector(0, -100, 95),
    FreeCAD.Vector(0, -90, 95)
])
face_dmd_bot = Part.Face(poly_dmd_bot).extrude(FreeCAD.Vector(100, 0, 0)).translate(FreeCAD.Vector(-50, 0, 0))

for obj in optics:
    if hasattr(obj, 'Shape') and not obj.Shape.isNull():
        if face_dmd_bot.common(obj.Shape).Volume > 1e-4: print("WARNING: dmd_bot intersects", obj.Label)
        
print("Check complete.")
