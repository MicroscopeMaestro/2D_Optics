import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part

doc = FreeCAD.open('model1.FCStd')
optics = doc.getObject('Optical_System_Assembly').Group

# We want to cut the sharp bottom-left corner of the DMD housing.
# DMD is at Y = -66.5, Z = 105.1.
# Its box goes down to Z = 95.
# Let's slice from Y = -85, Z = 115 down to Y = -21, Z = 50.
poly_dmd_bot = Part.makePolygon([
    FreeCAD.Vector(0, -85, 115),
    FreeCAD.Vector(0, -21, 50),
    FreeCAD.Vector(0, -100, 50),
    FreeCAD.Vector(0, -100, 115),
    FreeCAD.Vector(0, -85, 115)
])
face_dmd_bot = Part.Face(poly_dmd_bot).extrude(FreeCAD.Vector(100, 0, 0)).translate(FreeCAD.Vector(-50, 0, 0))

# Also the sensor pad has sharp bottom corners?
# Sensor is at Z = 227, Y = -33.
# top_pad is at Y = -60, Z = 233. Wait, top_pad is Z=233 to Z=248.
# Let's check for intersections with optics for this new dmd cut:
for obj in optics:
    if hasattr(obj, 'Shape') and not obj.Shape.isNull():
        if face_dmd_bot.common(obj.Shape).Volume > 1e-4: print("WARNING: dmd_bot intersects", obj.Label)
        
print("Check complete.")
