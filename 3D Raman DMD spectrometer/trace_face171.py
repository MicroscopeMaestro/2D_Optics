import sys
FREECAD_BIN = r"C:\Program Files\FreeCAD 1.1\bin"
if FREECAD_BIN not in sys.path:
    sys.path.append(FREECAD_BIN)
import FreeCAD
import Part
import build_spectrometer

def find_target_face(shape):
    for f in shape.Faces:
        if type(f.Surface) == Part.Plane and abs(f.Surface.Axis.x) < 1e-3:
            if abs(f.Surface.Axis.y - 0.9640878) < 1e-3 and abs(f.Surface.Axis.z - (-0.2655835)) < 1e-3:
                return f
    return None

doc = FreeCAD.open("model1.FCStd")
sketch = doc.getObject("Sketch")
top_half = Part.Face(sketch.Shape.Wires[0]).extrude(FreeCAD.Vector(17.5, 0, 0))
bot_half = Part.Face(sketch.Shape.Wires[0]).extrude(FreeCAD.Vector(-17.5, 0, 0))

poly_tl = Part.makePolygon([
    FreeCAD.Vector(0, -120, 120), FreeCAD.Vector(0, -55, 200), FreeCAD.Vector(0, -55, 240),
    FreeCAD.Vector(0, -150, 240), FreeCAD.Vector(0, -150, 120), FreeCAD.Vector(0, -120, 120)
])
taper_tl = Part.Face(poly_tl).extrude(FreeCAD.Vector(100, 0, 0)).translate(FreeCAD.Vector(-50, 0, 0))
top_half = top_half.cut(taper_tl)
bot_half = bot_half.cut(taper_tl)

top_pad = Part.makeBox(30.0, 55.0, 45.0, FreeCAD.Vector(0, -63.0, 205.0))
bot_pad = Part.makeBox(30.0, 55.0, 45.0, FreeCAD.Vector(-30.0, -63.0, 205.0))
top_half = top_half.fuse(top_pad)
bot_half = bot_half.fuse(bot_pad)

sensor = next((o for o in doc.getObject("Optical_System_Assembly").Group if "23_s23" in o.Label.lower()), None)
sensor_z = sensor.Placement.Rotation.multVec(FreeCAD.Vector(0,0,1))
sensor_cut_box = Part.makeBox(200, 200, 200, FreeCAD.Vector(-100, -100, 0))
sensor_cut_box.transformShape(FreeCAD.Placement(sensor.Placement.Base + sensor_z * 2.0, sensor.Placement.Rotation).Matrix)
top_half = top_half.cut(sensor_cut_box)
bot_half = bot_half.cut(sensor_cut_box)

top_half = top_half.cut(taper_tl)
bot_half = bot_half.cut(taper_tl)

clean_top_x0 = top_half.removeSplitter()
x0_faces = [f for f in clean_top_x0.Faces if abs(f.BoundBox.XMin) < 1e-4 and abs(f.BoundBox.XMax) < 1e-4]
outer_face = Part.Face(x0_faces[0].OuterWire)
t1 = outer_face.makeOffset2D(-1.2)
t2 = outer_face.makeOffset2D(-3.6)
t_cut = t1.cut(t2)
main_t_face = max(t_cut.Faces, key=lambda f: f.Area)
tongue_solid = main_t_face.extrude(FreeCAD.Vector(3.0, 0, 0))
bot_half = bot_half.fuse(tongue_solid)
print("After fuse tongue_solid:", find_target_face(bot_half) is not None)
f = find_target_face(bot_half)
if f:
    print("Found Face BB:", f.BoundBox, "Area:", f.Area)
