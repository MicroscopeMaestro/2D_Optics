import sys
FREECAD_BIN = r"C:\Program Files\FreeCAD 1.1\bin"
if FREECAD_BIN not in sys.path:
    sys.path.append(FREECAD_BIN)
import FreeCAD
import Part
import build_spectrometer

def check_shape(name, shape):
    count = 0
    for f in shape.Faces:
        if type(f.Surface) == Part.Plane and abs(f.Surface.Axis.x) < 1e-3:
            if abs(f.Surface.Axis.y - 0.9640878) < 1e-3 and abs(f.Surface.Axis.z - (-0.2655835)) < 1e-3:
                # Check distance of P0 to plane
                P0 = FreeCAD.Vector(0, -55.367353, 208.717276)
                if abs((f.Surface.Position - P0).dot(f.Surface.Axis)) < 1e-3:
                    count += 1
                    print(f"[{name}] Found matching plane: BB={f.BoundBox}")
    return count

# Let's inspect step by step
doc = FreeCAD.open("model1.FCStd")
sketch = doc.getObject("Sketch")
top_half = Part.Face(sketch.Shape.Wires[0]).extrude(FreeCAD.Vector(17.5, 0, 0))
bot_half = Part.Face(sketch.Shape.Wires[0]).extrude(FreeCAD.Vector(-17.5, 0, 0))

check_shape("Initial Extrude", bot_half)

# Check pad
top_pad = Part.makeBox(30.0, 55.0, 45.0, FreeCAD.Vector(0, -63.0, 205.0))
bot_pad = Part.makeBox(30.0, 55.0, 45.0, FreeCAD.Vector(-30.0, -63.0, 205.0))
bot_half = bot_half.fuse(bot_pad)
check_shape("After Pad Fuse", bot_half)

sensor = next((o for o in doc.getObject("Optical_System_Assembly").Group if "23_s23" in o.Label.lower()), None)
sensor_z = sensor.Placement.Rotation.multVec(FreeCAD.Vector(0,0,1))
sensor_cut_box = Part.makeBox(200, 200, 200, FreeCAD.Vector(-100, -100, 0))
sensor_cut_box.transformShape(FreeCAD.Placement(sensor.Placement.Base + sensor_z * 2.0, sensor.Placement.Rotation).Matrix)
bot_half = bot_half.cut(sensor_cut_box)
check_shape("After Sensor Cut Box", bot_half)
