import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part
from build_spectrometer import *

doc = FreeCAD.openDocument('model1.FCStd')
sketch = doc.getObject('Sketch')
face = Part.Face(sketch.Shape.Wires[0])
top_half = face.extrude(FreeCAD.Vector(17.5, 0, 0))

top_shifted = top_half.copy()
top_shifted.translate(FreeCAD.Vector(0, -60.0, 0))
top_half = top_half.fuse(top_shifted)

top_block = Part.makeBox(17.5, 60.0, 15.0, FreeCAD.Vector(0, -60.0, 233.0))
top_half = top_half.fuse(top_block)

asm = doc.getObject('Optical_System_Assembly')
sensor = None
for obj in asm.Group:
    if '23_S23' in obj.Label:
        sensor = obj
        break

sensor_z = sensor.Placement.Rotation.multVec(FreeCAD.Vector(0,0,1))
plane_pos = sensor.Placement.Base + sensor_z * 2.0
sensor_cut_box = Part.makeBox(200, 200, 200, FreeCAD.Vector(-100, -100, 0))
sensor_cut_box.transformShape(FreeCAD.Placement(plane_pos, sensor.Placement.Rotation).Matrix)

top_half = top_half.cut(sensor_cut_box)

spacing = 30.0
half = spacing / 2.0
for dx in [-half, half]:
    for dy in [-half, half]:
        offset = FreeCAD.Vector(dx, dy, 0)
        global_offset = sensor.Placement.Rotation.multVec(offset)
        start_pos = sensor.Placement.Base + sensor_z * 3.0 + global_offset
        if start_pos.x > 0:
            tap_cyl = Part.makeCylinder(1.65, 15.0, start_pos, sensor_z * -1.0)
            top_half = top_half.cut(tap_cyl)

print("Volume:", top_half.Volume)
print("Solids:", len(top_half.Solids))
