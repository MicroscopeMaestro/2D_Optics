import os
import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part

width = 32.0
height = 22.0
thickness = 3.0
box = Part.makeBox(width, height, thickness, FreeCAD.Vector(-14.0, -11.0, -1.5))
slit_cut = Part.makeBox(8.0, 0.2, thickness + 0.2, FreeCAD.Vector(-4.0, -0.1, -1.6))
cartridge = box.cut(slit_cut)
right_channel = Part.makeBox(3.0, 1.3, thickness + 0.2, FreeCAD.Vector(2.5, 9.8, -1.6))
left_channel = Part.makeBox(3.0, 1.3, thickness + 0.2, FreeCAD.Vector(2.5, -11.1, -1.6))
cartridge = cartridge.cut(right_channel).cut(left_channel)
pull_hole = Part.makeCylinder(1.65, 5.0, FreeCAD.Vector(15.0, 0, 0), FreeCAD.Vector(-1, 0, 0))
cartridge = cartridge.cut(pull_hole)
cartridge.exportStep('external_slit_cartridge.step')
print("Exported external_slit_cartridge.step!")
