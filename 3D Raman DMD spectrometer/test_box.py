import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part

s_top = Part.read('support_top_half.step')
box = Part.makeBox(25.0, 66.0, 68.0, FreeCAD.Vector(-2.0, -86.0, 98.0))
test_dmd = s_top.common(box)
print(f'Test Vol: {test_dmd.Volume}')
