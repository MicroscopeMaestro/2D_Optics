import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part
shape = Part.Shape()
shape.read('top_dmd_aperture_focused.step')
bb = shape.BoundBox
print(f'top_dmd_aperture_focused.step: X[{bb.XMin}, {bb.XMax}], Y[{bb.YMin}, {bb.YMax}], Z[{bb.ZMin}, {bb.ZMax}]')
