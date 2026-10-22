import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part

shape = Part.Shape()
shape.read('support_top_half.step')
bb = shape.BoundBox
print(f'Top half BB: X[{bb.XMin}, {bb.XMax}], Y[{bb.YMin}, {bb.YMax}], Z[{bb.ZMin}, {bb.ZMax}]')

dmd = Part.Shape()
dmd.read('top_dmd_region.step')
bb = dmd.BoundBox
print(f'DMD piece BB: X[{bb.XMin}, {bb.XMax}], Y[{bb.YMin}, {bb.YMax}], Z[{bb.ZMin}, {bb.ZMax}]')

opt = Part.Shape()
opt.read('top_optics_region.step')
bb = opt.BoundBox
print(f'Optics piece BB: X[{bb.XMin}, {bb.XMax}], Y[{bb.YMin}, {bb.YMax}], Z[{bb.ZMin}, {bb.ZMax}]')
