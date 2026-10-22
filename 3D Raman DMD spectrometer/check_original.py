import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part
f = 'top_dmd_region_original.step'
try:
    shape = Part.Shape()
    shape.read(f)
    bb = shape.BoundBox
    print(f'{f}: X[{bb.XMin}, {bb.XMax}], Y[{bb.YMin}, {bb.YMax}], Z[{bb.ZMin}, {bb.ZMax}], Vol={shape.Volume}')
except Exception as e:
    pass
