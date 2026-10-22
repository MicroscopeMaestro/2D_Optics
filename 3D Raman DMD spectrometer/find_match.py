import sys
import glob
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part

for f in glob.glob('top_dmd*.step'):
    try:
        shape = Part.Shape()
        shape.read(f)
        bb = shape.BoundBox
        if abs(bb.YMin - -86.0) < 1.0 and abs(bb.YMax - -20.0) < 1.0:
            print(f'Match: {f} -> Y[{bb.YMin}, {bb.YMax}], Z[{bb.ZMin}, {bb.ZMax}], Vol={shape.Volume}')
    except Exception as e:
        pass
