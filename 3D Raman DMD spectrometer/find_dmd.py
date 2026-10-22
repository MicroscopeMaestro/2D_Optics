import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part

for f in ['DMD_Adapter.step', 'DMD4500_Alignment_Frame_Final.step', 'top_dmd_with_m4.step']:
    try:
        shape = Part.Shape()
        shape.read(f)
        bb = shape.BoundBox
        print(f'{f}: Y[{bb.YMin}, {bb.YMax}], Z[{bb.ZMin}, {bb.ZMax}], Vol={shape.Volume}')
    except Exception as e:
        pass
