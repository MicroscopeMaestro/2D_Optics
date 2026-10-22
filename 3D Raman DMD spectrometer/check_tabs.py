import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part

f = r'C:\Users\Juan\Desktop\Spectrometer_Print_Files\Top_DMD_M4_with_tabs.step'
try:
    s = Part.read(f)
    bb = s.BoundBox
    print(f'Top_DMD_M4_with_tabs BB: X[{bb.XMin}, {bb.XMax}], Y[{bb.YMin}, {bb.YMax}], Z[{bb.ZMin}, {bb.ZMax}], Vol={s.Volume}')
except Exception as e:
    pass
