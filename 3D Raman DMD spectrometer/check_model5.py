import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part

s = Part.read(r'C:\Users\Juan\Desktop\Spectrometer_Print_Files\support_top_half_model5.step')
bb = s.BoundBox
print(f'Model5 Top BB: X[{bb.XMin}, {bb.XMax}], Y[{bb.YMin}, {bb.YMax}], Z[{bb.ZMin}, {bb.ZMax}]')
