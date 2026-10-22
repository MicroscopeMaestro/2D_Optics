import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part

try:
    b1 = Part.read('support_bottom_half.step').BoundBox
    print(f'Original Bottom BB: X[{b1.XMin}, {b1.XMax}]')
    b2 = Part.read('model5_bottom_refined.brep').BoundBox
    print(f'Model5 Bottom BB: X[{b2.XMin}, {b2.XMax}]')
except:
    pass
