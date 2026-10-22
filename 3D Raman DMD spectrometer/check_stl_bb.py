import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Mesh

mesh = Mesh.Mesh('DMD_support_top.stl')
bb = mesh.BoundBox
print(f'DMD_support_top.stl BB: X[{bb.XMin}, {bb.XMax}], Y[{bb.YMin}, {bb.YMax}], Z[{bb.ZMin}, {bb.ZMax}]')
print(f'Mesh Volume: {mesh.Volume}')
