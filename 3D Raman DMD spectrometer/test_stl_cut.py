import sys
sys.path.append(r'C:\Program Files\FreeCAD 1.1\bin')
import FreeCAD, Part, Mesh, MeshPart

print('Loading mesh...')
mesh = Mesh.Mesh('DMD_support_top.stl')
print('Converting to shape...')
shape = Part.Shape()
shape.makeShapeFromMesh(mesh.Topology, 0.05)
print('Making solid...')
solid = Part.Solid(shape)
if not solid.isValid():
    print('Solid invalid, fixing...')
    solid = solid.removeSplitter()

print('Loading top half...')
s_top = Part.read('support_top_half.step')

print('Enlarging solid for clearance...')
try:
    cutter = solid.makeOffsetShape(0.25, 0.01)
    print('Cutting...')
    rest_piece = s_top.cut(cutter)
    print(f'Cut volume: {rest_piece.Volume}')
    rest_piece.exportStep('top_rest_adapted.step')
    print('Success')
except Exception as e:
    print(f'Error: {e}')
