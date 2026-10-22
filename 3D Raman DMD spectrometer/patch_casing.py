import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# poly_tr modification
old_tr = '''poly_tr = Part.makePolygon([
        FreeCAD.Vector(0, 20, 175), FreeCAD.Vector(0, -10, 240), FreeCAD.Vector(0, 50, 240),
        FreeCAD.Vector(0, 50, 175), FreeCAD.Vector(0, 20, 175)
    ])'''
new_tr = '''poly_tr = Part.makePolygon([
        FreeCAD.Vector(0, 24, 175), FreeCAD.Vector(0, -10, 240), FreeCAD.Vector(0, 50, 240),
        FreeCAD.Vector(0, 50, 175), FreeCAD.Vector(0, 24, 175)
    ])'''
content = content.replace(old_tr, new_tr)

# poly_br modification
old_br = '''poly_br = Part.makePolygon([
        FreeCAD.Vector(0, 18, 20), FreeCAD.Vector(0, 20, 175), FreeCAD.Vector(0, 50, 175), FreeCAD.Vector(0, 50, 20), FreeCAD.Vector(0, 18, 20)
    ])'''
new_br = '''poly_br = Part.makePolygon([
        FreeCAD.Vector(0, 22, 20), FreeCAD.Vector(0, 24, 175), FreeCAD.Vector(0, 50, 175), FreeCAD.Vector(0, 50, 20), FreeCAD.Vector(0, 22, 20)
    ])'''
content = content.replace(old_br, new_br)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Expanded right-side casing!")
