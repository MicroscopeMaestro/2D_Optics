import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_poly = '''    poly_br = Part.makePolygon([
        FreeCAD.Vector(0, 18, 20), FreeCAD.Vector(0, 18, 160), FreeCAD.Vector(0, 50, 160), FreeCAD.Vector(0, 50, 20), FreeCAD.Vector(0, 18, 20)
    ])'''

new_poly = '''    poly_br = Part.makePolygon([
        FreeCAD.Vector(0, 18, 20), FreeCAD.Vector(0, 20, 175), FreeCAD.Vector(0, 50, 175), FreeCAD.Vector(0, 50, 20), FreeCAD.Vector(0, 18, 20)
    ])'''

if old_poly in content:
    content = content.replace(old_poly, new_poly)
    print("Replaced poly_br successfully!")
else:
    print("Failed to replace poly_br!")

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
