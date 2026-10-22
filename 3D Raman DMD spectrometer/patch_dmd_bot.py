import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_poly = '''    poly_dmd_bot = Part.makePolygon([
        FreeCAD.Vector(0, -90, 95), FreeCAD.Vector(0, -40, 50), FreeCAD.Vector(0, -100, 50),
        FreeCAD.Vector(0, -100, 95), FreeCAD.Vector(0, -90, 95)
    ])'''

new_poly = '''    poly_dmd_bot = Part.makePolygon([
        FreeCAD.Vector(0, -82, 95), FreeCAD.Vector(0, -25, 50), FreeCAD.Vector(0, -100, 50),
        FreeCAD.Vector(0, -100, 95), FreeCAD.Vector(0, -82, 95)
    ])'''

if old_poly in content:
    content = content.replace(old_poly, new_poly)
    print("Replaced poly_dmd_bot successfully!")
else:
    print("Failed to replace poly_dmd_bot!")

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
