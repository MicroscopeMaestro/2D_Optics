with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix poly_base_left overlap bug
old_poly_base_left = '''    poly_base_left = Part.makePolygon([
        FreeCAD.Vector(0, -22, -10), FreeCAD.Vector(0, -50, -10), FreeCAD.Vector(0, -50, 55),
        FreeCAD.Vector(0, -22, 55), FreeCAD.Vector(0, -22, -10)
    ])'''
new_poly_base_left = '''    poly_base_left = Part.makePolygon([
        FreeCAD.Vector(0, -22, -10), FreeCAD.Vector(0, -50, -10), FreeCAD.Vector(0, -50, 50),
        FreeCAD.Vector(0, -22, 50), FreeCAD.Vector(0, -22, -10)
    ])'''
content = content.replace(old_poly_base_left, new_poly_base_left)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Fixed poly_base_left overlap bug!")
