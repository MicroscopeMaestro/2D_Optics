with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_poly_tl = '''    poly_tl = Part.makePolygon([
        FreeCAD.Vector(0, -110, 120), FreeCAD.Vector(0, -80, 200), FreeCAD.Vector(0, -58.5, 240),
        FreeCAD.Vector(0, -150, 240), FreeCAD.Vector(0, -150, 120), FreeCAD.Vector(0, -110, 120)
    ])'''
new_poly_tl = '''    poly_tl = Part.makePolygon([
        FreeCAD.Vector(0, -110, 120), FreeCAD.Vector(0, -55, 200), FreeCAD.Vector(0, -50, 240),
        FreeCAD.Vector(0, -150, 240), FreeCAD.Vector(0, -150, 120), FreeCAD.Vector(0, -110, 120)
    ])'''
content = content.replace(old_poly_tl, new_poly_tl)

# Also let's push the front left corner in!
old_poly_base_left = '''    poly_base_left = Part.makePolygon([
        FreeCAD.Vector(0, -18, -10), FreeCAD.Vector(0, -50, -10), FreeCAD.Vector(0, -50, 55),
        FreeCAD.Vector(0, -18, 55), FreeCAD.Vector(0, -18, -10)
    ])'''
new_poly_base_left = '''    poly_base_left = Part.makePolygon([
        FreeCAD.Vector(0, -19, -10), FreeCAD.Vector(0, -50, -10), FreeCAD.Vector(0, -50, 55),
        FreeCAD.Vector(0, -19, 55), FreeCAD.Vector(0, -19, -10)
    ])'''
content = content.replace(old_poly_base_left, new_poly_base_left)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Trimmed left side!")
