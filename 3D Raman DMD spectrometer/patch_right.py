with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Make the right side much slimmer! The optics max out at Y=12.7.
# We keep Y=18 at the entrance for the screws.
old_poly_br = '''    poly_br = Part.makePolygon([
        FreeCAD.Vector(0, 22, 20), FreeCAD.Vector(0, 24, 175), FreeCAD.Vector(0, 50, 175), FreeCAD.Vector(0, 50, 20), FreeCAD.Vector(0, 22, 20)
    ])'''
new_poly_br = '''    poly_br = Part.makePolygon([
        FreeCAD.Vector(0, 19, 50), FreeCAD.Vector(0, 18, 175), FreeCAD.Vector(0, 50, 175), FreeCAD.Vector(0, 50, 50), FreeCAD.Vector(0, 19, 50)
    ])'''
content = content.replace(old_poly_br, new_poly_br)

old_poly_tr = '''    poly_tr = Part.makePolygon([
        FreeCAD.Vector(0, 24, 175), FreeCAD.Vector(0, -10, 240), FreeCAD.Vector(0, 50, 240),
        FreeCAD.Vector(0, 50, 175), FreeCAD.Vector(0, 24, 175)
    ])'''
new_poly_tr = '''    poly_tr = Part.makePolygon([
        FreeCAD.Vector(0, 18, 175), FreeCAD.Vector(0, -15, 240), FreeCAD.Vector(0, 50, 240),
        FreeCAD.Vector(0, 50, 175), FreeCAD.Vector(0, 18, 175)
    ])'''
content = content.replace(old_poly_tr, new_poly_tr)

old_poly_base_right = '''    poly_base_right = Part.makePolygon([
        FreeCAD.Vector(0, 18, -10), FreeCAD.Vector(0, 50, -10), FreeCAD.Vector(0, 50, 50),
        FreeCAD.Vector(0, 18, 50), FreeCAD.Vector(0, 18, -10)
    ])'''
new_poly_base_right = '''    poly_base_right = Part.makePolygon([
        FreeCAD.Vector(0, 19, -10), FreeCAD.Vector(0, 50, -10), FreeCAD.Vector(0, 50, 50),
        FreeCAD.Vector(0, 19, 50), FreeCAD.Vector(0, 19, -10)
    ])'''
content = content.replace(old_poly_base_right, new_poly_base_right)


with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Trimmed right side!")
