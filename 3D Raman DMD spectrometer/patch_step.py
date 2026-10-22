with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_poly_tr = '''    poly_tr = Part.makePolygon([
        FreeCAD.Vector(0, 18, 175), FreeCAD.Vector(0, -15, 240), FreeCAD.Vector(0, 50, 240),
        FreeCAD.Vector(0, 50, 175), FreeCAD.Vector(0, 18, 175)
    ])'''
new_poly_tr = '''    poly_tr = Part.makePolygon([
        FreeCAD.Vector(0, 22, 175), FreeCAD.Vector(0, -15, 240), FreeCAD.Vector(0, 50, 240),
        FreeCAD.Vector(0, 50, 175), FreeCAD.Vector(0, 22, 175)
    ])'''
content = content.replace(old_poly_tr, new_poly_tr)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Smoothed right taper step!")
