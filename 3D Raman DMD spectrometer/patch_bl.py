with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_poly_dmd_bot = '''    poly_dmd_bot = Part.makePolygon([
        FreeCAD.Vector(0, -82, 95), FreeCAD.Vector(0, -25, 50), FreeCAD.Vector(0, -100, 50),
        FreeCAD.Vector(0, -100, 95), FreeCAD.Vector(0, -82, 95)
    ])'''
new_poly_dmd_bot = '''    poly_dmd_bot = Part.makePolygon([
        FreeCAD.Vector(0, -100, 95), FreeCAD.Vector(0, -19, 50), FreeCAD.Vector(0, -150, 50),
        FreeCAD.Vector(0, -150, 95), FreeCAD.Vector(0, -100, 95)
    ])'''
content = content.replace(old_poly_dmd_bot, new_poly_dmd_bot)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Trimmed bottom left side!")
