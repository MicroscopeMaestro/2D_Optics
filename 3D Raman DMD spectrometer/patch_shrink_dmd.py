with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Shrink DMD housing to -86
old_dmd = '''    dmd_housing_top = Part.makeBox(CLAMSHELL_HALF_THICKNESS, 93.0, 140.0, FreeCAD.Vector(0, -110.0, 95.0))
    dmd_housing_bot = Part.makeBox(CLAMSHELL_HALF_THICKNESS, 93.0, 140.0, FreeCAD.Vector(-CLAMSHELL_HALF_THICKNESS, -110.0, 95.0))'''
new_dmd = '''    dmd_housing_top = Part.makeBox(CLAMSHELL_HALF_THICKNESS, 69.0, 140.0, FreeCAD.Vector(0, -86.0, 95.0))
    dmd_housing_bot = Part.makeBox(CLAMSHELL_HALF_THICKNESS, 69.0, 140.0, FreeCAD.Vector(-CLAMSHELL_HALF_THICKNESS, -86.0, 95.0))'''
content = content.replace(old_dmd, new_dmd)

# 2. Adjust poly_tl
old_poly_tl = '''    poly_tl = Part.makePolygon([
        FreeCAD.Vector(0, -120, 125), FreeCAD.Vector(0, -55, 200), FreeCAD.Vector(0, -50, 240),
        FreeCAD.Vector(0, -150, 240), FreeCAD.Vector(0, -150, 120), FreeCAD.Vector(0, -115, 120), FreeCAD.Vector(0, -120, 125)
    ])'''
new_poly_tl = '''    poly_tl = Part.makePolygon([
        FreeCAD.Vector(0, -86, 120), FreeCAD.Vector(0, -55, 200), FreeCAD.Vector(0, -50, 240),
        FreeCAD.Vector(0, -150, 240), FreeCAD.Vector(0, -150, 120), FreeCAD.Vector(0, -86, 120)
    ])'''
content = content.replace(old_poly_tl, new_poly_tl)

# 3. Adjust poly_dmd_bot
old_poly_dmd_bot = '''    poly_dmd_bot = Part.makePolygon([
        FreeCAD.Vector(0, -110, 90), FreeCAD.Vector(0, -27, 50), FreeCAD.Vector(0, -150, 50),
        FreeCAD.Vector(0, -150, 95), FreeCAD.Vector(0, -105, 95), FreeCAD.Vector(0, -110, 90)
    ])'''
new_poly_dmd_bot = '''    poly_dmd_bot = Part.makePolygon([
        FreeCAD.Vector(0, -86, 95), FreeCAD.Vector(0, -22, 50), FreeCAD.Vector(0, -150, 50),
        FreeCAD.Vector(0, -150, 95), FreeCAD.Vector(0, -86, 95)
    ])'''
content = content.replace(old_poly_dmd_bot, new_poly_dmd_bot)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Shrunk DMD housing back to -86.0")
