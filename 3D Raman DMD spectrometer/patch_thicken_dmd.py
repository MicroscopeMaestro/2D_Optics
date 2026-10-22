with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Thicken DMD housing
old_dmd_housing = '''    dmd_housing_top = Part.makeBox(CLAMSHELL_HALF_THICKNESS, 83.0, 140.0, FreeCAD.Vector(0, -100.0, 95.0))
    dmd_housing_bot = Part.makeBox(CLAMSHELL_HALF_THICKNESS, 83.0, 140.0, FreeCAD.Vector(-CLAMSHELL_HALF_THICKNESS, -100.0, 95.0))'''
new_dmd_housing = '''    dmd_housing_top = Part.makeBox(CLAMSHELL_HALF_THICKNESS, 93.0, 140.0, FreeCAD.Vector(0, -110.0, 95.0))
    dmd_housing_bot = Part.makeBox(CLAMSHELL_HALF_THICKNESS, 93.0, 140.0, FreeCAD.Vector(-CLAMSHELL_HALF_THICKNESS, -110.0, 95.0))'''
content = content.replace(old_dmd_housing, new_dmd_housing)

# 2. Adjust top-left taper to avoid shaving the new thickness
old_poly_tl = '''    poly_tl = Part.makePolygon([
        FreeCAD.Vector(0, -110, 120), FreeCAD.Vector(0, -55, 200), FreeCAD.Vector(0, -50, 240),
        FreeCAD.Vector(0, -150, 240), FreeCAD.Vector(0, -150, 120), FreeCAD.Vector(0, -110, 120)
    ])'''
new_poly_tl = '''    poly_tl = Part.makePolygon([
        FreeCAD.Vector(0, -120, 120), FreeCAD.Vector(0, -55, 200), FreeCAD.Vector(0, -50, 240),
        FreeCAD.Vector(0, -150, 240), FreeCAD.Vector(0, -150, 120), FreeCAD.Vector(0, -120, 120)
    ])'''
content = content.replace(old_poly_tl, new_poly_tl)

# 3. Adjust bottom-left taper
old_poly_dmd_bot = '''    poly_dmd_bot = Part.makePolygon([
        FreeCAD.Vector(0, -100, 95), FreeCAD.Vector(0, -22, 50), FreeCAD.Vector(0, -150, 50),
        FreeCAD.Vector(0, -150, 95), FreeCAD.Vector(0, -100, 95)
    ])'''
new_poly_dmd_bot = '''    poly_dmd_bot = Part.makePolygon([
        FreeCAD.Vector(0, -110, 95), FreeCAD.Vector(0, -22, 50), FreeCAD.Vector(0, -150, 50),
        FreeCAD.Vector(0, -150, 95), FreeCAD.Vector(0, -110, 95)
    ])'''
content = content.replace(old_poly_dmd_bot, new_poly_dmd_bot)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Thickened lateral DMD wall!")
