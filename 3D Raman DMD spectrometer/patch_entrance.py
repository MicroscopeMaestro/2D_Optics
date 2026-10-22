with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add the entrance screws
old_screws = '''MANUAL_SCREW_POSITIONS = [
    (-33.0, 115.0),'''
new_screws = '''MANUAL_SCREW_POSITIONS = [
    ( 17.5, 20.0),   # ENTRANCE RIGHT
    (-17.5, 20.0),   # ENTRANCE LEFT
    (-33.0, 115.0),'''
content = content.replace(old_screws, new_screws)

# 2. Widen the base tapers to Y=22 to safely encase the new entrance screws
old_base_left = '''    poly_base_left = Part.makePolygon([
        FreeCAD.Vector(0, -19, -10), FreeCAD.Vector(0, -50, -10), FreeCAD.Vector(0, -50, 55),
        FreeCAD.Vector(0, -19, 55), FreeCAD.Vector(0, -19, -10)
    ])'''
new_base_left = '''    poly_base_left = Part.makePolygon([
        FreeCAD.Vector(0, -22, -10), FreeCAD.Vector(0, -50, -10), FreeCAD.Vector(0, -50, 55),
        FreeCAD.Vector(0, -22, 55), FreeCAD.Vector(0, -22, -10)
    ])'''
content = content.replace(old_base_left, new_base_left)

old_base_right = '''    poly_base_right = Part.makePolygon([
        FreeCAD.Vector(0, 19, -10), FreeCAD.Vector(0, 50, -10), FreeCAD.Vector(0, 50, 50),
        FreeCAD.Vector(0, 19, 50), FreeCAD.Vector(0, 19, -10)
    ])'''
new_base_right = '''    poly_base_right = Part.makePolygon([
        FreeCAD.Vector(0, 22, -10), FreeCAD.Vector(0, 50, -10), FreeCAD.Vector(0, 50, 50),
        FreeCAD.Vector(0, 22, 50), FreeCAD.Vector(0, 22, -10)
    ])'''
content = content.replace(old_base_right, new_base_right)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Added entrance screws and expanded entrance walls to 22mm!")
