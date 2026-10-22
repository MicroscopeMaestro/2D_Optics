with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add back 4 very robust middle screws
old_screws = '''MANUAL_SCREW_POSITIONS = [
    ( 17.5, 20.0),   # ENTRANCE RIGHT
    (-17.5, 20.0),   # ENTRANCE LEFT
    (-15.0, 227.0),  # EXIT RIGHT
    (-46.0, 227.0)   # EXIT LEFT
]'''
new_screws = '''MANUAL_SCREW_POSITIONS = [
    ( 17.5, 20.0),   # ENTRANCE RIGHT
    (-17.5, 20.0),   # ENTRANCE LEFT
    ( 17.5, 100.0),  # RIGHT MID 1
    (-70.0, 115.0),  # LEFT MID 1
    ( 17.5, 150.0),  # RIGHT MID 2
    (-50.0, 170.0),  # LEFT MID 2
    (-15.0, 227.0),  # EXIT RIGHT
    (-46.0, 227.0)   # EXIT LEFT
]'''
content = content.replace(old_screws, new_screws)

# 2. Widen the right wall slightly to constant 22mm to safely house the right middle screws
old_poly_br = '''    poly_br = Part.makePolygon([
        FreeCAD.Vector(0, 22, 50), FreeCAD.Vector(0, 18, 175), FreeCAD.Vector(0, 50, 175), FreeCAD.Vector(0, 50, 50), FreeCAD.Vector(0, 22, 50)
    ])'''
new_poly_br = '''    poly_br = Part.makePolygon([
        FreeCAD.Vector(0, 22, 50), FreeCAD.Vector(0, 22, 175), FreeCAD.Vector(0, 50, 175), FreeCAD.Vector(0, 50, 50), FreeCAD.Vector(0, 22, 50)
    ])'''
content = content.replace(old_poly_br, new_poly_br)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Restored middle screws securely!")
