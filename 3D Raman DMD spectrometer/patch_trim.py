import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_poly_tl = '''    poly_tl = Part.makePolygon([
        FreeCAD.Vector(0, -125, 140), FreeCAD.Vector(0, -85, 210), FreeCAD.Vector(0, -35, 245),
        FreeCAD.Vector(0, -35, 300), FreeCAD.Vector(0, -125, 300), FreeCAD.Vector(0, -125, 140)
    ])
    taper_tl = Part.Face(poly_tl).extrude(FreeCAD.Vector(100, 0, 0))
    taper_tl.translate(FreeCAD.Vector(-50, 0, 0))
    top_half, bottom_half = top_half.cut(taper_tl), bottom_half.cut(taper_tl)'''

new_poly_tl = '''    poly_tl = Part.makePolygon([
        FreeCAD.Vector(0, -125, 120),  # Start way out left
        FreeCAD.Vector(0, -82, 120),   # Come in to DMD edge
        FreeCAD.Vector(0, -60, 205),   # Slant to Relay lens
        FreeCAD.Vector(0, -45, 235),   # Slant to Sensor
        FreeCAD.Vector(0, -45, 300),   # Go up
        FreeCAD.Vector(0, -125, 300),  # Go left
        FreeCAD.Vector(0, -125, 120)   # Close
    ])
    taper_tl = Part.Face(poly_tl).extrude(FreeCAD.Vector(100, 0, 0))
    taper_tl.translate(FreeCAD.Vector(-50, 0, 0))
    top_half, bottom_half = top_half.cut(taper_tl), bottom_half.cut(taper_tl)
    
    # Aggressive Top-Right trim to remove empty plastic above Grating
    poly_tr = Part.makePolygon([
        FreeCAD.Vector(0, 50, 185),    # Start way out right, above grating
        FreeCAD.Vector(0, -10, 235),   # Slant to Sensor right edge
        FreeCAD.Vector(0, -10, 300),   # Go up
        FreeCAD.Vector(0, 50, 300),    # Go right
        FreeCAD.Vector(0, 50, 185)     # Close
    ])
    taper_tr = Part.Face(poly_tr).extrude(FreeCAD.Vector(100, 0, 0))
    taper_tr.translate(FreeCAD.Vector(-50, 0, 0))
    top_half, bottom_half = top_half.cut(taper_tr), bottom_half.cut(taper_tr)'''

content = content.replace(old_poly_tl, new_poly_tl)

with open('build_spectrometer_trim.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Trim script created")
