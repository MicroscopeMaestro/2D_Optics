import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the old poly_tl and poly_tr logic with the new aggressive 4-corner logic
old_taper = '''    poly_tl = Part.makePolygon([
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

new_taper = '''    # MASSIVE 4-corner trim to remove all empty plastic and drastically reduce print time!
    # 1. Top-Left (Above DMD)
    poly_tl = Part.makePolygon([
        FreeCAD.Vector(0, -90, 120),
        FreeCAD.Vector(0, -60, 200),
        FreeCAD.Vector(0, -55, 240),
        FreeCAD.Vector(0, -150, 240),
        FreeCAD.Vector(0, -150, 120),
        FreeCAD.Vector(0, -90, 120)
    ])
    taper_tl = Part.Face(poly_tl).extrude(FreeCAD.Vector(100, 0, 0)).translate(FreeCAD.Vector(-50, 0, 0))
    
    # 2. Top-Right (Above Grating)
    poly_tr = Part.makePolygon([
        FreeCAD.Vector(0, 20, 175),
        FreeCAD.Vector(0, -10, 240),
        FreeCAD.Vector(0, 50, 240),
        FreeCAD.Vector(0, 50, 175),
        FreeCAD.Vector(0, 20, 175)
    ])
    taper_tr = Part.Face(poly_tr).extrude(FreeCAD.Vector(100, 0, 0)).translate(FreeCAD.Vector(-50, 0, 0))
    
    # 3. Inner Triangle (Beside DMD)
    poly_inner = Part.makePolygon([
        FreeCAD.Vector(0, -18, 50),
        FreeCAD.Vector(0, -50, 95),
        FreeCAD.Vector(0, -35, 120),
        FreeCAD.Vector(0, -18, 120),
        FreeCAD.Vector(0, -18, 50)
    ])
    taper_inner = Part.Face(poly_inner).extrude(FreeCAD.Vector(100, 0, 0)).translate(FreeCAD.Vector(-50, 0, 0))
    
    # 4. Bottom-Right (Entrance edge)
    poly_br = Part.makePolygon([
        FreeCAD.Vector(0, 18, 20),
        FreeCAD.Vector(0, 18, 160),
        FreeCAD.Vector(0, 50, 160),
        FreeCAD.Vector(0, 50, 20),
        FreeCAD.Vector(0, 18, 20)
    ])
    taper_br = Part.Face(poly_br).extrude(FreeCAD.Vector(100, 0, 0)).translate(FreeCAD.Vector(-50, 0, 0))
    
    # Apply all cuts!
    for taper in [taper_tl, taper_tr, taper_inner, taper_br]:
        top_half = top_half.cut(taper)
        bottom_half = bottom_half.cut(taper)'''

if old_taper in content:
    content = content.replace(old_taper, new_taper)
    print("Replaced successfully!")
else:
    print("Failed to replace!")

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
