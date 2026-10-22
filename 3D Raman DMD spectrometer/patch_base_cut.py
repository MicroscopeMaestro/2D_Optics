import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add two new polygons to chop off the squared base at Z=0
old_tapers = '''    for taper in [taper_tl, taper_tr, taper_inner, taper_br, taper_dmd_bot, taper_sensor_tr]:
        top_half, bottom_half = top_half.cut(taper), bottom_half.cut(taper)'''

new_tapers = '''    # Extra cuts to remove the squared slabs at the very bottom base (Z=0)
    poly_base_left = Part.makePolygon([
        FreeCAD.Vector(0, -18, -10), FreeCAD.Vector(0, -50, -10), FreeCAD.Vector(0, -50, 50),
        FreeCAD.Vector(0, -18, 50), FreeCAD.Vector(0, -18, -10)
    ])
    taper_base_left = Part.Face(poly_base_left).extrude(FreeCAD.Vector(100, 0, 0)).translate(FreeCAD.Vector(-50, 0, 0))

    poly_base_right = Part.makePolygon([
        FreeCAD.Vector(0, 18, -10), FreeCAD.Vector(0, 50, -10), FreeCAD.Vector(0, 50, 50),
        FreeCAD.Vector(0, 18, 50), FreeCAD.Vector(0, 18, -10)
    ])
    taper_base_right = Part.Face(poly_base_right).extrude(FreeCAD.Vector(100, 0, 0)).translate(FreeCAD.Vector(-50, 0, 0))

    for taper in [taper_tl, taper_tr, taper_inner, taper_br, taper_dmd_bot, taper_sensor_tr, taper_base_left, taper_base_right]:
        top_half, bottom_half = top_half.cut(taper), bottom_half.cut(taper)'''

if old_tapers in content:
    content = content.replace(old_tapers, new_tapers)
    print("Replaced tapers successfully!")
else:
    print("Failed to replace tapers!")

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
