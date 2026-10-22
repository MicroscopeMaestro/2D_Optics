import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_tapers = '''    for taper in [taper_tl, taper_tr, taper_inner, taper_br]:
        top_half, bottom_half = top_half.cut(taper), bottom_half.cut(taper)'''

new_tapers = '''    # Extra smoothing cuts for "squared pronuntiations" (DMD bottom pad and Sensor top-right pad)
    poly_dmd_bot = Part.makePolygon([
        FreeCAD.Vector(0, -90, 95), FreeCAD.Vector(0, -40, 50), FreeCAD.Vector(0, -100, 50),
        FreeCAD.Vector(0, -100, 95), FreeCAD.Vector(0, -90, 95)
    ])
    taper_dmd_bot = Part.Face(poly_dmd_bot).extrude(FreeCAD.Vector(100, 0, 0)).translate(FreeCAD.Vector(-50, 0, 0))

    poly_sensor_tr = Part.makePolygon([
        FreeCAD.Vector(0, -15, 233), FreeCAD.Vector(0, 0, 250), FreeCAD.Vector(0, 50, 250),
        FreeCAD.Vector(0, 50, 233), FreeCAD.Vector(0, -15, 233)
    ])
    taper_sensor_tr = Part.Face(poly_sensor_tr).extrude(FreeCAD.Vector(100, 0, 0)).translate(FreeCAD.Vector(-50, 0, 0))

    for taper in [taper_tl, taper_tr, taper_inner, taper_br, taper_dmd_bot, taper_sensor_tr]:
        top_half, bottom_half = top_half.cut(taper), bottom_half.cut(taper)'''

if old_tapers in content:
    content = content.replace(old_tapers, new_tapers)
    print("Replaced tapers successfully!")
else:
    print("Failed to replace tapers!")

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
