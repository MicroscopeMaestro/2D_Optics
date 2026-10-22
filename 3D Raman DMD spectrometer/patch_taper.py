import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

new_poly = '''
    # New cut between Grating and Sensor
    poly_grating_sensor = Part.makePolygon([
        FreeCAD.Vector(0, -5, 190), FreeCAD.Vector(0, -25, 235), FreeCAD.Vector(0, 20, 235), FreeCAD.Vector(0, 20, 190), FreeCAD.Vector(0, -5, 190)
    ])
    taper_grating_sensor = Part.Face(poly_grating_sensor).extrude(FreeCAD.Vector(100, 0, 0)).translate(FreeCAD.Vector(-50, 0, 0))
'''

content = content.replace('taper_tr = Part.Face(poly_tr).extrude(FreeCAD.Vector(100, 0, 0)).translate(FreeCAD.Vector(-50, 0, 0))', 'taper_tr = Part.Face(poly_tr).extrude(FreeCAD.Vector(100, 0, 0)).translate(FreeCAD.Vector(-50, 0, 0))\n' + new_poly)

content = content.replace('for taper in [taper_tl, taper_tr, taper_inner, taper_br, taper_dmd_bot, taper_sensor_tr, taper_base_left, taper_base_right]:', 'for taper in [taper_tl, taper_tr, taper_inner, taper_br, taper_dmd_bot, taper_sensor_tr, taper_base_left, taper_base_right, taper_grating_sensor]:')

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Changes applied!")
