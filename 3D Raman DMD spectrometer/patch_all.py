import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Remove entrance_tunnel
content = re.sub(r'entrance_tunnel = Part\.makeCylinder\(.*?\n\s+top_half, bottom_half = top_half\.cut\(entrance_tunnel\), bottom_half\.cut\(entrance_tunnel\)', '', content)

# 2. Update makeCone to 18mm clear aperture (9.0mm radius)
content = content.replace('cone = Part.makeCone(max(2.0, r_min)', 'cone = Part.makeCone(max(9.0, r_min)')
content = content.replace('cone = Part.makeCone(OPTICS_POCKET_RADIUS, max(2.0, r_max)', 'cone = Part.makeCone(OPTICS_POCKET_RADIUS, max(9.0, r_max)')

# 3. Add poly_grating_sensor to remove the square structure
new_poly = '''
    # New cut between Grating and Sensor
    poly_grating_sensor = Part.makePolygon([
        FreeCAD.Vector(0, -5, 190), FreeCAD.Vector(0, -25, 235), FreeCAD.Vector(0, 20, 235), FreeCAD.Vector(0, 20, 190), FreeCAD.Vector(0, -5, 190)
    ])
    taper_grating_sensor = Part.Face(poly_grating_sensor).extrude(FreeCAD.Vector(100, 0, 0)).translate(FreeCAD.Vector(-50, 0, 0))
    top_half, bottom_half = top_half.cut(taper_grating_sensor), bottom_half.cut(taper_grating_sensor)
'''

# Insert it after taper_tr
pattern = r'(taper_tr = Part\.Face.*?bottom_half\.cut\(taper_tr\))'
content = re.sub(pattern, r'\1\n' + new_poly, content)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Changes applied!")
