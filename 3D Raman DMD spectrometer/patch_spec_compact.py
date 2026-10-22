with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

import re

# Update z_min and z_max in make_aspheric_cradle
content = re.sub(r'z_min = bb\.ZMin - .*', 'z_min = bb.ZMin - 0.5', content)
content = re.sub(r'z_max = bb\.ZMax \+ .*', 'z_max = bb.ZMax + 0.5', content)

# Update the pocket dimensions
content = re.sub(r'p_width = .*', 'p_width = 35.4', content)
content = re.sub(r'p_height = .*', 'p_height = 31.4', content)

# Update box placement
content = re.sub(r'FreeCAD\.Vector\(-14\.7, -16\.7, p_z_min\)', 'FreeCAD.Vector(-14.7, -15.7, p_z_min)', content)

# Update dynamic set screws in make_aspheric_cradle
content = re.sub(r'FreeCAD\.Vector\(4\.0, 26\.0, 4\.64\)', 'FreeCAD.Vector(4.0, 25.0, 5.5)', content)
content = re.sub(r'FreeCAD\.Vector\(4\.0, -26\.0, 4\.64\)', 'FreeCAD.Vector(4.0, -25.0, 5.5)', content)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated build_spectrometer.py with ultra-compact pocket dimensions!")
