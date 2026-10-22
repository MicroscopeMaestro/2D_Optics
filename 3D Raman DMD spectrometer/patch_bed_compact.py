with open('build_external_bed.py', 'r', encoding='utf-8') as f:
    content = f.read()

import re

# 1. Update z_min and z_max for compact thickness
content = re.sub(r'z_min = bb\.ZMin - .*', 'z_min = bb.ZMin - 0.5', content)
content = re.sub(r'z_max = bb\.ZMax \+ .*', 'z_max = bb.ZMax + 0.5', content)

# 2. Update width and height
content = re.sub(r'width = 36\.0', 'width = 34.0', content)
content = re.sub(r'height = 32\.0', 'height = 30.0', content)

# 3. Update box position
content = re.sub(r'Part\.makeBox\(width, height, thickness, FreeCAD\.Vector\(-14\.0, -16\.0, z_min\)\)', 
                 'Part.makeBox(width, height, thickness, FreeCAD.Vector(-14.0, -15.0, z_min))', content)

# 4. Update pts array (ensure full barrel diameter at the back so lens can be inserted)
content = re.sub(r'pts = \[FreeCAD\.Vector\(0, 0, z_min\), FreeCAD\.Vector\([^)]*\)\]',
                 'pts = [FreeCAD.Vector(0, 0, z_min), FreeCAD.Vector(p_radius, 0, z_min)]', content)

# 5. Update top set screw position (move to X=20.0, Z=bb.ZMax-2.5)
content = re.sub(r'FreeCAD\.Vector\(22\.0, 0, bb\.ZMax - 1\.5\)', 'FreeCAD.Vector(20.0, 0, bb.ZMax - 2.5)', content)
content = re.sub(r'FreeCAD\.Vector\(22\.0 - cb_depth, 0, bb\.ZMax - 1\.5\)', 'FreeCAD.Vector(20.0 - cb_depth, 0, bb.ZMax - 2.5)', content)

# 6. Update channels for compact size
content = re.sub(r'channel_z_min = 2\.0', 'channel_z_min = 1.0', content)
content = re.sub(r'channel_z_max = 16\.0', 'channel_z_max = bb.ZMax - 1.0', content)
content = re.sub(r'channel_width = 5\.0', 'channel_width = 3.0', content)
content = re.sub(r'16\.0 - channel_depth', '15.0 - channel_depth', content)
content = re.sub(r'-16\.0 - 0\.1', '-15.0 - 0.1', content)
# Also fix Z placement of channels
content = re.sub(r'z_min \+ 1\.0\)\)', 'z_min + 1.0))', content)

with open('build_external_bed.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated build_external_bed.py to be ultra-compact!")
