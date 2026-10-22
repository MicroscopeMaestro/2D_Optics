with open('build_external_bed.py', 'r', encoding='utf-8') as f:
    content = f.read()

import re

# We will set XMin to -13.0
content = re.sub(r'FreeCAD\.Vector\(-14\.0, -16\.0', 'FreeCAD.Vector(-13.0, -16.0', content)

with open('build_external_bed.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated external bed to XMin = -13.0!")
