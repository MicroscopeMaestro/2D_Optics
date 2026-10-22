with open('build_external_bed.py', 'r', encoding='utf-8') as f:
    content = f.read()

import re

# Set the bottom of the cartridge to -13.0 instead of -14.0
content = re.sub(r'FreeCAD\.Vector\(-14\.0', 'FreeCAD.Vector(-13.0', content)

with open('build_external_bed.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated cartridge to XMin = -13.0!")
