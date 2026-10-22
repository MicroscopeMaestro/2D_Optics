with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

import re

# We will set the pocket XMin to -13.7
content = re.sub(r'FreeCAD\.Vector\(-14\.7, -16\.7', 'FreeCAD.Vector(-13.7, -16.7', content)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated casing pocket to XMin = -13.7!")
