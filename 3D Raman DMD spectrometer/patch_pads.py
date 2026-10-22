with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

import re

content = re.sub(r'top_pad = Part\.makeBox\(30\.0, 55\.0, 45\.0, FreeCAD\.Vector\(0, -63\.0, 205\.0\)\)', 
                 'top_pad = Part.makeBox(CLAMSHELL_HALF_THICKNESS, 55.0, 45.0, FreeCAD.Vector(0, -63.0, 205.0))', content)

content = re.sub(r'bot_pad = Part\.makeBox\(30\.0, 55\.0, 45\.0, FreeCAD\.Vector\(-30\.0, -63\.0, 205\.0\)\)', 
                 'bot_pad = Part.makeBox(CLAMSHELL_HALF_THICKNESS, 55.0, 45.0, FreeCAD.Vector(-CLAMSHELL_HALF_THICKNESS, -63.0, 205.0))', content)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Sensor pads made flush with casing!")
