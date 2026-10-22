import re

with open('build_spectrometer_trim.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Change top_pad and bot_pad size
old_pads = '''        top_pad = Part.makeBox(CLAMSHELL_HALF_THICKNESS, 85.0, 15.0, FreeCAD.Vector(0, -100.0, 233.0))
        bot_pad = Part.makeBox(CLAMSHELL_HALF_THICKNESS, 85.0, 15.0, FreeCAD.Vector(-CLAMSHELL_HALF_THICKNESS, -100.0, 233.0))'''

new_pads = '''        # Optimized sensor pad to reduce excess material footprint
        top_pad = Part.makeBox(CLAMSHELL_HALF_THICKNESS, 45.0, 15.0, FreeCAD.Vector(0, -60.0, 233.0))
        bot_pad = Part.makeBox(CLAMSHELL_HALF_THICKNESS, 45.0, 15.0, FreeCAD.Vector(-CLAMSHELL_HALF_THICKNESS, -60.0, 233.0))'''

content = content.replace(old_pads, new_pads)

with open('build_spectrometer_trim.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Sensor pad optimized")
