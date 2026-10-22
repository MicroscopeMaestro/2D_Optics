with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

import re

old_cradle_logic = '''    if USE_EXTERNAL_ASPHERIC_MOUNTS and "acl25416u" in obj.Label.lower():
        z_min = bb.ZMin - 0.1
        z_max = bb.ZMax + 1.0
        p_z_min = z_min - sliding_tolerance - 0.2
        p_z_max = z_max + sliding_tolerance + 0.2
        p_thickness = p_z_max - p_z_min
        p_width = 35.4
        p_height = 33.4
        box = Part.makeBox(p_width, p_height, p_thickness, FreeCAD.Vector(-13.7, -16.7, p_z_min))'''

new_cradle_logic = '''    if USE_EXTERNAL_ASPHERIC_MOUNTS and "acl25416u" in obj.Label.lower():
        # Standardized fixed pocket dimensions guaranteeing 100% identical entrance and exit gaps
        p_z_min = -0.30 - sliding_tolerance
        p_thickness = 11.20 + 2 * sliding_tolerance
        p_width = 35.4
        p_height = 33.4
        box = Part.makeBox(p_width, p_height, p_thickness, FreeCAD.Vector(-13.7, -16.7, p_z_min))'''

if old_cradle_logic in content:
    content = content.replace(old_cradle_logic, new_cradle_logic)
    with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Successfully updated make_aspheric_cradle with fixed standardized dimensions!")
else:
    print("Pattern not found, checking...")
