import sys
with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_fastener_logic = '''    for (y, z) in screws:
        top_half = top_half.cut(Part.makeCylinder(SCREW_CLEARANCE_RADIUS, SCREW_HOLE_LENGTH + 5.0, FreeCAD.Vector(0, y, z), FreeCAD.Vector(1, 0, 0)))
        bottom_half = bottom_half.cut(Part.makeCylinder(SCREW_TAP_RADIUS, SCREW_HOLE_LENGTH + 5.0, FreeCAD.Vector(-CLAMSHELL_HALF_THICKNESS - RIB_HEIGHT - 1.0, y, z), FreeCAD.Vector(1, 0, 0)))'''

new_fastener_logic = '''    for (y, z) in screws:
        top_half = top_half.cut(Part.makeCylinder(SCREW_CLEARANCE_RADIUS, SCREW_HOLE_LENGTH + 5.0, FreeCAD.Vector(0, y, z), FreeCAD.Vector(1, 0, 0)))
        tap_rad = 2.8 if z < 25.0 else SCREW_TAP_RADIUS
        bottom_half = bottom_half.cut(Part.makeCylinder(tap_rad, SCREW_HOLE_LENGTH + 5.0, FreeCAD.Vector(-CLAMSHELL_HALF_THICKNESS - RIB_HEIGHT - 1.0, y, z), FreeCAD.Vector(1, 0, 0)))'''

if old_fastener_logic not in content:
    print("Could not find fastener logic!")
    sys.exit(1)

content = content.replace(old_fastener_logic, new_fastener_logic)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Successfully patched fastener logic!")
