with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Enlarge the sensor pad (increase X width and Y width)
# old: Part.makeBox(CLAMSHELL_HALF_THICKNESS, 45.0, 15.0, FreeCAD.Vector(0, -60.0, 233.0))
# new: Part.makeBox(30.0, 50.0, 25.0, FreeCAD.Vector(0, -60.0, 225.0))
old_top_pad = 'top_pad = Part.makeBox(CLAMSHELL_HALF_THICKNESS, 45.0, 15.0, FreeCAD.Vector(0, -60.0, 233.0))'
new_top_pad = 'top_pad = Part.makeBox(30.0, 55.0, 25.0, FreeCAD.Vector(0, -63.0, 225.0))'
content = content.replace(old_top_pad, new_top_pad)

old_bot_pad = 'bot_pad = Part.makeBox(CLAMSHELL_HALF_THICKNESS, 45.0, 15.0, FreeCAD.Vector(-CLAMSHELL_HALF_THICKNESS, -60.0, 233.0))'
new_bot_pad = 'bot_pad = Part.makeBox(30.0, 55.0, 25.0, FreeCAD.Vector(-30.0, -63.0, 225.0))'
content = content.replace(old_bot_pad, new_bot_pad)

# 2. Restore the 4-hole M4 pattern!
old_sensor_holes = '''if sensor:
        # The PDAPC3 Adapter mounts via two holes at (0, 15) and (0, -15).
        # We place M4 tap holes (radius 1.65) in the spectrometer to match this.
        z_vec = sensor.Placement.Rotation.multVec(vZ)
        for dx, dy in [(0.0, 15.0), (0.0, -15.0)]:
            pos = sensor.Placement.Base + z_vec * 3.0 + sensor.Placement.Rotation.multVec(FreeCAD.Vector(dx, dy, 0))
            dir_vec = sensor.Placement.Rotation.multVec(vZ) * (-1.0)
            cyl = Part.makeCylinder(1.65, SENSOR_M4_HOLE_DEPTH, pos, dir_vec)
            # The holes lie EXACTLY on the X=0 split line!
            top_half = top_half.cut(cyl)
            bottom_half = bottom_half.cut(cyl)'''

new_sensor_holes = '''if sensor:
        # User wants the PDAPC3 mounted directly to the spectrometer structure!
        # We restore the 4-hole (±15, ±15) M4 pilot hole pattern.
        z_vec = sensor.Placement.Rotation.multVec(vZ)
        for dx, dy in [(-15.0, -15.0), (-15.0, 15.0), (15.0, -15.0), (15.0, 15.0)]:
            pos = sensor.Placement.Base + z_vec * 3.0 + sensor.Placement.Rotation.multVec(FreeCAD.Vector(dx, dy, 0))
            dir_vec = sensor.Placement.Rotation.multVec(vZ) * (-1.0)
            cyl = Part.makeCylinder(1.65, SENSOR_M4_HOLE_DEPTH, pos, dir_vec)
            top_half = top_half.cut(cyl)
            bottom_half = bottom_half.cut(cyl)'''

content = content.replace(old_sensor_holes, new_sensor_holes)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Enlarged the back structure and restored 4-hole mounting pattern!")
