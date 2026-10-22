with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# I will replace the sensor mounting hole logic
old_sensor_logic = '''if sensor:
        apply_mount_holes(sensor, 30.0, SENSOR_M4_HOLE_DEPTH, 3.0, sensor.Placement.Rotation.multVec(vZ), True)'''

new_sensor_logic = '''if sensor:
        # The PDAPC3 Adapter mounts via two holes at (0, 15) and (0, -15).
        # We place M4 tap holes (radius 1.65) in the spectrometer to match this.
        z_vec = sensor.Placement.Rotation.multVec(vZ)
        for dx, dy in [(0.0, 15.0), (0.0, -15.0)]:
            pos = sensor.Placement.Base + z_vec * 3.0 + sensor.Placement.Rotation.multVec(FreeCAD.Vector(dx, dy, 0))
            dir_vec = sensor.Placement.Rotation.multVec(vZ) * (-1.0)
            cyl = Part.makeCylinder(1.65, SENSOR_M4_HOLE_DEPTH, pos, dir_vec)
            if pos.x > 0: top_half = top_half.cut(cyl)
            else: bottom_half = bottom_half.cut(cyl)'''

content = content.replace(old_sensor_logic, new_sensor_logic)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated spectrometer sensor holes to match adapter!")
