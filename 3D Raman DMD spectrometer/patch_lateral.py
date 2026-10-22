with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_sensor_holes = '''    if sensor:
        apply_mount_holes(sensor, 20.0, SENSOR_M4_HOLE_DEPTH, 3.0, sensor.Placement.Rotation.multVec(vZ), True)
        apply_mount_holes(sensor, 30.0, SENSOR_M4_HOLE_DEPTH, 3.0, sensor.Placement.Rotation.multVec(vZ), True)
        apply_mount_holes(sensor, 40.0, SENSOR_M4_HOLE_DEPTH, 3.0, sensor.Placement.Rotation.multVec(vZ), True)
    if dmd:
        apply_mount_holes(dmd, 20.0, DMD_M4_HOLE_DEPTH, -2.0, dmd.Placement.Rotation.multVec(vZ))
        apply_mount_holes(dmd, 30.0, DMD_M4_HOLE_DEPTH, -2.0, dmd.Placement.Rotation.multVec(vZ))
        apply_mount_holes(dmd, 40.0, DMD_M4_HOLE_DEPTH, -2.0, dmd.Placement.Rotation.multVec(vZ))
        apply_mount_holes(dmd, 50.0, DMD_M4_HOLE_DEPTH, -2.0, dmd.Placement.Rotation.multVec(vZ))'''

new_sensor_holes = '''    if sensor:
        apply_mount_holes(sensor, 30.0, SENSOR_M4_HOLE_DEPTH, 3.0, sensor.Placement.Rotation.multVec(vZ), True)
    if dmd:
        apply_mount_holes(dmd, 30.0, DMD_M4_HOLE_DEPTH, -2.0, dmd.Placement.Rotation.multVec(vZ))'''

if old_sensor_holes not in content:
    print("Could not find the sensor holes block!")
else:
    content = content.replace(old_sensor_holes, new_sensor_holes)
    with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Removed all extra lateral M4 holes, keeping only the 30x30mm grids!")
