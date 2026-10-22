# -*- coding: utf-8 -*-
with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# We need to find the sensor and DMD hole logic and revert it to the original apply_mount_holes.
import re

old_sensor_logic = '''if sensor:
        # Direct mount for PDAPC3
        z_vec = sensor.Placement.Rotation.multVec(vZ)
        for dx, dy in [(-15.0, -15.0), (-15.0, 15.0), (15.0, -15.0), (15.0, 15.0)]:
            pos = sensor.Placement.Base + z_vec * 3.0 + sensor.Placement.Rotation.multVec(FreeCAD.Vector(dx, dy, 0))
            dir_vec = sensor.Placement.Rotation.multVec(vZ) * (-1.0)
            cyl = Part.makeCylinder(1.65, SENSOR_M4_HOLE_DEPTH, pos, dir_vec)
            top_half = top_half.cut(cyl)
            bottom_half = bottom_half.cut(cyl)'''

new_sensor_logic = '''if sensor:
        apply_mount_holes(sensor, 30.0, SENSOR_M4_HOLE_DEPTH, 3.0, sensor.Placement.Rotation.multVec(vZ), True)'''

content = content.replace(old_sensor_logic, new_sensor_logic)

old_dmd_logic = '''if dmd:
        # IPD1231G specialized asymmetric mount pattern
        dmd_z_vec = dmd.Placement.Rotation.multVec(vZ)
        for dx, dy in [(3.5, 17.5), (-3.5, 17.5), (-6.0, -15.5)]:
            pos = dmd.Placement.Base + dmd_z_vec * (-2.0) + dmd.Placement.Rotation.multVec(FreeCAD.Vector(dx, dy, 0))
            dir_vec = dmd_z_vec
            cyl = Part.makeCylinder(0.8, DMD_M4_HOLE_DEPTH, pos, dir_vec)
            # The holes lie EXACTLY on the X=0 split line!
            top_half = top_half.cut(cyl)
            bottom_half = bottom_half.cut(cyl)'''

new_dmd_logic = '''if dmd:
        apply_mount_holes(dmd, 30.0, DMD_M4_HOLE_DEPTH, -2.0, dmd.Placement.Rotation.multVec(vZ))'''

content = content.replace(old_dmd_logic, new_dmd_logic)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Reverted both to original apply_mount_holes logic!")
