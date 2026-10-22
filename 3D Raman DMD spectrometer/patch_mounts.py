import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_mount = '''    if sensor: apply_mount_holes(sensor, SENSOR_M4_HOLES_SPACING, SENSOR_M4_HOLE_DEPTH, 3.0, sensor.Placement.Rotation.multVec(vZ), True)
    if dmd: apply_mount_holes(dmd, DMD_M4_HOLES_SPACING, DMD_M4_HOLE_DEPTH, -2.0, dmd.Placement.Rotation.multVec(vZ))'''

new_mount = '''    if sensor: 
        apply_mount_holes(sensor, 20.0, SENSOR_M4_HOLE_DEPTH, 3.0, sensor.Placement.Rotation.multVec(vZ), True)
        apply_mount_holes(sensor, 30.0, SENSOR_M4_HOLE_DEPTH, 3.0, sensor.Placement.Rotation.multVec(vZ), True)
        apply_mount_holes(sensor, 40.0, SENSOR_M4_HOLE_DEPTH, 3.0, sensor.Placement.Rotation.multVec(vZ), True)
    if dmd: 
        apply_mount_holes(dmd, 20.0, DMD_M4_HOLE_DEPTH, -2.0, dmd.Placement.Rotation.multVec(vZ))
        apply_mount_holes(dmd, 30.0, DMD_M4_HOLE_DEPTH, -2.0, dmd.Placement.Rotation.multVec(vZ))
        apply_mount_holes(dmd, 40.0, DMD_M4_HOLE_DEPTH, -2.0, dmd.Placement.Rotation.multVec(vZ))
        apply_mount_holes(dmd, 50.0, DMD_M4_HOLE_DEPTH, -2.0, dmd.Placement.Rotation.multVec(vZ))
        # Add some extra rectangular ones for the DMD since it's a long PCB (65x140)
        z_vec = dmd.Placement.Rotation.multVec(FreeCAD.Vector(0,0,1))
        target_z = dmd.Placement.Rotation.multVec(FreeCAD.Vector(0,0,1))
        for dx in [-25.0, 25.0]:
            for dy in [-60.0, -45.0, 45.0, 60.0]:
                pos = dmd.Placement.Base + z_vec * (-2.0) + dmd.Placement.Rotation.multVec(FreeCAD.Vector(dx, dy, 0))
                cyl = Part.makeCylinder(THREAD_TAP_RADIUS if enable_threads else SCREW_TAP_RADIUS, DMD_M4_HOLE_DEPTH, pos, target_z)
                if enable_threads:
                    try:
                        th = Part.makeThread(THREAD_PITCH, THREAD_DEPTH, DMD_M4_HOLE_DEPTH, THREAD_TAP_RADIUS)
                        rot = FreeCAD.Rotation(FreeCAD.Vector(0,0,1), target_z)
                        th.transformShape(FreeCAD.Placement(pos, rot).Matrix)
                        cyl = cyl.fuse(th)
                    except: pass
                if pos.x > 0: top_half = top_half.cut(cyl)
                else: bottom_half = bottom_half.cut(cyl)
'''

content = content.replace(old_mount, new_mount)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Added more mount holes")
