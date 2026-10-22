with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

import re

# Update DMD holes
new_dmd = '''if dmd:
        # IPD1231G specialized asymmetric mount pattern
        dmd_z_vec = dmd.Placement.Rotation.multVec(vZ)
        for dx, dy in [(3.5, 17.5), (-3.5, 17.5), (-6.0, -15.5)]:
            pos = dmd.Placement.Base + dmd_z_vec * (-2.0) + dmd.Placement.Rotation.multVec(FreeCAD.Vector(dx, dy, 0))
            dir_vec = dmd_z_vec
            cyl = Part.makeCylinder(0.8, DMD_M4_HOLE_DEPTH, pos, dir_vec)
            if pos.x > 0: top_half = top_half.cut(cyl)
            else: bottom_half = bottom_half.cut(cyl)'''

content = re.sub(r'if dmd:\s+apply_mount_holes\(dmd, 30\.0, DMD_M4_HOLE_DEPTH, -2\.0, dmd\.Placement\.Rotation\.multVec\(vZ\), False\)', new_dmd, content)

# Also ensure Grating is 8mm!
content = content.replace('gr_box = Part.makeBox(26.0, 26.0, 6.7, FreeCAD.Vector(-13.0, -13.0, -3.35))', 'gr_box = Part.makeBox(26.0, 26.0, 8.0, FreeCAD.Vector(-13.0, -13.0, -4.0))')

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated DMD and Grating!")
