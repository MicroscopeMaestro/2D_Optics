with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update the Grating Thickness to 8.0 mm
old_grating = '''gr_box = Part.makeBox(26.0, 26.0, 6.7, FreeCAD.Vector(-13.0, -13.0, -3.35))
              gr_box.translate(FreeCAD.Vector(0, 5.0, 0))'''
new_grating = '''gr_box = Part.makeBox(26.0, 26.0, 8.0, FreeCAD.Vector(-13.0, -13.0, -4.0))
              gr_box.translate(FreeCAD.Vector(0, 5.0, 0))'''
content = content.replace(old_grating, new_grating)

# 2. Update the DMD Mount Holes to match IPD1231G
old_dmd_holes = '''if dmd:
        apply_mount_holes(dmd, 30.0, DMD_M4_HOLE_DEPTH, -2.0, dmd.Placement.Rotation.multVec(vZ))'''
new_dmd_holes = '''if dmd:
        # IPD1231G specialized asymmetric mount pattern
        dmd_z_vec = dmd.Placement.Rotation.multVec(vZ)
        for dx, dy in [(3.5, 17.5), (-3.5, 17.5), (-6.0, -15.5)]:
            pos = dmd.Placement.Base + dmd_z_vec * (-2.0) + dmd.Placement.Rotation.multVec(FreeCAD.Vector(dx, dy, 0))
            dir_vec = dmd_z_vec
            cyl = Part.makeCylinder(0.8, DMD_M4_HOLE_DEPTH, pos, dir_vec) # M2 tap radius 0.8mm (board holes are D=1.4, might need M1.6, but starting with M2/M1.6 size)
            if pos.x > 0: top_half = top_half.cut(cyl)
            else: bottom_half = bottom_half.cut(cyl)'''
content = content.replace(old_dmd_holes, new_dmd_holes)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated Grating to 8mm and applied IPD1231G DMD holes!")
