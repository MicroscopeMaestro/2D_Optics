with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Remove the old hardcoded set screws block
old_screw_block = '''    # Dual horizontal M4 set screw holes to lock external lens cartridge
    if USE_EXTERNAL_ASPHERIC_MOUNTS:
        right_screw_hole = Part.makeCylinder(cartridge_locking_screw_radius, 15.0,
                                             FreeCAD.Vector(cartridge_locking_screw_x, 26.0, cartridge_locking_screw_z),
                                             FreeCAD.Vector(0, -1, 0))
        left_screw_hole = Part.makeCylinder(cartridge_locking_screw_radius, 15.0,
                                            FreeCAD.Vector(cartridge_locking_screw_x, -26.0, cartridge_locking_screw_z),
                                            FreeCAD.Vector(0, 1, 0))
        top_half = top_half.cut(right_screw_hole).cut(left_screw_hole)'''

if old_screw_block in content:
    content = content.replace(old_screw_block, '')
else:
    print("Warning: old screw block not found exactly as written.")

# 2. Add the dynamic screw generation inside the make_aspheric_cradle function so it cuts them for every lens!
# Actually, make_aspheric_cradle only returns the pocket box. I can just fuse the screw holes to the pocket box!
# If I fuse the screw holes to the pocket box, then when the pocket box is cut from the casing, it will also cut the screw holes!
# But wait, screw holes are cylinders. If I fuse them to the box, they will just be empty space in the casing.
# This is EXACTLY what we want! We are cutting the pocket AND the screw holes from the solid casing.

injection = '''
        if axial_dz != 0.0:
            box.translate(FreeCAD.Vector(0, 0, axial_dz))
        
        # Inject set screw cylinders locally before transform
        r_screw = Part.makeCylinder(1.65, 15.0, FreeCAD.Vector(4.0, 26.0, 4.64), FreeCAD.Vector(0, -1, 0))
        l_screw = Part.makeCylinder(1.65, 15.0, FreeCAD.Vector(4.0, -26.0, 4.64), FreeCAD.Vector(0, 1, 0))
        box = box.fuse(r_screw).fuse(l_screw)
        
        box.transformShape(obj.Placement.Matrix)
'''

content = content.replace('''
        if axial_dz != 0.0:
            box.translate(FreeCAD.Vector(0, 0, axial_dz))
        box.transformShape(obj.Placement.Matrix)
''', injection)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Dynamic screws injected into cradle!")
